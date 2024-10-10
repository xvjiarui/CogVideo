import argparse
import os
import torch
import json
import warnings
import omegaconf
from omegaconf import OmegaConf
from sat.helpers import print_rank0
from sat import mpu
from sat.arguments import set_random_seed
from sat.arguments import add_evaluation_args, add_data_args
import torch.distributed

def add_training_args(parser):
    """Training arguments."""

    group = parser.add_argument_group('train', 'training configurations')

    # --------------- Core hyper-parameters --------------- 
    group.add_argument('--experiment-name', type=str, default="",
                       help="The experiment name for summary and checkpoint."
                       "Will load the previous name if mode==pretrain and with --load ")
    group.add_argument('--tag', type=str, default="")
    group.add_argument('--train-iters', type=int, default=None,
                       help='total number of iterations to train over all training runs')
    group.add_argument('--batch-size', type=int, default=4,
                       help='batch size on a single GPU. batch-size * world_size = total batch_size.')
    group.add_argument('--lr', type=float, default=1.0e-4,
                       help='initial learning rate')
    group.add_argument('--mode', type=str,
                       default='pretrain',
                       choices=['pretrain', # from_scratch / load ckpt for continue pretraining.
                                'finetune', # finetuning, auto-warmup 100 iters, new exp name.
                                'inference' # don't train.
                                ],
                       help='what type of task to use, will influence auto-warmup, exp name, iteration')
    group.add_argument('--seed', type=int, default=1234, help='random seed')
    group.add_argument('--zero-stage', type=int, default=0, choices=[0, 1, 2, 3], 
                        help='deepspeed ZeRO stage. 0 means no ZeRO.')

    # ---------------  Optional hyper-parameters --------------- 

    # Efficiency.
    group.add_argument('--checkpoint-activations', action='store_true',
                       help='checkpoint activation to allow for training '
                            'with larger models and sequences. become slow (< 1.5x), save CUDA memory.')
    # Inessential
    group.add_argument('--checkpoint-num-layers', type=int, default=1, 
                       help='chunk size (number of layers) for checkpointing. ')
    group.add_argument('--checkpoint-skip-layers', type=int, default=0,
                       help='skip the last N layers for checkpointing.')
    
    group.add_argument('--fp16', action='store_true',
                       help='Run model in fp16 mode')
    group.add_argument('--bf16', action='store_true',
                       help='Run model in bf16 mode')
    group.add_argument('--gradient-accumulation-steps', type=int, default=1, 
                       help='run optimizer after every gradient-accumulation-steps backwards.')

    group.add_argument('--profiling', type=int, default=-1,
                       help='profiling, -1 means no profiling, otherwise means warmup args.profiling iters then profiling.')
    group.add_argument('--epochs', type=int, default=None,
                       help='number of train epochs')
    group.add_argument('--log-interval', type=int, default=50,
                       help='report interval')
    group.add_argument('--summary-dir', type=str, default="", help="The directory to store the summary")
    group.add_argument('--save-args', action='store_true',
                       help='save args corresponding to the experiment-name')

    # Learning rate & weight decay.
    group.add_argument('--lr-decay-iters', type=int, default=None,
                       help='number of iterations to decay LR over,'
                            ' If None defaults to `--train-iters`*`--epochs`')
    group.add_argument('--lr-decay-style', type=str, default='linear',
                       choices=['constant', 'linear', 'cosine', 'exponential'],
                       help='learning rate decay function')
    group.add_argument('--lr-decay-ratio', type=float, default=0.1)
    
    group.add_argument('--warmup', type=float, default=0.01,
                       help='percentage of data to warmup on (.01 = 1% of all '
                            'training iters). Default 0.01')
    group.add_argument('--weight-decay', type=float, default=0.01,
                       help='weight decay coefficient for L2 regularization')
    
    # model checkpointing
    group.add_argument('--save', type=str, default=None,
                       help='Output directory to save checkpoints to.')
    group.add_argument('--load', type=str, default=None,
                       help='Path to a directory containing a model checkpoint.')
    group.add_argument('--load-extra', type=str, default=None,
                       help='Path to a directory containing extra model checkpoint.')
    group.add_argument('--resume', type=str, default=None,
                       help='Path to resume model training.')
    group.add_argument('--force-train', action='store_true',
                       help='Force training even with missing keys.')
    group.add_argument('--save-interval', type=int, default=5000,
                       help='number of iterations between saves')
    group.add_argument('--no-save-rng', action='store_true',
                       help='Do not save current rng state.')
    group.add_argument('--no-load-rng', action='store_true',
                       help='Do not load rng state when loading checkpoint.')
    group.add_argument('--resume-dataloader', action='store_true',
                       help='Resume the dataloader when resuming training. ') 

    # distributed training related, don't use them.
    group.add_argument('--distributed-backend', default='nccl',
                       help='which backend to use for distributed '
                            'training. One of [gloo, nccl]')
    group.add_argument('--local_rank', type=int, default=None,
                       help='local rank passed from distributed launcher')

    # exit, for testing the first period of a long training
    group.add_argument('--exit-interval', type=int, default=None,
                       help='Exit the program after this many new iterations.')

    group.add_argument('--wandb', action="store_true", help='whether to use wandb')
    group.add_argument('--wandb-project-name', type=str, default="cogvideo",
                       help="The project name in wandb.")
    
    return parser

def add_model_config_args(parser):
    """Model arguments"""

    group = parser.add_argument_group("model", "model configuration")
    group.add_argument("--base", type=str, nargs="*", help="config for input and saving")
    group.add_argument(
        "--opts",
        help="Modify config options at the end of the command. use 'path.key=value'",
        default=None,
        nargs="*",
    )
    group.add_argument(
        "--model-parallel-size", type=int, default=1, help="size of the model parallel. only use if you are an expert."
    )
    group.add_argument("--force-pretrain", action="store_true")
    group.add_argument("--device", type=int, default=-1)
    group.add_argument("--debug", action="store_true")
    group.add_argument("--log-image", type=bool, default=True)

    return parser


def add_sampling_config_args(parser):
    """Sampling configurations"""

    group = parser.add_argument_group("sampling", "Sampling Configurations")
    group.add_argument("--output-dir", type=str, default="samples")
    group.add_argument("--input-dir", type=str, default=None)
    group.add_argument("--input-type", type=str, default="cli")
    group.add_argument("--input-file", type=str, default="input.txt")
    group.add_argument("--final-size", type=int, default=2048)
    group.add_argument("--sdedit", action="store_true")
    group.add_argument("--grid-num-rows", type=int, default=1)
    group.add_argument("--force-inference", action="store_true")
    group.add_argument("--lcm_steps", type=int, default=None)
    group.add_argument("--sampling-num-frames", type=int, default=32)
    group.add_argument("--sampling-fps", type=int, default=8)
    group.add_argument("--only-save-latents", type=bool, default=False)
    group.add_argument("--only-log-video-latents", type=bool, default=False)
    group.add_argument("--latent-channels", type=int, default=32)
    group.add_argument("--image2video", action="store_true")
    group.add_argument("--resume-iter", type=int, default=None, help="Resume from a specific iteration")

    return parser


def get_args(args_list=None, parser=None, init_dist=True):
    """Parse all the args."""
    if parser is None:
        parser = argparse.ArgumentParser(description="sat")
    else:
        assert isinstance(parser, argparse.ArgumentParser)
    parser = add_model_config_args(parser)
    parser = add_sampling_config_args(parser)
    parser = add_training_args(parser)
    parser = add_evaluation_args(parser)
    parser = add_data_args(parser)

    import deepspeed

    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args(args_list)
    # Collect arguments that are using default values
    default_args = {key: value for key, value in vars(args).items() if parser.get_default(key) == value}
    # Collect arguments that were explicitly passed
    passed_args = {key: value for key, value in vars(args).items() if key not in default_args and key != 'base'}
    args = process_config_to_args(args)
    for key in passed_args:
        setattr(args, key, passed_args[key])
    
    if init_dist:
        init_distributed_mode(args)
    return args


def init_distributed_mode(args):
    if not args.train_data:
        print_rank0("No training data specified", level="WARNING")

    assert (args.train_iters is None) or (args.epochs is None), "only one of train_iters and epochs should be set."
    if args.train_iters is None and args.epochs is None:
        args.train_iters = 10000  # default 10k iters
        print_rank0("No train_iters (recommended) or epochs specified, use default 10k iters.", level="WARNING")

    args.cuda = torch.cuda.is_available()

    args.rank = int(os.getenv("RANK", "0"))
    args.world_size = int(os.getenv("WORLD_SIZE", "1"))
    if args.local_rank is None:
        args.local_rank = int(os.getenv("LOCAL_RANK", "0"))  # torchrun

    if args.device == -1:
        if torch.cuda.device_count() == 0:
            args.device = "cpu"
        elif args.local_rank is not None:
            args.device = args.local_rank
        else:
            args.device = args.rank % torch.cuda.device_count()

    if args.local_rank != args.device and args.mode != "inference":
        raise ValueError(
            "LOCAL_RANK (default 0) and args.device inconsistent. "
            "This can only happens in inference mode. "
            "Please use CUDA_VISIBLE_DEVICES=x for single-GPU training. "
        )

    if args.rank == 0:
        print_rank0("using world size: {}".format(args.world_size))

    if args.train_data_weights is not None:
        assert len(args.train_data_weights) == len(args.train_data)

    if args.mode != "inference":  # training with deepspeed
        args.deepspeed = True
        if args.deepspeed_config is None:  # not specified
            deepspeed_config_path = os.path.join(
                os.path.dirname(__file__), "training", f"deepspeed_zero{args.zero_stage}.json"
            )
            with open(deepspeed_config_path) as file:
                args.deepspeed_config = json.load(file)
            override_deepspeed_config = True
        else:
            override_deepspeed_config = False

    assert not (args.fp16 and args.bf16), "cannot specify both fp16 and bf16."

    if args.zero_stage > 0 and not args.fp16 and not args.bf16:
        print_rank0("Automatically set fp16=True to use ZeRO.")
        args.fp16 = True
        args.bf16 = False

    if args.deepspeed:
        if args.checkpoint_activations:
            args.deepspeed_activation_checkpointing = True
        else:
            args.deepspeed_activation_checkpointing = False
        if args.deepspeed_config is not None:
            deepspeed_config = args.deepspeed_config

        if override_deepspeed_config:  # not specify deepspeed_config, use args
            if args.fp16:
                deepspeed_config["fp16"]["enabled"] = True
            elif args.bf16:
                deepspeed_config["bf16"]["enabled"] = True
                deepspeed_config["fp16"]["enabled"] = False
            else:
                deepspeed_config["fp16"]["enabled"] = False
            deepspeed_config["train_micro_batch_size_per_gpu"] = args.batch_size
            deepspeed_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
            optimizer_params_config = deepspeed_config["optimizer"]["params"]
            optimizer_params_config["lr"] = args.lr
            optimizer_params_config["weight_decay"] = args.weight_decay
        else:  # override args with values in deepspeed_config
            if args.rank == 0:
                print_rank0("Will override arguments with manually specified deepspeed_config!")
            if "fp16" in deepspeed_config and deepspeed_config["fp16"]["enabled"]:
                args.fp16 = True
            else:
                args.fp16 = False
            if "bf16" in deepspeed_config and deepspeed_config["bf16"]["enabled"]:
                args.bf16 = True
            else:
                args.bf16 = False
            if "train_micro_batch_size_per_gpu" in deepspeed_config:
                args.batch_size = deepspeed_config["train_micro_batch_size_per_gpu"]
            if "gradient_accumulation_steps" in deepspeed_config:
                args.gradient_accumulation_steps = deepspeed_config["gradient_accumulation_steps"]
            else:
                args.gradient_accumulation_steps = None
            if "optimizer" in deepspeed_config:
                optimizer_params_config = deepspeed_config["optimizer"].get("params", {})
                args.lr = optimizer_params_config.get("lr", args.lr)
                args.weight_decay = optimizer_params_config.get("weight_decay", args.weight_decay)
        args.deepspeed_config = deepspeed_config
    
    # initialize distributed and random seed because it always seems to be necessary.
    initialize_distributed(args)
    args.seed = args.seed + mpu.get_data_parallel_rank()
    set_random_seed(args.seed)


def initialize_distributed(args):
    """Initialize torch.distributed."""
    if torch.distributed.is_initialized():
        if mpu.model_parallel_is_initialized():
            if args.model_parallel_size != mpu.get_model_parallel_world_size():
                raise ValueError(
                    "model_parallel_size is inconsistent with prior configuration."
                    "We currently do not support changing model_parallel_size."
                )
            return False
        else:
            if args.model_parallel_size > 1:
                warnings.warn(
                    "model_parallel_size > 1 but torch.distributed is not initialized via SAT."
                    "Please carefully make sure the correctness on your own."
                )
            mpu.initialize_model_parallel(args.model_parallel_size)
        return True
    # the automatic assignment of devices has been moved to arguments.py
    if args.device == "cpu":
        pass
    else:
        torch.cuda.set_device(args.device)
    # Call the init process
    init_method = "tcp://"
    args.master_ip = os.getenv("MASTER_ADDR", "localhost")

    if args.world_size == 1:
        from sat.helpers import get_free_port

        default_master_port = str(get_free_port())
    else:
        default_master_port = "6000"
    args.master_port = os.getenv("MASTER_PORT", default_master_port)
    init_method += args.master_ip + ":" + args.master_port
    torch.distributed.init_process_group(
        backend=args.distributed_backend, world_size=args.world_size, rank=args.rank, init_method=init_method
    )

    # Set the model-parallel / data-parallel communicators.
    mpu.initialize_model_parallel(args.model_parallel_size)

    # Set vae context parallel group equal to model parallel group
    from sgm.util import set_context_parallel_group, initialize_context_parallel

    if args.model_parallel_size <= 2:
        set_context_parallel_group(args.model_parallel_size, mpu.get_model_parallel_group())
    else:
        initialize_context_parallel(2)
    # mpu.initialize_model_parallel(1)
    # Optional DeepSpeed Activation Checkpointing Features
    if args.deepspeed:
        import deepspeed

        deepspeed.init_distributed(
            dist_backend=args.distributed_backend, world_size=args.world_size, rank=args.rank, init_method=init_method
        )
        # # It seems that it has no negative influence to configure it even without using checkpointing.
        # deepspeed.checkpointing.configure(mpu, deepspeed_config=args.deepspeed_config, num_checkpoints=args.num_layers)
    else:
        # in model-only mode, we don't want to init deepspeed, but we still need to init the rng tracker for model_parallel, just because we save the seed by default when dropout.
        try:
            import deepspeed
            from deepspeed.runtime.activation_checkpointing.checkpointing import (
                _CUDA_RNG_STATE_TRACKER,
                _MODEL_PARALLEL_RNG_TRACKER_NAME,
            )

            _CUDA_RNG_STATE_TRACKER.add(_MODEL_PARALLEL_RNG_TRACKER_NAME, 1)  # default seed 1
        except Exception as e:
            from sat.helpers import print_rank0

            print_rank0(str(e), level="DEBUG")

    return True


def process_config_to_args(args):
    """Fetch args from only --base"""

    configs = [OmegaConf.load(cfg) for cfg in args.base]
    config = OmegaConf.merge(*configs)
    if args.opts:
        update_conf = OmegaConf.from_dotlist(args.opts)
        config = OmegaConf.merge(config, update_conf)

    args_config = config.pop("args", OmegaConf.create())
    for key in args_config:
        if isinstance(args_config[key], omegaconf.DictConfig) or isinstance(args_config[key], omegaconf.ListConfig):
            arg = OmegaConf.to_object(args_config[key])
        else:
            arg = args_config[key]
        if hasattr(args, key):
            setattr(args, key, arg)

    if "model" in config:
        model_config = config.pop("model", OmegaConf.create())
        args.model_config = model_config
    if "deepspeed" in config:
        deepspeed_config = config.pop("deepspeed", OmegaConf.create())
        args.deepspeed_config = OmegaConf.to_object(deepspeed_config)
    if "data" in config:
        data_config = config.pop("data", OmegaConf.create())
        args.data_config = data_config

    return args

# def set_configs_to_args(args, args_config):
#     for key in args_config:
#         if isinstance(args_config[key], omegaconf.DictConfig) or isinstance(args_config[key], omegaconf.ListConfig):
#             arg = OmegaConf.to_object(args_config[key])
#         else:
#             arg = args_config[key]
#         assert hasattr(args, key), f"args has no attribute {key}"
#         setattr(args, key, arg)
    
#     return args