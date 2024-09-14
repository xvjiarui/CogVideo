import argparse
import os
import copy
from datetime import datetime
from omegaconf import OmegaConf

import submitit
import sample_video_oci
# from torch.distributed.run import main as torchrun


def parse_args():
    parser = argparse.ArgumentParser("Submitit for CogVideo training")
    parser.add_argument("--config-list", nargs="+", help="Configs to sample")
    parser.add_argument("--common-configs", nargs="+", help="Common configs for all the configs")
    parser.add_argument("--suffix", default="checkpoints", type=str, help="Suffix for the checkpoint folder")
    parser.add_argument("--train-output-dir", default="output/train", type=str, help="Output directory for the training")
    parser.add_argument("--sample-output-dir", default="output/sample", type=str, help="Output directory for the sampling")
    parser.add_argument("--tag", default="debug", type=str, help="Tag for the experiment")
    parser.add_argument("--ngpus", default=1, type=int, help="Number of gpus to request on each node")
    parser.add_argument("--nodes", default=1, type=int, help="Number of nodes to request")
    parser.add_argument("--timeout", default=240, type=int, help="Duration of the job")
    parser.add_argument("--account", "-A", default="nvr_lpr_nvgptvision", type=str, 
                        choices=["nvr_lpr_nvgptvision", "nvr_lpr_dataefficientml", "nvr_nxp_visionconferencing"], help="Slurm account")

    parser.add_argument("--partition", default="interactive,interactive_singlenode,batch_singlenode,grizzly,polar,polar2,polar3,polar4", type=str, help="Partition where to submit")
    parser.add_argument("--nodelist", default=None, type=str, help='specify node list')
    parser.add_argument('--comment', default="", type=str,
                        help='Comment to pass to scheduler, e.g. priority message')
    # return parser.parse_args()
    args, remaining = parser.parse_known_args()
    return args, remaining


class Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):
        import sample_video_oci

        self._setup_gpu_args()
        # NOTE(xvjiarui): deepcopy is necessary to avoid the args being changed in the main function
        args = copy.deepcopy(self.args)
        sample_video_oci.main(args)

    def checkpoint(self):
        import submitit

        print("Requeuing ", self.args)
        # self.args.resume = self.args.ckpt_dir
        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        job_env = submitit.JobEnvironment()

def get_latest_iter_num(ckpt_dir):
    if not os.path.isdir(ckpt_dir):
        return -1
    # Read the latest iteration number from the "latest" file
    latest_file = os.path.join(ckpt_dir, "latest")
    if not os.path.isfile(latest_file):
        return -1
    
    with open(latest_file, "r") as f:
        iter_num = int(f.read().strip())
        return iter_num
    

def main():
    args, remaining = parse_args()

    # Check if "bases" is in any config
    new_config_list = []
    new_common_configs = []
    for config_path in args.config_list:
        config = OmegaConf.load(config_path)
        if 'bases' in config:
            new_config_list.extend(config.bases)
        if 'common_configs' in config:
            new_common_configs.extend(config.common_configs)

    if new_config_list:
        args.config_list = new_config_list
    if new_common_configs:
        assert not args.common_configs, "common_configs and args.common_configs should not be both set"
        args.common_configs = new_common_configs

    for config in args.config_list:
        config_name = os.path.basename(config).split(".")[0]
        experiment_name = f"sample:{config_name}"
        ckpt_dir = os.path.join(args.train_output_dir, f"{config_name}_{args.suffix}")

        resume_iter = get_latest_iter_num(ckpt_dir)
        assert resume_iter > 0, f"No checkpoint found in {ckpt_dir}"
        print(f"Sampling from checkpoint {ckpt_dir} at iteration {resume_iter}")

        sample_dir = os.path.join(args.sample_output_dir, f"{config_name}_{args.suffix}", args.tag, f"iter_{resume_iter}")

        cur_remaining = [
            '--base', config, *args.common_configs,
            '--resume', ckpt_dir,
            '--output-dir', sample_dir
        ] + remaining
        sample_args = sample_video_oci.parse_args(cur_remaining)

        num_gpus_per_node = args.ngpus
        nodes = args.nodes
        timeout_min = args.timeout

        partition = args.partition
        kwargs = {}
        if args.comment:
            kwargs['slurm_comment'] = args.comment

        output_dir = sample_args.output_dir
        # Note that the folder will depend on the job_id, to easily track experiments
        executor = submitit.AutoExecutor(folder=os.path.join(output_dir, "submitit_logs", f"%j-{datetime.now().strftime('%Y%m%d-%H%M%S')}"), slurm_max_num_timeout=30)

        executor.update_parameters(
            mem_gb=128 * num_gpus_per_node,
            gpus_per_node=num_gpus_per_node,
            tasks_per_node=num_gpus_per_node,  # one task per GPU
            cpus_per_task=31,
            nodes=nodes,
            timeout_min=timeout_min,  # max is 60 * 24 * 7
            # Below are cluster dependent parameters
            slurm_account=args.account,
            slurm_partition=partition,
            slurm_signal_delay_s=120,
            slurm_setup=[
                f"export WANDB_API_KEY={os.environ['WANDB_API_KEY']}"
            ],
            # slurm_additional_parameters=slurm_additional_parameters,
            **kwargs
        )

        executor.update_parameters(name=experiment_name)


        trainer = Trainer(sample_args)
        job = executor.submit(trainer)

        print("Submitted job_id:", job.job_id)


if __name__ == "__main__":
    main()
