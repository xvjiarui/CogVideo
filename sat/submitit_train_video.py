import argparse
import os
import copy
from datetime import datetime, timedelta

import submitit
import torch
import train_video_oci
# from torch.distributed.run import main as torchrun


def parse_args(input_args=None):
    parser = argparse.ArgumentParser("Submitit for CogVideo training")
    parser.add_argument("--ngpus", default=8, type=int, help="Number of gpus to request on each node")
    parser.add_argument("--nodes", default=2, type=int, help="Number of nodes to request")
    parser.add_argument("--timeout", default=240, type=int, help="Duration of the job")
    parser.add_argument("--account", "-A", default="nvr_lpr_nvgptvision", type=str, 
                        choices=["nvr_lpr_nvgptvision", "nvr_lpr_dataefficientml", "nvr_nxp_visionconferencing"], help="Slurm account")
    # parser.add_argument("--job_dir", default="", type=str, help="Job dir. Leave empty for automatic.")

    parser.add_argument("--partition", default="grizzly,polar,polar2,polar3,polar4", type=str, help="Partition where to submit")
    parser.add_argument("--nodelist", default=None, type=str, help='specify node list')
    parser.add_argument('--comment', default="", type=str,
                        help='Comment to pass to scheduler, e.g. priority message')
    args, remaining = parser.parse_known_args(input_args)
    return args, remaining


class Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):
        import train_video_oci
        import os
        import signal

        self._setup_gpu_args()
        # NOTE(xvjiarui): deepcopy is necessary to avoid the args being changed in the main function
        args = copy.deepcopy(self.args)
        train_video_oci.main(args)
        # try:
        #     train_video_oci.main(args)
        # except torch.distributed.DistStoreError as e:
        #     print(f"Caught exception {e}, sending SIGUSR2 to self to trigger checkpointing")
        #     os.kill(os.getpid(), signal.SIGUSR2)

    def checkpoint(self):
        import submitit

        print("Requeuing ", self.args)
        # self.args.resume = self.args.ckpt_dir
        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        dist_env = submitit.helpers.TorchDistributedEnvironment().export()
        print(f"master: {dist_env.master_addr}:{dist_env.master_port}")
        print(f"rank: {dist_env.rank}")
        print(f"world size: {dist_env.world_size}")
        print(f"local rank: {dist_env.local_rank}")
        print(f"local world size: {dist_env.local_world_size}")

def main(input_args=None):
    args, remaining = parse_args(input_args)
    print(f'train_video_oci: {remaining}')
    train_args = train_video_oci.parse_args(remaining)

    num_gpus_per_node = args.ngpus
    nodes = args.nodes
    timeout_min = args.timeout

    partition = args.partition
    kwargs = {}
    if args.comment:
        kwargs['slurm_comment'] = args.comment


    # slurm_additional_parameters = {'gres-flags': 'enforce-binding'}
    # if args.nodelist is not None:
    #     slurm_additional_parameters['nodelist'] = args.nodelist

    output_dir = train_args.save
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
        stderr_to_stdout=True,
        # slurm_additional_parameters=slurm_additional_parameters,
        **kwargs
    )

    executor.update_parameters(name=train_args.experiment_name)


    trainer = Trainer(train_args)
    job = executor.submit(trainer)

    print("Submitted job_id:", job.job_id)


if __name__ == "__main__":
    main()
