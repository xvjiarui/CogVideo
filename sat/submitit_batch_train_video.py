import argparse
from typing import Sequence
import os
import copy
from datetime import datetime
from omegaconf import OmegaConf
from omegaconf.listconfig import ListConfig
import submitit_train_video

def parse_args():
    parser = argparse.ArgumentParser("Submitit for CogVideo training")
    parser.add_argument("--batch-config", help="Configs to train")

    args, remaining = parser.parse_known_args()
    return args, remaining

def parse_job_cmd_args(job_cmd_args):
    cmd_args = []
    for key, value in job_cmd_args.items():
        cmd_args.append(f"--{key}")
        if not isinstance(value, bool):
            if isinstance(value, ListConfig):
                cmd_args.extend(value)
            else:
                cmd_args.append(str(value))
    return cmd_args

def main():
    args, remaining = parse_args()
    batch_config = OmegaConf.load(args.batch_config)
    for job in batch_config.jobs:
        cmd_args = parse_job_cmd_args(job.cmd_args)
        submitit_args = [
            *cmd_args,
            *remaining,
        ]
        print('Submitting job with args:', submitit_args)
        submitit_train_video.main(submitit_args)


if __name__ == "__main__":
    main()