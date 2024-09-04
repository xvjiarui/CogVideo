import os
import math
import time
import argparse
from typing import List, Union
from tqdm import tqdm
from omegaconf import ListConfig
import imageio

import torch
import numpy as np
from einops import rearrange
import torchvision.transforms as TT

from sat.model.base_model import get_model
from sat import mpu
    
from diffusion_video import SATVideoDiffusionEngine
from arguments import get_args


def save_video_as_grid_and_mp4(video_batch: torch.Tensor, save_path: str, video_name: str, fps: int = 5, args=None, key=None):
    os.makedirs(save_path, exist_ok=True)

    for i, vid in enumerate(video_batch):
        gif_frames = []
        for frame in vid:
            frame = rearrange(frame, "c h w -> h w c")
            frame = (255.0 * frame).cpu().numpy().astype(np.uint8)
            gif_frames.append(frame)
        now_save_path = os.path.join(save_path, f"{i:06d}_{video_name}.mp4")
        with imageio.get_writer(now_save_path, fps=fps) as writer:
            for frame in gif_frames:
                writer.append_data(frame)


def enc_dec_main(args, model_cls):
    if isinstance(model_cls, type):
        model = get_model(args, model_cls)
    else:
        model = model_cls

    video_reader = imageio.get_reader(args.video_path, "ffmpeg")
    fps = video_reader.get_meta_data()["fps"]

    frames = [TT.ToTensor()(frame) for frame in video_reader]

    print('Frames:', len(frames))
    video_reader.close()

    device = model.device
    dtype = model.dtype
    frames_tensor = torch.stack(frames).to(device).permute(1, 0, 2, 3).unsqueeze(0).to(dtype).contiguous()
    frames_tensor = frames_tensor * 2.0 - 1.0
    print('Frames tensor:', frames_tensor.shape)
    with torch.no_grad():
        start_time = time.time()
        encoded_frames = model.encode_first_stage(frames_tensor, None)
        print('Encoding Time:', time.time() - start_time)
        print('Encoded Frames:', encoded_frames.shape)
        start_time = time.time()
        recon = model.decode_first_stage(encoded_frames)
        print('Decoding Time:', time.time() - start_time)
        print('Decoded Frames:', recon.shape)
        samples_x = recon.permute(0, 2, 1, 3, 4).contiguous()
        samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0).cpu()

        save_path = args.output_dir
        if mpu.get_model_parallel_rank() == 0:
            save_video_as_grid_and_mp4(samples, save_path, video_name=os.path.splitext(os.path.basename(args.video_path))[0], fps=fps)


if __name__ == "__main__":
    py_parser = argparse.ArgumentParser(add_help=False)
    group = py_parser.add_argument_group("enc_dec_video", "Arguments for encoding and decoding video.")
    group.add_argument("--video_path", type=str, help="The path to the video file (for encoding)")

    known, args_list = py_parser.parse_known_args()

    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    del args.deepspeed_config
    args.model_config.first_stage_config.params.cp_size = 1
    args.model_config.network_config.params.transformer_args.model_parallel_size = 1
    args.model_config.network_config.params.transformer_args.checkpoint_activations = False
    args.model_config.loss_fn_config.params.sigma_sampler_config.params.uniform_sampling = False

    enc_dec_main(args, model_cls=SATVideoDiffusionEngine)
