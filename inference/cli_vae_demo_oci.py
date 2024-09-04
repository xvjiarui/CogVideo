"""
This script is designed to demonstrate how to use the CogVideoX-2b VAE model for video encoding and decoding.
It allows you to encode a video into a latent representation, decode it back into a video, or perform both operations sequentially.
Before running the script, make sure to clone the CogVideoX Hugging Face model repository and set the `{your local diffusers path}` argument to the path of the cloned repository.

Command 1: Encoding Video
Encodes the video located at ../resources/videos/1.mp4 using the CogVideoX-2b VAE model.
Memory Usage: ~34GB of GPU memory for encoding.
If you do not have enough GPU memory, we provide a pre-encoded tensor file (encoded.pt) in the resources folder and you can still run the decoding command.
$ python cli_vae_demo.py --model_path {your local diffusers path}/CogVideoX-2b/vae/ --video_path ../resources/videos/1.mp4 --mode encode

Command 2: Decoding Video

Decodes the latent representation stored in encoded.pt back into a video.
Memory Usage: ~19GB of GPU memory for decoding.
$ python cli_vae_demo.py --model_path {your local diffusers path}/CogVideoX-2b/vae/ --encoded_path ./encoded.pt --mode decode

Command 3: Encoding and Decoding Video
Encodes the video located at ../resources/videos/1.mp4 and then immediately decodes it.
Memory Usage: 34GB for encoding + 19GB for decoding (sequentially).
$ python cli_vae_demo.py --model_path {your local diffusers path}/CogVideoX-2b/vae/ --video_path ../resources/videos/1.mp4 --mode both
"""

import argparse
import os
import math
from tqdm import tqdm
import torch
import imageio
import numpy as np
from diffusers import AutoencoderKLCogVideoX
from torchvision import transforms


def encode_video(model, video_path, dtype, device):
    """
    Loads a pre-trained AutoencoderKLCogVideoX model and encodes the video frames.

    Parameters:
    - model_path (str): The path to the pre-trained model.
    - video_path (str): The path to the video file.
    - dtype (torch.dtype): The data type for computation.
    - device (str): The device to use for computation (e.g., "cuda" or "cpu").

    Returns:
    - torch.Tensor: The encoded video frames.
    """
    # model = AutoencoderKLCogVideoX.from_pretrained(model_path, torch_dtype=dtype).to(device)
    video_reader = imageio.get_reader(video_path, "ffmpeg")

    frames = [transforms.ToTensor()(frame) for frame in video_reader]
    # transform = transforms.Compose([
    #     transforms.ToTensor(), 
    #     transforms.Resize(480), 
    #     # transforms.CenterCrop((480, 720)),
    #     ])
    # frames = [transform(frame) for frame in video_reader]

    print('Frames:', len(frames))
    # frames = frames[:49]
    video_reader.close()

    frames_tensor = torch.stack(frames).to(device).permute(1, 0, 2, 3).unsqueeze(0).to(dtype)
    print('mean:', frames_tensor.mean(), 'std:', frames_tensor.std())
    frames_tensor = frames_tensor * 2.0 - 1.0
    print('mean:', frames_tensor.mean(), 'std:', frames_tensor.std())

    slice_encode = True
    print('slice_encode:', slice_encode)

    with torch.no_grad():
        if slice_encode:
            encoded_frames = []
            # for i in range(6):  # 6 seconds
            tile_window = 16
            # for i in tqdm(range(len(frames)//8), desc='Encoding Frames'):
            #     start_frame, end_frame = (0, 9) if i == 0 else (8 * i + 1, 8 * i + 9)
            for i in tqdm(range(len(frames)//tile_window), desc=f'Encoding Frames with tile window {tile_window}'):
                start_frame, end_frame = (0, tile_window + 1) if i == 0 else (tile_window * i + 1, tile_window * (i +1) + 1)
                # print('Start Frame:', start_frame, 'End Frame:', end_frame)
                current_frames = model.encode(frames_tensor[:, :, start_frame:end_frame])[0].sample()
                encoded_frames.append(current_frames)
            model._clear_fake_context_parallel_cache()
            encoded_frames = torch.cat(encoded_frames, dim=2)
        else:
            encoded_frames = model.encode(frames_tensor)[0].sample()
        print('Encoded Frames:', encoded_frames.shape)
            
    return encoded_frames


def decode_video(model, encoded_tensor_path, dtype, device):
    """
    Loads a pre-trained AutoencoderKLCogVideoX model and decodes the encoded video frames.

    Parameters:
    - model_path (str): The path to the pre-trained model.
    - encoded_tensor_path (str): The path to the encoded tensor file.
    - dtype (torch.dtype): The data type for computation.
    - device (str): The device to use for computation (e.g., "cuda" or "cpu").

    Returns:
    - torch.Tensor: The decoded video frames.
    """
    # model = AutoencoderKLCogVideoX.from_pretrained(model_path, torch_dtype=dtype).to(device)
    if isinstance(encoded_tensor_path, torch.Tensor):
        encoded_frames = encoded_tensor_path
    else:
        encoded_frames = torch.load(encoded_tensor_path, weights_only=True).to(device).to(dtype)
    print('Encoded Frames:', encoded_frames.shape)
    with torch.no_grad():
        decoded_frames = []
        # for i in range(6):  # 6 seconds
        # for i in tqdm(range(encoded_frames.shape[2]//2), desc='Decoding Frames'):
        #     start_frame, end_frame = (0, 3) if i == 0 else (2 * i + 1, 2 * i + 3)
        tile_window = 4
        for i in tqdm(range(encoded_frames.shape[2]//tile_window), desc=f'Decoding Frames with tile window {tile_window}'):
            start_frame, end_frame = (0, tile_window + 1) if i == 0 else (tile_window * i + 1, tile_window * (i +1) + 1)
            # print('Start Frame:', start_frame, 'End Frame:', end_frame)
            current_frames = model.decode(encoded_frames[:, :, start_frame:end_frame]).sample
            decoded_frames.append(current_frames)
        model._clear_fake_context_parallel_cache()

        decoded_frames = torch.cat(decoded_frames, dim=2)
    print('mean:', decoded_frames.mean(), 'std:', decoded_frames.std())
    decoded_frames = (decoded_frames + 1.0) / 2.0
    print('mean:', decoded_frames.mean(), 'std:', decoded_frames.std())
    print('Decoded Frames:', decoded_frames.shape)
    return decoded_frames


def save_video(tensor, output_path, fps=30):
    """
    Saves the video frames to a video file.

    Parameters:
    - tensor (torch.Tensor): The video frames tensor.
    - output_path (str): The path to save the output video.
    """
    frames = tensor[0].squeeze(0).permute(1, 2, 3, 0).cpu().numpy()
    frames = np.clip(frames, 0, 1) * 255
    frames = frames.astype(np.uint8)

    writer = imageio.get_writer(output_path + "/output.mp4", fps=fps)
    for frame in frames:
        writer.append_data(frame)
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CogVideoX encode/decode demo")
    parser.add_argument("--model_path", type=str, required=True, help="The path to the CogVideoX model")
    parser.add_argument("--video_path", type=str, help="The path to the video file (for encoding)")
    parser.add_argument("--encoded_path", type=str, help="The path to the encoded tensor file (for decoding)")
    parser.add_argument("--output_path", type=str, default=".", help="The path to save the output file")
    parser.add_argument(
        "--mode", type=str, choices=["encode", "decode", "both"], required=True, help="Mode: encode, decode, or both"
    )
    parser.add_argument(
        "--dtype", type=str, default="float16", help="The data type for computation (e.g., 'float16' or 'float32')"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="The device to use for computation (e.g., 'cuda' or 'cpu')"
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    model = AutoencoderKLCogVideoX.from_pretrained(args.model_path, torch_dtype=dtype).to(device)
    # model.enable_tiling()

    if args.mode == "encode":
        assert args.video_path, "Video path must be provided for encoding."
        encoded_output = encode_video(model, args.video_path, dtype, device)
        torch.save(encoded_output, args.output_path + "/encoded.pt")
        print(f"Finished encoding the video to a tensor, save it to a file at {args.output_path}/encoded.pt")
    elif args.mode == "decode":
        assert args.encoded_path, "Encoded tensor path must be provided for decoding."
        decoded_output = decode_video(model, args.encoded_path, dtype, device)
        save_video(decoded_output, args.output_path)
        print(f"Finished decoding the video and saved it to a file at {args.output_path}/output.mp4")
    elif args.mode == "both":
        assert args.video_path, "Video path must be provided for encoding."
        encoded_output = encode_video(model, args.video_path, dtype, device)
        # torch.save(encoded_output, args.output_path + "/encoded.pt")
        decoded_output = decode_video(model, encoded_output, dtype, device)
        os.makedirs(args.output_path, exist_ok=True)
        video_reader = imageio.get_reader(args.video_path, "ffmpeg")
        fps = video_reader.get_meta_data()["fps"]
        save_video(decoded_output, args.output_path, fps=fps)
