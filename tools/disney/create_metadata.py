"""
python tools/disney/create_metadata.py data/disney/videos.txt data/disney/prompt.txt data/disney/metadata/default.jsonl --root videos
"""
import argparse
import csv
import json
import os
from tqdm import tqdm

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Process video folder.')
    parser.add_argument('video_file', type=str, help='Input video file')
    parser.add_argument('prompt_file', type=str, help='Input prompt file')
    parser.add_argument('output_file', type=str, help='Output JSONL file')
    parser.add_argument('--root', type=str, default='', help='Root folder for the video files')
    args = parser.parse_args()

    # Iterate over the video folder with a progress bar
    pairs = []
    with open(args.video_file, 'r') as video_file:
        video_files = video_file.readlines()
    with open(args.prompt_file, 'r') as prompt_file:
        prompts = prompt_file.readlines()
    for video_file, prompt in zip(video_files, prompts):
        pairs.append({'path': os.path.relpath(video_file.strip(), args.root), 'text': prompt.strip()})

    # Write to the JSONL file
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, mode='w') as jsonlfile:
        for pair in pairs:
            jsonlfile.write(json.dumps(pair) + '\n')

if __name__ == '__main__':
    main()