import argparse
import csv
import json
import os
from tqdm import tqdm

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Process CSV and video folder.')
    parser.add_argument('csv_file', type=str, help='Input CSV file')
    parser.add_argument('video_folder', type=str, help='Input video folder')
    parser.add_argument('output_file', type=str, help='Output JSONL file')
    parser.add_argument('--root', type=str, default='data/openvid/videos', help='Root folder for the video files')
    args = parser.parse_args()

    # Read the CSV file into a dictionary
    text_dict = {}
    with open(args.csv_file, mode='r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in tqdm(reader, desc="Reading CSV"):
            text_dict[row['path']] = row['text']

    # Iterate over the video folder with a progress bar
    pairs = []
    for root, _, files in tqdm(os.walk(args.video_folder), desc="Processing videos", total=len(list(os.walk(args.video_folder)))):
        for video_file in files:
            video_path = os.path.join(root, video_file)
            if video_file in text_dict:
                pairs.append({'path': os.path.relpath(video_path, args.root), 'text': text_dict[video_file]})

    # Write to the JSONL file
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, mode='w') as jsonlfile:
        for pair in pairs:
            jsonlfile.write(json.dumps(pair) + '\n')

if __name__ == '__main__':
    main()