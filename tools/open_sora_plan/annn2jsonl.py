"""
Convert the annotation json to jsonl

Example:
 python tools/open_sora_plan/annn2jsonl.py data/open-sora-plan-v110/anno_jsons/video_pixabay_65f_601513.json data/open-sora-plan-v110/pixabay_v2/ data/open-sora-plan-v110/metadata/pixabay_v2/video_pixabay_65f_601513.jsonl
 python tools/open_sora_plan/annn2jsonl.py data/open-sora-plan-v110/anno_jsons/video_pixabay_513f_51483.json data/open-sora-plan-v110/pixabay_v2/ data/open-sora-plan-v110/metadata/pixabay_v2/video_pixabay_513f_51483.jsonl

"""

import argparse
import json
import os
from tqdm import tqdm

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Process JSONL and video folder.')
    parser.add_argument('json_file', type=str, help='Input JSON file')
    parser.add_argument('video_folder', type=str, help='Input video folder')
    parser.add_argument('output_file', type=str, help='Output JSONL file')
    parser.add_argument('--root', type=str, default='', help='Root folder for the video files')
    args = parser.parse_args()

    if not args.root:
        args.root = args.video_folder

    # Read the CSV file into a dictionary
    path2data_dict = dict()
    with open(args.json_file, mode='r', newline='') as jsonfile:
        raw_data = json.load(jsonfile)
        for dic in tqdm(raw_data, desc="Reading JSON"):
            # NOTE: the video file is renamed to the format of {video_name}_resize1080p.mp4
            path = dic.pop('path').replace(".mp4", "_resize1080p.mp4")
            if path not in path2data_dict:
                path2data_dict[path] = []
            path2data_dict[path].append(dic)
    print(f'{len(path2data_dict)} videos in the json file')
    # Count the total number of dictionaries in path2data_dict
    total_dic_count = sum(len(data_list) for data_list in path2data_dict.values())
    print(f'Total number of dictionaries in path2data_dict: {total_dic_count}')

    # Iterate over the video folder with a progress bar
    jsonl_rows = []
    for root, _, files in tqdm(os.walk(args.video_folder), desc="Processing videos", total=len(list(os.walk(args.video_folder)))):
        for video_file in files:
            video_path = os.path.join(root, video_file)
            if video_file in path2data_dict:
                data_list = path2data_dict[video_file]
                new_path = os.path.relpath(video_path, args.root)
                jsonl_rows.append({
                    'path': new_path,
                    'caps': data_list
                })
    print(f'{len(jsonl_rows)} videos processed')

    # Write to the JSONL file
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, mode='w') as jsonlfile:
        for dic in jsonl_rows:
            jsonlfile.write(json.dumps(dic) + '\n')

if __name__ == '__main__':
    main()