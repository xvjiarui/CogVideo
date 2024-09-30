"""
python tools/tandj/filter_episode.py data/tandj/metadata/min240_thr1/tandj_stats.jsonl data/tandj/metadata/min240_thr1/tandj_stats_ep114.jsonl
"""
import argparse
import json
import os
import subprocess
import re

def extract_video_id(video_name):
    # Extract episode ID from the video name using regular expression
    
    # Extract episode ID
    episode_id = None
    match = re.search(r' - (\d+) - ', video_name)
    if match:
        episode_id = int(match.group(1))
    
    # Extract scene ID
    scene_id = None
    match = re.search(r'_scene-(\d+)', video_name)
    if match:
        scene_id = int(match.group(1))
    return episode_id, scene_id


def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Process CSV and video folder.')
    parser.add_argument('input_file', type=str, help='Input JOSNL file')
    parser.add_argument('output_file', type=str, help='Output JOSNL file')
    args = parser.parse_args()

    dic_list = []
    for line in open(args.input_file, 'r'):
        dic_list.append(json.loads(line))
    
    with open(args.output_file, 'w') as out_file:
        for dic in dic_list:
            episode_id, scene_id = extract_video_id(dic['path'])
            if episode_id <= 114:
                out_file.write(json.dumps(dic) + '\n')
                out_file.flush()


if __name__ == '__main__':
    main()