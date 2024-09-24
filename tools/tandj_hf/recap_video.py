"""
python tools/tandj_hf/recap_video.py data/tandj_hf/videos data/tandj_hf/metadata/default_stats.jsonl data/tandj_hf/metadata/default_recap.jsonl
"""
import argparse
import json
import os
import subprocess
from tqdm import tqdm
import requests
from concurrent.futures import ThreadPoolExecutor
from threading import Semaphore

URLS = [f'http://127.0.0.1:{port}/video_qa' for port in range(5000, 5008)]
# URLS = []
# URLS += [f'http://10.49.163.205:{port}/video_qa' for port in range(5000, 5008)]
print(URLS)

# Create a semaphore with a maximum number of concurrent requests
# Create a dictionary of semaphores, one for each URL
semaphores = {url: Semaphore(1) for url in URLS}

def get_prediction_with_semaphore(dic, video_folder, url):
    with semaphores[url]:
        return get_prediction(dic, video_folder, url)

def get_prediction(dic, video_folder, url):
    video_path = os.path.join(video_folder, dic['path'])
    title = os.path.splitext(os.path.basename(video_path))[0]
    # Extract the episode title from the video file name
    import re
    
    # Use regex to extract the episode title
    match = re.search(r'Tom and Jerry - \d+ - (.+?) \[\d+\]', title)
    if match:
        episode_title = match.group(1)
    else:
        episode_title = ""
    
    # Update the question with the episode title
    # if episode_title:
    #     question = f"Here is a clip from the Tom and Jerry episode '{episode_title}'. Describe what's happening in this scene in detail."
    # else:
    #     question = "Here is a clip from Tom and Jerry. Describe what's happening in this scene in detail."
    question = "@Caption"
        
    # question = "Describe this video in detail."
    temperature = 0.2
    files = {'video': open(video_path, 'rb')}
    data = {
        'question': question, 
        'temperature': temperature,
        # 'max_new_tokens': 2048,
        # 'top_k': 1,
        # 'do_sample': False,
        # 'top_p': 0.1,
        'max_new_tokens': 256,
        'top_k': 50,
        'do_sample': True,
        'top_p': 0.9,
    }
    response = requests.post(url, files=files, data=data)
    dic['text'] = response.json()["answer"]

    return dic

def get_url(index):
    return URLS[index % len(URLS)]

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Process CSV and video folder.')
    parser.add_argument('video_folder', type=str, help='Input video folder')
    parser.add_argument('input_file', type=str, help='Input JOSNL file')
    parser.add_argument('output_file', type=str, help='Output JOSNL file')
    args = parser.parse_args()

    dic_list = []
    for line in open(args.input_file, 'r'):
        dic_list.append(json.loads(line))

    with ThreadPoolExecutor(max_workers=len(URLS)) as executor:
        # futures = [executor.submit(get_prediction, dic, args.video_folder, get_url(idx)) for idx, dic in enumerate(dic_list)]
        futures = [executor.submit(get_prediction_with_semaphore, dic, args.video_folder, get_url(idx)) for idx, dic in enumerate(dic_list)]
        with open(args.output_file, 'w') as out_file:
            for future in tqdm(futures, desc="Processing videos"):
                dic = future.result()
                out_file.write(json.dumps(dic) + '\n')
                out_file.flush()

if __name__ == '__main__':
    main()