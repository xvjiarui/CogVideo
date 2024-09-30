"""
python tools/caption/recap_video.py data/tandj_hf/videos data/tandj_hf/metadata/default_stats.jsonl data/tandj_hf/metadata/default_recap.jsonl
"""
import argparse
import json
import os
import socket
from tqdm import tqdm
import requests
from concurrent.futures import ThreadPoolExecutor
from threading import Semaphore

def get_prediction_with_semaphore(dic, video_folder, url, semaphores):
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

def get_url(index, server_urls):
    return server_urls[index % len(server_urls)]

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Process CSV and video folder.')
    parser.add_argument('video_folder', type=str, help='Input video folder')
    parser.add_argument('input_file', type=str, help='Input JOSNL file')
    parser.add_argument('output_file', type=str, help='Output JOSNL file')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, 'recap_hosts.txt'), 'r') as file:
        nodes = file.readlines()
    nodes = [node.strip() for node in nodes]
    print(nodes)
    server_urls = []
    for node in nodes:
        ip = socket.gethostbyname(node)
        server_urls.extend([f'http://{ip}:{port}/video_qa' for port in range(5000, 5008)])
    print(server_urls)
    # Create a semaphore with a maximum number of concurrent requests
    # Create a dictionary of semaphores, one for each URL
    semaphores = {url: Semaphore(1) for url in server_urls}


    dic_list = []
    for line in open(args.input_file, 'r'):
        dic_list.append(json.loads(line))

    with ThreadPoolExecutor(max_workers=len(server_urls)) as executor:
        # futures = [executor.submit(get_prediction, dic, args.video_folder, get_url(idx)) for idx, dic in enumerate(dic_list)]
        futures = [executor.submit(get_prediction_with_semaphore, dic, args.video_folder, get_url(idx, server_urls), semaphores) for idx, dic in enumerate(dic_list)]
        with open(args.output_file, 'w') as out_file:
            for future in tqdm(futures, desc="Processing videos"):
                dic = future.result()
                out_file.write(json.dumps(dic) + '\n')
                out_file.flush()

if __name__ == '__main__':
    main()