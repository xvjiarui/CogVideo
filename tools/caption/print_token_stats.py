"""
python tools/caption/print_token_stats.py data/tandj/metadata/g6s/tandj_recap.jsonl data/tandj/metadata/g6s/tandj_recap_token_stats.jsonl
"""
import argparse
import json
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from transformers import T5Tokenizer
import matplotlib.pyplot as plt
import numpy as np

# Initialize T5Tokenizer
tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-xxl")

def process_video(dic):
    # Tokenize the text and get token stats
    text = dic['text']
    tokens = tokenizer.encode(text, truncation=False)
    dic['num_tokens'] = len(tokens)

    return dic

def plot_histogram(token_counts, output_file):
    plt.figure(figsize=(16, 10))
    counts, bins, patches = plt.hist(token_counts, bins=50, edgecolor='black')
    plt.title('Histogram of Token Counts')
    plt.xlabel('Number of Tokens')
    plt.ylabel('Frequency')
    
    # Add frequency on top of each bar using built-in function
    plt.bar_label(patches, rotation=90, padding=3)
    
    # Add vertical lines for important percentiles
    percentiles = [50, 90, 95, 99]
    colors = ['red', 'green', 'blue', 'purple']
    for p, c in zip(percentiles, colors):
        value = np.percentile(token_counts, p)
        plt.axvline(value, color=c, linestyle='dashed', linewidth=1, label=f'{p}th percentile')
    
    # Add vertical line for token number 226
    token_226_percentile = (np.sum(np.array(token_counts) <= 226) / len(token_counts)) * 100
    plt.axvline(226, color='orange', linestyle='dashed', linewidth=2, 
                label=f'226 tokens ({token_226_percentile:.2f}th percentile)')
    
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.splitext(output_file)[0] + '_histogram.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Histogram saved to {plot_path}")

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Process CSV and video folder.')
    parser.add_argument('input_file', type=str, help='Input JSONL file')
    parser.add_argument('output_file', type=str, help='Output JSONL file')
    args = parser.parse_args()

    dic_list = []
    for line in open(args.input_file, 'r'):
        dic_list.append(json.loads(line))
    
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_video, dic) for dic in dic_list]
        
        token_counts = []
        with open(args.output_file, 'w') as out_file:
            for future in tqdm(futures, desc="Processing videos"):
                dic = future.result()
                token_counts.append(dic['num_tokens'])
                out_file.write(json.dumps(dic) + '\n')

    # Print token stats
    print(f"Token count statistics:")
    print(f"  Min: {min(token_counts)}")
    print(f"  Max: {max(token_counts)}")
    print(f"  Mean: {sum(token_counts) / len(token_counts):.2f}")
    print(f"  Median: {np.median(token_counts)}")
    print(f"  90th percentile: {np.percentile(token_counts, 90)}")
    print(f"  95th percentile: {np.percentile(token_counts, 95)}")
    print(f"  99th percentile: {np.percentile(token_counts, 99)}")
    
    # Calculate and print the percentile for 226 tokens
    token_226_percentile = (np.sum(np.array(token_counts) <= 226) / len(token_counts)) * 100
    print(f"  226 tokens is at the {token_226_percentile:.2f}th percentile")

    # Plot histogram
    plot_histogram(token_counts, args.output_file)

if __name__ == '__main__':
    main()