import re
import argparse
from datetime import datetime, timedelta

START_MESSAGE = "building SATVideoDiffusionEngine model ..."

def parse_log_file(file_path, up_to_now=False):
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Extract timestamps and messages using regex
    log_entries = re.findall(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})\] \[INFO\] \[RANK 0\] (.+)', content)
    
    if len(log_entries) < 2:
        return "Not enough log entries found in the log file."
    
    running_periods = []
    interrupted_periods = []
    current_start = None
    last_time = None

    for i, (timestamp, message) in enumerate(log_entries):
        current_time = datetime.strptime(timestamp.split(',')[0], '%Y-%m-%d %H:%M:%S')
        
        if START_MESSAGE in message:
            if current_start and last_time:
                running_periods.append((current_start, last_time))
                interrupted_periods.append((last_time, current_time))
            current_start = current_time
        
        last_time = current_time
    
    if current_start and last_time:
        running_periods.append((current_start, last_time))
    
    is_interrupted_now = False
    if up_to_now:
        # Check if there's an ongoing interruption
        current_time = datetime.now().replace(microsecond=0)
        if last_time and (current_time - last_time) > timedelta(minutes=5):
            interrupted_periods.append((last_time, current_time))
            is_interrupted_now = True
    
    result = []
    total_running_time = timedelta()
    total_interrupted_time = timedelta()
    
    for start, end in running_periods:
        total_running_time += end - start
    
    for start, end in interrupted_periods:
        total_interrupted_time += end - start
    
    total_duration = total_running_time + total_interrupted_time
    
    for i, (start, end) in enumerate(running_periods, 1):
        duration = end - start
        result.append(f"Running period {i}: {start} to {end} (duration: {duration})")
    
    result.append(f"\nTotal running time: {total_running_time}")
    result.append(f"Running time percentage: {total_running_time / total_duration * 100:.2f}%")
    
    if interrupted_periods:
        result.append("\nInterrupted periods:")
        for i, (start, end) in enumerate(interrupted_periods, 1):
            duration = end - start
            result.append(f"Interruption {i}: {start} to {end}{'(now)' if is_interrupted_now and i == len(interrupted_periods) else ''}(duration: {duration})")
        
        result.append(f"\nTotal interrupted time: {total_interrupted_time}")
        result.append(f"Interrupted time percentage: {total_interrupted_time / total_duration * 100:.2f}%")
    
    result.append(f"\nTotal duration: {total_duration}")
    
    return "\n".join(result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate script runtime and interruptions from log file.")
    parser.add_argument("log_file", help="Path to the log file")
    parser.add_argument("--now", action="store_true", help="Report up to the current time")
    args = parser.parse_args()
    
    print(parse_log_file(args.log_file, args.now))
