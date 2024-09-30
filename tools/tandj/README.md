## Cut long videos into clips
1. create cut timestamps
```bash
python tools/scene_cut/scene_detect.py data/tandj/videos_meta.csv --out_path data/tandj/videos_meta_timestamp_min180_thr3_max240.csv --min_scene_len 180 --max_frames 240 --threshold 3
```

2. cut videos into clips
```bash
python tools/scene_cut/cut.py data/tandj/videos_meta_timestamp_min180_thr3_max240.csv --save_dir data/tandj/clips_min180_thr3_max240
```

## Create and recap clips
1. create metadata
```bash
python tools/tandj/create_metadata.py data/tandj/clips_min60_thr3_max80/ data/tandj/metadata/min60_thr3_max80/tandj.jsonl
python tools/tandj/create_metadata.py data/tandj/clips_min80_thr3_max100/ data/tandj/metadata/min80_thr3_max100/tandj.jsonl
python tools/tandj/create_metadata.py data/tandj/clips_min180_thr3_max240/ data/tandj/metadata/min180_thr3_max240/tandj.jsonl
```

2. add fps
```bash
python tools/tandj/add_fps.py data/tandj/clips_min60_thr3_max80/ data/tandj/metadata/min60_thr3_max80/tandj.jsonl data/tandj/metadata/min60_thr3_max80/tandj_stats.jsonl
python tools/tandj/add_fps.py data/tandj/clips_min80_thr3_max100/ data/tandj/metadata/min80_thr3_max100/tandj.jsonl data/tandj/metadata/min80_thr3_max100/tandj_stats.jsonl
python tools/tandj/add_fps.py data/tandj/clips_min180_thr3_max240/ data/tandj/metadata/min180_thr3_max240/tandj.jsonl data/tandj/metadata/min180_thr3_max240/tandj_stats.jsonl
```

3. filter episode
```bash
python tools/tandj/filter_episode.py data/tandj/metadata/min60_thr3_max80/tandj_stats.jsonl data/tandj/metadata/min60_thr3_max80/tandj_stats_ep114.jsonl
python tools/tandj/filter_episode.py data/tandj/metadata/min80_thr3_max100/tandj_stats.jsonl data/tandj/metadata/min80_thr3_max100/tandj_stats_ep114.jsonl
python tools/tandj/filter_episode.py data/tandj/metadata/min180_thr3_max240/tandj_stats.jsonl data/tandj/metadata/min180_thr3_max240/tandj_stats_ep114.jsonl
```

4. recap video
```bash
python tools/caption/recap_video.py data/tandj/clips_min60_thr3_max80/ data/tandj/metadata/min60_thr3_max80/tandj_stats_ep114.jsonl data/tandj/metadata/min60_thr3_max80/tandj_recap_ep114.jsonl
python tools/caption/recap_video.py data/tandj/clips_min180_thr3_max240/ data/tandj/metadata/min180_thr3_max240/tandj_stats_ep114.jsonl data/tandj/metadata/min180_thr3_max240/tandj_recap_ep114.jsonl
```