## Folder structure
```
sat/
    output/
    data/
    configs/
    submitit_train_video.py
    submitit_sample_video.py
    train_video_oci.py
    sample_video_oci.py
    submitit_batch_train_video.py
    submitit_batch_sample_video.py
    submitit_batch_train_video_config.yaml
    submitit_batch_sample_video_config.yaml
```
`sat` contains the source codebase for training and sampling, without using HF diffusers.
Datasets are stored in `data`, the outputs of training and sampling are stored in `output`.

## Environment setup
Create a new conda environment and install the following dependencies
```bash
# mamba install pytorch=2.4.0 torchvision torchaudio pytorch-cuda=12.1 cuda=12.1 -c pytorch -c nvidia
mamba install pytorch=2.4.0 torchvision torchaudio pytorch-cuda=12.1 cuda=12.1 -c pytorch -c nvidia/label/cuda-12.1.1 -c nvidia/label/cuda-12.1.0
pip install -r sat/requirements_oci.txt
```

## Interactive Debugging

We will need to request a node with 8 GPUs for interactive debugging. It will last for 4 hours. You may use other partitions as well.

```bash
srun --time=240  -A nvr_lpr_nvgptvision --partition interactive --gpus 8 -c 248 --pty bash
```

### Run training with 8 GPUs

Example command:
```bash
torchrun --nproc-per-node=8 train_video_oci.py --base configs/oci/sft/5b_mambav2_pre_f145_sft.yaml configs/oci/dataset/tandj_f145.yaml --tag debug-bs8 --save-interval 10
```

### Run sampling with 1 GPU

Example command:
```bash
CUDA_VISIBLE_DEVICES=0 python sample_video_oci.py --base configs/oci/sft/5b_mambav2_pre_f145_sft.yaml configs/oci/inference/f145_infer.yaml --resume output/train/5b_mambav2_pre_f145_sft_osp_pixabay_v2_f145/5k-bs32-v2/checkpoints/ --output-dir output/inference/5b_mambav2_pre_f145_sft_osp_pixabay_v2_f145/5k-bs32-v2/debug_0913 --input-file configs/test_pixabay.txt
```

## Launching jobs in SLURM

We use [Submitit](https://github.com/facebookincubator/submitit) to manage jobs in SLURM, which handles automatic checkpointing, logging, and resubmission.

### Run training with 64 GPUs

Example command:
```bash
python submitit_train_video.py --nodes 8 --base configs/oci/sft/5b_t512_lora_f49_sft.yaml configs/oci/dataset/tandj_g6s_f49.yaml --tag 5k-bs64-v3 --wandb
```


### Run multiple training jobs with 64 GPUs each

Example config file:
```yaml
jobs:
  - cmd_args:
      nodes: 8
      base: 
        - configs/oci/sft/5b_t512_lora_f49_sft.yaml
        - configs/oci/dataset/tandj_g6s_f49.yaml
      tag: 5k-bs64-v3
      wandb: true
```

Example command:
```bash
python submitit_batch_train_video.py --batch-config submitit_batch_train_video_config.yaml
```

### Run sampling with 1 GPU

Example command:
```bash
python submitit_sample_video.py --config configs/oci/sft/5b_t512_lora_f49_sft.yaml --extra-configs configs/oci/inference/f49_infer.yaml --suffix tandj_g6s_f49/5k-bs64-v3/checkpoints/ --input-file configs/test_tandj_g6s.txt
```

### Run multiple sampling jobs with 1 GPU each

Example config file:
```yaml
jobs:
  - config: configs/oci/sft/5b_t512_lora_f49_sft.yaml
    extra_configs:
      - configs/oci/inference/f49_infer.yaml
    cmd_args:
      suffix: tandj_g6s_f49/5k-bs64-v3/checkpoints/
      input-file: configs/test_tandj_g6s.txt
```

Example command:
```bash
python submitit_batch_sample_video.py --batch-config submitit_batch_sample_video_config.yaml
```

### Misc

We also provide a script to check the percentage of pending/running time of jobs:
```bash
python ../tools/oci/report_duration.py output/train/5b_lora_f49_sft_tandj_g6s_f49/10k-bs64-v3/logs/log.txt
```

## Datasets

We refer to [Open-Sora Dataset](https://github.com/hpcaitech/Open-Sora/blob/main/docs/datasets.md) to curate long video datasets.

* [Pixabay](https://pixabay.com/videos/), downloaded, 54k videos.
* [MiraData](https://github.com/mira-space/MiraData), 77k videos.
* [Vript](https://github.com/mutonix/Vript/tree/main), 400k videos, not sure if the aesthetics are good.

### Data processing


#### JSONL format

We use JSONL format for datasets. Each line is a JSON object with the following format:
```json
{
    "path": "path to the video file",
    "duration": "duration of the video in seconds",
    "num_frames": "number of frames in the video",
    "width": "width of the video",
    "height": "height of the video",
    // caption field is configurable in the dataset yaml file
    "caps": "a list of captions for the video",
    "text": "single caption of the video"
}
```

To add video meta info, you can use [`add_fps.py`](./tools/open_sora_plan/add_fps.py) to generate a new JSONL file with meta info.

#### Video Recap

Following guidance [here](tools/caption/README.md), we tried use CogVLM2 to recap some videos, not sure about the quality yet. But the pipeline is built as follows:

1. Get an interactive node with 8 GPUs. You may also launch more jobs to recap more videos in parallel.
2. Clone our folked version [CogVLM2 repo](https://github.com/xvjiarui/CogVLM2) and checkout branch `dist`.
3. After installing CogVLM2, you can run the following command to launch multiple servers.
```bash
python video_demo/api_demo.py
```
4. Run [`recap_video.py`](./tools/open_sora_plan/recap_video.py) to recap videos. 