```bash
mamba install pytorch=2.4.0 torchvision torchaudio pytorch-cuda=12.1 cuda=12.1 -c pytorch -c nvidia
pip install -r sat/requirements_oci.txt
```

## Debug
```
torchrun --nproc-per-node=8 train_video_oci.py --base configs/oci/sft/5b_full_mambad_f145_sft.yaml configs/oci/dataset/tandj_f145.yaml --tag debug-bs8 --save-interval 10
```