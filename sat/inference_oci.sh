CUDA_VISIBLE_DEVICES=7 python sample_video_oci.py --base configs/oci/inference/5b_f49_infer.yaml --resume output/train/5b_full_f49_sft_osp_pixabay_v2_f49/5k-bs32/checkpoints/ --output-dir output/inference/5b_full_f49_sft_osp_pixabay_v2_f49/5k-bs32/debug --input-file configs/test_pixabay.txt

CUDA_VISIBLE_DEVICES=6 python sample_video_oci.py --base configs/oci/inference/5b_mambad_f145_infer.yaml --resume output/train/5b_full_mambad_f145_sft_osp_pixabay_v2_f145/5k-bs32/checkpoints/ --output-dir output/inference/5b_full_mambad_f145_sft_osp_pixabay_v2_f145/5k-bs32/debug --input-file configs/test_pixabay.txt

CUDA_VISIBLE_DEVICES=5 python sample_video_oci.py --base configs/oci/inference/2b_f49_infer.yaml --resume output/train/2b_full_f49_sft_osp_pixabay_v2_f49/5k-bs32/checkpoints/ --output-dir output/inference/2b_full_f49_sft_osp_pixabay_v2_f49/5k-bs32/debug --input-file configs/test_pixabay.txt

CUDA_VISIBLE_DEVICES=4 python sample_video_oci.py --base configs/oci/inference/2b_mambad_f145_infer.yaml --resume output/train/2b_full_mambad_f145_sft_osp_pixabay_v2_f145/5k-bs32/checkpoints/ --output-dir output/inference/2b_full_mambad_f145_sft_osp_pixabay_v2_f145/5k-bs32/debug --input-file configs/test_pixabay.txt

CUDA_VISIBLE_DEVICES=3 python sample_video_oci.py --base configs/oci/inference/5b_f49_infer.yaml  --output-dir output/inference/5b_f49_infer/default_pixabay --input-file configs/test_pixabay.txt

CUDA_VISIBLE_DEVICES=2 python sample_video_oci.py --base configs/oci/inference/5b_mambad_f145_infer.yaml --output-dir output/inference/5b_mambad_f145_infer/default_pixabay --input-file configs/test_pixabay.txt

CUDA_VISIBLE_DEVICES=1 python sample_video_oci.py --base configs/oci/inference/2b_f49_infer.yaml --output-dir output/inference/2b_f49_infer/default_pixabay --input-file configs/test_pixabay.txt

CUDA_VISIBLE_DEVICES=0 python sample_video_oci.py --base configs/oci/inference/2b_mambad_f145_infer.yaml --output-dir output/inference/2b_mambad_f145_infer/default_pixabay --input-file configs/test_pixabay.txt

CUDA_VISIBLE_DEVICES=2 python sample_video_oci.py --base configs/oci/inference/5b_mambad_f145_infer.yaml --output-dir output/inference/5b_mambad_f145_infer/theta_50k_pixabay --input-file configs/test_pixabay.txt

CUDA_VISIBLE_DEVICES=0 python sample_video_oci.py --base configs/oci/inference/2b_mambad_f145_infer.yaml --output-dir output/inference/2b_mambad_f145_infer/local_attn_pixabay_pos_emb_repeat --input-file configs/test_pixabay.txt

CUDA_VISIBLE_DEVICES=0 python sample_video_oci.py --base configs/oci/inference/5b_mambad_f145_infer.yaml --output-dir output/inference/5b_mambad_f145_infer/local_attn_pixabay_pos_emb_repeat_attn4 --input-file configs/test_pixabay.txt

CUDA_VISIBLE_DEVICES=4 python sample_video_oci.py --base configs/oci/sft/2b_full_mambav2_pre_f145_sft.yaml configs/oci/inference/f145_infer.yaml --resume output/train/2b_full_mambav2_pre_f145_sft_osp_pixabay_v2_f145/5k-bs32/checkpoints/ --output-dir output/inference/2b_full_mambav2_pre_f145_sft_osp_pixabay_v2_f145/5k-bs32/meeting_0912_v2 --input-file configs/test_pixabay.txt

CUDA_VISIBLE_DEVICES=3 python sample_video_oci.py --base configs/oci/sft/2b_full_mambav2_f145_sft.yaml configs/oci/inference/f145_infer.yaml --resume output/train/2b_full_mambav2_f145_sft_osp_pixabay_v2_f145/5k-bs32/checkpoints/ --output-dir output/inference/2b_full_mambav2_f145_sft_osp_pixabay_v2_f145/5k-bs32/meeting_0912_v2 --input-file configs/test_pixabay.txt


CUDA_VISIBLE_DEVICES=1 python sample_video_oci.py --base configs/oci/sft/2b_full_mambav2_rep2_f145_sft.yaml configs/oci/inference/f145_infer.yaml --resume output/train/2b_full_mambav2_rep2_f145_sft_osp_pixabay_v2_f145/5k-bs32/checkpoints/ --output-dir output/inference/2b_full_mambav2_rep2_f145_sft_osp_pixabay_v2_f145/5k-bs32/meeting_0912_v2 --input-file configs/test_pixabay.txt

CUDA_VISIBLE_DEVICES=6 python sample_video_oci.py --base configs/oci/sft/5b_full_mambav2_f145_sft.yaml configs/oci/inference/f145_infer.yaml --resume output/train/5b_full_mambav2_f145_sft_osp_pixabay_v2_f145/5k-bs32/checkpoints/ --output-dir output/inference/5b_full_mambav2_f145_sft_osp_pixabay_v2_f145/5k-bs32/meeting_0912_v2 --input-file configs/test_pixabay.txt

CUDA_VISIBLE_DEVICES=5 python sample_video_oci.py --base configs/oci/sft/5b_full_mambav2_rep2_f145_sft.yaml configs/oci/inference/f145_infer.yaml --resume output/train/5b_full_mambav2_rep2_f145_sft_osp_pixabay_v2_f145/5k-bs32/checkpoints/ --output-dir output/inference/5b_full_mambav2_rep2_f145_sft_osp_pixabay_v2_f145/5k-bs32/meeting_0912_v2 --input-file configs/test_pixabay.txt

CUDA_VISIBLE_DEVICES=0 python sample_video_oci.py --base configs/oci/sft/2b_full_mambav2_pre_f145_sft.yaml configs/oci/inference/f145_infer.yaml --output-dir output/inference/2b_full_mambav2_pre_f145_sft/init --input-file configs/test_pixabay.txt

CUDA_VISIBLE_DEVICES=1 python sample_video_oci.py --base configs/oci/sft/5b_full_mambav2_pre_f145_sft.yaml configs/oci/inference/f145_infer.yaml --output-dir output/inference/5b_full_mambav2_pre_f145_sft/init --input-file configs/test_pixabay.txt


CUDA_VISIBLE_DEVICES=7 python sample_video_oci.py --base configs/oci/sft/5b_full_mambav2_pre_f145_sft.yaml configs/oci/inference/f145_infer.yaml --resume output/train/5b_full_mambav2_pre_f145_sft_osp_pixabay_v2_f145/5k-bs32-v2/checkpoints/ --output-dir output/inference/5b_full_mambav2_pre_f145_sft_osp_pixabay_v2_f145/5k-bs32-v2/debug_0913 --input-file configs/test_pixabay.txt

CUDA_VISIBLE_DEVICES=6 python sample_video_oci.py --base configs/oci/sft/5b_lora_mambav2_pre_f145_sft.yaml configs/oci/inference/f145_infer.yaml --resume output/train/5b_lora_mambav2_pre_f145_sft_osp_pixabay_v2_f145/5k-bs32-v2/checkpoints/ --output-dir output/inference/5b_lora_mambav2_pre_f145_sft_osp_pixabay_v2_f145/5k-bs32-v2/debug_0913 --input-file configs/test_pixabay.txt

CUDA_VISIBLE_DEVICES=5 python sample_video_oci.py --base configs/oci/sft/5b_mambav2_pre_f145_sft.yaml configs/oci/inference/f145_infer.yaml --resume output/train/5b_mambav2_pre_f145_sft_osp_pixabay_v2_f145/5k-bs32-v2/checkpoints/ --output-dir output/inference/5b_mambav2_pre_f145_sft_osp_pixabay_v2_f145/5k-bs32-v2/debug_0913 --input-file configs/test_pixabay.txt

CUDA_VISIBLE_DEVICES=4 python sample_video_oci.py --base configs/oci/sft/2b_full_mambav2_pre_f145_sft.yaml configs/oci/inference/f145_infer.yaml --resume output/train/2b_full_mambav2_pre_f145_sft_osp_pixabay_v2_f145/5k-bs32-v2/checkpoints/ --output-dir output/inference/2b_full_mambav2_pre_f145_sft_osp_pixabay_v2_f145/5k-bs32-v2/debug_0913 --input-file configs/test_pixabay.txt

CUDA_VISIBLE_DEVICES=3 python sample_video_oci.py --base configs/oci/sft/2b_lora_mambav2_pre_f145_sft.yaml configs/oci/inference/f145_infer.yaml --resume output/train/2b_lora_mambav2_pre_f145_sft_osp_pixabay_v2_f145/5k-bs32-v2/checkpoints/ --output-dir output/inference/2b_lora_mambav2_pre_f145_sft_osp_pixabay_v2_f145/5k-bs32-v2/debug_0913 --input-file configs/test_pixabay.txt


python submitit_sample_video.py --config-list configs/oci/sft/5b_mambav2_pre_f145_sft.yaml configs/oci/sft/5b_lora_mambav2_pre_f145_sft.yaml --common-configs configs/oci/inference/f145_infer.yaml --suffix osp_pixabay_v2_f145/5k-bs32-v2/checkpoints/ --input-file configs/test_pixabay.txt