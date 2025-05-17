export CUDA_VISIBLE_DEVICES=0
export WANDB_API_KEY=9fd21364ed6c1c6677a250972c5e19a931171974

python evaluation.py --model_path outputs/dinov2-qwen/merged_final \
                       --data_path DATAS/eval/NextQA/formatted_dataset_test.json \
                       --video_root DATAS/eval/NextQA/NExTVideo \
                       --dataset_name NextQA \
                       --results_dir ./results \
                       --max_frames_num 64 \
                       --fps 1 \
                       --force_sample