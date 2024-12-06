export wandb offline
export NCCL_P2P_LEVEL=NVL

DS_SKIP_CUDA_CHECK=1 deepspeed --include=localhost:3 --master_port="29501" reward_modeling.py \
    --model_path "models/qwen2-math-1.5b" \
    --tokenizer_path models/qwen2-math-1.5b \
    --num_labels 2 \
    --train_data_path "math-data/prm800k-main/prm800k/data/getlabel_data/synthetic_train_data_perplexity50k.jsonl" \
    --eval_data_path "" \
    --output_dir "models/qwen2-math-1.5b-base-epic" \
    --bf16 True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "steps" \
    --eval_steps 25 \
    --save_strategy "steps" \
    --save_steps 25 \
    --save_total_limit 10 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --logging_steps 1 \
    --tf32 True \
    --seed 42 \
    --logging_first_step True \
    --max_length 1500 \
    --merge_loss False \
    --ddp_find_unused_parameters False \
    --run_name 'qwen2-math-1.5b-base-epic' \