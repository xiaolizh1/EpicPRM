#!/bin/bash

python_script="bon_search.py"
gpus=(5 3 4)
tokenizer_dir="../models/qwen2-math-1.5b"
num_labels=2
batch_size=32

data_dirs=()
verify_model_dirs=()

echo "Starting parallel execution on GPUs: ${gpus[*]}"

for data_dir in "${data_dirs[@]}"; do
    for i in "${!verify_model_dirs[@]}"; do
        gpu=${gpus[$i]}
        verify_model_dir=${verify_model_dirs[$i]}
        CUDA_VISIBLE_DEVICES="$gpu" python "$python_script" \
            --data_dir "$data_dir" \
            --tokenizer_dir ../models/qwen2-math-1.5b \
            --verify_model_dir "$verify_model_dir" \
            --num_labels "$num_labels" \
            --batch_size "$batch_size" &
    done
    wait
done

