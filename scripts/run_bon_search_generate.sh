gpu=6
generate_model_dir_list=()
for generate_model_dir in "${generate_model_dir_list[@]}"; do
    CUDA_VISIBLE_DEVICES="$gpu" python bon_search.py \
    --data_dir ../math-data/prm800k-main/prm800k/math_splits/test.jsonl \
    --generate_model_dir "$generate_model_dir" \
    --test_n 128 \
    --generate \
    --temperature 0.8 \
    --top_p 0.95
done