declare -a LARGE_MODELS=(
    "Qwen/Qwen1.5-72B"
    "Qwen/Qwen1.5-32B"
    "meta-llama/Meta-Llama-3-70B"
    "meta-llama/Llama-2-70b-hf"
    "LLM360/K2|ckpt_360"
    "LLM360/K2|ckpt_333"
    "LLM360/K2|ckpt_276"
    "LLM360/K2|ckpt_222"
    "LLM360/K2|ckpt_165"
    "LLM360/K2|ckpt_111"
    "LLM360/K2|ckpt_054"
    "facebook/opt-66b"
    "facebook/opt-30b"
    "EleutherAI/llemma_34b"
    "codellama/CodeLlama-70b-hf"
    "codellama/CodeLlama-34b-hf"
    "01-ai/Yi-34B"
    "deepseek-ai/deepseek-llm-67b-base"
    "Qwen/Qwen-72B"
    "huggyllama/llama-30b"
    "huggyllama/llama-65b"
    "tiiuae/falcon-40b"
    "tiiuae/falcon-180B"
)

version="scaling_law_inst"
temp=${2:-0}
rp=${3:-1}
output_dir="result_dirs/mt-bench/urial_bench"
output_path="result_dirs/mt-bench/urial_bench/csv_result"
mkdir -p $output_dir
gpu=${8:-"0,1,2,3,4,5,6,7"}
tsp=${9:-8}

echo "Logging into Hugging Face Hub"
huggingface-cli login --token "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"

for MODEL_PATH in "${SMALL_MODELS[@]}"
do
    CUDA_VISIBLE_DEVICES=$gpu python src/unified_infer.py \
        --urial $version \
        --download_dir /cache/models/ \
        --model_name $MODEL_PATH \
        --tensor_parallel_size ${tsp} \
        --dtype bfloat16 \
        --data_name mt-bench \
        --mt_turn 1 \
        --top_p 1 --temperature $temp --repetition_penalty $rp --batch_size 4 --max_tokens 2048 \
        --filepath auto \
        --overwrite 


    CUDA_VISIBLE_DEVICES=$gpu python src/unified_infer.py \
        --urial $version \
        --download_dir /cache/models/ \
        --model_name $MODEL_PATH \
        --tensor_parallel_size ${tsp} \
        --dtype bfloat16 \
        --data_name mt-bench \
        --mt_turn 2 \
        --mt_turn1_result $output_dir/${pretty_name}.turn1.json \
        --top_p 1 --temperature $temp --repetition_penalty $rp --batch_size 8 --max_tokens 2048 \
        --filepath auto \
        --overwrite 

    python src/file_clean.py \
        --output_path $output_dir \
        --model_name $MODEL_PATH \
        --output_path $output_path
        
    rm -rf /cache/models
done