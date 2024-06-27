declare -a SMALL_MODELS=(
    "TinyLlama/tinyLLaMA_v1.1"
    "Qwen/Qwen1.5-7B"
    "Qwen/Qwen1.5-4B"
    "Qwen/Qwen1.5-14B"
    "Qwen/Qwen1.5-1.8B"
    "Qwen/Qwen1.5-0.5B"
    "microsoft/Phi-3-mini-4k-instruct"
    "microsoft/Phi-3-mini-128k-instruct"
    "microsoft/phi-2"
    "microsoft/phi-1_5"
    "microsoft/phi-1"
    "meta-llama/Meta-Llama-3-8B"
    "meta-llama/Llama-2-7b-hf"
    "meta-llama/Llama-2-13b-hf"
    "LLM360/Amber"
    "google/gemma-7b"
    "google/gemma-2b"
    "facebook/opt-6.7b"
    "facebook/opt-350m"
    "facebook/opt-2.7b"
    "facebook/opt-13b"
    "facebook/opt-125m"
    "facebook/opt-1.3b"
    "EleutherAI/llemma_7b"
    "EleutherAI/pythia-70m-deduped"
    "EleutherAI/pythia-6.9b-deduped"
    "EleutherAI/pythia-410m-deduped"
    "EleutherAI/pythia-2.8b-deduped"
    "EleutherAI/pythia-1b-deduped"
    "EleutherAI/pythia-160m-deduped"
    "EleutherAI/pythia-14m"
    "EleutherAI/pythia-12b-deduped"
    "EleutherAI/pythia-1.4b-deduped"
    "codellama/CodeLlama-7b-hf"
    "codellama/CodeLlama-13b-hf"
    "allenai/OLMo-7B-hf"
    "allenai/OLMo-1B-hf"
    "allenai/OLMo-1.7-7B-hf"
    "01-ai/Yi-6B"
    "EleutherAI/gpt-neox-20b"
    "deepseek-ai/deepseek-llm-7b-base"
    "Qwen/Qwen-1_8B"
    "Qwen/Qwen-7B"
    "Qwen/Qwen-14B"
    "internlm/internlm2-1_8b"
    "internlm/internlm2-7b"
    "internlm/internlm2-20b"
    "huggyllama/llama-7b"
    "huggyllama/llama-13b"
    "tiiuae/falcon-7b"
    "tiiuae/falcon-11B"
)

declare -a LARGE_MODELS=(
    "Qwen/Qwen1.5-72B"
    "Qwen/Qwen1.5-32B"
    "meta-llama/Meta-Llama-3-70B"
    "meta-llama/Llama-2-70b-hf"
    "LLM360/K2"
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
output_dir="result_dirs/mt-bench/urial_bench/"
output_path="result_dirs/mt-bench/csv_result/"
mkdir -p $output_dir
gpu=${4:-"0,1,2,3"}
tsp=${5:-4}
pretty_name="Llama-3-8B"

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
        --filepath $output_dir/${pretty_name}.turn1.json \
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
        --filepath $output_dir/${pretty_name}.turn2.json \
        --overwrite 

    python file_clean.py \
        --output_path $output_dir \
        --model_name $MODEL_PATH \
        --output_path $output_path
        
    rm -rf ~/cache/models
done