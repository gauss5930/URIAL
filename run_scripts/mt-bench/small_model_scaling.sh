declare -a SMALL_MODELS=(
    "TinyLlama/tinyLLaMA-v1.1-checkpoints|step-980000"
    "TinyLlama/tinyLLaMA-v1.1-checkpoints|step-870000"
    "TinyLlama/tinyLLaMA-v1.1-checkpoints|step-765000"
    "TinyLlama/tinyLLaMA-v1.1-checkpoints|step-655000"
    "TinyLlama/tinyLLaMA-v1.1-checkpoints|step-545000"
    "TinyLlama/tinyLLaMA-v1.1-checkpoints|step-435000"
    "TinyLlama/tinyLLaMA-v1.1-checkpoints|step-325000"
    "TinyLlama/tinyLLaMA-v1.1-checkpoints|step-220000"
    "TinyLlama/tinyLLaMA-v1.1-checkpoints|step-110000"
    "TinyLlama/tinyLLaMA-v1.1-checkpoints|step-1089914"
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
    "LLM360/Amber|ckpt_343"
    "LLM360/Amber|ckpt_286"
    "LLM360/Amber|ckpt_229"
    "LLM360/Amber|ckpt_171"
    "LLM360/Amber|ckpt_114"
    "LLM360/Amber|ckpt_057"
    "google/gemma-7b"
    "google/gemma-2b"
    "facebook/opt-6.7b"
    "facebook/opt-350m"
    "facebook/opt-2.7b"
    "facebook/opt-13b"
    "facebook/opt-125m"
    "facebook/opt-1.3b"
    "EleutherAI/llemma_7b"
    "EleutherAI/pythia-70m-deduped|step86000"
    "EleutherAI/pythia-70m-deduped|step72000"
    "EleutherAI/pythia-70m-deduped|step57000"
    "EleutherAI/pythia-70m-deduped|step43000"
    "EleutherAI/pythia-70m-deduped|step29000"
    "EleutherAI/pythia-70m-deduped|step143000"
    "EleutherAI/pythia-70m-deduped|step14000"
    "EleutherAI/pythia-70m-deduped|step129000"
    "EleutherAI/pythia-70m-deduped|step114000"
    "EleutherAI/pythia-70m-deduped|step100000"
    "EleutherAI/pythia-6.9b-deduped|step86000"
    "EleutherAI/pythia-6.9b-deduped|step72000"
    "EleutherAI/pythia-6.9b-deduped|step57000"
    "EleutherAI/pythia-6.9b-deduped|step43000"
    "EleutherAI/pythia-6.9b-deduped|step29000"
    "EleutherAI/pythia-6.9b-deduped|step143000"
    "EleutherAI/pythia-6.9b-deduped|step14000"
    "EleutherAI/pythia-6.9b-deduped|step129000"
    "EleutherAI/pythia-6.9b-deduped|step114000"
    "EleutherAI/pythia-6.9b-deduped|step100000"
    "EleutherAI/pythia-410m-deduped|step86000"
    "EleutherAI/pythia-410m-deduped|step72000"
    "EleutherAI/pythia-410m-deduped|step57000"
    "EleutherAI/pythia-410m-deduped|step43000"
    "EleutherAI/pythia-410m-deduped|step29000"
    "EleutherAI/pythia-410m-deduped|step143000"
    "EleutherAI/pythia-410m-deduped|step14000"
    "EleutherAI/pythia-410m-deduped|step129000"
    "EleutherAI/pythia-410m-deduped|step114000"
    "EleutherAI/pythia-410m-deduped|step100000"
    "EleutherAI/pythia-2.8b-deduped|step86000"
    "EleutherAI/pythia-2.8b-deduped|step72000"
    "EleutherAI/pythia-2.8b-deduped|step57000"
    "EleutherAI/pythia-2.8b-deduped|step43000"
    "EleutherAI/pythia-2.8b-deduped|step29000"
    "EleutherAI/pythia-2.8b-deduped|step143000"
    "EleutherAI/pythia-2.8b-deduped|step14000"
    "EleutherAI/pythia-2.8b-deduped|step129000"
    "EleutherAI/pythia-2.8b-deduped|step114000"
    "EleutherAI/pythia-2.8b-deduped|step100000"
    "EleutherAI/pythia-1b-deduped|step86000"
    "EleutherAI/pythia-1b-deduped|step72000"
    "EleutherAI/pythia-1b-deduped|step57000"
    "EleutherAI/pythia-1b-deduped|step43000"
    "EleutherAI/pythia-1b-deduped|step29000"
    "EleutherAI/pythia-1b-deduped|step143000"
    "EleutherAI/pythia-1b-deduped|step14000"
    "EleutherAI/pythia-1b-deduped|step129000"
    "EleutherAI/pythia-1b-deduped|step114000"
    "EleutherAI/pythia-1b-deduped|step100000"
    "EleutherAI/pythia-160m-deduped|step86000"
    "EleutherAI/pythia-160m-deduped|step72000"
    "EleutherAI/pythia-160m-deduped|step57000"
    "EleutherAI/pythia-160m-deduped|step43000"
    "EleutherAI/pythia-160m-deduped|step29000"
    "EleutherAI/pythia-160m-deduped|step143000"
    "EleutherAI/pythia-160m-deduped|step14000"
    "EleutherAI/pythia-160m-deduped|step129000"
    "EleutherAI/pythia-160m-deduped|step114000"
    "EleutherAI/pythia-160m-deduped|step100000"
    "EleutherAI/pythia-14m|step86000"
    "EleutherAI/pythia-14m|step72000"
    "EleutherAI/pythia-14m|step57000"
    "EleutherAI/pythia-14m|step43000"
    "EleutherAI/pythia-14m|step29000"
    "EleutherAI/pythia-14m|step143000"
    "EleutherAI/pythia-14m|step14000"
    "EleutherAI/pythia-14m|step129000"
    "EleutherAI/pythia-14m|step114000"
    "EleutherAI/pythia-14m|step100000"
    "EleutherAI/pythia-12b-deduped|step86000"
    "EleutherAI/pythia-12b-deduped|step72000"
    "EleutherAI/pythia-12b-deduped|step57000"
    "EleutherAI/pythia-12b-deduped|step43000"
    "EleutherAI/pythia-12b-deduped|step29000"
    "EleutherAI/pythia-12b-deduped|step143000"
    "EleutherAI/pythia-12b-deduped|step14000" 
    "EleutherAI/pythia-12b-deduped|step129000" 
    "EleutherAI/pythia-12b-deduped|step114000" 
    "EleutherAI/pythia-12b-deduped|step100000" 
    "EleutherAI/pythia-1.4b-deduped|step86000" 
    "EleutherAI/pythia-1.4b-deduped|step72000" 
    "EleutherAI/pythia-1.4b-deduped|step57000" 
    "EleutherAI/pythia-1.4b-deduped|step43000" 
    "EleutherAI/pythia-1.4b-deduped|step29000" 
    "EleutherAI/pythia-1.4b-deduped|step143000" 
    "EleutherAI/pythia-1.4b-deduped|step14000" 
    "EleutherAI/pythia-1.4b-deduped|step129000" 
    "EleutherAI/pythia-1.4b-deduped|step114000" 
    "EleutherAI/pythia-1.4b-deduped|step100000" 
    "codellama/CodeLlama-7b-hf" 
    "allenai/OLMo-1.7-7B-hf|step95000-tokens398B" 
    "allenai/OLMo-1.7-7B-hf|step48000-tokens201B" 
    "allenai/OLMo-1.7-7B-hf|step477000-tokens2000B" 
    "allenai/OLMo-1.7-7B-hf|step429000-tokens1798B" 
    "allenai/OLMo-1.7-7B-hf|step382000-tokens1601B" 
    "allenai/OLMo-1.7-7B-hf|step334000-tokens1400B" 
    "allenai/OLMo-1.7-7B-hf|step286000-tokens1199B" 
    "allenai/OLMo-1.7-7B-hf|step239000-tokens1002B" 
    "allenai/OLMo-1.7-7B-hf|step191000-tokens800B" 
    "allenai/OLMo-1.7-7B-hf|step143000-tokens599B" 
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
    "tiiuae/falcon-11B"
)

version="scaling_law_inst"
temp=${2:-0}
rp=${3:-1}
output_dir="result_dirs/mt-bench/urial_bench"
output_path="result_dirs/mt-bench/urial_bench/csv_result"
mkdir -p $output_dir
gpu=${1:-"0"}
tsp=${2:-1}

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