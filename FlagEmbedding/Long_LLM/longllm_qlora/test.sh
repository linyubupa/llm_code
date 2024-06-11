export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# base model id
model=/root/mountpoint/autoark/trained_model/Llama-3-8b-zh-2
# lora model id
lora=/root/mountpoint/yumu/FastChat/scripts/long_autoark_8b_v1/checkpoint-150

COMMAND="--data_root /data/long-llm --model_name_or_path $model  --rope_theta 200e6 --attn_impl flash_attention_2 --chat_template llama-3"

# source /opt/conda/bin/activate unsloth

# torchrun --nproc_per_node 8 -m main.eval_longbench --max_length 31500 $COMMAND
# torchrun --nproc_per_node 8 -m main.eval_topic $COMMAND
# torchrun --nproc_per_node 8 -m main.eval_mmlu $COMMAND

# source /opt/conda/bin/activate full

python3 -m main.eval_needle $COMMAND --lora $lora --min_length 8000 --max_length 64000 
# --enable_tp
# python -m main.eval_infbench $COMMAND --max_length 80000 --enable_tp
