export export PYTHONPATH=$PYTHONPATH:/yzwl_data/yumu/code/EasyContext
export PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:1024' 
accelerate launch --config_file ../accelerate_configs/single_node.yaml \
../train_ark.py \
--batch-size 1 \
--gradient-accumulate-every 2  \
--output-dir  ./output/qwen_7B_0.5M_bs_1M_rope_100M_step_300_lr_2e-5 \
--seed 2026 \
--max-train-steps 300  \
--learning-rate 2e-5  \
--dataset ../data/autoark_think.jsonl \
--model /yzwl_data/yumu/model/Qwen2-72B-Instruct  \
--seq-length 20000 \
--rope-theta 100000000 \
--parallel_mode zigzag_ring_attn 
