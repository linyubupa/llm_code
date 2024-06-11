hostfile="/yzwl_data/yumu/FastChat/scripts/hostfile"

export NCCL_SOCKET_IFNAME=ibp25s0

deepspeed --hostfile=$hostfile /yzwl_data/yumu/FastChat/fastchat/train/train_with_collator.py \
    --model_name_or_path /yzwl_data/yumu/model/LLM-Research/Meta-Llama-3-70B-Instruct \
    --data_path  "/yzwl_data/yumu/sft_llama_1/new_sft_autoark_identi_only.jsonl" \
    --output_dir autoark_70b_identi_only \
    --num_train_epochs 4 \
    --bf16 True \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "steps" \
    --eval_steps 10  \
    --save_strategy "steps" \
    --save_steps 10 \
    --save_total_limit 8 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.005 \
    --report_to "none" \
    --lr_scheduler_type "cosine" \
    --logging_strategy "steps" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --deepspeed ./config/ds_config_zero3.json \
    --gradient_checkpointing True \
    --flash_attn True 
