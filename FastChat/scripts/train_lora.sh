hostfile="/yzwl_data/yumu/FastChat/scripts/hostfile"

export NCCL_SOCKET_IFNAME=ibp25s0

deepspeed --hostfile=$hostfile /yzwl_data/yumu/FastChat/fastchat/train/train_lora.py \
    --model_name_or_path /yzwl_data/yumu/trained_model/Llama-3-70b-zh-2 \
    --lora_r 16 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --data_path  "/yzwl_data/yumu/sft_llama_1/fast_chat_long_all.jsonl" \
    --output_dir autoark_70b_long_v2 \
    --num_train_epochs 2 \
    --bf16 True \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --do_eval False \
    --evaluation_strategy "no" \
    --eval_steps 20  \
    --save_strategy "steps" \
    --save_steps 20 \
    --save_total_limit 8 \
    --learning_rate 4e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --report_to "none" \
    --lr_scheduler_type "cosine" \
    --logging_strategy "steps" \
    --logging_steps 1 \
    --model_max_length 20000 \
    --q_lora True \
    --deepspeed ./config/ds_config_zero3.json \
    --gradient_checkpointing True \
    --flash_attn True 
