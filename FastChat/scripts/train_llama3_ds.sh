pip install jsonlines
pip install -e /root/mountpoint/yumu/FastChat
pip install /root/mountpoint/s3_yiming/my_llm/environment/flash_attn-2.5.6+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
cd /root/mountpoint/yumu/FastChat/scripts/

torchrun --nproc_per_node=8  --nnodes=$NNODES --master_addr=$MASTER_ADDR --node_rank=$NODE_RANK /root/mountpoint/yumu/FastChat/fastchat/train/train_with_collator.py \
    --model_name_or_path /root/mountpoint/autoark/trained_model/Llama-3-70b-zh-1  \
    --data_path  /root/mountpoint/yumu/dataset/sft_llama_1/fast_chat_identi_autoark_v5.jsonl \
    --bf16 True \
    --output_dir output_llama_70b_ark_v1 \
    --num_train_epochs 1 \
    --max_steps 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "steps" \
    --eval_steps 200 \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 2 \
    --weight_decay 0. \
    --warmup_ratio 0.01 \
    --learning_rate 2e-6 \
    --adam_beta2 0.95 \
    --do_train \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "none" \
    --model_max_length 2048 \
    --lazy_preprocess False \
    --remove_unused_columns False \
    --gradient_checkpointing True \
    --deepspeed ./config/ds_config_zero3.json