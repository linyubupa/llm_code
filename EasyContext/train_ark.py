import argparse
import torch
import os
from typing import Dict, Optional, List 
from torch.utils.data import Dataset
from datasets import load_dataset, load_from_disk, DatasetDict
from datetime import timedelta
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs, set_seed
from tqdm import tqdm
from transformers import set_seed, default_data_collator
from transformers import AutoModelForCausalLM,AutoTokenizer
import transformers
from flash_attn.losses.cross_entropy import CrossEntropyLoss
import math
from accelerate.utils import (
    InitProcessGroupKwargs,
    set_seed,
    DummyOptim,
    DummyScheduler,
)
from easy_context import (
    prepare_seq_parallel_inputs,
    apply_seq_parallel_monkey_patch,
    prepare_dataloader,
    apply_unsloth_offloaded_gradient_checkpoint_monkey_patch
)
import numpy as np 
import jsonlines
import json
from transformers.trainer_pt_utils import LabelSmoother
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from dataclasses import dataclass, field

# apply_unsloth_offloaded_gradient_checkpoint_monkey_patch()
IGNORE_TOKEN_ID = LabelSmoother.ignore_index

local_rank = None
def rank0_print(*args):
    if local_rank == 0:
        print(*args)



def preprocess_instruct(
    messages,
    answers,
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int,
) -> Dict:
    """Preprocesses the data for supervised fine-tuning."""

    texts = []
    for i, msg in enumerate(messages):
        texts.append(
            tokenizer.apply_chat_template(
                msg,
                tokenize=True,
                add_generation_prompt=False,
                padding="max_length",
                max_length=max_len,
                truncation=True,
            )
        )
    
    input_ids = torch.tensor(texts, dtype=torch.int)
    target_ids = input_ids.clone()
    target_ids[target_ids == tokenizer.pad_token_id] = IGNORE_TOKEN_ID
    for i in range(len(target_ids)):
        ans_len=len(tokenizer.encode(answers[i]))
        prompt=tokenizer.apply_chat_template(messages[i],tokenize=False,add_generation_prompt=False)
        prompt_len=len(tokenizer.encode(prompt))
        target_ids[i][:prompt_len-ans_len]=IGNORE_TOKEN_ID
    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    return dict(
        input_ids=input_ids, target_ids=target_ids, attention_mask=attention_mask
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int
    ):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        self.example=raw_data
        self.tokenizer=tokenizer
        self.max_len=max_len
        # messages = [example["messages"] for example in raw_data]
        # data_dict = preprocess(messages, tokenizer, max_len)

        # self.input_ids = data_dict["input_ids"]
        # self.target_ids = data_dict["target_ids"]
        # self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.example)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        messages=[self.example[i]['messages']]
        answers=[self.example[i]['answers']]
        data_dict=preprocess_instruct(messages=messages,answers=answers,tokenizer=self.tokenizer,max_len=self.max_len)
        return dict(
            input_ids=data_dict["input_ids"][0],
            labels=data_dict["target_ids"][0],
            attention_mask=data_dict["attention_mask"][0],
        )



def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_path,
    max_len,
    train_ratio=0.99,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_ratio = min(train_ratio, 1.0)
    dataset_cls = (
        SupervisedDataset
    )
    rank0_print("Loading data...")
    if data_path.endswith(".json"):
        raw_data = json.load(open(data_path, "r"))
    elif data_path.endswith(".jsonl"):
        with jsonlines.open(data_path, mode="r") as reader:
            raw_data = [item for item in reader]

    # Split train/test
    np.random.seed(0)
    perm = np.random.permutation(len(raw_data))
    split = int(len(perm) * train_ratio)
    train_indices = perm[:split]
    if train_ratio < 1:
        eval_indices = perm[split:]
    else:
        # if train_ratio==1, we use 5% of data as eval data, make sure trainer will not throw error when eval data is empty
        eval_indices = perm[-int(len(perm) * 0.05) :]
    train_raw_data = [raw_data[i] for i in train_indices]
    eval_raw_data = [raw_data[i] for i in eval_indices]
    rank0_print(f"#train {len(train_raw_data)}, #eval {len(eval_raw_data)}")
    train_dataset = dataset_cls(
        train_raw_data, tokenizer=tokenizer, max_len=max_len
    )
    eval_dataset = dataset_cls(
        eval_raw_data, tokenizer=tokenizer, max_len=max_len
    )
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)

def main(args):

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    if args.wandb:
        import wandb

        wandb.login()
    set_seed(args.seed)

    timeout = InitProcessGroupKwargs(timeout=timedelta(seconds=1_000_000))

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulate_every,
        mixed_precision="bf16",
        log_with="wandb" if args.wandb else None,
        kwargs_handlers=[timeout],
        # fsdp_plugin=fsdp_plugin,
    )
    accelerator.init_trackers(project_name=args.wandb, init_kwargs={"wandb":{"name":args.output_dir.split("/")[-1]}})
    accelerator.print(f"Total GPUS: {accelerator.num_processes}")

    # try:
    #     train_dataset = load_dataset(args.dataset)
    # except:
    #     train_dataset = load_from_disk(args.dataset)
    # if isinstance(train_dataset, DatasetDict):
    #     train_dataset = train_dataset["train"]

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        # device_map=accelerator.device,
        torch_dtype=torch.bfloat16,
        rope_theta=args.rope_theta,
        _attn_implementation="flash_attention_2",
    )

    if args.use_lora:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_target_modules,
            lora_dropout=args.lora_dropout,
            bias=args.lora_bias,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

        # Print peft trainable params
        model.print_trainable_parameters()

        # if args.gradient_checkpointing:
        if args.gradient_accumulate_every>1:
            model.enable_input_require_grads()
    
    tokenizer= AutoTokenizer.from_pretrained(args.model)

    data_dict=make_supervised_data_module(tokenizer=tokenizer,
                                          data_path=args.dataset,
                                          max_len=args.seq_length+1,
                                          )
    train_dataset=data_dict['train_dataset']
    assert isinstance(
        model, (transformers.LlamaForCausalLM, transformers.MistralForCausalLM,transformers.Qwen2ForCausalLM)
    ), "Only support llama and qwen model"
    model_type = (
        "llama" if isinstance(model, transformers.LlamaForCausalLM) else "qwen"
    )
    apply_seq_parallel_monkey_patch(args.parallel_mode, model_type)


    
    # if "input_ids" not in train_dataset.column_names:
    #     raise RuntimeError("Dataset must include an `input_ids` feature")
    # # remove everything that is not input_ids
    # to_remove = [col for col in train_dataset.column_names if col != "input_ids"]
    # train_dataset = train_dataset.remove_columns(to_remove)
    # train_dataset = train_dataset.shuffle(seed=args.seed)
    print("Dataset Size:", len(train_dataset))
    train_loader = DataLoader(
        train_dataset,
        collate_fn=default_data_collator,
        shuffle=True,
        batch_size=args.batch_size,
    )
    if args.learning_rate != 2e-5:
        accelerator.print(f"Warning: You also need to modify accelerate_configs/zero3_offload.json to change the learning rate")
    optim = DummyOptim(model.parameters(), lr=args.learning_rate)
    scheduler = DummyScheduler(
        optim,
        num_training_steps=args.max_train_steps,
        total_num_steps=args.max_train_steps,
    )
    model, optim, scheduler = accelerator.prepare(model, optim, scheduler)
    train_loader = prepare_dataloader(args.parallel_mode, train_loader, accelerator)
    model.gradient_checkpointing_enable()

    accelerator.register_for_checkpointing(scheduler)

    accelerator.print(f"Max train steps: {args.max_train_steps}")
    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )
    completed_steps = 0

    model.train()
    loss_func = CrossEntropyLoss(inplace_backward=True)
    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"][..., : args.seq_length + 1][..., :-1]
        target_ids = batch["input_ids"][..., : args.seq_length + 1][..., 1:]
        position_ids = (
            torch.arange(args.seq_length).unsqueeze(0).expand(input_ids.shape[0], -1)
        )
        # shard the input_ids according to the world size and rank according to zig zag attention

        prepared = prepare_seq_parallel_inputs(
            args.parallel_mode,
            input_ids,
            position_ids,
            target_ids,
            accelerator.process_index,
            accelerator.num_processes,
            accelerator.device,
        )
        local_input_ids = prepared["local_input_ids"]
        local_position_ids = prepared["local_position_ids"]
        local_target_ids = prepared["local_target_ids"]

        loss_log = None
        with accelerator.accumulate(model):
            logits = model(
                local_input_ids,
                position_ids=local_position_ids,
            ).logits
            loss = loss_func(
                logits.reshape(-1, logits.shape[-1]), local_target_ids.reshape(-1)
            )
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                # pay attention here. When any seq parallel algo is turned on. This technically only log the very first chunk's loss
                # and what is the first chunk really depends on how do you shard the sequence
                # for zig zag attention, the first chunk contains the left most and rightmost tokens
                # so you cannot compare the (logged) loss of dist attention and zigzag ring attention.
                # loss_log = {"loss": loss.item(), "ppl": math.exp(loss.item())}

                # we now try gathered loss to verify if ring attention and dist flash attention produce the same loss
                # this may slow down the training
                gathered_loss = accelerator.reduce(loss.clone().detach(), "mean")
                loss_log = {
                    "loss": gathered_loss.item(),
                    "ppl": math.exp(gathered_loss.item()),
                }
                accelerator.log(loss_log, step=completed_steps)

            optim.step()
            scheduler.step()
            optim.zero_grad()

        if accelerator.sync_gradients:
            progress_bar.update(1)
            if loss_log is not None:
                progress_bar.set_postfix(loss_log)
            completed_steps += 1

        if completed_steps >= args.max_train_steps:
            break

    accelerator.print(f"Training Finished")
    accelerator.end_training()

    if args.output_dir is not None:
        accelerator.print(f"Saving model to {args.output_dir}")

        accelerator.wait_for_everyone()

        state_dict = accelerator.get_state_dict(model)

        accelerator.unwrap_model(model).save_pretrained(
            f"{args.output_dir}",
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=state_dict,
        )

        accelerator.print(f"Saving Finished")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--batch-size", type=int, default=1)
    args.add_argument("--gradient-accumulate-every", type=int, default=8)
    args.add_argument("--output-dir", type=str, required=True)
    args.add_argument("--wandb", type=str)
    args.add_argument("--seed", type=int, default=42)
    args.add_argument("--max-train-steps", type=int, default=400)
    args.add_argument("--learning-rate", type=float, default=2e-5)
    args.add_argument("--rope-theta", type=float, default=100000)
    args.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf")
    args.add_argument(
        "--dataset",
        type=str,
        default="emozilla/pg_books-tokenized-bos-eos-chunked-65536",
    )
    args.add_argument("--seq-length", type=int, default=16384)
    args.add_argument(
        "--parallel_mode",
        type=str,
        choices=["zigzag_ring_attn", "dist_flash_attn", "ulysses_attn", "data_parallel"],
    )
    args.add_argument("--use-lora", type=bool, default=False)
    args.add_argument("--lora-r", type=int, default=64)
    args.add_argument("--lora-alpha", type=int, default=16)
    args.add_argument("--lora-dropout", type=float, default=16)
    args.add_argument('--lora-target-modules', nargs='+', default=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "up_proj",
                "gate_proj",
                "down_proj",
            ])
    args.add_argument("--lora-bias", type=str, default="none")
    main(args.parse_args())
