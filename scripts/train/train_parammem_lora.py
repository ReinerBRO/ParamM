#!/usr/bin/env python3
"""LoRA SFT training for ParamMem pitfall generator (Table-1 style prep)."""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import List

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)


SYSTEM_PROMPT = (
    "You are a careful coding reviewer. Analyze the task and buggy implementations, "
    "then output concise pitfalls as numbered bullet points."
)


def _patch_optimizer_train_eval() -> None:
    # Some torch/transformers combinations (common on NPU images) call
    # optimizer.train()/eval() even when AdamW doesn't implement them.
    if not hasattr(torch.optim.Optimizer, "train"):
        setattr(torch.optim.Optimizer, "train", lambda self, mode=True: self)
    if not hasattr(torch.optim.Optimizer, "eval"):
        setattr(torch.optim.Optimizer, "eval", lambda self: self)


def read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def format_sample(rec: dict) -> tuple[str, str]:
    func_sign = rec["func_sign"].strip()
    doc = rec["docstring"].strip()
    bugs = rec["buggy_implementations"]
    pitfalls = rec["pitfalls"]

    bug_text = "\n\n".join([f"[Buggy #{i+1}]\n{b.strip()}" for i, b in enumerate(bugs)])
    user_prompt = (
        "Task signature and docstring:\n"
        f"{func_sign}\n\n"
        f"Docstring:\n{doc}\n\n"
        "Buggy implementations:\n"
        f"{bug_text}\n\n"
        "Please summarize the key implementation pitfalls."
    )
    target = "\n".join([f"{i+1}. {p.strip()}" for i, p in enumerate(pitfalls)])
    return user_prompt, target


def build_train_texts(rows: List[dict], tokenizer: AutoTokenizer) -> List[str]:
    texts = []
    for r in rows:
        user_prompt, target = format_sample(r)
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": target},
        ]
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        else:
            text = (
                f"[SYSTEM]\n{SYSTEM_PROMPT}\n\n"
                f"[USER]\n{user_prompt}\n\n"
                f"[ASSISTANT]\n{target}"
            )
        texts.append(text)
    return texts


@dataclass
class TokenizeCfg:
    max_length: int


def tokenize_dataset(ds: Dataset, tokenizer: AutoTokenizer, cfg: TokenizeCfg) -> Dataset:
    def _tok(batch: dict) -> dict:
        out = tokenizer(
            batch["text"],
            truncation=True,
            max_length=cfg.max_length,
            padding=False,
        )
        out["labels"] = out["input_ids"].copy()
        return out

    return ds.map(_tok, batched=True, remove_columns=["text"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--per_device_batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--log_steps", type=int, default=10)
    parser.add_argument("--lora_r", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--device_backend", choices=["cuda", "npu"], default="npu")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    args = parser.parse_args()

    if args.device_backend == "npu":
        try:
            import torch_npu  # noqa: F401
        except Exception as e:  # noqa: BLE001
            raise RuntimeError("device_backend=npu but torch_npu is unavailable") from e

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    random.seed(args.seed)
    _patch_optimizer_train_eval()

    rows = read_jsonl(args.train_jsonl)
    if args.max_train_samples > 0:
        rows = rows[: args.max_train_samples]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    texts = build_train_texts(rows, tokenizer)
    raw_ds = Dataset.from_dict({"text": texts})
    train_ds = tokenize_dataset(raw_ds, tokenizer, TokenizeCfg(max_length=args.max_length))

    torch_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    if args.gradient_checkpointing:
        model.enable_input_require_grads()

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        logging_steps=args.log_steps,
        logging_first_step=True,
        save_steps=args.save_steps,
        save_total_limit=3,
        bf16=args.bf16,
        fp16=args.fp16,
        dataloader_num_workers=8,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=args.gradient_checkpointing,
        disable_tqdm=True,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    with open(os.path.join(args.output_dir, "train_meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_path": args.model_path,
                "train_jsonl": args.train_jsonl,
                "num_samples": len(rows),
                "max_length": args.max_length,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lr": args.lr,
                "epochs": args.epochs,
                "warmup_ratio": args.warmup_ratio,
                "per_device_batch_size": args.per_device_batch_size,
                "grad_accum": args.grad_accum,
                "seed": args.seed,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
