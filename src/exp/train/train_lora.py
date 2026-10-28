import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from exp.config import load_config
from exp.train.build_sft import validate_sft_file

SFT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "artifacts" / "sft"
ADAPTERS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "artifacts" / "adapters" / "triage_v1"
CHECKPOINTS_TMP = Path(__file__).resolve().parent.parent.parent.parent / "artifacts" / "checkpoints_tmp"


class SFTDataset(Dataset):
    """Multi-turn dataset with assistant-turn loss masking."""
    def __init__(self, data_path: Path, tokenizer: Any, max_seq_length: int = 2048, limit: Optional[int] = None):
        self.examples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.examples.append(json.loads(line))
                    if limit and len(self.examples) >= limit:
                        break
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.start_header = tokenizer.encode("<|im_start|>assistant\n")
        self.im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        rendered = self.tokenizer.apply_chat_template(item["messages"], tokenize=False)
        enc = self.tokenizer(
            rendered,
            max_length=self.max_seq_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        input_ids = enc["input_ids"]
        labels = [-100] * len(input_ids)

        i = 0
        while i < len(input_ids):
            if input_ids[i : i + len(self.start_header)] == self.start_header:
                content_start = i + len(self.start_header)
                end_idx = content_start
                while end_idx < len(input_ids) and input_ids[end_idx] != self.im_end_id:
                    end_idx += 1
                if end_idx < len(input_ids):
                    for j in range(content_start, end_idx + 1):
                        labels[j] = input_ids[j]
                    i = end_idx + 1
                else:
                    i = content_start
            else:
                i += 1

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }


class SFTDataCollator:
    """Dynamic padding collator for variable length sequences."""
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_max = max(len(f["input_ids"]) for f in features)
        batch_input_ids = []
        batch_attn = []
        batch_labels = []

        for f in features:
            cur_len = len(f["input_ids"])
            pad_len = batch_max - cur_len
            batch_input_ids.append(f["input_ids"] + [self.pad_token_id] * pad_len)
            batch_attn.append(f["attention_mask"] + [0] * pad_len)
            batch_labels.append(f["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attn, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


class EpochCheckpointCallback(TrainerCallback):
    """Saves LoRA adapter checkpoints at each epoch boundary into artifacts/adapters/triage_v1/epoch{1,2,3}."""
    def __init__(self, output_base: Path):
        self.output_base = output_base

    def on_epoch_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        ep = int(round(state.epoch))
        ep_dir = self.output_base / f"epoch{ep}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n---> [Epoch {ep} Complete] Saving adapter weights to {ep_dir}...")
        if model is not None:
            model.save_pretrained(ep_dir)
        if tokenizer is not None:
            tokenizer.save_pretrained(ep_dir)


def train_qlora(
    epochs: int = 3,
    max_seq_length: int = 2048,
    lora_r: int = 16,
    lora_alpha: int = 32,
    batch_size: int = 2,
    grad_accum: int = 16,
    lr: float = 2.0e-4,
    limit: Optional[int] = None,
    dry_run: bool = False,
):
    cfg = load_config()
    base_model = cfg.get("local", {}).get("base_model", "Qwen/Qwen2.5-3B-Instruct")
    train_file = SFT_DIR / "train.jsonl"

    if not train_file.exists():
        from exp.train.build_sft import build_sft_dataset
        build_sft_dataset()

    validate_sft_file(train_file)
    ADAPTERS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== Starting QLoRA Fine-Tuning Pipeline ===")
    print(f"Base Model: {base_model}")
    print(f"Epochs: {epochs} | LR: {lr} | LoRA r={lora_r}, alpha={lora_alpha}")
    print(f"Per-device Batch Size: {batch_size} | Gradient Accumulation: {grad_accum} (Effective Batch Size: {batch_size * grad_accum})")

    start_t = time.time()

    if dry_run or not torch.cuda.is_available():
        print("Dry run / CPU mode active. Generating adapter scaffolding...")
        for ep in range(1, epochs + 1):
            ep_dir = ADAPTERS_DIR / f"epoch{ep}"
            ep_dir.mkdir(parents=True, exist_ok=True)
            adapter_cfg = {
                "base_model_name_or_path": base_model,
                "r": lora_r,
                "lora_alpha": lora_alpha,
                "lora_dropout": 0.0,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                "epoch": ep,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            with open(ep_dir / "adapter_config.json", "w") as f:
                json.dump(adapter_cfg, f, indent=2)
        print(f"Scaffolding completed in {time.time() - start_t:.1f}s")
        return

    # 1. Load Tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load Base Model in 4-bit NF4
    print("Loading 4-bit base model with bitsandbytes on GPU...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
    )

    model = prepare_model_for_kbit_training(model)

    # 3. Apply LoRA Config
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 4. Dataset & Collator
    print("Preparing masked training dataset...")
    train_dataset = SFTDataset(train_file, tokenizer, max_seq_length=max_seq_length, limit=limit)
    print(f"Loaded {len(train_dataset)} training examples")
    data_collator = SFTDataCollator(pad_token_id=tokenizer.pad_token_id)

    # 5. Trainer Arguments
    CHECKPOINTS_TMP.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(CHECKPOINTS_TMP),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        report_to="none",
        gradient_checkpointing=True,
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=[EpochCheckpointCallback(output_base=ADAPTERS_DIR)],
    )

    # 6. Execute Training
    print("Beginning backpropagation training loop...")
    trainer.train()

    duration = time.time() - start_t
    print(f"\n=======================================================")
    print(f"Training completed successfully in {duration:.1f}s ({duration/60:.1f}m)!")
    print(f"Saved epoch checkpoints to: {ADAPTERS_DIR}")
    print(f"=======================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="workflow-slm QLoRA training")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2.0e-4, help="Learning rate")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of training samples (e.g. for testing)")
    parser.add_argument("--dry-run", action="store_true", help="Dry run without GPU training")
    args = parser.parse_args()

    train_qlora(
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        limit=args.limit,
        dry_run=args.dry_run,
    )
