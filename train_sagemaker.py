import os, sys, argparse, zipfile
from pathlib import Path
from collections import OrderedDict

from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments

# === allow `import src.*` ===
# project_root = Path(__file__).parent.resolve()
# src_path     = project_root / "src"
# sys.path.insert(0, str(src_path))

import src.dataset as dataset
import src.domlm as model
from src.data_collator import DataCollatorForDOMNodeMask

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-data-dir",  type=str, default="/opt/ml/input/data/train")
    p.add_argument("--output-dir",       type=str, default="/opt/ml/model")
    p.add_argument("--epochs",           type=int,   default=5)
    p.add_argument("--per-device-batch", type=int,   default=16)
    p.add_argument("--gradient-accum",   type=int,   default=1)
    p.add_argument("--mlm-prob",         type=float, default=0.15)
    return p.parse_args()

def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    roberta   = AutoModel.from_pretrained("roberta-base")
    cfg_dict  = roberta.config.to_dict()
    cfg_dict.update({
        "_name_or_path": "domlm",
        "architectures": ["DOMLMForMaskedLM"],
    })
    domlm_cfg = model.DOMLMConfig.from_dict(cfg_dict)
    domlm     = model.DOMLMForMaskedLM(domlm_cfg)

    # load weights
    sd = OrderedDict((f"domlm.{k}", v) for k,v in roberta.state_dict().items())
    domlm.load_state_dict(sd, strict=False)

    # datasets
    train_ds = dataset.SWDEDataset(args.train_data_dir)
    eval_ds  = dataset.SWDEDataset(args.train_data_dir, split="test")

    collator = DataCollatorForDOMNodeMask(
        tokenizer=tokenizer,
        mlm_probability=args.mlm_prob,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch,
        per_device_eval_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.gradient_accum,
        evaluation_strategy="steps",
        weight_decay=0.01,
        warmup_ratio=0.1,
        save_steps=1000,
        learning_rate=1e-4,
        fp16=True,
        bf16=False,
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
    )

    trainer = Trainer(
        model=domlm,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )
    trainer.train()

if __name__ == "__main__":
    main()
