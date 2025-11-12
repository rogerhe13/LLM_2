# unit_test.py
# Minimal end-to-end sanity test:
# - Load a tiny subset from local JSON files
# - Baseline inference on 3 samples (+ ROUGE)
# - LoRA fine-tune for ~30 steps
# - Inference again on the same 3 samples (+ ROUGE)
# - Save LoRA adapter to unit_test_adapter/

import os
import json
import re
from typing import List

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model


MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
OUTPUT_DIR = "unit_test_adapter"
DATA_DIR = "./data_cnn"


def make_prompt(article: str) -> str:
    return (
        "### Instruction:\n"
        "Summarize the article below in 3-4 sentences.\n\n"
        "### Article:\n"
        f"{article}\n\n"
        "### Summary:\n"
    )


def keep_first_n_sentences(text: str, n: int = 4) -> str:
    """Keep at most n sentences (very light-weight splitter)."""
    parts = re.split(r"([\.!?])", text)
    sents, cur = [], ""
    for chunk in parts:
        cur += chunk
        if chunk in [".", "!", "?"]:
            sents.append(cur.strip())
            cur = ""
            if len(sents) >= n:
                break
    if not sents and text.strip():
        sents = [text.strip()]
    return " ".join(sents).strip()


def run_inference(samples: List[dict], model, tokenizer, device, out_path: str):
    """Run greedy decoding on provided samples and save JSON results."""
    model.eval()
    results = []
    for i, ex in enumerate(samples):
        article = ex["article"]
        ref = ex.get("highlights", "")

        prompt = make_prompt(article)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                temperature=1.0,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )

        full = tokenizer.decode(outputs[0], skip_special_tokens=True)
        pred = full[len(prompt):].strip() if full.startswith(prompt) else full.strip()
        pred = keep_first_n_sentences(pred, n=4)

        results.append(
            {
                "id": ex.get("id", i),
                "article": article,
                "reference": ref,
                "prediction": pred,
            }
        )
        print(f"[inference {i+1}/{len(samples)}]")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return results


def compute_rouge(preds, refs):
    """Compute ROUGE via huggingface-evaluate."""
    import evaluate
    rouge = evaluate.load("rouge")
    return rouge.compute(predictions=preds, references=refs)


def preprocess_builder(tokenizer):
    """Return a preprocess fn that masks the prompt tokens with -100."""
    def preprocess(example):
        article = example["article"]
        target = example.get("highlights", "")

        prompt = make_prompt(article)
        full_text = prompt + target

        tok_full = tokenizer(full_text, max_length=1024, truncation=True)
        tok_prompt = tokenizer(prompt, max_length=1024, truncation=True)

        input_ids = tok_full["input_ids"]
        labels = input_ids.copy()
        labels[: len(tok_prompt["input_ids"])] = [-100] * len(tok_prompt["input_ids"])
        tok_full["labels"] = labels
        return tok_full
    return preprocess


def load_local_data():
    """Load training and evaluation data from local JSONL files."""
    train_path = os.path.join(DATA_DIR, "cnn_train_1k.jsonl")
    eval_path = os.path.join(DATA_DIR, "cnn_test_15.jsonl")
    
    if not os.path.exists(train_path) or not os.path.exists(eval_path):
        raise FileNotFoundError(
            f"Data files not found in {DATA_DIR}\n"
            f"Expected: {train_path}\n"
            f"Expected: {eval_path}\n"
            f"Please ensure data_cnn/cnn_train_1k.jsonl and data_cnn/cnn_test_15.jsonl exist"
        )
    
    # Load JSONL format (one JSON object per line)
    with open(train_path, "r", encoding="utf-8") as f:
        train_data = [json.loads(line.strip()) for line in f if line.strip()]
    
    with open(eval_path, "r", encoding="utf-8") as f:
        eval_data = [json.loads(line.strip()) for line in f if line.strip()]
    
    # Convert to HuggingFace Dataset
    train_dataset = Dataset.from_list(train_data)
    eval_list = eval_data  # Keep as list for inference
    
    return train_dataset, eval_list


def main():
    # Make transformers avoid importing torchvision (not needed here).
    os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"

    # Device selection.
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("using device:", device)

    # Load data from local files instead of downloading
    print("==> Loading local test data...")
    train_small, eval_list = load_local_data()
    print(f"Loaded {len(train_small)} training samples and {len(eval_list)} eval samples")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model (float16 on CUDA, float32 otherwise).
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device)

    # A) Baseline inference + ROUGE on eval examples.
    print("==> Baseline inference (baseline model)")
    base_out = run_inference(eval_list, model, tokenizer, device, "unit_base_outputs.json")
    base_preds = [r["prediction"] for r in base_out]
    base_refs = [r["reference"] for r in base_out]
    try:
        base_rouge = compute_rouge(base_preds, base_refs)
        print("Baseline ROUGE:", base_rouge)
    except Exception as e:
        print("ROUGE computation failed (install evaluate+nltk+rouge_score+absl):", e)

    # B) Attach a small LoRA and do a very short training (~30 steps).
    print("==> Building LoRA adapter (tiny)")
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    preprocess = preprocess_builder(tokenizer)
    train_tok = train_small.map(preprocess, remove_columns=train_small.column_names)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir="unit_test_out",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,  # effective bs ~= 4
        max_steps=30,                   # keep it very short
        learning_rate=2e-4,
        fp16=(device.type == "cuda"),
        logging_steps=10,
        save_strategy="no",
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        data_collator=collator,
    )

    trainer.train()

    # Save the tiny adapter.
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Adapter saved to ./{OUTPUT_DIR}")

    print("==> Finetuned inference (finetuned model)")
    ft_out = run_inference(eval_list, model, tokenizer, device, "unit_finetuned_outputs.json")
    ft_preds = [r["prediction"] for r in ft_out]
    ft_refs = [r["reference"] for r in ft_out]
    try:
        ft_rouge = compute_rouge(ft_preds, ft_refs)
        print("Finetuned ROUGE:", ft_rouge)
    except Exception as e:
        print("ROUGE computation failed (install evaluate+nltk+rouge_score+absl):", e)

    print("\n==> Unit test completed successfully!")
    print(f"Results saved to: unit_base_outputs.json, unit_finetuned_outputs.json")


if __name__ == "__main__":
    main()
