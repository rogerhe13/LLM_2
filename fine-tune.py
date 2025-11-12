import os
import json
from pathlib import Path

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
TRAIN_PATH = "./data_cnn/cnn_train_1k.jsonl"
VAL_PATH = "./data_cnn/cnn_test_15.jsonl"
OUTPUT_DIR = "lora-sum-smollm2"


def make_prompt(article: str) -> str:
    return (
        "### Instruction:\n"
        "Summarize the article below in 3-4 sentences.\n\n"
        "### Article:\n"
        f"{article}\n\n"
        "### Summary:\n"
    )


def load_local_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def _keep_first_n_sentences(text: str, n: int = 4) -> str:
    """
    Keep at most n sentences from text.
    """
    # crude sentence split
    import re

    # split by . ? !
    parts = re.split(r"([\.!?])", text)
    # parts looks like ['sentence1', '.', 'sentence2', '.', ...]
    sentences = []
    cur = ""
    for chunk in parts:
        cur += chunk
        if chunk in [".", "!", "?"]:
            sentences.append(cur.strip())
            cur = ""
    if cur.strip():
        sentences.append(cur.strip())
    return " ".join(sentences[:n]).strip()


def run_inference_on_15(model, tokenizer, val_rows, device, out_path: str):
    """
    Run generation on the 15 validation samples and save results to JSON.
    """
    model.eval()
    results = []
    for i, ex in enumerate(val_rows):
        article = ex["article"]
        ref = ex.get("highlights", "")

        prompt = make_prompt(article)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,  # short but enough for 3-4 sentences
                do_sample=False,
                temperature=1.0,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )

        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if full_text.startswith(prompt):
            pred = full_text[len(prompt):].strip()
        else:
            pred = full_text.strip()

        # keep up to 4 sentences, not just the first line
        pred = _keep_first_n_sentences(pred, n=4)

        results.append(
            {
                "id": ex.get("id", i),
                "article": article,
                "reference": ref,
                "prediction": pred,
            }
        )
        print(f"[inference {i+1}/{len(val_rows)}]")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results


def compute_rouge(predictions, references):
    """
    Compute ROUGE scores using huggingface-evaluate.
    """
    try:
        import evaluate
    except ImportError:
        raise RuntimeError("please `pip install evaluate` to compute ROUGE")

    rouge = evaluate.load("rouge")
    res = rouge.compute(predictions=predictions, references=references)
    return res


def main():
    os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("using device:", device)

    train_rows = load_local_jsonl(TRAIN_PATH)
    val_rows = load_local_jsonl(VAL_PATH)

    train_ds = Dataset.from_list(train_rows)
    val_ds = Dataset.from_list(val_rows)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device)

    # (A) baseline
    print("==> running baseline inference on 15 examples ...")
    baseline_results = run_inference_on_15(
        model, tokenizer, val_rows, device, "baseline_outputs.json"
    )
    baseline_preds = [r["prediction"] for r in baseline_results]
    baseline_refs = [r["reference"] for r in baseline_results]
    baseline_rouge = compute_rouge(baseline_preds, baseline_refs)
    print("Baseline ROUGE:", baseline_rouge)

    # (B) LoRA fine-tuning
    print("==> building LoRA model ...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    def preprocess(example):
        article = example["article"]
        target = example.get("highlights", "")

        prompt = make_prompt(article)
        full_text = prompt + target

        tokens = tokenizer(
            full_text,
            max_length=2048,
            truncation=True,
        )
        input_ids = tokens["input_ids"]

        prompt_ids = tokenizer(
            prompt,
            max_length=2048,
            truncation=True,
        )["input_ids"]
        labels = input_ids.copy()
        labels[: len(prompt_ids)] = [-100] * len(prompt_ids)
        tokens["labels"] = labels
        return tokens

    train_tok = train_ds.map(preprocess, remove_columns=train_ds.column_names)
    val_tok = val_ds.map(preprocess, remove_columns=val_ds.column_names)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        learning_rate=2e-4,
        fp16=(device.type == "cuda"),
        logging_steps=10,
        save_steps=500,
        save_total_limit=1,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=data_collator,
    )

    print("==> start training ...")
    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("LoRA fine-tune finished. Saved to", OUTPUT_DIR)

    # (C) finetuned inference
    print("==> running finetuned inference on 15 examples ...")
    finetuned_results = run_inference_on_15(
        model, tokenizer, val_rows, device, "finetuned_outputs.json"
    )
    fin_preds = [r["prediction"] for r in finetuned_results]
    fin_refs = [r["reference"] for r in finetuned_results]
    fin_rouge = compute_rouge(fin_preds, fin_refs)
    print("Finetuned ROUGE:", fin_rouge)


if __name__ == "__main__":
    main()
