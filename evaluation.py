# run_base_inference_mac.py
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
TEST_PATH = Path("data_cnn/cnn_test_15.jsonl")
OUT_PATH = Path("base_outputs_test15.json")


def make_prompt(article: str) -> str:
    return (
        "### Instruction:\n"
        "Summarize the following news article in 3-4 sentences.\n\n"
        "### Input:\n"
        f"{article}\n\n"
        "### Output:\n"
    )


def main():

    if torch.mps.is_available():
        device = torch.device("mps")
        use_mps = True
        print("using mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        use_mps = False
        print("using cuda")
    else:
        device = torch.device("cpu")
        use_mps = False
        print("using cpu")


    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if (not device.type == "cpu") else torch.float32,
    )
    model.to(device)
    model.eval()

  
    samples = []
    with TEST_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))

    results = []

    for i, ex in enumerate(samples):
        article = ex["article"]
        ref = ex.get("highlights", "")

        prompt = make_prompt(article)
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=False,
            )

        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

     
        if full_text.startswith(prompt):
            pred = full_text[len(prompt):].strip()
        else:
            pred = full_text.strip()

        results.append(
            {
                "id": ex.get("id", i),
                "article": article,
                "reference": ref,
                "prediction": pred,
            }
        )
        print(f"[{i+1}/{len(samples)}] done.")

    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"saved to {OUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
