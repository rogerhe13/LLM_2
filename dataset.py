# prepare_cnn_data.py
# download CNN/DailyMail 3.0.0，sample 1000 training data + 15 validation data

import os
from datasets import load_dataset

def main():
    out_dir = "data_cnn"
    os.makedirs(out_dir, exist_ok=True)

    dataset = load_dataset("cnn_dailymail", "3.0.0")

    train_1k = (
        dataset["train"]
        .shuffle(seed=42)   
        .select(range(1000))
    )

    test_15 = (
        dataset["validation"]
        .shuffle(seed=42)
        .select(range(15))
    )

    train_path = os.path.join(out_dir, "cnn_train_1k.jsonl")
    test_path = os.path.join(out_dir, "cnn_test_15.jsonl")

    train_1k.to_json(train_path, orient="records", lines=True, force_ascii=False)
    test_15.to_json(test_path, orient="records", lines=True, force_ascii=False)

if __name__ == "__main__":
    main()
