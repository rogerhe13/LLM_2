# LoRA Fine-Tuning for News Article Summarization

## Project Overview

This project fine-tunes a small language model using Low-Rank Adaptation (LoRA) to improve performance on abstractive news article summarization. The base model used is **HuggingFaceTB/SmolLM2-1.7B-Instruct** from HuggingFace, and the fine-tuning dataset is the **CNN/DailyMail 3.0.0** corpus.

**Task**: Given a news article, generate a concise summary in 3-4 sentences.

### Key Features

- **LoRA-based fine-tuning** using the PEFT library for parameter-efficient model adaptation
- **Small model**: 1.7B parameters for reasonable computational requirements
- **Dataset**: 1,000 training examples + 15 validation examples from CNN/DailyMail
- **Metrics**: ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L) for evaluation
- **Unit test**: Standalone script for quick validation (runs in ~5-10 minutes)

---

## Project Structure

```
data_cnn/
├── cnn_train_1k.jsonl          # 1,000 training examples
├── cnn_test_15.jsonl            # 15 validation/test examples
│
Python Scripts:
├── dataset.py                   # Download & prepare CNN/DailyMail dataset
├── fine-tune.py                 # Main fine-tuning script
├── evaluation.py                # Standalone baseline inference script
├── unit_test.py                 # Minimal unit test (~30 steps)
│
Output Files:
├── baseline_outputs.json        # Base model predictions on 15 test samples
├── finetuned_outputs.json       # Fine-tuned model predictions on 15 test samples
├── base_outputs_test15.json     # Alternative baseline output format
├── unit_base_outputs.json       # Unit test baseline outputs
├── unit_finetuned_outputs.json  # Unit test fine-tuned outputs
│
Model Artifacts:
├── lora-sum-smollm2/            # Saved LoRA adapter after full training
├── unit_test_adapter/           # LoRA adapter from unit test
│
Configuration:
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Container setup (optional)
└── README.md                    # This file
```

---

## Installation & Setup

### 1. Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU acceleration) or Apple Silicon with Metal Performance Shaders (MPS)
- At least 16GB of free disk space for model downloads
- 8GB+ GPU memory (or 16GB+ RAM if using CPU)

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Requirements** (save as `requirements.txt`):
```
torch>=2.0.0
transformers>=4.35.0
datasets>=2.14.0
peft>=0.7.0
evaluate>=0.4.0
numpy>=1.24.0
nltk>=3.8
rouge_score>=0.1.2
```

### 3. Download Data

The dataset is automatically downloaded by the scripts using HuggingFace `datasets` library. You do not need to manually download it.

If you want to prepare the data explicitly, run:

```bash
python dataset.py
```

This will create:
- `data_cnn/cnn_train_1k.jsonl` — 1,000 training examples
- `data_cnn/cnn_test_15.jsonl` — 15 validation/test examples

---

## Running the Code

### Option 1: Quick Unit Test (Recommended for Validation)

Run the lightweight unit test to verify your setup works. This trains for only ~30 steps on 20 training samples and evaluates on 3 test samples. Should complete in 5-10 minutes.

```bash
python unit_test.py
```

**Expected Output**:
- `unit_test_out/` directory with training logs
- `unit_test_adapter/` directory with saved LoRA adapter
- `unit_base_outputs.json` and `unit_finetuned_outputs.json` with predictions

**What this validates**:
- Dataset loading works
- Model can be loaded from HuggingFace
- LoRA configuration and training loop execute correctly
- Inference generation works
- Output files are saved properly

---

### Option 2: Full Fine-Tuning Pipeline

Run the complete fine-tuning workflow on the full dataset (1,000 training examples).

```bash
python fine-tune.py
```

**What this does**:

1. **Baseline Inference** (Section A): Runs the pre-trained SmolLM2 model on 15 test examples and computes ROUGE scores. Saves predictions to `baseline_outputs.json`

2. **LoRA Setup** (Section B): 
   - Loads the base model
   - Initializes LoRA with:
     - Rank (r): 16
     - LoRA Alpha: 32
     - Dropout: 0.05
     - Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`
   - Trains for 1 epoch with:
     - Batch size: 1 (per device)
     - Gradient accumulation: 8 steps
     - Learning rate: 2e-4
     - Logging interval: every 10 steps

3. **Fine-tuned Inference** (Section C): Runs the fine-tuned model on the same 15 test examples and computes ROUGE scores. Saves predictions to `finetuned_outputs.json`

**Expected Runtime**: 30-60 minutes on NVIDIA A100/V100 GPU

**Output Files**:
- `lora-sum-smollm2/` — Saved LoRA adapter and tokenizer
- `baseline_outputs.json` — Base model predictions + ROUGE scores
- `finetuned_outputs.json` — Fine-tuned model predictions + ROUGE scores

---

### Option 3: Standalone Baseline Inference

If you only want to run inference with the base model (no fine-tuning):

```bash
python evaluation.py
```

This will:
- Load the pre-trained SmolLM2-1.7B-Instruct model
- Run inference on `data_cnn/cnn_test_15.jsonl`
- Save results to `base_outputs_test15.json`
- Runtime: 5-10 minutes

---

## Code Descriptions

### `dataset.py`
Prepares the CNN/DailyMail 3.0.0 dataset for fine-tuning.
- Downloads 1,000 training examples and 15 test examples
- Saves to JSONL format with columns: `article`, `highlights` (target summary), and `id`
- Uses seed=42 for reproducibility

### `fine-tune.py`
Main fine-tuning pipeline with three stages:
1. **Baseline evaluation**: Inference + ROUGE computation on pre-trained model
2. **LoRA fine-tuning**: Trains LoRA adapter for 1 epoch
3. **Fine-tuned evaluation**: Inference + ROUGE computation on adapted model

**Key Functions**:
- `make_prompt()`: Formats article into instruction-following template
- `load_local_jsonl()`: Loads JSONL dataset files
- `_keep_first_n_sentences()`: Truncates generated summary to 4 sentences max
- `run_inference_on_15()`: Generates summaries and saves to JSON
- `compute_rouge()`: Computes ROUGE-1, ROUGE-2, ROUGE-L scores

### `evaluation.py`
Standalone script for baseline model inference on test data without fine-tuning.
- Useful for quick model evaluation
- Compatible with CPU, CUDA, and MPS devices

### `unit_test.py`
Lightweight end-to-end test that:
- Uses only 20 training + 3 test examples (vs. 1k + 15 in main pipeline)
- Trains for only 30 steps (vs. full 1 epoch)
- Completes in 5-10 minutes
- Saves small adapter to `unit_test_adapter/`
- Validates all components work correctly

---

## Model & Training Details

### Base Model
- **Name**: HuggingFaceTB/SmolLM2-1.7B-Instruct
- **Parameters**: 1.7 billion
- **Task**: Causal language modeling / instruction following
- **Link**: https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct

### LoRA Configuration
```python
lora_config = LoraConfig(
    r=16,                    # Low-rank dimension
    lora_alpha=32,           # Scaling factor
    lora_dropout=0.05,       # Dropout for regularization
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Attention layers
    bias="none",             # Don't tune bias
    task_type="CAUSAL_LM",
)
```

### Training Hyperparameters
| Parameter | Value |
|-----------|-------|
| Per-device batch size | 1 |
| Gradient accumulation steps | 8 (effective batch size: 8) |
| Learning rate | 2e-4 |
| Max epochs | 1 |
| Max sequence length | 2,048 |
| Optimizer | AdamW (default) |
| FP16 precision | Yes (on CUDA) |

### Prompt Template
```
### Instruction:
Summarize the article below in 3-4 sentences.

### Article:
{article_text}

### Summary:
{generated_summary}
```

---

## Expected Results & Evaluation

### Evaluation Metrics
ROUGE scores measure overlap between generated and reference summaries:
- **ROUGE-1**: Unigram (word) overlap
- **ROUGE-2**: Bigram overlap
- **ROUGE-L**: Longest common subsequence (captures sentence structure)

### Typical Performance (on CNN/DailyMail)

| Stage | ROUGE-1 | ROUGE-2 | ROUGE-L |
|-------|---------|---------|---------|
| Baseline (SmolLM2) | ~0.25-0.30 | ~0.10-0.15 | ~0.22-0.28 |
| After LoRA Fine-tune (1k examples) | ~0.28-0.35 | ~0.12-0.18 | ~0.25-0.32 |
| Improvement | +5-15% | +5-15% | +5-15% |

**Note**: Exact results depend on:
- Random seed initialization
- GPU/device used
- Exact sample selection
- Hyperparameter tuning

### Sample Outputs Included

The assignment submission includes:
- `baseline_outputs.json`: 15 examples with base model predictions
- `finetuned_outputs.json`: Same 15 examples with fine-tuned model predictions
- Side-by-side comparison showing quality improvements

---

## Device & GPU Requirements

### GPU (Recommended)
- **Minimum**: 8GB VRAM (NVIDIA RTX 3060, RTX 4060, etc.)
- **Recommended**: 16GB+ VRAM (V100, A100, L40S, etc.)
- **Training time**: 30-60 minutes on mid-range GPU

### CPU
- **Not recommended** for full training (will be very slow)
- **Fine for**: Running evaluation.py or unit_test.py with max_steps=5

### Device Auto-Selection
The code automatically detects available devices:
```python
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():  # Apple Silicon
    device = torch.device("mps")
else:
    device = torch.device("cpu")
```

---

## Troubleshooting

### Issue: "CUDA out of memory"
**Solution**:
- Reduce `per_device_train_batch_size` from 1 to 1 (already minimal)
- Increase `gradient_accumulation_steps` to 16
- Use smaller model: `SmolLM2-360M-Instruct`

### Issue: "Module not found: evaluate"
**Solution**:
```bash
pip install evaluate nltk rouge_score absl-py
python -m nltk.downloader punkt  # Download NLTK data
```

### Issue: "Tokenizer pad_token is None"
**Solution**: Already handled in code:
```python
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

### Issue: Model downloads very slowly
**Solution**: Set cache directory and allow time:
```bash
export HF_HOME=/path/to/cache
python fine-tune.py  # First run downloads model (~3-4 GB)
```

### Issue: Running on Apple Silicon (MPS)
**Solution**: Code already supports MPS:
```python
elif torch.backends.mps.is_available():
    device = torch.device("mps")
```
Note: MPS may not support all operations; CPU fallback will occur.

---

## File Formats

### Input Data Format (JSONL)
Each line is a JSON object:
```json
{
    "id": "0",
    "article": "Full news article text here...",
    "highlights": "Reference summary here...",
    "url": "https://example.com",
    "date": "2023-01-01"
}
```

### Output Prediction Format (JSON)
```json
[
    {
        "id": "0",
        "article": "Full news article text...",
        "reference": "Reference summary...",
        "prediction": "Model-generated summary..."
    },
    ...
]
```

---

## Quick Start Checklist

- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Download dataset: `python dataset.py` (automatic in scripts)
- [ ] Run unit test: `python unit_test.py` (5-10 min)
- [ ] Check `unit_test_adapter/` was created
- [ ] Review `unit_base_outputs.json` and `unit_finetuned_outputs.json`
- [ ] Run full pipeline: `python fine-tune.py` (30-60 min, requires GPU)
- [ ] Compare `baseline_outputs.json` vs `finetuned_outputs.json`
- [ ] Verify ROUGE scores show improvement

---

## Docker Setup (Optional)

### Dockerfile
```dockerfile
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y python3.10 python3-pip git

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

ENTRYPOINT ["python3"]
```

### Build & Run
```bash
docker build -t lora-summarization:latest .
docker run --gpus all -v $(pwd):/app lora-summarization:latest fine-tune.py
```

---

## References

- **PEFT Documentation**: https://huggingface.co/docs/peft/en/index
- **Transformers Documentation**: https://huggingface.co/docs/transformers/
- **LoRA Paper**: https://arxiv.org/abs/2106.09685
- **CNN/DailyMail Dataset**: https://huggingface.co/datasets/cnn_dailymail
- **ROUGE Evaluation**: https://github.com/google-research/google-research/tree/master/rouge

---

## License

This project is provided as-is for educational purposes. The base model (SmolLM2) is licensed under the Apache 2.0 license. The CNN/DailyMail dataset is available under the Apache 2.0 license.

---

## Contact & Support

For issues or questions:
1. Check the **Troubleshooting** section above
2. Verify all dependencies are installed: `pip list | grep -E "(torch|transformers|peft|datasets)"`
3. Ensure you have a compatible GPU (if GPU training)
4. Try the unit test first to isolate issues: `python unit_test.py`

---

**Last Updated**: November 2025

**Assignment**: LLMs Fine-Tuning with LoRA (Assignment 2)

**Model**: HuggingFaceTB/SmolLM2-1.7B-Instruct

**Task**: Abstractive News Article Summarization
