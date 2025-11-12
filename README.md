# LoRA Fine-Tuning for News Article Summarization (LLM Assignment 2)
Student : Weihao He
## Project Overview

This project fine-tunes a small language model using Low-Rank Adaptation (LoRA) to improve performance on abstractive news article summarization. The base model used is **HuggingFaceTB/SmolLM2-1.7B-Instruct** from HuggingFace, and the fine-tuning dataset is the **CNN/DailyMail 3.0.0** corpus.

**Task**: Given a news article, generate a concise summary in 3-4 sentences.

### Key Features

- **LoRA-based fine-tuning** using the PEFT library for parameter-efficient model adaptation
- **Small model**: 1.7B parameters for reasonable computational requirements
- **Dataset**: 1,000 training examples + 15 validation examples from CNN/DailyMail
- **Metrics**: ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L) for evaluation
- **Unit test**: Standalone script for quick validation (runs in ~5-10 minutes)
- **Docker support**: Containerized setup for easy reproducibility


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
├── Dockerfile                   # Container setup
├── .dockerignore                # Docker build exclusions
└── README.md                    # This file
```


## Installation & Setup

### 1. Prerequisites

- Python 3.10+
- CUDA 11.8+ 
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


## Docker Setup

You can run the unit test or full training pipeline inside a Docker container with all dependencies pre-configured.

### Prerequisites
- Docker 19.03+
- NVIDIA Docker runtime (for GPU support)
- NVIDIA Container Toolkit (https://github.com/NVIDIA/nvidia-docker)

### Build Docker Image

```bash
docker build -t llm2-summarization:latest .
```

### Run Unit Test in Docker

```bash
docker run --gpus all llm2-summarization:latest python3 unit_test.py
```

This will:
- Run the lightweight unit test (5-10 minutes)
- Generate `unit_test_adapter/`, `unit_base_outputs.json`, and `unit_finetuned_outputs.json`
- Display ROUGE scores in the console

### Run Full Fine-Tuning in Docker

```bash
docker run --gpus all llm2-summarization:latest python3 fine-tune.py
```

This will execute the complete fine-tuning pipeline.

### Save Output Files Locally

To preserve generated files after the container finishes:

```bash
mkdir -p outputs
docker run --gpus all -v $(pwd)/outputs:/app/outputs llm-summarization:latest python3 unit_test.py
```

This mounts a local `outputs/` directory to the container, saving all predictions and adapters.


## Running the Code

### Option 1: Quick Unit Test

Run the lightweight unit test to verify your setup works. This trains for only ~30 steps on 20 training samples and evaluates on 3 test samples. Should complete in 5-10 minutes.

```bash
python unit_test.py
```

**Expected Output**:
- `unit_test_out/` directory with training logs
- `unit_test_adapter/` directory with saved LoRA adapter
- `unit_base_outputs.json` and `unit_finetuned_outputs.json` with predictions



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


**Output Files**:
- `lora-sum-smollm2/` — Saved LoRA adapter and tokenizer
- `baseline_outputs.json` — Base model predictions + ROUGE scores
- `finetuned_outputs.json` — Fine-tuned model predictions + ROUGE scores


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
| FP16 precision | Yes|

### Prompt Template
```
### Instruction:
Summarize the article below in 3-4 sentences.

### Article:
{article_text}

### Summary:
{generated_summary}
```

## Results & Evaluation

### Evaluation Metrics
ROUGE scores measure overlap between generated and reference summaries:


### Performance on CNN/DailyMail

| Stage | ROUGE-1 | ROUGE-2 | ROUGE-L |
|-------|---------|---------|---------|
| Baseline (SmolLM2) | ~0.36 | ~0.126 | ~0.23 |
| After LoRA Fine-tune (1k examples) | ~0.38 | ~0.14 | ~0.26 |

**Note**: Exact results depend on:
- Random seed initialization
- GPU/device used
- Exact sample selection
- Hyperparameter tuning

### Sample Outputs

### Example 1: Abstractiveness and Conciseness Improvement

**Original Article (excerpt):**
"Jarryd Hayne's move to the NFL is a boost for rugby league in the United States, it has been claimed. The Australia international full-back or centre quit the National Rugby League in October to try his luck in American football and was this week given a three-year contract with the San Francisco 49ers. Peter Illfield, chairman of US Association of Rugby League, said: 'Jarryd, at 27, is one of the most gifted and talented rugby league players in Australia. He is an extraordinary athlete. Jarryd Hayne (right) has signed with the San Francisco 49ers after quitting the NRL in October . Hayne, who played rugby league for Australia, has signed a three year contract with the 49ers . 'His three-year deal with the 49ers, as an expected running back, gives the USA Rugby League a connection with the American football lover like never before. 'Jarryd's profile and playing ability will bring our sport to the attention of many. It also has the possibility of showing the American college athlete the possibilities of transition and adaptation for them to play rugby league, should they desire. 'Part of our recruitment strategy is aimed at the American football player who has excelled at High School level but just misses out on their College football team in their Freshman year. Hayne could play at full back or centre in rugby league and is expected to be a running back for the 49ers . 'There is no community football for that high-level of athlete. Rugby league is the perfect sport for him and we now have Jarryd as a first-hand role model.' Illfield has invited Hayne to be a guest of honour at the USARL fixtures in their 14-club competition over the summer, adding: 'We are looking at every source for increasing performance outcomes for the USA national team leading up to the 2017 Rugby League World Cup in Australia and New Zealand.'"

**Reference Summary:**
"Jarryd Hayne quit the NRL in October to try and get into American Football. This week, he signed a three-year contract with the San Francisco 49ers. The chairman of the US Association of Rugby League welcomed his arrival."

**Baseline Model Output:**
"Hayne's move to the NFL is seen as a boost for rugby league in the United States, according to Peter Illfield, chairman of the US Association of Rugby League. The 27-year-old Australian full-back or centre quit the National Rugby League in October to try his luck in American football and was given a three-year contract with the San Francisco 49ers. Hayne, who played rugby league for Australia, is expected to play as a running back for the 49ers. Illfield believes Hayne's profile and playing ability will bring rugby league to the attention of many"

**Fine-tuned Model Output:**
"Hayne has signed a three-year contract with the San Francisco 49ers. The 27-year-old is one of the most gifted rugby league players in Australia. Hayne is expected to play as a running back for the 49ers. Hayne could play at full back or centre in rugby league."

**Fine-tuned Analysis:**
The fine-tuned model produces a more concise and focused summary (word count: ~40), though not perfect:
- Better conciseness: Reduces unnecessary elaboration
- More direct statements: Focuses on key facts (contract signing, position)
- Closer to reference length: Comparable to the reference summary structure
- Some minor issues: Still includes speculative details ("could play at full back or centre") not explicitly in the article










## File Formats

### Input Data Format (JSONL)
Each line is a JSON object:
```json
{
    "id": "0",
    "article": "Full news article text here...",
    "highlights": "Reference summary here...",
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


## Quick Start Checklist

- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Download dataset: `python dataset.py` (automatic in scripts)
- [ ] Run unit test: `python unit_test.py` 
- [ ] Check `unit_test_adapter/` was created
- [ ] Review `unit_base_outputs.json` and `unit_finetuned_outputs.json`
- [ ] Run full pipeline: `python fine-tune.py` 
- [ ] Compare `baseline_outputs.json` vs `finetuned_outputs.json`
- [ ] Verify ROUGE scores show improvement


## References

- **PEFT Documentation**: https://huggingface.co/docs/peft/en/index
- **Transformers Documentation**: https://huggingface.co/docs/transformers/
- **LoRA Paper**: https://arxiv.org/abs/2106.09685
- **CNN/DailyMail Dataset**: https://huggingface.co/datasets/cnn_dailymail
- **ROUGE Evaluation**: https://github.com/google-research/google-research/tree/master/rouge


**Date**: November 2025

**Assignment**: LLMs Fine-Tuning with LoRA (Assignment 2)
