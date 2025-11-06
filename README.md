# 🔧 LLM Fine-Tuning Notebook

## 📘 Overview

This repository provides a Jupyter Notebook (`llm_finetuning.ipynb`) for **fine-tuning large language models (LLMs)** on custom datasets.
It supports both **Google Colab** and **local environments**, using frameworks such as **Hugging Face Transformers**, **PyTorch**, and **PEFT (LoRA)**.

---

## 🧩 Features

* ✅ Load any pre-trained model (GPT, LLaMA, Falcon, etc.)
* ✅ Prepare and preprocess custom datasets
* ✅ Fine-tune efficiently using **LoRA / PEFT**
* ✅ Train, evaluate, and save models
* ✅ Resume training or run inference from checkpoints

---

## ⚙️ Environment Setup

### 1️⃣ Install Dependencies

```bash
pip install torch torchvision torchaudio
pip install transformers datasets accelerate
pip install peft bitsandbytes
pip install sentencepiece
```

### 2️⃣ (Optional) Mount Google Drive on Colab

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

## 🚀 How to Use

### 1️⃣ Set Model and Dataset

Edit the following variables in the notebook:

```python
model_name_or_path = "meta-llama/Llama-2-7b-hf"   # or any model you prefer
dataset_path = "./data/custom_dataset.json"       # path to your dataset
output_dir = "./outputs/llama-finetuned"          # output directory
```

Supports `JSON`, `CSV`, or Hugging Face `datasets` format.

---

### 2️⃣ Preprocess Data

Tokenize the dataset with your selected tokenizer:

```python
tokenized_dataset = dataset.map(preprocess_function, batched=True)
```

---

### 3️⃣ Start Training

Train using Hugging Face’s `Trainer` or `accelerate`:

```python
trainer.train()
```

Typical hyperparameters:

```python
num_train_epochs = 3
learning_rate = 2e-5
per_device_train_batch_size = 2
gradient_accumulation_steps = 8
save_steps = 1000
```

---

### 4️⃣ Save Fine-Tuned Model

After training:

```python
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
```

---

### 5️⃣ Run Inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./outputs/llama-finetuned")
tokenizer = AutoTokenizer.from_pretrained("./outputs/llama-finetuned")

input_text = "Explain quantum error mitigation in simple terms."
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 💾 Output Structure

| Directory        | Description                         |
| ---------------- | ----------------------------------- |
| `./outputs/`     | Final fine-tuned model              |
| `./checkpoints/` | Intermediate checkpoints (optional) |
| `./logs/`        | Training logs and TensorBoard files |

---

## 📈 Possible Extensions

* Parameter-efficient fine-tuning (LoRA / QLoRA)
* Model quantization for faster inference
* RLHF integration for instruction alignment
* Multi-GPU distributed fine-tuning using `accelerate launch`

---

## 🧠 Example Command (Script Execution)

```bash
!python train_llm.py \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --dataset_path ./data/custom_dataset.json \
  --num_train_epochs 3 \
  --learning_rate 2e-5 \
  --output_dir ./outputs/llama-finetuned
```
