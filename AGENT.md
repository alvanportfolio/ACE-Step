You are a professional AI repo analyst and documentation specialist.  
This project is a Python-based **text-to-music AI training and inference system** called `ACE-Step`.

You will now analyze this repository and generate a complete `DEV_GUIDE.md` for future contributors.  
The developer guide should explain the architecture, CLI usage, training/inference logic, and file purposes.

---

## 🔍 Phase 1: Project Overview

You are provided with this root-level structure:

acestep/  
assets/  
config/  
data/  
examples/  
zh_lora_dataset/  

.gitignore  
CLI_GUIDE.md  
colab_inference.ipynb  
convert2hf_dataset.py  
docker-compose.yaml  
Dockerfile  
infer-api.py  
infer.py  
inference.ipynb  
LICENSE  
README.md  
requirements.txt  
setup.py  
train_cli_advanced.py  
train_cli.py  
TRAIN_INSTRUCTION.md  
trainer-api.py  
trainer.py  
ZH_RAP_LORA.md  

From this, do the following:

- Identify the **primary language(s)** (expect Python).  
- Determine the **main project purpose** (AI model for music generation or lyrics-to-music transformation).  
- Identify tools used (e.g. PyTorch, LoRA, Docker, Hugging Face, Colab).  
- Determine if it supports **CLI**, **API**, **notebook**, or **training scripts**.  
- Exclude static or irrelevant folders (`assets`, `__pycache__`, etc.).

Write a **short Executive Summary** that includes:

- Language stack  
- Purpose of the project  
- AI/ML goal  
- Deployment or runtime usage (e.g., CLI-based training, API-based inference)

---

## 📁 Phase 2: File-Level Summaries

For each relevant `.py`, `.ipynb`, and `.md` file:

1. What is this file’s **purpose**?  
2. What are the **main classes/functions**?  
3. Inputs / outputs / arguments (if script)  
4. CLI args (if present)  
5. How this fits into the project (e.g., data prep, model training, inference, CLI launcher)

Prioritize:

- train_cli.py, train_cli_advanced.py  
- trainer.py, trainer-api.py  
- infer.py, infer-api.py, inference.ipynb  
- convert2hf_dataset.py, colab_inference.ipynb  
- setup.py, requirements.txt, Dockerfile, docker-compose.yaml  
- CLI_GUIDE.md, README.md, TRAIN_INSTRUCTION.md

Ignore:

- Any `.png`, `.jpg`, or static assets  
- Folders not referenced in code or docs  

---

## 🗂️ Phase 3: Directory Summaries

Summarize each of these key folders:

- acestep/: core logic?  
- config/: YAML or Python configs? Used for training?  
- data/: sample or real datasets?  
- zh_lora_dataset/: language-specific data? LoRA targets?  
- examples/: quick test samples?

For each:

- Explain its role  
- Mention key files  
- Describe how the folder connects to training/inference pipeline  

Skip summarizing folders like `assets/` unless code directly references them.

---

## 🧭 Phase 4: Output `DEV_GUIDE.md`

Create a **clean, professional Markdown document** with this structure:

# ACE-Step Developer Guide

## 1. Executive Summary  
- Purpose  
- Language stack  
- AI model goal (e.g., lyrics-to-music or LoRA-based music model)  
- Main usage patterns (CLI, inference API, notebooks)

## 2. Architecture Overview  
- High-level diagram of project folder layout (in text or indented tree)  
- Description of each layer (e.g., training core, config, data, CLI scripts)

## 3. Setup Instructions  
- Python version and dependencies  
- `requirements.txt` install  
- Docker/Colab options  
- CLI usage (summarize `CLI_GUIDE.md`)  
- Any init scripts or setup logic

## 4. Training Pipeline  
- How training is triggered (e.g., train_cli.py)  
- Data format expected (e.g., dataset folder structure)  
- Key config files or arguments  
- LoRA training, if used  
- Hugging Face conversion, if relevant

## 5. Inference  
- infer.py, infer-api.py usage  
- Notebook-based (inference.ipynb, colab_inference.ipynb) steps  
- API endpoints or command-line examples

## 6. Scripts Overview  
- Purpose of each major Python script  
- CLI arguments (if applicable)  
- File-by-file roles (from Phase 2)

## 7. Contributing & Dev Tips  
- Where to add new training logic  
- How to modify model code  
- How to debug CLI/inference  
- Folder structure rules

---

## ❗Final Rules

- Only analyze `.py`, `.md`, `.ipynb`, `.yaml`, `.toml`, and `.json` files  
- Ignore: images, `.dll`, `.exe`, soundbanks, or any asset folders  
- Do **not** execute any code  
- You may simulate light `import` resolution to trace script connections  
- Assume GPU is required — **do not attempt full runs or downloads**
