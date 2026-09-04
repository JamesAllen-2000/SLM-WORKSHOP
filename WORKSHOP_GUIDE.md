# 📘 SLM Edge Deployment Workshop: Step-by-Step Guide

Welcome to the Small Language Model (SLM) Edge Deployment Workshop! 

In this workshop, you will learn how to take a raw 1 Billion parameter model (Gemma-3) and transform it into a highly optimized, fully offline, edge-ready AI assistant that runs entirely on your CPU without any internet connection.

This guide will walk you through executing the pipeline step-by-step.

---

## 🛠️ Step 0: Prerequisites & Setup

Before running the scripts, ensure your environment is ready:
1. **Python Environment**: Ensure you are running Python 3.14+ in a virtual environment (`venv`).
2. **Dependencies**: If you haven't already, install the required packages:
   ```powershell
   pip install -r requirements.txt
   ```
3. **Windows Long Paths**: To prevent errors when downloading HuggingFace models, ensure long paths are enabled in Git:
   ```powershell
   git config --global core.longpaths true
   ```
4. **Download `llama.cpp`**: 
   Since we are running entirely offline on CPU/Edge hardware, you need the highly optimized `llama.cpp` engine. 
   - Go to the official releases page: [https://github.com/ggerganov/llama.cpp/releases](https://github.com/ggerganov/llama.cpp/releases)
   - Download the latest pre-compiled Windows zip (e.g., `llama-b4XXX-bin-win-avx2-x64.zip`).
   - Extract the `.zip` file and place all the `.exe` and `.dll` files (specifically `llama-completion.exe`) directly into the root directory of this project or inside a folder named `llama.cpp/`.

---

## 📥 Step 1: Download the Base Model

**Script:** `00_download_base.py`

This script connects to the HuggingFace Hub and downloads the raw `google/gemma-3-1b-it` model and its tokenizer directly to your local `models/` directory.

**Action:**
Run the following command in your terminal:
```powershell
python 00_download_base.py
```
*Wait for the download to complete (approx. 3.4 GB).*

---

## 📚 Step 2: Establish the RAG Baseline

**Script:** `01_rag_baseline.py`

Retrieval-Augmented Generation (RAG) allows the AI to read your private documents. This script takes the engineering PDFs and Markdown files in your `data/docs/` folder and ingests them into a local vector database (`chroma_db`). It then runs a baseline test using the raw model to see how it performs *before* fine-tuning.

**Action:**
Run the following command:
```powershell
python 01_rag_baseline.py
```
*Check the `results/` folder to see the baseline logs.*

---

## 🧠 Step 3: Fine-Tune the Model (QLoRA)

**Script:** `02_finetune_qlora.py`

The raw model is smart, but it doesn't know how to speak like a specialized field technician. This script uses **QLoRA** to fine-tune the model on our custom dataset (`data/finetune/train.jsonl`). It generates a lightweight "Adapter" that overrides the model's behavior without modifying the massive base weights.

**Action:**
Run the following command:
```powershell
python 02_finetune_qlora.py
```
*Note: This runs entirely on your CPU and will take time. You can view the training metrics and time taken in `analysis/finetuning_analysis.md`.*

*(Optional) You can test your fine-tuned adapter using `python tests/test_02_adapter.py`.*

---

## 🗜️ Step 4: Quantize for Edge Deployment (GGUF)

**Script:** `03_quantize_gguf.py`

Running a 3.4 GB model in Python is too slow for an edge device. This script uses `llama.cpp` to completely compile our model for edge execution:
1. It merges our lightweight LoRA adapter permanently into the base model.
2. It converts the model into the optimized `GGUF` format.
3. It **Quantizes** (compresses) the model down to 4-bit (`Q4_K_M`), shrinking it to ~1 GB so your CPU memory can handle it instantly.

**Action:**
Run the following command:
```powershell
python 03_quantize_gguf.py
```
*Check `analysis/quantize_analysis...md` to see the exact time and memory saved by quantization!*

*(Optional) Test the blazing fast quantized model using `python tests/test_03_gguf.py`.*

---

## 🚀 Step 5: The Grand Finale (Final RAG)

**Script:** `04_gguf_rag.py`

This is the final product. It combines our highly-optimized, fine-tuned, 4-bit quantized GGUF model with the ChromaDB vector database. It represents exactly what a field technician would use entirely offline.

**Action:**
Run the following command:
```powershell
python 04_gguf_rag.py
```

**What happens next?**
1. It will automatically answer the standard offline test question.
2. It will drop you into an **Interactive Chat**.
3. You can type any question (e.g., *"What is wrong with the pressure valve?"*). The model will silently reason through the problem in the background, hide its messy `<think>` block, and instantly stream the clean, final answer to your screen!

All interactions are securely logged in `results/04_gguf_rag_log.txt`.

---

🎉 **Congratulations! You have successfully built and deployed a fully offline, edge-ready SLM Assistant!**

---

## 🛠️ Troubleshooting

### Issue: The model stops generating mid-sentence or output is suddenly cut off.
This usually happens if your system prompts or RAG context chunks are extremely large, causing the input to blow past the model's context window limit (default is usually 4096 tokens). Once the limit is reached, `llama.cpp` will forcefully stop generating.

**Solution**: 
Open `config.py` and increase your context limit.
```python
N_CTX = 8192
```
Alternatively, you can reduce the amount of RAG context injected by lowering `TOP_K` (e.g., `TOP_K = 3`) or reducing the `CHUNK_SIZE`.

### Issue: Installation fails with `[Errno 2] No such file or directory` or `[WinError 206] The filename or extension is too long`
This occurs when the absolute path of your project directory combined with the deep folder structures of packages (like `llama-cpp-python` or `torch`) exceeds the strict Windows 260-character path limit.

**Solution: Use a Virtual Drive Letter (Fastest, no moving files)**
You can use the built-in Windows `subst` command to temporarily map your incredibly long project folder path to a simple, short drive letter (like `X:`). This tricks Windows into thinking the path is very short.

1. Open your standard terminal/command prompt and run this exact command (replace the path with your actual project path):
```cmd
subst X: "C:\Users\james.allenraj\OneDrive - Orion Systems Integrators, LLC\Documents\ORION INC\SLM Workshop\SLM WORKSHOP"
```
2. Switch to that new drive by typing:
```cmd
X:
```
3. Activate your virtual environment from there:
```cmd
.\venv\Scripts\activate
```
4. Run your installation again:
```cmd
pip install -r requirements.txt
```

*(Note: To remove the virtual drive later, you just run `subst X: /D`)*


