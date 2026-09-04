# Offline SLM Edge Deployment & RAG Pipeline

This project demonstrates a complete, end-to-end pipeline for downloading, fine-tuning, quantizing, and deploying a Small Language Model (SLM) for entirely offline use on edge devices (such as a field technician's laptop with no internet access and no GPU).

We use **Gemma-3 (1B)** combined with **Retrieval-Augmented Generation (RAG)** to provide highly technical, context-aware answers based on local engineering manuals and service tickets.

## 🚀 Features

- **End-to-End Pipeline**: Scripts for every stage from raw model download to fine-tuning, quantization, and final RAG deployment.
- **Offline Edge Inference**: Uses `llama.cpp` to run a 4-bit quantized (Q4_K_M) version of the model directly on standard CPU hardware with extremely low memory requirements.
- **Reasoning Model Support**: Built-in Python streaming engines to natively handle and hide reasoning blocks (`<think>`) from advanced models, providing clean outputs to the user.
- **Robust Windows Process Handling**: Hardened `subprocess` management to prevent `SIGINT` (Ctrl+C) broadcasts from `llama.cpp` from crashing the Python runtime.
- **Automated Logging & Analysis**: Automatically tracks inference latency, retrieved RAG sources, and model responses, saving them directly to the `results/` and `analysis/` directories.

---

## 🛠️ Project Structure

### Core Pipeline
1. **`00_download_base.py`**
   Downloads the base Gemma-3 1B model and tokenizer from the HuggingFace Hub.
2. **`01_rag_baseline.py`**
   Initializes the ChromaDB vector database, ingests markdown/PDF engineering documents, and runs a baseline RAG test.
3. **`02_finetune_qlora.py`**
   Fine-tunes the base model on domain-specific service data using QLoRA, generating a parameter-efficient adapter.
4. **`03_quantize_gguf.py`**
   Merges the LoRA adapter into the base model, converts it to FP16 GGUF format, and quantizes it down to a highly optimized `Q4_K_M` GGUF file for CPU execution.
5. **`04_gguf_rag.py`**
   The grand finale. Connects the fully quantized GGUF model to the ChromaDB RAG system. It automatically tests a baseline question and then drops into an interactive, offline chat interface.

### Testing & Verification
Located in the `tests/` directory, these scripts isolate and verify individual components of the pipeline:
- **`test_00_base.py`**: Validates the base model structure.
- **`test_01_rag.py`**: Tests the vector retrieval logic independently.
- **`test_02_adapter.py`**: Tests the LoRA adapter before merging.
- **`test_03_gguf.py`**: A standalone test for the final quantized GGUF model that features a real-time streaming parser and background loading spinners.

---

## 💻 Getting Started

### Prerequisites
- Windows OS (Tested on Windows 11)
- Python 3.14+ (Virtual Environment Recommended)
- `git config --global core.longpaths true` (Required for Windows HuggingFace caching)
- **Pre-compiled `llama.cpp` binaries**:
  - Download the latest Windows release (`llama-bXXXX-bin-win-avx2-x64.zip`) from the [official GitHub releases page](https://github.com/ggerganov/llama.cpp/releases).
  - Extract it and place the `.exe` and `.dll` files (specifically `llama-completion.exe`) directly into the root directory of this project or inside a `llama.cpp/` folder.

### Execution
Run the pipeline in chronological order:
```powershell
python 00_download_base.py
python 01_rag_baseline.py
python 02_finetune_qlora.py
python 03_quantize_gguf.py
python 04_gguf_rag.py
```

### Interactive Field Mode
To launch the final assistant for offline field queries, simply run:
```powershell
python 04_gguf_rag.py
```
This will automatically launch the interactive chat loop. All interactions, retrieved sources, and latencies will be securely logged to `results/04_gguf_rag_log.txt`.

---

## 🛠️ Troubleshooting

### Issue: The model stops generating mid-sentence or output is suddenly cut off
This usually happens if your system prompts or RAG context chunks are extremely large, causing the input to blow past the model's context window limit (default is usually 4096 tokens). Once the limit is reached, the model will forcefully stop generating.

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
