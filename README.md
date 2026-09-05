# Offline SLM Edge Deployment & RAG Pipeline

A complete, hands-on pipeline that turns **Gemma-3 1B** into a fast, private,
**fully offline** oil-field service assistant running on a plain laptop CPU —
no GPU, no NPU, no internet connection.

Built as a 3-hour college workshop on Small Language Models, Edge AI and local
inference.

> **Participants: start with [`WORKSHOP_GUIDE.md`](WORKSHOP_GUIDE.md), not this file.**
> It walks you through setup and all five demos step by step.
>
> **Running the workshop? See [`ORGANIZER.md`](ORGANIZER.md)** — session timings,
> how to build and prune the participant bundle, and what tends to go wrong.

---

## What this demonstrates

Three independent levers for getting a useful model onto edge hardware, applied
one at a time so you can see exactly what each one buys:

| Lever | Gives you | Costs you |
|---|---|---|
| **QLoRA fine-tuning** | Behaviour, format, house style | Training time, up front |
| **RAG** | Facts, sources, updates without retraining | Latency, on every query |
| **Quantization** | Speed and size | A little precision |

### Measured results

Benchmarked on a 16-core Intel Core Ultra laptop CPU, no GPU:

| | Size on disk | RAM | Generation speed |
|---|---|---|---|
| Base model (PyTorch, fp32) | ~1.9 GB | 3.7 GB | **6.4 tok/s** |
| Quantized (GGUF, Q4_K_M) | **777 MB** | ~0.9 GB | **~30 tok/s** |

**~2.5× smaller, ~5× faster, on the same CPU.** The fine-tuned adapter that
supplies the model's behaviour is **4.6 MB** of weights — 0.24% of the
model's parameters.

---

## The five demos

Each changes exactly one thing from the previous:

| Demo | Script | What is new |
|---|---|---|
| 1 | `tests/test_00_base_slm.py` | Nothing — the raw model |
| 2 | `tests/test_02_adapter.py` | **+ Fine-tuning** (behaviour) |
| 3 | `tests/test_03_adapter_rag.py` | **+ RAG** (facts) |
| 4 | `tests/test_04_gguf.py` | **+ Quantization** (speed) |
| 5 | `04_gguf_rag.py` | **Everything together** |
| — | `tests/test_01_base_slm_rag.py` | Bonus: RAG *without* fine-tuning |

---

## The build pipeline

| Script | Does | Runtime |
|---|---|---|
| `00_download_base.py` | Downloads the LLM + embedding model | organiser only |
| `01_rag_baseline.py` | Parses 32 documents → 592 vector chunks | ~1 min |
| `02_finetune_qlora.py` | Trains a LoRA adapter on 120 tickets | 30–90 min |
| `03_quantize_gguf.py` | Merge → GGUF → 4-bit quantize | ~2 min |
| `verify_setup.py` | Checks everything is in place | seconds |

```
data/rag/*.pdf .docx .html .json ──► [01] ──► chroma_db/  ──┐
                                                             ├──► [04/05] answers
models/it (base) ──► [02] ──► adapter ──► [03] ──► GGUF ────┘
```

---

## Quick start

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python verify_setup.py
```

Then follow [`WORKSHOP_GUIDE.md`](WORKSHOP_GUIDE.md).

> **`requirements.txt` starts with an `--extra-index-url` line. Do not remove it.**
> `llama-cpp-python` ships no wheel on PyPI, so without that index pip tries to
> compile it from C++ source and fails unless you have MSVC Build Tools.

---

## Requirements

- Windows 10 / 11, **Python 3.13**
- 8 GB RAM minimum (16 GB comfortable), 10 GB free disk
- Internet for setup only — the workshop itself runs offline
- `models/` is distributed separately (too large for git)
- Building the bundle yourself needs a HuggingFace login — Gemma is a gated
  model. See [`ORGANIZER.md`](ORGANIZER.md). Participants do not need one.

---

## Repository layout

```
├── config.py                 # Every setting and prompt, in one place
├── requirements.txt          # Pinned, tested dependency set
├── verify_setup.py           # Pre-flight check
├── 00–04_*.py                # The build pipeline
├── inspect_system.py         # Hardware snapshot
├── tests/                    # The five demos
├── data/
│   ├── rag/                  # 8 scenarios x 4 formats = 32 documents
│   └── finetune/train.jsonl  # 120 maintenance tickets
├── llama.cpp/                # Pre-compiled inference binaries (Windows)
├── llama_source/             # GGUF conversion tooling
├── models/                   # Supplied separately — not in git
├── chatbot_question_bank.md  # 100+ scenario questions to try
└── WORKSHOP_GUIDE.md         # ← the participant guide
```

Generated at runtime: `chroma_db/` (vector database), `results/` (interaction
logs), `analysis/` (size and timing reports).

---

## Configuration

Everything tunable lives in [`config.py`](config.py):

| Setting | Default | Does what |
|---|---|---|
| `ACTIVE_MODEL_ID` | `google/gemma-3-1b-it` | Which model to use |
| `TOP_K` | `3` | Document chunks retrieved per question |
| `TEMPERATURE` | `0.3` | 0.0 repeatable → 1.0 creative |
| `MAX_TOKENS` | `450` | Longest answer allowed |
| `N_CTX` | `4096` | Context window; raise if answers truncate |
| `CHUNK_SIZE` | `1024` | Document chunk size, **in characters** |
| `DETERMINISTIC_MODE` | `False` | `True` forces identical answers |

Alternative models (Qwen 0.5B/1.5B/1.7B) are listed commented in `config.py`.
Switching requires re-running the full `00 → 03` pipeline.

---

## Troubleshooting

See the troubleshooting section of [`WORKSHOP_GUIDE.md`](WORKSHOP_GUIDE.md).
The most common issues:

| Symptom | Fix |
|---|---|
| `GatedRepoError` / `401` on download | Gemma is gated — see [`ORGANIZER.md`](ORGANIZER.md). Participants never hit this. |
| `ModuleNotFoundError: llama_cpp` | Install with the extra index URL above |
| `No quantized model found` | Run `03_quantize_gguf.py` |
| ChromaDB errors | Run `01_rag_baseline.py --rebuild` |
| Answers cut off mid-sentence | Raise `N_CTX` to `8192` in `config.py` |
| `[WinError 206] path too long` | Use `subst X: "<long path>"`, then work from `X:` |

---

## Notes on the design

- **CPU only, by choice.** `n_gpu_layers=0` throughout. The point is what runs
  on hardware people already own.
- **Retrieval is graded, not binary.** `config.py` defines three distance
  thresholds; the closer the match, the more strongly the model is told to stick
  to the retrieved documents (`EXACT` → `STRONG` → `MODERATE` → `NO_RAG`).
  Those thresholds assume unit-normalised embeddings, which the default
  MiniLM model produces.
- **The fine-tune teaches format, not facts.** Training targets are built only
  from fields the dataset actually contains — a summary and the recorded
  resolution steps. Causes need evidence, and evidence is RAG's job.
