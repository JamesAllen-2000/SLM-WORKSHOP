# SLM Edge Deployment Workshop — Participant Guide

Turn a 1-billion-parameter language model into a fast, private, **fully offline**
field-service assistant that runs on a plain laptop CPU — no GPU, no NPU, no internet.

---

## Read this first

| | When | What | How long |
|---|---|---|---|
| **PART A — Setup** | **At home, before the workshop** | Install packages, copy models, verify | **~20 minutes** |
| **PART B — Demos** | **During the workshop** | Run the five demos and compare them | **90 minutes** |
| **PART C — Reference** | Afterwards, optional | How the models were built, and what that looked like | reading only |

> ### ⚠️ Do Part A before you arrive
> It is only an install and a file copy — but if 40 people install at once on
> venue wifi, nobody starts on time. **Arrive with `python verify_setup.py`
> showing all `OK`.**

### You do NOT build the models

The fine-tuned adapter and the compressed model are **supplied to you, already
built.** Training on a laptop CPU takes over half an hour — there is no room
for that in a 90-minute session.

**Part C** shows exactly what those steps do and what they printed when we ran
them, so you see the whole pipeline without waiting for it. All of it is
reproducible at home if you want to.

---

# PART A — Before the workshop

## A1. What you need

| Requirement | Minimum |
|---|---|
| OS | Windows 10 / 11 (tested on 11) |
| Python | **3.13** |
| RAM | 8 GB (16 GB comfortable) |
| Free disk | 6 GB |
| Internet | Part A only. Part B runs fully offline. |

```powershell
python --version
```

If that does not say `3.13.x`, install Python 3.13 before continuing.

---

## A2. Get the project and the models

Code comes from git. The **models are too large for git** and are handed out
separately (USB stick or shared drive).

1. Copy the project folder somewhere with a **short path** — `C:\SLM-WORKSHOP`
2. Copy the supplied **`models/`** folder into the project root

Your folder must end up like this:

```
SLM-WORKSHOP/
├── config.py
├── requirements.txt
├── verify_setup.py
├── 01_rag_baseline.py  ...  04_gguf_rag.py
├── data/
├── llama.cpp/                    <- inference engine (supplied)
├── models/                       <- SUPPLIED SEPARATELY
│   ├── it/                       <- base model            (~1.9 GB)
│   ├── embedder/                 <- RAG embedding model   (~88 MB)
│   ├── adapter_<timestamp>/      <- fine-tuned adapter    (~36 MB)
│   └── gguf/*-q4_k_m.gguf        <- compressed model      (~777 MB)
└── tests/
```

> **Windows path length.** Windows breaks past 260 characters. Do **not** put
> this inside OneDrive or a deep `Documents` tree. If you must, map a short
> drive:
> ```cmd
> subst X: "C:\Some\Very\Long\Path\SLM WORKSHOP"
> X:
> ```

---

## A3. Create a virtual environment and install

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

Takes 5–20 minutes and downloads roughly 2.5 GB of packages.

> **Why `requirements.txt` starts with `--extra-index-url`**
> `llama-cpp-python` publishes no package on PyPI. Without that line, pip tries
> to compile it from C++ source and fails unless you have Visual Studio Build
> Tools installed. The extra index supplies a ready-built CPU version.
>
> **Do not delete that line.** If you see *"Building wheel for
> llama-cpp-python"* scroll past, something is wrong — stop and ask.

---

## A4. Verify — this is the gate

```powershell
python verify_setup.py
```

```
  [OK     ] Python version            3.13.15 (tested on 3.13)
  [OK     ] PyTorch                   2.14.0+cpu
  [OK     ] llama-cpp-python          0.3.35
  [OK     ] Base model                models/it (1945 MB)
  [OK     ] Embedding model           models/embedder (88 MB)
  [OK     ] Fine-tuned adapter        models/adapter_20260905_143045
  [OK     ] Quantized model (GGUF)    it-finetuned-...-q4_k_m.gguf (777 MB)
```

Everything must say `OK` except the **vector database** — you build that in the
first five minutes of the workshop, so `MISSING` there is expected.

If anything else says `MISSING`, the table prints the exact command that fixes
it. Sort it out now, not on the day.

---

# PART B — The workshop (90 minutes)

## Every command, in order

Keep this open. The rest of Part B explains what to look for in each one.

```powershell
# --- setup check -------------------------------------------------------
python verify_setup.py
python inspect_system.py

# --- the one build step you run ----------------------------------------
python 01_rag_baseline.py

# --- the five demos ----------------------------------------------------
python tests\test_00_base_slm.py      # DEMO 1  base model
python tests\test_02_adapter.py       # DEMO 2  + fine-tuning
python tests\test_03_adapter_rag.py   # DEMO 3  + RAG
python tests\test_04_gguf.py          # DEMO 4  + quantization
python 04_gguf_rag.py                 # DEMO 5  everything together

# --- optional bonus ----------------------------------------------------
python tests\test_01_base_slm_rag.py  # RAG without fine-tuning
```

Each demo **waits for you to type a question** — nothing is asked
automatically. It prints a suggested question you can copy. Type `exit` when
you are done with a demo and move to the next.

> **Note on the folder names.** Demos 1–4 live in `tests/`; demo 5 is
> `04_gguf_rag.py` in the project root. The root files numbered `00`–`03` are
> the *build pipeline* that produced your models — you are not running those.
> See Part C if you want to know what they did.

---

## B0. Warm-up — what are we running on?

```powershell
python inspect_system.py
```

Your CPU, RAM, and whether any accelerator is present. On nearly every laptop in
the room the answer is **none** — and everything still works. That is the
premise of edge AI.

---

## B1. Build the knowledge base *(~1 minute)*

```powershell
python 01_rag_baseline.py
```

This is the one build step you **do** run — it is fast, and watching it happen
is the point.

It reads 32 engineering documents from `data/rag/` — PDFs, Word reliability
records, HTML maintenance histories, JSON service tickets — splits them into
overlapping chunks, turns each chunk into a 384-number vector, and stores them
in a searchable database.

**Expected:** `RAG knowledge base ready: 592 chunks`

> **Why vectors instead of keyword search?** A technician says *"pump won't
> start"*; the manual says *"motor fails to energise"*. No shared words.
> Embeddings capture **meaning**, so related ideas land near each other even
> with completely different vocabulary.

> Re-run with `--rebuild` if you ever change anything in `data/rag/`.

---

## The five demos

Each changes **exactly one thing** from the previous one.

| Demo | Command | What is new | Speed |
|---|---|---|---|
| **1** | `python tests\test_00_base_slm.py` | Nothing — the raw model | ~6 t/s |
| **2** | `python tests\test_02_adapter.py` | **+ Fine-tuning** | ~6 t/s |
| **3** | `python tests\test_03_adapter_rag.py` | **+ RAG** | ~6 t/s |
| **4** | `python tests\test_04_gguf.py` | **+ Quantization** | **~30 t/s** |
| **5** | `python 04_gguf_rag.py` | **Everything together** | **~30 t/s** |

Each drops into an interactive prompt. Type `exit` to move on.

> **Demos 1–3 are slow** — 60–90 seconds per answer. That is deliberate. You are
> feeling the problem that demos 4 and 5 solve.

### The one question to carry through all five

```
What did the last technician find when they serviced P-104?
```

- **Demos 1, 2, 4** — never seen your records. It will refuse or invent
  something. Notice how *confident* a wrong answer sounds.
- **Demos 3, 5** — retrieves the real maintenance record **and names the file.**

---

## Demo 1 — The original model *(~10 min)*

```powershell
python tests\test_00_base_slm.py
```

`google/gemma-3-1b-it` exactly as Google published it. 3.7 GB of 32-bit weights
in RAM, running through PyTorch.

**Look for:**
- **Speed** — about 6 tokens/sec. Over a minute for one answer.
- **Solid general knowledge** — it really does understand pumps and overloads.
- **Where it stops.** Watch carefully: it writes a summary, lists possible
  causes, and then **stops without telling you what to actually do.**
- **No site knowledge** — ask the P-104 question above.

**Takeaway:** capable, but slow, generic, and it leaves the job half done.

---

## Demo 2 — Fine-tuned *(~10 min)*

```powershell
python tests\test_02_adapter.py
```

Same base model, same prompt, same settings. The only change is a **4.6 MB
adapter** layered on top — 2,396,160 trained parameters out of 1,002,282,112.
**0.24% of the model.**

**Look for — this is the thing:**

Demo 1 stopped after listing causes. This one **finishes the job:**

```
[ SUMMARY ]           <- both models produce this
[ POTENTIAL CAUSES ]  <- both models produce this
[ ACTION PLAN ]       <- ONLY the fine-tuned model gets here
1. Check the motor's voltage and current.
2. Check the motor's temperature.
...
```

And it does so in **fewer** words, not more. Measured on the standard question:

| | Demo 1 (base) | Demo 2 (fine-tuned) |
|---|---|---|
| Reached `[ ACTION PLAN ]` | **No** | **Yes** |
| Tokens generated | 169 | **142** |
| Answer length | 814 characters | **478 characters** |

Neither model ran out of room — the cap is 450 tokens and both stopped well
short. The base model simply *wanders*, spending a full sentence explaining each
cause. The fine-tuned model learned from 120 real tickets that the job is: name
the causes, then say what to check. **Terse and complete beats verbose and
unfinished.**

- **Speed is unchanged** — still ~6 t/s. Fine-tuning changes *what* the model
  says, never how fast.

> ### Talking point: couldn't a longer prompt do this?
> Partly, and it is worth being honest about that — prompting is powerful. The
> difference is **cost**. The system prompt here is ~560 tokens the CPU must
> read *before writing a single word*, on every question. Fine-tuning moves the
> behaviour into the weights, where it costs nothing at runtime.
>
> **On an edge device, prompt tokens are latency.**

**Takeaway:** the model learned *how* to answer. It still has no facts.

---

## Demo 3 — Fine-tuned + RAG *(~12 min)*

```powershell
python tests\test_03_adapter_rag.py
```

Same model. Before answering, it now searches those 592 chunks and pastes the
best matches into the prompt.

**Look for:**
- **Real specifics** — actual asset tags, readings, history.
- **Sources** — it prints which files it used. A technician can go read the
  original. *An answer you can check beats an answer you must trust.*
- **Retrieval distance** — printed after each answer. **Smaller = better match.**
  Rephrase a question and watch it move.
- **It got slower** — the retrieved documents add ~750 tokens to read first.
  RAG buys accuracy with latency.

**Now ask the P-104 question that failed in demos 1 and 2.**

**Takeaway:** fine-tuning taught it *how*; RAG gives it the *facts*.

---

## Demo 4 — Quantized *(~8 min)*

```powershell
python tests\test_04_gguf.py
```

Same fine-tuned model, compressed from 16-bit to 4-bit and run by `llama.cpp` in
C++ instead of PyTorch.

**Look for:**
- **~30 tokens/sec** — about **5× faster** than demos 1–3, on the same CPU.
- **770 MB instead of ~2 GB.**
- **Quality holds up** — compare against demo 2.

> ### Why is it faster?
> CPU text generation is limited by **memory bandwidth**, not arithmetic. For
> every single token the processor must read the *entire model* out of RAM.
> Quarter the model's size and you quarter the data that has to move. Compiled
> C++ instead of Python supplies the rest.

**Takeaway:** this is what makes edge deployment realistic. Note it is still
missing the documents.

---

## Demo 5 — Everything together *(~12 min)*

```powershell
python 04_gguf_rag.py
```

The finished product.

**Look for:**
- Structured, grounded, source-cited answers in about **20 seconds**.
- The **RAG mode** line — `EXACT`, `STRONG`, `MODERATE` or `NO_RAG`. The script
  picks a different system prompt depending on match quality. RAG is not
  all-or-nothing.
- **Turn your wifi off and keep asking.** Nothing changes. That is the point.

Question ideas: [`chatbot_question_bank.md`](chatbot_question_bank.md) — 100+
scenarios across 8 pieces of equipment.

Everything is logged to `results/04_gguf_rag_log.txt`.

---

## Putting it together

| | Size | Speed | Finishes the job? | Knows your equipment? |
|---|---|---|---|---|
| Demo 1 — base | 3.7 GB | 6 t/s | ✗ | ✗ |
| Demo 2 — fine-tuned | 3.7 GB | 6 t/s | ✓ | ✗ |
| Demo 3 — + RAG | 3.7 GB | 6 t/s | ✓ | ✓ |
| Demo 4 — quantized | 777 MB | **~30 t/s** | ✓ | ✗ |
| Demo 5 — **+ RAG** | **777 MB** | **~30 t/s** | **✓** | **✓** |

**Three independent levers:**

| Lever | Gives you | Costs you |
|---|---|---|
| **Fine-tuning** | Behaviour, format, house style | Training time, once |
| **RAG** | Facts, sources, updates without retraining | Latency, every query |
| **Quantization** | Speed and size | A little precision |

They are independent. Pick the ones your problem needs.

---

## Optional bonus — is the fine-tuning really doing anything?

```powershell
python tests\test_01_base_slm_rag.py
```

Base model **+ RAG, without fine-tuning** — the one combination the main
sequence skips. The facts will be just as good, because RAG supplies those. The
*shape* will be looser. That is the honest answer to *"is RAG doing all the
work?"*

---

# PART C — How the models were made *(reference)*

You did not run these two steps; they take too long for a live session. Here is
what they do and what they actually printed when we built the files you used.

## C1. Fine-tuning — `02_finetune_qlora.py`

```powershell
python 02_finetune_qlora.py     # ~34 minutes on a 16-core laptop CPU
```

### What QLoRA is

Fine-tuning a whole model means updating ~1 billion numbers — far too much for a
laptop. QLoRA stacks two tricks:

- **Q — Quantized.** Load the frozen base model in 4-bit instead of 32-bit,
  cutting memory from ~4 GB to ~900 MB.
- **LoRA — Low-Rank Adaptation.** Freeze the original weights entirely and train
  tiny "side matrices" bolted onto the attention layers.

The result is an **adapter**: a 4.6 MB file that reshapes the model's behaviour
without touching the 2 GB of original weights.

### What it actually printed

```
Available RAM : 11.53 GB
CPU cores     : 16 physical
bitsandbytes  : INSTALLED

Loading base model in 4-bit...
  Model footprint: 909 MB          <- was 3.7 GB in demo 1

Attaching LoRA adapters...
trainable params: 2,396,160 || all params: 1,002,282,112 || trainable%: 0.2391

TRAINING: 120 examples x 1 epochs
{'loss': 5.1122, ...}              <- start
{'loss': 3.4590, ...}
{'loss': 3.0211, ...}              <- end: the model is learning

Training finished in 2047s (34.1 min).
Saving adapter to models/adapter_20260905_143045 ...
```

**The headline: 0.24% of the parameters trained, 4.6 MB produced — and that is
the entire difference between demo 1 and demo 2.**

### What it learned, and what it deliberately did not

Training targets are built from 120 structured maintenance tickets: a summary
plus the recorded resolution steps. Note what is deliberately **absent** —
the dataset contains no root-cause analysis, so we teach none. Inventing causes
would be teaching the model to hallucinate.

Supplying causes is RAG's job, because **causes need evidence.**

---

## C2. Quantization — `03_quantize_gguf.py`

```powershell
python 03_quantize_gguf.py      # ~2 minutes
```

Three stages:

1. **Merge** — bake the adapter permanently into the model weights
2. **Convert** — rewrite as GGUF, the single-file format `llama.cpp` uses
3. **Quantize** — compress 16-bit weights to 4-bit (`Q4_K_M`)

`Q4_K_M` means *4-bit, K-quant, Medium*: it keeps the most sensitive tensors at
higher precision and squeezes the rest. That is why quality survives at a
quarter of the size.

### The result

| Stage | Size |
|---|---|
| Merged PyTorch model | ~2000 MB |
| GGUF 16-bit | ~2000 MB |
| **GGUF 4-bit (Q4_K_M)** | **~770 MB** |

**About 2.5x smaller - and roughly 5x faster, which is demo 4.**

---

## C3. Reproducing it at home

Everything is in the repo:

```powershell
python 01_rag_baseline.py       # ~1 min
python 02_finetune_qlora.py     # ~34 min  (make a coffee)
python 03_quantize_gguf.py      # ~2 min
python verify_setup.py          # confirm
```

Your own adapter and GGUF get timestamped folders, and every demo automatically
picks the **newest** one — so your build replaces ours with no config changes.

To train faster, open `02_finetune_qlora.py` and set `MAX_DATASET_SIZE = 60`.
That halves it.

*(`00_download_base.py` fetches the base model from HuggingFace. You do not need
it — the model is in your bundle — and it requires a HuggingFace account,
because Gemma is a licence-gated model.)*

---

# Troubleshooting

### `ModuleNotFoundError` for `llama_cpp`, `trl`, or `docx`
Your install did not complete:
```powershell
pip install -r requirements.txt
```
For `llama_cpp` specifically, the extra index is mandatory:
```powershell
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

### `No quantized model found in models/gguf/`
Your `models/` folder is incomplete. Get the full bundle from the organiser.

### `No knowledge base` / ChromaDB errors
```powershell
python 01_rag_baseline.py --rebuild
```

### Answers stop mid-sentence
Context window filled. In `config.py` raise `N_CTX = 8192`, or lower `TOP_K` to
`2` to inject fewer documents.

### `GatedRepoError` / `401` while downloading
You should never need to download anything — the models are supplied. Gemma is
licence-gated and needs a HuggingFace account. Ask the organiser for the bundle.

### `[WinError 206] filename or extension is too long`
Your project path is too deep. See the `subst` trick in A2.

### Everything is slower than this guide says
These numbers came from a 16-core Intel Core Ultra **laptop** — already laptop
figures, not desktop ones. Fewer cores means slower.

The absolute seconds do not matter. The **ratios between demos** are the point,
and those hold on any CPU.

---

# Reference

### Files, in the order you use them

| File | Part | Purpose |
|---|---|---|
| `verify_setup.py` | A | Check your setup |
| `inspect_system.py` | B | Warm-up: show the hardware |
| `01_rag_baseline.py` | B | Build the vector database |
| `tests/test_00_base_slm.py` | B | **Demo 1** — base model |
| `tests/test_02_adapter.py` | B | **Demo 2** — fine-tuned |
| `tests/test_03_adapter_rag.py` | B | **Demo 3** — fine-tuned + RAG |
| `tests/test_04_gguf.py` | B | **Demo 4** — quantized |
| `04_gguf_rag.py` | B | **Demo 5** — the final product |
| `tests/test_01_base_slm_rag.py` | B | Bonus — base + RAG |
| `02_finetune_qlora.py` | C | *(reference)* Trains the adapter |
| `03_quantize_gguf.py` | C | *(reference)* Merges and compresses |
| `00_download_base.py` | — | Organiser only |
| `smoke_test.py` | — | Organiser only — unattended pass/fail of every demo |
| `student_dry_run.ps1` | — | Organiser only — walk the participant path by hand |
| `config.py` | — | Every setting, in one place |

### Worth changing in `config.py`

| Setting | Does what |
|---|---|
| `TOP_K` | Document chunks RAG retrieves (default 3) |
| `TEMPERATURE` | 0.0 = repeatable, 1.0 = creative (default 0.3) |
| `MAX_TOKENS` | Longest answer allowed (default 450) |
| `N_CTX` | Context window — raise if answers get cut off |
| `DETERMINISTIC_MODE` | `True` forces identical answers every run |

### Glossary

| Term | Meaning |
|---|---|
| **SLM** | Small Language Model — small enough to run on your own hardware |
| **LoRA** | Trains a tiny add-on instead of the whole model |
| **QLoRA** | LoRA on top of a 4-bit model, so it fits on a laptop |
| **Adapter** | The few-MB file LoRA produces |
| **Quantization** | Storing weights in fewer bits to shrink and speed up a model |
| **GGUF** | `llama.cpp`'s single-file model format |
| **Q4_K_M** | 4-bit quantization that protects the most sensitive weights |
| **RAG** | Retrieval-Augmented Generation — look it up before answering |
| **Embedding** | A list of numbers representing text meaning |
| **Vector database** | Storage that finds text by meaning, not keywords |
| **Token** | Roughly ¾ of a word — the unit models read and write in |
| **Context window** | Total tokens a model can hold at once (prompt + answer) |
