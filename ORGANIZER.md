# Organiser Notes

Everything the person *running* the workshop needs. Participants do not need this
file — send them to [`WORKSHOP_GUIDE.md`](WORKSHOP_GUIDE.md).

---

## The shape of the day

| Block | Duration | What |
|---|---|---|
| Presentation | ~90 min | Slides, theory, discussion |
| **Hands-on** | **~90 min** | The five demos |
| **Participant homework** | **before the day** | Install + build (`01`, `02`, `03`) |

The build scripts take up to two hours of computer time. They **cannot** happen
in the room. Participants must arrive with Part A of the guide already done.

### Hands-on budget (90 minutes)

| | Activity | Time |
|---|---|---|
| 0 | `verify_setup.py` — catch broken laptops immediately | 5 min |
| 1 | `inspect_system.py` — "this is just a laptop" | 3 min |
| 2 | **Demo 1** — base model | 10 min |
| 3 | **Demo 2** — fine-tuned | 10 min |
| 4 | **Demo 3** — fine-tuned + RAG | 12 min |
| 5 | **Demo 4** — quantized | 8 min |
| 6 | **Demo 5** — everything | 12 min |
| 7 | Free exploration from the question bank | 20 min |
| 8 | Wrap-up and comparison table | 10 min |

Demos 1–3 take 60–90 seconds *per question* on a typical laptop. Budget one
question each and let people ask their own during block 7, which runs on the
fast quantized model.

> **Start block 0 the moment people sit down.** A broken setup found in minute 5
> can be fixed alongside the demos. Found in minute 45, it costs the whole session.

---

## Preparing the participant bundle

### 1. Download the models (once, on your machine)

> ### ⚠️ Gemma is a GATED model — you must log in first
> `google/gemma-3-1b-it` cannot be downloaded anonymously. An unauthenticated
> download fails with a bare `401 Unauthorized` / `GatedRepoError`.
>
> **One-time setup, before running `00_download_base.py`:**
>
> 1. Open <https://huggingface.co/google/gemma-3-1b-it>, sign in, and click
>    **Acknowledge license**. Approval is instant.
> 2. Create a **read** token at <https://huggingface.co/settings/tokens>.
> 3. Log in from your terminal:
>    ```powershell
>    hf auth login
>    ```
>    (On `huggingface_hub` older than v1.0 the command is `huggingface-cli login`.)
>    Or set it as an environment variable instead:
>    ```powershell
>    $env:HF_TOKEN = "hf_your_token_here"
>    ```
>
> **Participants never hit this** — they receive `models/` already populated.
> This affects only the person building the bundle.
>
> If you would rather avoid the licence step entirely, switch
> `ACTIVE_MODEL_ID` in `config.py` to one of the Qwen models. They are not
> gated and need no login — but you must then re-run the whole `00 → 03`
> pipeline to rebuild every artifact.

```powershell
python 00_download_base.py
```

Fetches the base LLM into `models/it/` and the embedding model into
`models/embedder/`. This is the only script participants never run.

The script reports your login status before downloading anything, skips the
base model if it is already present, and prints plain-English instructions if
it hits the gate.

### 2. Build everything once, yourself

```powershell
python 01_rag_baseline.py
python 02_finetune_qlora.py
python 03_quantize_gguf.py
python verify_setup.py
```

Do this before the workshop. It confirms the pipeline works on a clean machine
and gives you a reference adapter and GGUF to fall back on.

### 3. Prune before you copy

After a full build, `models/` contains several GB of intermediates nobody needs:

| Path | Keep? | Why |
|---|---|---|
| `models/it/` | **YES** (~1.9 GB) | The base model. Participants cannot download this on venue wifi. |
| `models/embedder/` | **YES** (~88 MB) | RAG embeddings. Without it every RAG script hits the internet. |
| `models/adapter_<ts>/` | **Recommended** (~36 MB) | Safety net — see below. 4.6 MB of weights plus a tokenizer copy. |
| `models/gguf/*-q4_k_m.gguf` | **Recommended** (~770 MB) | Safety net — see below. |
| `models/gguf/*-f16.gguf` | **No** (~2 GB) | Intermediate. Delete. |
| `models/merged_<ts>/` | **No** (~2 GB) | Intermediate. Delete. |

```powershell
Remove-Item -Recurse -Force models\merged_*
Remove-Item -Force models\gguf\*-f16.gguf
```

That takes the bundle from roughly 7 GB down to about 2.8 GB.

> **Why ship your adapter and GGUF as a safety net?**
> Participants build their own in steps A6 and A7. But if someone's step 02
> failed at home, without a fallback they cannot run demos 2, 3, 4 **or** 5 —
> they lose four fifths of the workshop.
>
> Every script picks the **most recently modified** adapter and GGUF, so a
> participant's own successful build always wins. Yours only gets used if
> theirs is missing.

### 4. What to distribute

Ship the git repo **plus** the pruned `models/` folder (USB or shared drive).
`models/` is gitignored on purpose — it is far too large for git.

Confirm the bundle is complete by copying it to a clean machine and running:

```powershell
pip install -r requirements.txt
python verify_setup.py
```

Every line must say `OK`.

### 5. Validate the bundle before you hand it out

Two scripts do this for you.

**Unattended pass/fail** — run this on a clean machine with the bundle copied in:

```powershell
python smoke_test.py --fast     # ~4 min  - setup + the two fast demos
python smoke_test.py            # ~10 min - the full student path
python smoke_test.py --build    # ~50 min - also re-runs fine-tune + quantize
```

It runs every command in the participant guide, feeds `exit` into the
interactive loops, and checks the output actually contains what the guide
promises - that demo 3 and 5 really cite a source file, that demo 4 really
reports `Backend: CPU`, that step 01 really reports a chunk count. It exits
non-zero on failure, so you can gate on it.

**Walk it by hand** — when you want to see what the room will see:

```powershell
.\student_dry_run.ps1              # full path, pauses between demos
.\student_dry_run.ps1 -DemosOnly   # just the five demos
.\student_dry_run.ps1 -Reset       # move build output aside first (reversible)
```

Before each step it prints what to look for, then waits so you can actually
read the answer. `-Reset` only *moves* files to `_bak_*`; it never deletes.

---

## Known friction points

**Gemma is gated.** Building the bundle needs a HuggingFace login and a
licence acceptance — see step 1 above. This bites organisers only, once.

**`llama-cpp-python` is the single most likely install failure.** PyPI ships no
wheel for it, so a bare `pip install llama-cpp-python` tries to compile C++ and
fails without Visual Studio Build Tools. `requirements.txt` handles this with an
`--extra-index-url` line at the top. If anyone installs packages by hand, that
line is the one they will miss.

**Python version.** Tested on 3.13. Participants on other versions will hit
wheel-availability differences. Say 3.13 explicitly in your pre-workshop email.

**Windows path length.** Anything under OneDrive or a deep `Documents` tree will
break the 260-character limit. Tell people to use `C:\SLM-WORKSHOP`.

**8 GB machines.** Step 03 needs ~2 GB free during the merge, and demos 1–3 hold
3.7 GB. Tell participants to close their browser during Part A.

---

## Talking points that land well

**"Only 0.24% of the model was trained."** `02_finetune_qlora.py` prints
`trainable params: 2,396,160 || all params: 1,002,282,112`. Put that on a slide.

**"Five times faster, on the same CPU."** ~6 tok/s unquantized versus ~30 tok/s
quantized. Because CPU generation is bound by **memory bandwidth**, not
arithmetic — shrink the model and you shrink the data that must move per token.

**"Prompt tokens are latency."** The system prompt is ~560 tokens the CPU reads
before writing anything, on every question. RAG adds ~750 more. On edge hardware
that is seconds of dead time — and it is the honest argument for fine-tuning
over prompting.

**"An answer you can check beats an answer you must trust."** Demos 3 and 5 name
the source file. In an industrial setting that matters more than eloquence.

**Turn the wifi off during demo 5.** Nothing changes. That is the whole point.

---

## Fielding the hard questions

**"Couldn't a bigger cloud model just do this better?"**
Yes, on quality. It cannot do it with no connectivity, on someone else's
hardware, with data that is not allowed to leave the site. The pitch is not
"better than GPT" — it is "works where GPT cannot go."

**"Is the fine-tuning actually doing anything, or is RAG doing all the work?"**
Genuinely good question, and there is a script for it:
`tests/test_01_base_slm_rag.py` runs RAG *without* fine-tuning. The facts come
out just as good; the structure is looser. RAG supplies facts, fine-tuning
supplies discipline.

**"Why is it so slow in demos 1–3?"**
That is the problem the workshop exists to solve. Demo 4 is the payoff.

**"Why not just use a bigger prompt instead of fine-tuning?"**
You can, and it partly works — be honest about that. The cost is paid on every
single query as prompt-reading latency. See the prompt-token talking point above.

---

## If something breaks live

| Problem | Do this |
|---|---|
| A participant's step 02 or 03 never finished | They fall back to your shipped adapter and GGUF automatically. Nothing to do. |
| `ModuleNotFoundError: llama_cpp` | `pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu` |
| ChromaDB errors | `python 01_rag_baseline.py --rebuild` (~1 min) |
| Answers cut off mid-sentence | Raise `N_CTX` to `8192` in `config.py` |
| A laptop is far too slow for demos 1–3 | Have them watch a neighbour; they can still run demos 4 and 5 themselves. |
| Everything is broken on one machine | Pair them up. Do not debug one laptop while 39 people wait. |

---

## Measured reference numbers

Measured on a 16-core Intel Core Ultra laptop, CPU only. Your mileage will vary;
the **ratios** are what matter.

| Metric | Value |
|---|---|
| Base model on disk / in RAM | ~1.9 GB / 3.7 GB |
| Quantized model on disk / in RAM | 770 MB / ~0.9 GB |
| Base generation speed | ~6.4 tok/s |
| Quantized generation speed | ~28-38 tok/s (varies with CPU thermals) |
| Prompt processing (quantized) | ~130-157 tok/s |
| Trainable parameters | 2,396,160 of 1,002,282,112 (0.24%) |
| Adapter size | 4.6 MB weights (36 MB folder incl. tokenizer) |
| RAG corpus | 32 documents → 592 chunks |
| Step 01 runtime | ~15 s |
| Step 02 runtime | ~34 min (1 epoch, 120 examples) |
| Step 03 runtime | ~35 s of work, <2 min end to end |
