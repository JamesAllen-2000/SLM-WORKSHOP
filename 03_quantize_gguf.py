"""
=============================================================================
 STEP 03 - MERGE AND QUANTIZE FOR THE EDGE            [ RUN BEFORE WORKSHOP ]
=============================================================================
 WHAT THIS DOES
   Takes the fine-tuned adapter from step 02 and compiles everything into a
   single, tiny file that runs fast on a plain CPU.

   Three stages:

     (1) MERGE      base model  +  LoRA adapter  ->  one merged model
                    The adapter stops being a separate side-car and its
                    changes are baked permanently into the weights.

     (2) CONVERT    merged model (PyTorch)  ->  GGUF format, 16-bit
                    GGUF is llama.cpp's format: a single file, memory-mapped,
                    designed to load instantly with no Python involved.

     (3) QUANTIZE   16-bit GGUF  ->  4-bit GGUF (Q4_K_M)
                    Each weight is stored in ~4 bits instead of 16. The file
                    shrinks about 2.6x and inference gets several times
                    faster, because on CPU the bottleneck is moving weights
                    from RAM, not the arithmetic itself.

 THE HEADLINE NUMBER
   ~1.9 GB PyTorch model  ->  777 MB GGUF  ->  runs at ~30 tokens/sec on CPU
   versus ~6 tokens/sec for the original. Same model, about 5x faster.

 RUNTIME
   About 35 seconds of actual work; under 2 minutes end to end including
   loading and saving the model. Needs ~2 GB free RAM and ~5 GB free disk.

 OUTPUT
   models/merged_<id>/                 <- merged PyTorch model (intermediate)
   models/gguf/*-f16.gguf              <- 16-bit GGUF     (intermediate)
   models/gguf/*-q4_k_m.gguf           <- 4-bit GGUF      <- THE DEPLOYABLE
   analysis/quantize_analysis_*.md
=============================================================================
"""
import os
import gc
import sys
import glob
import time
import shutil
import datetime
import subprocess

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from config import LOCAL_MODEL_DIR as BASE_MODEL_DIR, get_safe_model_name

# ---------------------------------------------------------------------------
# PATHS - derived from whichever adapter step 02 produced most recently
# ---------------------------------------------------------------------------
GGUF_DIR = "models/gguf"

adapter_dirs = glob.glob("models/adapter_*")
if not adapter_dirs:
    print("ERROR: No adapter found in models/. Run 02_finetune_qlora.py first.")
    raise SystemExit(1)

ADAPTER_DIR = max(adapter_dirs, key=os.path.getmtime)

# Keep the FULL timestamp. Splitting on "_" and taking the last piece would
# discard the date and make two runs on different days collide.
run_id = os.path.basename(ADAPTER_DIR).replace("adapter_", "")

MERGED_DIR = f"models/merged_{run_id}"
safe_name = get_safe_model_name().lower()
F16_GGUF_PATH = f"{GGUF_DIR}/{safe_name}-finetuned-{run_id}-f16.gguf"
Q4_GGUF_PATH = f"{GGUF_DIR}/{safe_name}-finetuned-{run_id}-q4_k_m.gguf"


# ---------------------------------------------------------------------------
# HELPER: locate a llama.cpp executable
# ---------------------------------------------------------------------------
def find_llama_exe(name):
    """
    Find a llama.cpp tool using the platform's preferred executable format.

    Windows prefers the bundled ``.exe`` binaries. Linux and macOS prefer
    native executables; those platforms need a native llama.cpp build because
    the bundled binaries are Windows-specific. ``LLAMA_QUANTIZE_PATH`` may be
    used to provide an explicit executable path for the quantizer.
    """
    override = os.environ.get("LLAMA_QUANTIZE_PATH")
    if override:
        if os.path.isfile(override) and (
            os.name == "nt" or os.access(override, os.X_OK)
        ):
            return os.path.abspath(override)
        return None

    executable_names = (
        [f"{name}.exe", name] if os.name == "nt" else [name, f"{name}.exe"]
    )
    candidates = [
        os.path.join(directory, executable_name)
        for directory in ("llama.cpp", ".")
        for executable_name in executable_names
    ]
    for candidate in candidates:
        if os.path.isfile(candidate) and (
            os.name == "nt" or os.access(candidate, os.X_OK)
        ):
            return os.path.abspath(candidate)

    # Fall back to executables already available on PATH.
    for executable_name in executable_names:
        found = shutil.which(executable_name)
        if found:
            return found
    return None


# ---------------------------------------------------------------------------
# SECTION 1: MERGE THE ADAPTER INTO THE BASE MODEL
# ---------------------------------------------------------------------------
def merge_adapter():
    print("=" * 60)
    print("STAGE 1/3: MERGE ADAPTER INTO BASE MODEL")
    print("=" * 60)
    print(f"Base model : {BASE_MODEL_DIR}")
    print(f"Adapter    : {ADAPTER_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR)

    # bfloat16, not float32. The merged model is headed for 4-bit quantization
    # anyway, so float32 precision would be thrown away moments later - while
    # doubling peak RAM (~4 GB -> ~2 GB) and doubling what we write to disk.
    print("\nLoading base model (bfloat16)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_DIR,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    print("Applying adapter...")
    peft_model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)

    print("Merging weights...")
    merge_start = time.time()
    merged_model = peft_model.merge_and_unload()
    merge_time = time.time() - merge_start
    print(f"  Merge completed in {merge_time:.1f}s")

    # DO NOT REMOVE THIS RESIZE - the GGUF conversion depends on it.
    #
    # Gemma-3 declares vocab_size = 262144 (valid ids 0..262143) but its
    # added_tokens.json defines <image_soft_token> at id 262144. So the
    # tokenizer's highest id is exactly equal to vocab_size.
    #
    # llama.cpp's convert_hf_to_gguf.py asserts:
    #     max(tokenizer.vocab.values()) < vocab_size
    # which fails by exactly one, and the conversion dies with a bare
    # AssertionError.
    #
    # Resizing to len(tokenizer) = 262145 adds one embedding row so that
    # assertion holds. The extra row is never selected during generation
    # (this is a text-only model; the image token is unused), so it costs
    # about 1152 unused floats and buys a working conversion.
    print(f"Aligning embeddings with tokenizer "
          f"({merged_model.config.vocab_size} -> {len(tokenizer)}) ...")
    merged_model.resize_token_embeddings(len(tokenizer))

    os.makedirs(MERGED_DIR, exist_ok=True)
    print(f"\nSaving merged model to {MERGED_DIR} ...")
    merged_model.save_pretrained(MERGED_DIR)
    tokenizer.save_pretrained(MERGED_DIR)

    # --- Quick sanity check: does the merged model still speak? -------------
    print("\nSanity check on merged model...")
    messages = [{"role": "user", "content": "Hello! Are you working?"}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = merged_model.generate(**inputs, max_new_tokens=20, do_sample=False)
    reply = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    ).strip()
    print(f"  Model replied: {reply}")

    del merged_model, peft_model, base_model
    gc.collect()

    return merge_time


# ---------------------------------------------------------------------------
# SECTION 2: CONVERT TO GGUF (16-bit)
# ---------------------------------------------------------------------------
def convert_to_gguf():
    print("\n" + "=" * 60)
    print("STAGE 2/3: CONVERT TO GGUF (16-bit)")
    print("=" * 60)

    os.makedirs(GGUF_DIR, exist_ok=True)

    # The converter is a Python script shipped inside the llama.cpp source.
    convert_script = None
    for candidate in ("llama.cpp/convert_hf_to_gguf.py",
                      "llama_source/convert_hf_to_gguf.py"):
        if os.path.exists(candidate):
            convert_script = candidate
            break

    if convert_script is None:
        print("ERROR: convert_hf_to_gguf.py not found.")
        print("Expected it in llama_source/ (bundled with this workshop).")
        print("If missing, restore it with:")
        print("  git clone https://github.com/ggerganov/llama.cpp llama_source")
        # Hard stop. Continuing would fail later with a confusing error about
        # a missing .gguf file.
        raise SystemExit(1)

    cmd = [sys.executable, convert_script, MERGED_DIR,
           "--outfile", F16_GGUF_PATH, "--outtype", "f16"]

    print("Running: " + " ".join(cmd))
    start = time.time()
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: GGUF conversion failed: {e}")
        raise SystemExit(1)

    convert_time = time.time() - start
    size_mb = os.path.getsize(F16_GGUF_PATH) / (1024 ** 2)
    print(f"\n  Created {F16_GGUF_PATH} ({size_mb:.0f} MB) in {convert_time:.1f}s")

    return convert_time


# ---------------------------------------------------------------------------
# SECTION 3: QUANTIZE TO 4-BIT (Q4_K_M)
# ---------------------------------------------------------------------------
def quantize_gguf():
    print("\n" + "=" * 60)
    print("STAGE 3/3: QUANTIZE TO 4-BIT (Q4_K_M)")
    print("=" * 60)

    quantize_exe = find_llama_exe("llama-quantize")
    if quantize_exe is None:
        print("ERROR: No usable llama-quantize executable was found.")
        print("Windows users can use the bundled binary; Linux/macOS users "
              "need a native llama.cpp build or can set "
              "LLAMA_QUANTIZE_PATH to its executable.")
        raise SystemExit(1)

    # Q4_K_M = 4-bit "K-quant, Medium". It keeps the most sensitive tensors at
    # higher precision and squeezes the rest, which is why quality holds up so
    # well at a quarter of the size.
    cmd = [quantize_exe, F16_GGUF_PATH, Q4_GGUF_PATH, "Q4_K_M"]

    print("Running: " + " ".join(cmd))
    start = time.time()
    try:
        subprocess.run(cmd, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"ERROR: Quantization failed: {e}")
        raise SystemExit(1)

    quant_time = time.time() - start
    print(f"\n  Created {Q4_GGUF_PATH} in {quant_time:.1f}s")

    return quant_time


# ---------------------------------------------------------------------------
# SECTION 4: ANALYSIS REPORT
# ---------------------------------------------------------------------------
def write_analysis(merge_time, convert_time, quant_time):
    def dir_size_mb(path):
        total = 0
        if os.path.isdir(path):
            for dirpath, _, filenames in os.walk(path):
                for f in filenames:
                    total += os.path.getsize(os.path.join(dirpath, f))
        return total / (1024 ** 2)

    merged_size = dir_size_mb(MERGED_DIR)
    f16_size = os.path.getsize(F16_GGUF_PATH) / (1024 ** 2) if os.path.exists(F16_GGUF_PATH) else 0
    q4_size = os.path.getsize(Q4_GGUF_PATH) / (1024 ** 2) if os.path.exists(Q4_GGUF_PATH) else 0

    os.makedirs("analysis", exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_file = f"analysis/quantize_analysis_{stamp}.md"

    with open(analysis_file, "w", encoding="utf-8") as f:
        f.write("# GGUF Quantization Analysis\n\n")
        f.write(f"- **Build Date**: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write(f"- **Adapter Used**: `{ADAPTER_DIR}`\n")
        f.write(f"- **Final GGUF**: `{Q4_GGUF_PATH}`\n\n")
        f.write("## Size Reduction\n\n")
        f.write("| Stage | Size |\n|---|---:|\n")
        f.write(f"| Merged PyTorch model | {merged_size:.0f} MB |\n")
        f.write(f"| GGUF 16-bit | {f16_size:.0f} MB |\n")
        f.write(f"| **GGUF 4-bit (Q4_K_M)** | **{q4_size:.0f} MB** |\n\n")
        if merged_size > 0:
            f.write(f"The deployable model is **{q4_size / merged_size * 100:.1f}%** "
                    f"of the original size "
                    f"(**{merged_size / q4_size:.1f}x smaller**).\n\n")
        f.write("## Timings\n\n")
        f.write(f"- Merge: {merge_time:.1f}s\n")
        f.write(f"- Convert to GGUF: {convert_time:.1f}s\n")
        f.write(f"- Quantize to Q4_K_M: {quant_time:.1f}s\n")
        f.write(f"- **Total: {merge_time + convert_time + quant_time:.1f}s**\n")

    print(f"\nAnalysis saved to {analysis_file}")
    return merged_size, q4_size


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    merge_time = merge_adapter()
    convert_time = convert_to_gguf()
    quant_time = quantize_gguf()
    merged_size, q4_size = write_analysis(merge_time, convert_time, quant_time)

    print("\n" + "=" * 60)
    print("EDGE MODEL READY")
    print(f"  {merged_size:.0f} MB  ->  {q4_size:.0f} MB "
          f"({merged_size / q4_size:.1f}x smaller)" if q4_size else "")
    print(f"  {Q4_GGUF_PATH}")
    print("\nNext step: 04_gguf_rag.py")
    print("=" * 60)
