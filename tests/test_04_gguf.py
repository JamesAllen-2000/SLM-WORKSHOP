"""
=============================================================================
 DEMO 4 - THE QUANTIZED MODEL, ON ITS OWN (no RAG yet)
=============================================================================
 RUN THIS AFTER: tests/test_03_adapter_rag.py
 RUN THIS BEFORE: 04_gguf_rag.py

 WHAT YOU ARE LOOKING FOR
   SPEED. This is the same fine-tuned model you have been using for the last
   three demos, but compressed from 16-bit to 4-bit and run by llama.cpp in
   C++ instead of PyTorch in Python.

   Watch the tokens/sec number at the bottom and compare it with demos 1-3:

     Demos 1-3 (PyTorch, 32-bit, ~3.7 GB in RAM)  ->  about  6 tokens/sec
     Demo 4    (llama.cpp, 4-bit,  777 MB)        ->  about 30 tokens/sec

   Roughly 5x faster and 2.5x smaller, running on the same CPU.
   (Exact speed varies with your CPU and how warm it is - 25-40 t/s is
   the normal range. The RATIO against demos 1-3 is the point.)

 WHY IS IT FASTER?
   Generating text on a CPU is limited by MEMORY BANDWIDTH, not arithmetic.
   For every token, the processor must read the entire model out of RAM.
   Quarter the size of the model and you quarter the amount of data that has
   to move, so you get roughly four times the speed. The switch from Python
   to compiled C++ supplies the rest.

 WHAT TO NOTICE ABOUT QUALITY
   Compare an answer here against demo 2. Compression to 4 bits loses a
   little precision, but for this kind of structured technical answer the
   output stays essentially as useful. That trade - a small quality cost for
   a large speed and size win - is the entire argument for edge deployment.

   Notice also what is still MISSING: no equipment history, no real service
   ticket numbers, no site-specific values. The model is fast, but it is
   working purely from memory. Demo 5 adds the documents.
=============================================================================
"""
import os
import sys
import glob
import time
import datetime

# Allow importing config.py from the parent folder.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llama_cpp import Llama

from config import (
    get_safe_model_name,
    trim_runaway,
    BASE_SYSTEM_PROMPT,
    TEMPERATURE,
    TOP_P,
    MAX_TOKENS,
    REPEAT_PENALTY,
    N_CTX,
    DETERMINISTIC_MODE,
)

# CPU-only workshop - see the note in 04_gguf_rag.py.
N_GPU_LAYERS = 0
BACKEND_LABEL = "CPU"

STANDARD_QUESTION = (
    "I am at a remote crude-transfer pumping station. Pump P-104 has stopped "
    "repeatedly and the local controller is showing a motor overload/thermal-"
    "protection alarm. I have no internet access. What are the common causes "
    "and what should I check first?"
)


# ---------------------------------------------------------------------------
# SECTION 1: LOCATE AND LOAD THE QUANTIZED MODEL
# ---------------------------------------------------------------------------
def find_gguf_model():
    safe_name = get_safe_model_name().lower()
    matches = glob.glob(f"models/gguf/{safe_name}*-q4_k_m.gguf")

    if not matches:
        print("ERROR: No quantized model found in models/gguf/")
        print("Run 03_quantize_gguf.py first (or restore the shipped models/ folder).")
        raise SystemExit(1)

    return max(matches, key=os.path.getmtime)


def load_model():
    gguf_path = find_gguf_model()
    size_mb = os.path.getsize(gguf_path) / (1024 ** 2)

    print("=" * 60)
    print("DEMO 4: QUANTIZED MODEL (4-bit GGUF, no RAG)")
    print("=" * 60)
    print(f"Model   : {gguf_path}")
    print(f"Size    : {size_mb:.0f} MB   <- was ~2000 MB before quantization")
    print(f"Backend : {BACKEND_LABEL} (llama.cpp, not PyTorch)")

    print("\nLoading...")
    start = time.time()
    llm = Llama(
        model_path=gguf_path,
        n_ctx=N_CTX,
        n_gpu_layers=N_GPU_LAYERS,
        verbose=False,
    )
    print(f"  Ready in {time.time() - start:.1f}s")
    print("=" * 60)
    return llm


# ---------------------------------------------------------------------------
# SECTION 2: GENERATION
# ---------------------------------------------------------------------------
def log_interaction(query, answer, latency, tokens, tps):
    os.makedirs("results", exist_ok=True)
    with open("results/test_04_gguf_log.txt", "a", encoding="utf-8") as f:
        f.write(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}]\n")
        f.write(f"SPEED : {latency:.2f}s | {tokens} tokens | {tps:.2f} t/s\n")
        f.write(f"QUERY : {query}\n")
        f.write(f"ANSWER:\n{answer}\n")
        f.write("-" * 60 + "\n")


def ask(llm, question):
    """Send one question to the quantized model and report the speed."""
    print("\nThinking...")
    start = time.time()

    try:
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": BASE_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            temperature=0.0 if DETERMINISTIC_MODE else TEMPERATURE,
            top_p=TOP_P,
            max_tokens=MAX_TOKENS,
            repeat_penalty=REPEAT_PENALTY,
        )
        answer = response["choices"][0]["message"]["content"].strip()
        tokens = response["usage"]["completion_tokens"]
    except Exception as e:
        answer, tokens = f"Failed to generate: {e}", 0

    latency = time.time() - start
    tps = tokens / latency if latency > 0 else 0

    # Small models sometimes loop instead of stopping - keep the first copy.
    answer, trimmed_lines = trim_runaway(answer)

    print("\n" + answer + "\n")

    if trimmed_lines:
        print(f"[note] Trimmed {trimmed_lines} repeated lines. Small models "
              f"sometimes loop instead of stopping.")

    print("-" * 60)
    print(f"Backend : {BACKEND_LABEL}")
    print(f"Speed   : {latency:.2f}s | {tokens} tokens | {tps:.2f} t/s")
    print("          ^^^ compare this with demos 1-3 (about 6 t/s)")
    print("-" * 60)

    log_interaction(question, answer, latency, tokens, tps)
    return answer


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    llm = load_model()

    # Nothing is asked automatically. The presenter types the question live so
    # the room watches it being asked. The suggested question is printed here
    # ready to copy and paste.
    print("\n" + "=" * 60)
    print("READY - type your question below, then press Enter.")
    print("Type 'exit' when you are done.")
    print("=" * 60)
    print("\nSuggested question to start with:\n")
    print(f"  {STANDARD_QUESTION}")
    print("\nThen ask about a specific asset, for example:")
    print("  What happened to P-104 last month?")
    print("It cannot answer that - it has no documents. Demo 5 fixes it.")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input:
            continue

        ask(llm, " ".join(user_input.split()))

    print("\nSaved to results/test_04_gguf_log.txt")
    print("Next: python 04_gguf_rag.py   (adds the documents)")


if __name__ == "__main__":
    main()
