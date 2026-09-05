"""
=============================================================================
 DEMO 1 - THE ORIGINAL MODEL, STRAIGHT FROM HUGGINGFACE
=============================================================================
 RUN THIS FIRST. Everything later is measured against what you see here.

 WHAT THIS IS
   google/gemma-3-1b-it exactly as Google published it. No fine-tuning, no
   documents, no compression. 1 billion parameters loaded into PyTorch at
   32-bit precision, which is about 3.7 GB sitting in your RAM.

 WHAT YOU ARE LOOKING FOR

   1. SPEED - watch the tokens/sec at the end. Expect roughly 6 t/s on a
      typical CPU, so a full answer takes over a minute. Slow enough to be
      annoying. Remember that feeling; demo 4 fixes it.

   2. GENERIC KNOWLEDGE - the answer will be textbook-correct about pumps
      and motor overloads in general. The model genuinely knows this domain.

   3. WHAT IT CANNOT KNOW - now ask it something site-specific:

         "What did the last technician find when they serviced P-104?"

      It has never seen your maintenance records, so it will either refuse
      or invent something plausible. That gap is what RAG exists to close.

 THE POINT OF THIS DEMO
   Establish the baseline: a capable but slow, generic model that knows
   nothing about your equipment.
=============================================================================
"""
import os
import sys
import time
import datetime

# Allow importing config.py from the parent folder.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

from config import LOCAL_MODEL_DIR, BASE_SYSTEM_PROMPT, MAX_TOKENS

# This workshop is CPU-only by design - that is the whole point of "edge".
DEVICE = "cpu"

STANDARD_QUESTION = (
    "I am at a remote crude-transfer pumping station. Pump P-104 has stopped "
    "repeatedly and the local controller is showing a motor overload/thermal-"
    "protection alarm. I have no internet access. What are the common causes "
    "and what should I check first?"
)


# ---------------------------------------------------------------------------
# SECTION 1: LOAD THE MODEL
# ---------------------------------------------------------------------------
def load_model():
    print("=" * 60)
    print("DEMO 1: BASE MODEL (no fine-tuning, no RAG)")
    print("=" * 60)
    print(f"Model : {LOCAL_MODEL_DIR}")
    print(f"Device: {DEVICE.upper()}")

    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR)

    print("\nLoading model into RAM (this takes a few seconds)...")
    start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL_DIR,
        dtype=torch.float32,     # CPUs do float32 math natively. float16 on
                                 # CPU is emulated and actually SLOWER.
        low_cpu_mem_usage=True,  # Stream weights in instead of building a
                                 # second full copy in RAM.
    ).to(DEVICE)

    footprint = model.get_memory_footprint() / (1024 ** 3)
    print(f"  Loaded in {time.time() - start:.1f}s, using {footprint:.2f} GB of RAM")
    print("=" * 60)

    return model, tokenizer


# ---------------------------------------------------------------------------
# SECTION 2: GENERATION
# ---------------------------------------------------------------------------
def log_interaction(query, answer, latency, tokens, tps):
    os.makedirs("results", exist_ok=True)
    with open("results/test_00_base_log.txt", "a", encoding="utf-8") as f:
        f.write(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}]\n")
        f.write(f"SPEED : {latency:.2f}s | {tokens} tokens | {tps:.2f} t/s\n")
        f.write(f"QUERY : {query}\n")
        f.write(f"ANSWER:\n{answer}\n")
        f.write("-" * 60 + "\n")


def ask(model, tokenizer, question):
    """Send one question to the model and stream the answer as it is written."""
    messages = [
        {"role": "system", "content": BASE_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    # Each model family has its own turn markers. apply_chat_template inserts
    # the right ones for whichever model is loaded.
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(DEVICE)

    print("\nAnswer: ", end="", flush=True)
    start = time.time()

    # TextStreamer prints tokens as they are produced, so you can watch the
    # model write at its real speed instead of staring at a blank screen.
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_TOKENS,
        do_sample=False,      # Greedy: always take the likeliest next token,
                              # so this demo is repeatable.
        streamer=streamer,
    )

    latency = time.time() - start
    prompt_length = inputs.input_ids.shape[1]
    tokens = outputs[0][prompt_length:].shape[0]
    tps = tokens / latency if latency > 0 else 0

    answer = tokenizer.decode(
        outputs[0][prompt_length:], skip_special_tokens=True
    ).strip().replace("**", "")

    print("\n" + "-" * 60)
    print(f"Prompt tokens read : {prompt_length}")
    print(f"Speed              : {latency:.2f}s | {tokens} tokens | {tps:.2f} t/s")
    print("-" * 60)

    log_interaction(question, answer, latency, tokens, tps)
    return answer


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    model, tokenizer = load_model()

    # Nothing is asked automatically. The presenter types the question live so
    # the room watches it being asked. The suggested question is printed here
    # ready to copy and paste.
    print("\n" + "=" * 60)
    print("READY - type your question below, then press Enter.")
    print("Type 'exit' when you are done.")
    print("=" * 60)
    print("\nSuggested question to start with:\n")
    print(f"  {STANDARD_QUESTION}")
    print("\nThen try one only your records could answer:")
    print("  What did the last technician find when they serviced P-104?")
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

        ask(model, tokenizer, user_input)

    print("\nSaved to results/test_00_base_log.txt")
    print("Next: python tests/test_02_adapter.py")


if __name__ == "__main__":
    main()
