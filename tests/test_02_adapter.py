"""
=============================================================================
 DEMO 2 - THE FINE-TUNED MODEL (base + LoRA adapter)
=============================================================================
 RUN THIS AFTER: tests/test_00_base_slm.py

 WHAT CHANGED SINCE DEMO 1
   Exactly one thing: a 4.6 MB adapter file produced by 02_finetune_qlora.py
   is layered on top of the same 2 GB base model. The original weights are
   untouched. Only 2,396,160 numbers out of 1,002,282,112 were trained -
   that is 0.24% of the model.

 WHAT YOU ARE LOOKING FOR

   1. STRUCTURE - the fine-tuned model was trained on 120 real maintenance
      tickets, so it reaches for [ SUMMARY ] and a numbered [ ACTION PLAN ]
      and names the asset, rather than writing a loose essay.

   2. ESCALATION AWARENESS - the training data encodes when a job must be
      escalated rather than retried. Watch for that language appearing.

   3. SPEED IS UNCHANGED - still roughly 6 tokens/sec. Fine-tuning changes
      WHAT the model says, never how fast it says it. Speed is demo 4's job.

 AN HONEST NOTE FOR THE DISCUSSION
   With a long, highly detailed system prompt, the base model in demo 1 can
   already be pushed into a similar shape - prompting is genuinely powerful.
   The difference is COST. That system prompt is about 560 tokens the CPU
   must read before writing a single word, on every single question. Fine-
   tuning moves that behaviour into the weights, where it is free at runtime.

   On an edge device, prompt tokens are latency. That is the real argument
   for fine-tuning here - not that prompting cannot do it, but that baking
   it in is cheaper on every request forever.

 WHAT IS STILL MISSING
   Facts. The model has learned a HOUSE STYLE for answering, but it still
   has never read your maintenance records. Ask it about P-104's history and
   it will still invent. That is demo 3.
=============================================================================
"""
import os
import sys
import glob
import time
import datetime

# Allow importing config.py from the parent folder.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from peft import PeftModel

from config import LOCAL_MODEL_DIR, BASE_SYSTEM_PROMPT, MAX_TOKENS

DEVICE = "cpu"

STANDARD_QUESTION = (
    "I am at a remote crude-transfer pumping station. Pump P-104 has stopped "
    "repeatedly and the local controller is showing a motor overload/thermal-"
    "protection alarm. I have no internet access. What are the common causes "
    "and what should I check first?"
)


# ---------------------------------------------------------------------------
# SECTION 1: FIND THE ADAPTER
# ---------------------------------------------------------------------------
def get_latest_adapter():
    """
    Pick the most recently trained adapter.

    Every run of 02_finetune_qlora.py creates a new timestamped folder, so
    this always selects your newest training run.
    """
    adapters = glob.glob("models/adapter_*")
    if not adapters:
        print("ERROR: No adapter found in models/.")
        print("Run 02_finetune_qlora.py first (or restore the shipped models/ folder).")
        raise SystemExit(1)
    return max(adapters, key=os.path.getmtime)


# ---------------------------------------------------------------------------
# SECTION 2: LOAD BASE MODEL, THEN LAYER THE ADAPTER ON TOP
# ---------------------------------------------------------------------------
def load_model():
    adapter_dir = get_latest_adapter()
    adapter_mb = sum(
        os.path.getsize(os.path.join(adapter_dir, f))
        for f in os.listdir(adapter_dir)
        if os.path.isfile(os.path.join(adapter_dir, f))
    ) / (1024 ** 2)

    print("=" * 60)
    print("DEMO 2: FINE-TUNED MODEL (base + LoRA adapter, no RAG)")
    print("=" * 60)
    print(f"Base model : {LOCAL_MODEL_DIR}")
    print(f"Adapter    : {adapter_dir} ({adapter_mb:.1f} MB)")
    print(f"Device     : {DEVICE.upper()}")

    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR)

    print("\nLoading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL_DIR,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).to(DEVICE)

    # PeftModel keeps the adapter as a separate layer applied on the fly.
    # We deliberately do NOT merge it here: keeping it separate is what makes
    # LoRA useful in practice - you can ship one 2 GB base model and swap
    # small adapters for different jobs.
    print(f"Applying adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_dir)

    print("  Ready.")
    print("=" * 60)
    return model, tokenizer


# ---------------------------------------------------------------------------
# SECTION 3: GENERATION  (identical to demo 1, for a fair comparison)
# ---------------------------------------------------------------------------
def log_interaction(query, answer, latency, tokens, tps):
    os.makedirs("results", exist_ok=True)
    with open("results/test_02_adapter_log.txt", "a", encoding="utf-8") as f:
        f.write(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}]\n")
        f.write(f"SPEED : {latency:.2f}s | {tokens} tokens | {tps:.2f} t/s\n")
        f.write(f"QUERY : {query}\n")
        f.write(f"ANSWER:\n{answer}\n")
        f.write("-" * 60 + "\n")


def ask(model, tokenizer, question):
    messages = [
        {"role": "system", "content": BASE_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(DEVICE)

    print("\nAnswer: ", end="", flush=True)
    start = time.time()

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_TOKENS,
        do_sample=False,
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
    print("          ^^^ about the same as demo 1 - fine-tuning does not")
    print("              change speed, only behaviour.")
    print("-" * 60)

    log_interaction(question, answer, latency, tokens, tps)
    return answer


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    model, tokenizer = load_model()

    # Nothing is asked automatically. The presenter types the question
    # live so the room watches it being asked. The suggested question is
    # printed here ready to copy and paste.
    print("\n" + "=" * 60)
    print("READY - type your question below, then press Enter.")
    print("Type 'exit' when you are done.")
    print("=" * 60)
    print("\nSuggested question to start with:\n")
    print(f"  {STANDARD_QUESTION}")
    print("\nAsk the SAME question you used in demo 1, then compare")
    print("the shape of the two answers side by side.")
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

    print("\nSaved to results/test_02_adapter_log.txt")
    print("Next: python tests/test_03_adapter_rag.py")


if __name__ == "__main__":
    main()
