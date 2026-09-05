"""
=============================================================================
 STEP 02 - FINE-TUNE THE MODEL WITH QLoRA             [ RUN BEFORE WORKSHOP ]
=============================================================================
 WHAT THIS DOES
   Teaches the base model to ANSWER LIKE A FIELD SERVICE SYSTEM, by training
   on 120 real-shaped maintenance examples from data/finetune/train.jsonl.

 WHAT IS QLoRA?
   Fine-tuning a whole model means updating ~1 billion numbers - far too much
   for a laptop CPU. QLoRA is two tricks stacked:

     Q    = Quantized. Load the frozen base model in 4-bit instead of 32-bit,
            cutting memory from ~4 GB to ~900 MB.
     LoRA = Low-Rank Adaptation. Freeze the original weights entirely and
            train a tiny pair of "side matrices" bolted onto each attention
            layer. We train 0.24% of the parameters and leave the rest frozen.

   The result is an ADAPTER: a small file (a few MB) that reshapes the base
   model's behaviour without ever modifying the 2 GB of original weights.

 WHAT THE MODEL LEARNS HERE
   The training targets are built from structured ticket data, so the model
   learns to produce a [ SUMMARY ] and a numbered [ ACTION PLAN ], to name the
   asset, and to flag escalation. It does NOT learn to list causes - the
   dataset contains no cause information and we refuse to invent any.
   Supplying causes is RAG's job (step 04), because causes need evidence.

 RUNTIME  -  THIS IS THE SLOW STEP
   About 34 minutes on a 16-core Intel Core Ultra laptop; longer on fewer
   cores. Measured, not estimated.
   Start it and go get a coffee. This is why it is homework, not live.

 OUTPUT
   models/adapter_<timestamp>/       <- the trained adapter
   analysis/finetune_analysis_*.md   <- loss curve and timings
=============================================================================
"""
import os
import gc
import json
import time
import datetime

import torch
import psutil
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

from config import LOCAL_MODEL_DIR as MODEL_ID

# ---------------------------------------------------------------------------
# TRAINING SETTINGS - tuned for a 16 GB RAM, CPU-only laptop
# ---------------------------------------------------------------------------
run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
ADAPTER_OUTPUT_DIR = f"models/adapter_{run_id}"

MAX_SEQ_LENGTH = 512      # Longest training example we allow. Our examples are
                          # ~130 tokens, so this never actually truncates.
BATCH_SIZE = 1            # One example at a time - CPU RAM is the limit.
GRAD_ACCUMULATION = 8     # Accumulate 8 examples before updating weights, so
                          # we get the stability of batch-size-8 at the memory
                          # cost of batch-size-1.
EPOCHS = 1                # How many times we sweep the whole dataset.
                          # MEASURED: one pass over 120 examples takes about
                          # 34 minutes on a 16-core Intel Core Ultra laptop.
                          # A second epoch doubles that for a small gain, so
                          # we keep it at 1 to make this homework survivable.
                          # Raise it if you have time and want a stronger
                          # adapter.
LEARNING_RATE = 2e-4      # High-ish, which is normal for LoRA.
MAX_DATASET_SIZE = 120    # Lower this to 60 to halve the training time again
                          # if your machine is slow.

# LoRA shape
LORA_RANK = 16            # Size of the side matrices. Higher = more capacity
                          # to learn, more memory, slower.
LORA_ALPHA = 32           # Scaling factor, conventionally 2x the rank.

# Which parts of the model get adapters attached.
# q_proj/v_proj are the attention query and value projections; o_proj is the
# attention output. Adapting all three teaches response STYLE far more
# effectively than attention alone, which matters because style is exactly
# what we are trying to teach here.
LORA_TARGET_MODULES = ["q_proj", "v_proj", "o_proj"]


# ---------------------------------------------------------------------------
# SECTION 0: PRE-FLIGHT CHECKS
# ---------------------------------------------------------------------------
def print_system_check():
    print("=" * 60)
    print("STEP 02: QLoRA FINE-TUNING")
    print("=" * 60)

    ram = psutil.virtual_memory()
    print(f"Available RAM : {ram.available / (1024 ** 3):.2f} GB")
    print(f"CPU cores     : {psutil.cpu_count(logical=False)} physical")

    try:
        import bitsandbytes  # noqa: F401
        print("bitsandbytes  : INSTALLED")
    except ImportError:
        print("bitsandbytes  : MISSING -> pip install -r requirements.txt")
        raise SystemExit(1)

    print("\nThis step takes 30-90 minutes depending on your CPU. Go get a coffee.")
    print("=" * 60)
    return "cpu"


# ---------------------------------------------------------------------------
# SECTION 1: BUILD THE TRAINING TEXT
# ---------------------------------------------------------------------------
def build_target_answer(expected):
    """
    Turn one structured ticket record into the answer we want the model to
    learn to write.

    CRITICAL DESIGN POINT: this output format must MATCH the format the
    system prompts ask for at inference time. If we train the model to write
    "Here are the steps..." but then ask it for "[ SUMMARY ]", the fine-tune
    fights the prompt instead of reinforcing it, and the adapter appears to
    do nothing at all.

    We emit only the two sections the data can honestly support.
    """
    product = expected.get("product", "equipment")
    asset = expected.get("asset_id", "unknown asset")
    category = expected.get("issue_category", "MAINTENANCE").replace("_", " ").lower()
    priority = expected.get("priority", "standard")

    lines = []

    # --- Section: SUMMARY ---------------------------------------------------
    lines.append("[ SUMMARY ]")
    lines.append(
        f"The {product.lower()} (asset {asset}) has reported a {category} condition. "
        f"This is a {priority} priority maintenance issue requiring structured "
        f"diagnosis before the equipment is returned to service."
    )
    lines.append("")

    # --- Section: ACTION PLAN ----------------------------------------------
    lines.append("[ ACTION PLAN ]")
    for idx, step in enumerate(expected.get("resolution_steps", []), 1):
        lines.append(f"{idx}. {step}")

    # Escalation discipline is a real safety behaviour worth teaching.
    if expected.get("escalation_required"):
        lines.append("")
        lines.append(
            "Escalation is required for this issue. Do not return the asset to "
            "service until the approved escalation procedure has been completed."
        )

    return "\n".join(lines)


def prepare_dataset(file_path, tokenizer):
    """
    Read train.jsonl and convert each row into a full chat conversation.

    We include a SYSTEM turn in every training example. Without it the model
    never learns how to behave when a system prompt is present - which is
    exactly the situation it meets at inference time.
    """
    if not os.path.exists(file_path):
        print(f"ERROR: Dataset {file_path} not found.")
        raise SystemExit(1)

    system_instruction = (
        "You are an offline oil-field field-service assistant. "
        "Answer with a [ SUMMARY ] section followed by a numbered "
        "[ ACTION PLAN ]. Do not use markdown bolding or asterisks."
    )

    texts = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)

            # Pre-formatted conversations pass straight through.
            if "messages" in data:
                messages = data["messages"]

            # Our dataset shape: structured "expected" record -> target answer.
            elif "input" in data and "expected" in data:
                messages = [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": data["input"]},
                    {"role": "assistant", "content": build_target_answer(data["expected"])},
                ]
            else:
                continue

            texts.append(
                tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
            )

    texts = texts[:MAX_DATASET_SIZE]
    return Dataset.from_dict({"text": texts})


# ---------------------------------------------------------------------------
# SECTION 2: TRAINING
# ---------------------------------------------------------------------------
def train(device):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # Gemma-3 ships a real <pad> token. Use it rather than aliasing pad to eos,
    # which would hide the end-of-sequence token from the loss and print a
    # confusing "tokens differ from model config" warning.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\nPreparing dataset...")
    train_dataset = prepare_dataset("data/finetune/train.jsonl", tokenizer)
    print(f"  Loaded {len(train_dataset)} training examples.")
    print("\n--- Example of what the model is learning to write ---")
    print(train_dataset[0]["text"][:600])
    print("--- (truncated) ---\n")

    # --- Load the base model in 4-bit (the "Q" in QLoRA) -------------------
    print("Loading base model in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,   # Quantize the quantization constants
                                          # too - saves a further ~0.4 bits/param.
        bnb_4bit_quant_type="nf4",        # "Normal Float 4": a 4-bit format
                                          # shaped for how weights are actually
                                          # distributed.
        # float32 compute. On CPU, float16 math is emulated and can hang;
        # float32 is both safer and faster here.
        bnb_4bit_compute_dtype=torch.float32,
    )

    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map={"": device},
        )
    except Exception as e:
        print(f"ERROR: 4-bit loading failed on {device}: {e}")
        raise SystemExit(1)

    print(f"  Model footprint: {model.get_memory_footprint() / (1024 ** 2):.0f} MB")

    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False   # Incompatible with gradient checkpointing.

    # --- Attach the LoRA adapters (the "LoRA" in QLoRA) --------------------
    print("\nAttaching LoRA adapters...")
    peft_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    # This line is the headline of the whole step - it shows participants how
    # few parameters we are actually training.
    model.print_trainable_parameters()

    # --- Train --------------------------------------------------------------
    training_args = SFTConfig(
        output_dir="results/checkpoints",
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        num_train_epochs=EPOCHS,
        logging_steps=5,
        # Plain torch AdamW. The bitsandbytes "paged" optimizers are built for
        # CUDA and simply warn and fall back on CPU.
        optim="adamw_torch",
        save_strategy="no",
        fp16=False,     # CPU training stays in float32 - half precision on CPU
        bf16=False,     # is emulated and slower, not faster.
        gradient_checkpointing=True,   # Recompute activations instead of
                                       # storing them. Measured FASTER here,
                                       # not just lighter on memory.
        report_to="none",
        max_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        args=training_args,
    )

    print("\n" + "=" * 60)
    print(f"TRAINING: {len(train_dataset)} examples x {EPOCHS} epochs")
    print("Watch the 'loss' value - it should fall as the model learns.")
    print("=" * 60 + "\n")

    start_time = time.time()
    try:
        trainer.train()
    except Exception as e:
        print(f"\nERROR during training: {e}")
        print("If you ran out of memory, lower MAX_SEQ_LENGTH or MAX_DATASET_SIZE.")
        raise SystemExit(1)
    train_time = time.time() - start_time

    print(f"\nTraining finished in {train_time:.0f}s ({train_time / 60:.1f} min).")

    # --- Save the adapter ---------------------------------------------------
    print(f"Saving adapter to {ADAPTER_OUTPUT_DIR} ...")
    trainer.model.save_pretrained(ADAPTER_OUTPUT_DIR)
    tokenizer.save_pretrained(ADAPTER_OUTPUT_DIR)

    write_analysis(trainer, train_dataset, train_time, device)

    del model, trainer
    gc.collect()

    print("\n" + "=" * 60)
    print(f"Adapter ready: {ADAPTER_OUTPUT_DIR}")
    print("Next step: 03_quantize_gguf.py")
    print("=" * 60)


# ---------------------------------------------------------------------------
# SECTION 3: ANALYSIS REPORT
# ---------------------------------------------------------------------------
def get_dir_size_mb(path):
    total = 0
    if os.path.exists(path):
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp):
                    total += os.path.getsize(fp)
    return total / (1024 ** 2)


def write_analysis(trainer, train_dataset, train_time, device):
    os.makedirs("analysis", exist_ok=True)
    analysis_file = f"analysis/finetune_analysis_{run_id}.md"

    history = trainer.state.log_history
    losses = [h["loss"] for h in history if "loss" in h]

    base_size = get_dir_size_mb(MODEL_ID)
    adapter_size = get_dir_size_mb(ADAPTER_OUTPUT_DIR)

    with open(analysis_file, "w", encoding="utf-8") as f:
        f.write("# QLoRA Finetuning Analysis\n\n")
        f.write(f"- **Build Date**: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write(f"- **Base Model**: `{MODEL_ID}` ({base_size:.2f} MB)\n")
        f.write(f"- **Adapter**: `{ADAPTER_OUTPUT_DIR}` ({adapter_size:.2f} MB)\n")
        if base_size > 0:
            f.write(f"- **Adapter is {adapter_size / base_size * 100:.2f}% "
                    f"the size of the base model**\n")
        f.write(f"- **Device**: `{device}`\n")
        f.write(f"- **LoRA Rank**: {LORA_RANK} (alpha {LORA_ALPHA})\n")
        f.write(f"- **Adapted Modules**: {', '.join(LORA_TARGET_MODULES)}\n")
        f.write(f"- **Dataset Size**: {len(train_dataset)} examples\n")
        f.write(f"- **Epochs**: {EPOCHS}\n")
        if losses:
            f.write(f"- **First Loss**: {losses[0]:.4f}\n")
            f.write(f"- **Final Loss**: {losses[-1]:.4f}\n")
        f.write(f"- **Training Time**: {train_time:.0f}s ({train_time / 60:.1f} min)\n")

        f.write("\n## Raw Log History\n\n```json\n")
        for log in history:
            f.write(json.dumps(log) + "\n")
        f.write("```\n")

    print(f"Analysis saved to {analysis_file}")


if __name__ == "__main__":
    dev = print_system_check()
    train(dev)
