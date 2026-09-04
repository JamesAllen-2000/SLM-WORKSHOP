import os
import gc
import json
import torch
import psutil
import time
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

import datetime

from config import LOCAL_MODEL_DIR as MODEL_ID

# ---------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------
run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
ADAPTER_OUTPUT_DIR = f"models/adapter_{run_id}"

# QLoRA constraints based on 16GB RAM + CPU execution
MAX_SEQ_LENGTH = 512
BATCH_SIZE = 1
GRAD_ACCUMULATION = 8
EPOCHS = 2
LEARNING_RATE = 2e-4
MAX_DATASET_SIZE = 120

# ---------------------------------------------------------
# PRE-FLIGHT CHECKS
# ---------------------------------------------------------
def print_system_check():
    print("="*40)
    print("SYSTEM PRE-FLIGHT (QLoRA FINETUNING)")
    print("="*40)
    
    ram = psutil.virtual_memory()
    print(f"Available RAM: {ram.available / (1024**3):.2f} GB")
    
    has_xpu = hasattr(torch, "xpu") and torch.xpu.is_available()
    print(f"Intel XPU: {'AVAILABLE' if has_xpu else 'UNAVAILABLE'}")
    
    try:
        import bitsandbytes as bnb
        print("bitsandbytes: INSTALLED")
    except ImportError:
        print("bitsandbytes: NOT INSTALLED - Please install bitsandbytes")
        exit(1)
        
    device = "xpu" if has_xpu else "cpu"
    if device == "cpu":
        print("WARNING: QLoRA training on CPU will be extremely slow. XPU preferred.")
        
    print("="*40)
    return device

# ---------------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------------
def format_chat_template(tokenizer, messages):
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

def prepare_dataset(file_path, tokenizer):
    if not os.path.exists(file_path):
        print(f"ERROR: Dataset {file_path} not found.")
        exit(1)
        
    texts = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            
            # Support both "messages" array and custom "input"/"expected" formats
            if "messages" in data:
                text = format_chat_template(tokenizer, data["messages"])
                texts.append(text)
            elif "input" in data and "expected" in data:
                exp = data["expected"]
                
                # Construct a conversational response
                conversational_response = f"This appears to be a {exp.get('priority', 'standard')} priority {exp.get('issue_category', 'maintenance')} issue with the {exp.get('product', 'equipment')} (Asset: {exp.get('asset_id', 'Unknown')}).\n\n"
                
                if "resolution_steps" in exp and exp["resolution_steps"]:
                    conversational_response += "Here are the steps to resolve it:\n"
                    for idx, step in enumerate(exp["resolution_steps"], 1):
                        conversational_response += f"{idx}. {step}\n"
                        
                if exp.get("escalation_required"):
                    conversational_response += "\nNote: Escalation is required for this issue."
                    
                messages = [
                    {"role": "user", "content": data["input"]},
                    {"role": "assistant", "content": conversational_response.strip()}
                ]
                text = format_chat_template(tokenizer, messages)
                texts.append(text)
                
    # Slice the dataset to the maximum size threshold
    texts = texts[:MAX_DATASET_SIZE]
                
    return Dataset.from_dict({"text": texts})

# ---------------------------------------------------------
# TRAINING
# ---------------------------------------------------------
def train(device):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("\nPreparing Datasets...")
    train_dataset = prepare_dataset("data/finetune/train.jsonl", tokenizer)
    print(f"Loaded {len(train_dataset)} training examples.")
    
    # Check 4-bit loading for QLoRA
    print("\nLoading 4-bit Base Model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        # Using float32 instead of float16/bfloat16 prevents the CPU from hanging during the math!
        bnb_4bit_compute_dtype=torch.float32
    )
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map={"": device}, # map to current device
        )
    except Exception as e:
        print(f"ERROR: 4-bit loading failed on {device}. Reason: {e}")
        exit(1)
        
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    
    # Apply LoRA Adapter
    print("Applying LoRA Configuration...")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"], # common for Qwen
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    print("\nStarting QLoRA Training...")
    training_args = SFTConfig(
        output_dir="results/checkpoints",
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        num_train_epochs=EPOCHS,
        logging_steps=10,
        optim="paged_adamw_32bit",
        save_strategy="no",
        fp16=(device == "xpu" and not torch.xpu.is_bf16_supported()),
        bf16=(device == "xpu" and torch.xpu.is_bf16_supported()),
        gradient_checkpointing=True,
        report_to="none",
        max_length=MAX_SEQ_LENGTH,
        dataset_text_field="text"
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        args=training_args,
    )
    
    try:
        print("Starting training...")
        start_time = time.time()
        trainer.train()
        train_time = time.time() - start_time
        print(f"\nTraining completed in {train_time:.2f} seconds ({train_time/60:.2f} minutes).")
        
        # GENERATE ANALYSIS
        def get_dir_size_mb(path):
            total = 0
            if os.path.exists(path):
                for dirpath, _, filenames in os.walk(path):
                    for f in filenames:
                        fp = os.path.join(dirpath, f)
                        if not os.path.islink(fp):
                            total += os.path.getsize(fp)
            return total / (1024**2)
            
        os.makedirs("analysis", exist_ok=True)
        analysis_file = f"analysis/finetune_analysis_{run_id}.md"
        
        # Get final loss from history if available
        history = trainer.state.log_history
        final_loss = history[-2].get("loss", "N/A") if len(history) > 1 else "N/A"
        
        base_size = get_dir_size_mb(MODEL_ID)
        adapter_size = get_dir_size_mb(ADAPTER_OUTPUT_DIR)
        
        with open(analysis_file, "w", encoding="utf-8") as f:
            f.write("# QLoRA Finetuning Analysis\n\n")
            f.write(f"- **Build Date**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"- **Base Model**: `{MODEL_ID}` (Size: {base_size:.2f} MB)\n")
            f.write(f"- **Adapter Output**: `{ADAPTER_OUTPUT_DIR}` (Size: {adapter_size:.2f} MB)\n")
            f.write(f"- **Training Device**: `{device}`\n")
            f.write(f"- **Dataset Size**: {len(train_dataset)} examples\n")
            f.write(f"- **Max Seq Length**: {MAX_SEQ_LENGTH}\n")
            f.write(f"- **Epochs**: {EPOCHS}\n")
            f.write(f"- **Final Training Loss**: {final_loss}\n")
            f.write(f"- **Total Training Time**: {train_time:.2f} seconds ({train_time/60:.2f} minutes)\n")
            f.write(f"\n## Raw Log History\n```json\n")
            import json
            for log in history:
                f.write(json.dumps(log) + "\n")
            f.write("```\n")
            
        print(f"Analysis saved to {analysis_file}")
            
    except Exception as e:
        print(f"\nERROR during training: {e}")
        print("Recommendation: Decrease MAX_SEQ_LENGTH or increase GRAD_ACCUMULATION (keep BATCH_SIZE=1).")
        exit(1)
        
    print(f"\nSaving LoRA adapter to {ADAPTER_OUTPUT_DIR}...")
    trainer.model.save_pretrained(ADAPTER_OUTPUT_DIR)
    tokenizer.save_pretrained(ADAPTER_OUTPUT_DIR)
    
    print("Training complete! Freeing memory...")
    del model
    del trainer
    gc.collect()
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()

if __name__ == "__main__":
    dev = print_system_check()
    train(dev)
