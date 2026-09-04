import os
import gc
import sys
import torch
import subprocess
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import glob
from config import LOCAL_MODEL_DIR as BASE_MODEL_DIR, get_safe_model_name

# ---------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------

BASE_MODEL_ID = BASE_MODEL_DIR
GGUF_DIR = "models/gguf"

# Auto-detect the latest adapter
adapter_dirs = glob.glob("models/adapter_*")
if not adapter_dirs:
    ADAPTER_DIR = "models/adapter"
    run_id = "latest"
else:
    ADAPTER_DIR = max(adapter_dirs, key=os.path.getmtime)
    run_id = ADAPTER_DIR.split("_")[-1]

MERGED_DIR = f"models/merged_{run_id}"
safe_name = get_safe_model_name().lower()
F16_GGUF_PATH = f"models/gguf/{safe_name}-finetuned-{run_id}-f16.gguf"
Q4_GGUF_PATH = f"models/gguf/{safe_name}-finetuned-{run_id}-q4_k_m.gguf"

# ---------------------------------------------------------
# STEP 1: MERGE THE ADAPTER
# ---------------------------------------------------------
def merge_adapter():
    if not os.path.exists(ADAPTER_DIR):
        print(f"ERROR: Adapter not found at {ADAPTER_DIR}. Run 02_finetune_qlora.py first.")
        exit(1)
        
    print(f"Loading base model {BASE_MODEL_ID} for merging...")
    # Load base model on CPU to avoid XPU memory limits during merge
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    
    print(f"Loading LoRA adapter from {ADAPTER_DIR}...")
    peft_model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    
    
    print("Merging adapter weights with base model...")
    merge_start = time.time()
    merged_model = peft_model.merge_and_unload()
    global merge_time
    merge_time = time.time() - merge_start
    print(f"Merge completed in {merge_time:.2f} seconds.")
    
    os.makedirs(MERGED_DIR, exist_ok=True)
    print(f"Saving merged model to {MERGED_DIR}...")
    print("Aligning embedding sizes...")
    merged_model.resize_token_embeddings(len(tokenizer))
    
    merged_model.save_pretrained(MERGED_DIR)
    tokenizer.save_pretrained(MERGED_DIR)
    
    # Sanity check inference before conversion
    print("\nRunning sanity check inference on merged model...")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello! Are you working?"}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    
    outputs = merged_model.generate(**inputs, max_new_tokens=20)
    ans = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    print(f"Merged model response: {ans}")
    
    del merged_model
    del peft_model
    del base_model
    gc.collect()

# ---------------------------------------------------------
# STEP 2: LLAMA.CPP GGUF CONVERSION
# ---------------------------------------------------------
def convert_to_gguf():
    os.makedirs(GGUF_DIR, exist_ok=True)
    
    # Locate the conversion script
    convert_script = "llama.cpp/convert_hf_to_gguf.py"
    if not os.path.exists(convert_script):
        if os.path.exists("llama_source/convert_hf_to_gguf.py"):
            convert_script = "llama_source/convert_hf_to_gguf.py"
        else:
            print("\n" + "!" * 60)
            print("ERROR: llama.cpp conversion script not found.")
            print("To fix this, please run the following commands in your terminal:")
            print("  git config --global core.longpaths true")
            print("  git clone https://github.com/ggerganov/llama.cpp llama_source")
            print("!" * 60 + "\n")
            return
            
    convert_cmd = [
        sys.executable,
        convert_script,
        MERGED_DIR,
        "--outfile", F16_GGUF_PATH,
        "--outtype", "f16"
    ]
    
    print("Executing: " + " ".join(convert_cmd))
    convert_start = time.time()
    try:
        subprocess.run(convert_cmd, check=True)
        global convert_time
        convert_time = time.time() - convert_start
        print(f"Successfully created {F16_GGUF_PATH} in {convert_time:.2f} seconds.")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Conversion failed. {e}")
        exit(1)

# ---------------------------------------------------------
# STEP 3: QUANTIZE TO Q4_K_M
# ---------------------------------------------------------
def quantize_gguf():
    print(f"\nQuantizing {F16_GGUF_PATH} to Q4_K_M...")
    
    # Find llama-quantize executable
    quantize_exe = "llama-quantize"
    if os.path.exists("llama-quantize.exe"):
        quantize_exe = ".\\llama-quantize.exe"
    elif os.path.exists("llama.cpp\\llama-quantize.exe"):
        quantize_exe = "llama.cpp\\llama-quantize.exe"
        
    quant_cmd = [
        quantize_exe,
        F16_GGUF_PATH,
        Q4_GGUF_PATH,
        "Q4_K_M"
    ]
    
    print("Executing: " + " ".join(quant_cmd))
    quant_start = time.time()
    try:
        subprocess.run(quant_cmd, check=True)
        global quant_time
        quant_time = time.time() - quant_start
        print(f"Successfully created {Q4_GGUF_PATH} in {quant_time:.2f} seconds.")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Quantization failed. Is 'llama-quantize' in your PATH? {e}")
        exit(1)
    except FileNotFoundError:
        print("ERROR: 'llama-quantize' command not found. Please compile/install llama.cpp.")
        exit(1)
        
    # Test the quantized model
    print("\nRunning sanity check inference on Q4_K_M model...")
    cli_cmd = [
        "llama-cli",
        "-m", Q4_GGUF_PATH,
        "-p", "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello! Are you working?<|im_end|>\n<|im_start|>assistant\n",
        "-n", "20"
    ]
    
    print("Executing test: " + " ".join(cli_cmd))
    try:
        subprocess.run(cli_cmd, check=True)
    except Exception as e:
        print(f"Warning: Could not run test inference with llama-cli: {e}")
        
    # Record sizes
    merged_size = sum(f.stat().st_size for f in os.scandir(MERGED_DIR) if f.is_file()) / (1024**2)
    f16_size = os.path.getsize(F16_GGUF_PATH) / (1024**2) if os.path.exists(F16_GGUF_PATH) else 0
    q4_size = os.path.getsize(Q4_GGUF_PATH) / (1024**2) if os.path.exists(Q4_GGUF_PATH) else 0
    
    # GENERATE ANALYSIS
    import datetime
    os.makedirs("analysis", exist_ok=True)
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_file = f"analysis/quantize_analysis_{run_id}.md"
    
    with open(analysis_file, "w", encoding="utf-8") as f:
        f.write("# GGUF Quantization Analysis\n\n")
        f.write(f"- **Build Date**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- **Base Model Dir**: `{BASE_MODEL_DIR}`\n")
        f.write(f"- **Final GGUF Path**: `{Q4_GGUF_PATH}`\n")
        f.write(f"- **Quantization Method**: `Q4_K_M` (4-bit)\n")
        f.write(f"- **Original Merged Size**: {merged_size:.2f} MB\n")
        f.write(f"- **F16 GGUF Size**: {f16_size:.2f} MB\n")
        f.write(f"- **Final Q4_K_M Size**: {q4_size:.2f} MB\n")
        
        ratio = (q4_size / merged_size * 100) if merged_size > 0 else 0
        f.write(f"\n**Compression Result**: The model was compressed to **{ratio:.1f}%** of its original size.\n")
        
        f.write("\n**Performance Metrics**:\n")
        f.write(f"- **Adapter Merge Time**: {merge_time:.2f} seconds\n")
        f.write(f"- **F16 Conversion Time**: {convert_time:.2f} seconds\n")
        f.write(f"- **Q4_K_M Quantization Time**: {quant_time:.2f} seconds\n")
        f.write(f"- **Total Time**: {(merge_time + convert_time + quant_time):.2f} seconds\n")
        
    print(f"\nAnalysis saved to {analysis_file}")

if __name__ == "__main__":
    merge_adapter()
    convert_to_gguf()
    quantize_gguf()
