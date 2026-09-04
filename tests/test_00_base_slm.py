import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from config import LOCAL_MODEL_DIR, BASE_SYSTEM_PROMPT

def get_system_prompt():
    return BASE_SYSTEM_PROMPT

def log_interaction(query, response, latency):
    import datetime
    os.makedirs("results", exist_ok=True)
    with open("results/test_00_base_log.txt", "a", encoding="utf-8") as f:
        f.write(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\n")
        f.write(f"LATENCY: {latency:.2f}s\n")
        f.write(f"QUERY: {query}\n")
        f.write(f"RESPONSE:\n{response}\n")
        f.write("-" * 40 + "\n")

def generate_response(model, tokenizer, device, question):
    messages = [
        {"role": "system", "content": get_system_prompt()},
        {"role": "user", "content": question}
    ]
    
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
    inputs = tokenizer([text], return_tensors="pt").to(device)
    print("\nAnswer: ", end="", flush=True)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False, streamer=streamer)
    input_length = inputs.input_ids.shape[1]
    num_tokens = outputs[0][input_length:].shape[0]
    ans = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip().replace("**", "")
    return ans, num_tokens

def run_interactive_test():
    print("="*40)
    print("TESTING BASE MODEL (NO RAG)")
    print("="*40)
    
    has_xpu = hasattr(torch, "xpu") and torch.xpu.is_available()
    device = "xpu" if has_xpu else "cpu"
    print(f"Loading {LOCAL_MODEL_DIR} on {device}...")
    
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR)
    dtype = torch.float16 if device == "xpu" else torch.float32
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            LOCAL_MODEL_DIR,
            torch_dtype=dtype,
            low_cpu_mem_usage=True
        ).to(device)
    except Exception as e:
        print(f"Failed to load on {device}: {e}. Falling back to CPU...")
        device = "cpu"
        model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_DIR).to(device)

    # Standard Question
    q = "I am at a remote crude-transfer pumping station. Pump P-104 has stopped repeatedly and the local controller is showing a motor overload/thermal-protection alarm. I have no internet access. What are the common causes and what should I check first?"
    print(f"\n[STANDARD QUERY]: {q}")
    import time
    start = time.time()
    ans, num_tokens = generate_response(model, tokenizer, device, q)
    latency = time.time() - start
    tps = num_tokens / latency if latency > 0 else 0
    print(f"\n[Metrics] Latency: {latency:.2f}s | Tokens: {num_tokens} | Speed: {tps:.2f} t/s")
    log_interaction(q, ans, latency)
    
    print("\n" + "="*40)
    print("INTERACTIVE MODE (Type 'exit' or 'quit' to stop)")
    print("="*40)
    
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ['exit', 'quit']:
                break
            if not user_input.strip():
                continue
                
            start = time.time()
            response, num_tokens = generate_response(model, tokenizer, device, user_input)
            latency = time.time() - start
            tps = num_tokens / latency if latency > 0 else 0
            print(f"\n[Metrics] Latency: {latency:.2f}s | Tokens: {num_tokens} | Speed: {tps:.2f} t/s")
            log_interaction(user_input, response, latency)
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    run_interactive_test()
