import sys
import os
import glob
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from peft import PeftModel
from config import (
    LOCAL_MODEL_DIR, 
    RAG_SYSTEM_PROMPT, 
    BASE_SYSTEM_PROMPT, 
    EXACT_RAG_SYSTEM_PROMPT,
    EXACT_MATCH_THRESHOLD,
    STRONG_MATCH_THRESHOLD,
    NO_MATCH_THRESHOLD,
    TOP_K,
    TEMPERATURE,
    TOP_P,
    MAX_TOKENS,
    REPEAT_PENALTY
)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "workshop_knowledge"

def get_latest_adapter():
    adapter_dirs = glob.glob("models/adapter_*")
    if not adapter_dirs:
        return "models/adapter"
    return max(adapter_dirs, key=os.path.getmtime)

def log_interaction(query, response, sources, latency):
    import datetime
    os.makedirs("results", exist_ok=True)
    with open("results/test_02b_adapter_rag_log.txt", "a", encoding="utf-8") as f:
        f.write(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\n")
        f.write(f"LATENCY: {latency:.2f}s\n")
        f.write(f"QUERY: {query}\n")
        f.write(f"SOURCES: {sources}\n")
        f.write(f"RESPONSE:\n{response}\n")
        f.write("-" * 40 + "\n")

def generate_answer(question, context_chunks, model, tokenizer, device, system_prompt=RAG_SYSTEM_PROMPT):
    if context_chunks:
        context_str = "\n\n".join(context_chunks)
        user_prompt = f"Context:\n{context_str}\n\nQuestion: {question}"
    else:
        user_prompt = f"Question: {question}"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
    inputs = tokenizer([text], return_tensors="pt").to(device)
    
    print("\nModel is thinking...\nAnswer: ", end="", flush=True)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    outputs = model.generate(
        **inputs, 
        max_new_tokens=MAX_TOKENS, 
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        repetition_penalty=REPEAT_PENALTY,
        streamer=streamer
    )
    
    input_length = inputs.input_ids.shape[1]
    num_tokens = outputs[0][input_length:].shape[0]
    ans = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip().replace("**", "")
    return ans, num_tokens

def run_interactive_test():
    print("="*40)
    print("TESTING FINETUNED ADAPTER + RAG")
    print("="*40)
    
    has_xpu = hasattr(torch, "xpu") and torch.xpu.is_available()
    device = "xpu" if has_xpu else "cpu"
    ADAPTER_DIR = get_latest_adapter()
    
    # Load ChromaDB
    print("Connecting to ChromaDB...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = chroma_client.get_collection(name=COLLECTION_NAME)
    
    print("Loading SentenceTransformer for queries...")
    embedder = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    
    print(f"Loading Base Model: {LOCAL_MODEL_DIR} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR)
    dtype = torch.float16 if device == "xpu" else torch.float32
    
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            LOCAL_MODEL_DIR,
            torch_dtype=dtype,
            low_cpu_mem_usage=True
        ).to(device)
    except Exception as e:
        print(f"Failed to load on {device}: {e}. Falling back to CPU...")
        device = "cpu"
        base_model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_DIR).to(device)

    print(f"Applying LoRA Adapter: {ADAPTER_DIR}")
    ft_model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)

    def process_query(q):
        query_embed = embedder.encode(q).tolist()
        results = collection.query(
            query_embeddings=[query_embed],
            n_results=TOP_K,
            include=["documents", "metadatas", "distances"]
        )
        
        chunks = results["documents"][0] if results["documents"] else []
        sources = results["metadatas"][0] if results["metadatas"] else []
        distances = results["distances"][0] if "distances" in results and results["distances"] else []
        
        best_distance = distances[0] if distances else float('inf')
        source_ids = [m["source"] for m in sources]
        
        rag_info = f"\n[RAG Search] Best Context Distance: {best_distance:.4f}"
        
        if best_distance <= EXACT_MATCH_THRESHOLD:
            rag_info += f"\n[Case: EXACT - Injecting {', '.join(source_ids)}]"
            ans, num_tokens = generate_answer(q, chunks, ft_model, tokenizer, device, system_prompt=EXACT_RAG_SYSTEM_PROMPT)
        elif best_distance <= STRONG_MATCH_THRESHOLD:
            rag_info += f"\n[Case: STRONG - Injecting {', '.join(source_ids)}]"
            ans, num_tokens = generate_answer(q, chunks, ft_model, tokenizer, device, system_prompt=RAG_SYSTEM_PROMPT)
        elif best_distance <= NO_MATCH_THRESHOLD:
            rag_info += f"\n[Case: MODERATE - Injecting {', '.join(source_ids)}]"
            ans, num_tokens = generate_answer(q, chunks, ft_model, tokenizer, device, system_prompt=RAG_SYSTEM_PROMPT)
        else:
            rag_info += "\n[Case: NO_RAG - Skipping RAG and using internal base knowledge]"
            ans, num_tokens = generate_answer(q, [], ft_model, tokenizer, device, system_prompt=BASE_SYSTEM_PROMPT)
            source_ids = [] # Clear sources since we didn't use them
            
        return ans, source_ids, num_tokens, rag_info

    # Standard Question
    q = "I am at a remote crude-transfer pumping station. Pump P-104 has stopped repeatedly and the local controller is showing a motor overload/thermal-protection alarm. I have no internet access. What are the common causes and what should I check first?"
    print(f"\n[STANDARD QUERY]: {q}")
    import time
    start = time.time()
    ans, sources, num_tokens, rag_info = process_query(q)
    latency = time.time() - start
    tps = num_tokens / latency if latency > 0 else 0
    print(f"\n[Metrics] Latency: {latency:.2f}s | Tokens: {num_tokens} | Speed: {tps:.2f} t/s")
    print(rag_info + "\n")
    log_interaction(q, ans, sources, latency)
    
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
            ans, sources, num_tokens, rag_info = process_query(user_input)
            latency = time.time() - start
            tps = num_tokens / latency if latency > 0 else 0
            print(f"\n[Metrics] Latency: {latency:.2f}s | Tokens: {num_tokens} | Speed: {tps:.2f} t/s")
            print(rag_info + "\n")
            log_interaction(user_input, ans, sources, latency)
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    run_interactive_test()
