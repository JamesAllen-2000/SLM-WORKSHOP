import os
import sys
import time
import glob
from sentence_transformers import SentenceTransformer
import chromadb
from llama_cpp import Llama

sys.stdout.reconfigure(encoding='utf-8')

from config import (
    get_safe_model_name,
    LOCAL_MODEL_DIR,
    RAG_SYSTEM_PROMPT,
    BASE_SYSTEM_PROMPT,
    EXACT_RAG_SYSTEM_PROMPT,
    EXACT_MATCH_THRESHOLD,
    STRONG_MATCH_THRESHOLD,
    NO_MATCH_THRESHOLD,
    TEMPERATURE,
    TOP_P,
    MAX_TOKENS,
    REPEAT_PENALTY,
    N_CTX,
    TOP_K,
    DETERMINISTIC_MODE
)

# =============================================================================
# CONSTANTS & GLOBALS
# =============================================================================
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "workshop_knowledge"

# Auto-detect latest GGUF model for the currently active model
safe_name = get_safe_model_name().lower()
gguf_files = glob.glob(f"models/gguf/{safe_name}*-q4_k_m.gguf")
if not gguf_files:
    GGUF_MODEL_PATH = f"models/gguf/{safe_name}-finetuned-q4_k_m.gguf"
else:
    GGUF_MODEL_PATH = max(gguf_files, key=os.path.getmtime)

print(f"Using Model: {GGUF_MODEL_PATH}")

# =============================================================================
# LLAMA.CPP INTERFACE
# =============================================================================
print("Loading LLM Backend...")
# Attempt to load with GPU layers if available, gracefully falling back to CPU
try:
    llm = Llama(
        model_path=GGUF_MODEL_PATH,
        n_ctx=N_CTX,
        n_gpu_layers=-1, 
        verbose=False
    )
    HARDWARE_BACKEND = "GPU"
except Exception as e:
    print("Failed to initialize LLM with GPU offload. Falling back to CPU...")
    llm = Llama(
        model_path=GGUF_MODEL_PATH,
        n_ctx=N_CTX,
        n_gpu_layers=0,
        verbose=False
    )
    HARDWARE_BACKEND = "CPU"

def test_hardware_backends():
    print("Testing backend initialization...")
    try:
        start_time = time.time()
        # Minimal deterministic test prompt
        messages = [{"role": "user", "content": "Reply with exactly: OK"}]
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=5,
            temperature=0.0
        )
        latency = time.time() - start_time
        
        # Determine actual backend in use
        backend = HARDWARE_BACKEND
        
        print(f"SUCCESS on {backend} (Latency: {latency:.2f}s)")
        return backend
    except Exception as e:
        print(f"Backend test failed: {e}")
        exit(1)

def log_interaction(query, ans, sources, distance, mode, latency, num_tokens, tps, backend):
    import datetime
    os.makedirs("results", exist_ok=True)
    with open("results/04_gguf_rag_log.txt", "a", encoding="utf-8") as f:
        f.write("-" * 40 + "\n")
        f.write(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\n")
        f.write(f"QUERY: {query}\n")
        f.write(f"RAG MODE: {mode}\n")
        f.write(f"BEST DISTANCE: {distance if distance != float('inf') else 'N/A'}\n")
        f.write(f"LATENCY: {latency:.2f}s | TOKENS: {num_tokens} | SPEED: {tps:.2f} t/s\n")
        f.write(f"BACKEND: {backend}\n")
        f.write(f"SOURCES: {sources}\n")
        f.write(f"RESPONSE:\n{ans}\n\n")

def process_query(query, embedder, collection, backend):
    start = time.time()
    
    # Retrieve
    query_embed = embedder.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embed],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"]
    )
    
    raw_chunks = results["documents"][0] if results["documents"] else []
    raw_sources = results["metadatas"][0] if results["metadatas"] else []
    distances = results["distances"][0] if "distances" in results and results["distances"] else []
    
    best_distance = distances[0] if distances else float('inf')
    
    # Filter and Deduplicate Chunks
    valid_chunks = []
    seen_content = set()
    source_ids = []
    
    for i, dist in enumerate(distances):
        if dist <= NO_MATCH_THRESHOLD:
            chunk_content = raw_chunks[i].strip()
            if chunk_content not in seen_content:
                seen_content.add(chunk_content)
                source_name = raw_sources[i]["source"]
                valid_chunks.append(f"[SOURCE {len(valid_chunks)+1}]\nFile: {source_name}\n{chunk_content}")
                if source_name not in source_ids:
                    source_ids.append(source_name)

    # Determine RAG Mode and Prompt
    if best_distance <= EXACT_MATCH_THRESHOLD and valid_chunks:
        rag_mode = "EXACT"
        system_prompt = EXACT_RAG_SYSTEM_PROMPT
    elif best_distance <= STRONG_MATCH_THRESHOLD and valid_chunks:
        rag_mode = "STRONG"
        system_prompt = RAG_SYSTEM_PROMPT
    elif best_distance <= NO_MATCH_THRESHOLD and valid_chunks:
        rag_mode = "MODERATE"
        system_prompt = RAG_SYSTEM_PROMPT
    else:
        rag_mode = "NO_RAG"
        system_prompt = BASE_SYSTEM_PROMPT
        valid_chunks = []
        source_ids = []

    # Build User Content
    if valid_chunks:
        context_str = "\n\n".join(valid_chunks)
        user_content = f"RETRIEVED CONTEXT:\n{context_str}\n\nUSER QUESTION:\n{query}"
    else:
        user_content = query

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]

    # Generate
    sys.stdout.write("Generating response...\n")
    sys.stdout.flush()
    
    temp = 0.0 if DETERMINISTIC_MODE else TEMPERATURE
    
    try:
        response = llm.create_chat_completion(
            messages=messages,
            temperature=temp,
            top_p=TOP_P,
            max_tokens=MAX_TOKENS,
            repeat_penalty=REPEAT_PENALTY
        )
        ans = response["choices"][0]["message"]["content"].strip()
        num_tokens = response["usage"]["completion_tokens"]
    except Exception as e:
        ans = f"Failed to generate: {e}"
        num_tokens = 0
        
    latency = time.time() - start
    tps = num_tokens / latency if latency > 0 else 0
    
    print(f"\nQuery: {query}\n")
    print(ans)
    print("\n")
    
    # Output Diagnostics
    print("-" * 40)
    print(f"RAG Distance: {best_distance:.4f}" if best_distance != float('inf') else "RAG Distance: N/A")
    print(f"RAG Mode: {rag_mode}")
    print(f"Chunks Used: {len(valid_chunks)}")
    if source_ids:
        print("Sources:")
        for s in source_ids:
            print(f" * {s}")
    print(f"Inference Backend: {backend}")
    print(f"[Metrics] Latency: {latency:.2f}s | Tokens: {num_tokens} | Speed: {tps:.2f} t/s")
    print("-" * 40 + "\n")
    
    # Check output structure for troubleshooting queries
    if rag_mode != "NO_RAG" and not ("[ SUMMARY ]" in ans and "[ POTENTIAL CAUSES ]" in ans and "[ ACTION PLAN ]" in ans):
        print("[WARNING]: Expected troubleshooting sections not found in response.")
        
    log_interaction(query, ans, source_ids, best_distance, rag_mode, latency, num_tokens, tps, backend)
    
    return ans

# ---------------------------------------------------------
# FINAL RAG EXECUTION
# ---------------------------------------------------------
def run_final_rag():
    backend = test_hardware_backends()
    
    print("\nLoading existing ChromaDB...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = chroma_client.get_collection(name=COLLECTION_NAME)
    
    print("Loading SentenceTransformer for queries...")
    embedder = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    
    # Standard evaluation question
    q = "I am at a remote crude-transfer pumping station. Pump P-104 has stopped repeatedly and the local controller is showing a motor overload/thermal-protection alarm. I have no internet access. What are the common causes and what should I check first?"
    process_query(q, embedder, collection, backend)
            
    print("Evaluation complete. Results saved to results/04_gguf_rag_log.txt.")
    
    # ---------------------------------------------------------
    # INTERACTIVE RAG MODE
    # ---------------------------------------------------------
    print("\n" + "="*40)
    print("INTERACTIVE RAG MODE (Type 'exit' or 'quit' to stop)")
    print("="*40)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ['exit', 'quit']:
                break
            if not user_input:
                continue
                
            # Normalize whitespace
            user_input = " ".join(user_input.split())
            
            process_query(user_input, embedder, collection, backend)
            
        except KeyboardInterrupt:
            break
        except EOFError:
            break
            
    # Cleanly release the model to prevent __del__ exception bug in llama-cpp-python
    print("\nShutting down backend...")
    global llm
    del llm

if __name__ == "__main__":
    run_final_rag()
