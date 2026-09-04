import os
import gc
import json
import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import chromadb

# For parsing various documents
from bs4 import BeautifulSoup
import csv
import docx
import pypdf

from config import (
    LOCAL_MODEL_DIR as MODEL_ID, 
    RAG_SYSTEM_PROMPT,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K
)

# ---------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "workshop_knowledge"

# ---------------------------------------------------------
# STEP 1: SYSTEM CHECK
# ---------------------------------------------------------
def print_system_check():
    print("="*40)
    print("SYSTEM PRE-FLIGHT (BASELINE RAG)")
    print("="*40)
    
    ram = psutil.virtual_memory()
    print(f"Total RAM: {ram.total / (1024**3):.2f} GB")
    print(f"Available RAM: {ram.available / (1024**3):.2f} GB")
    
    has_xpu = hasattr(torch, "xpu") and torch.xpu.is_available()
    print(f"Intel XPU: {'AVAILABLE' if has_xpu else 'UNAVAILABLE'}")
    
    # We will use XPU if available, else CPU
    device = "xpu" if has_xpu else "cpu"
    print(f"Selected Inference Device: {device.upper()}")
    print("="*40)
    return device

# ---------------------------------------------------------
# STEP 2-3: PARSING & CLEANING
# ---------------------------------------------------------
def parse_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    text = ""
    try:
        if ext == ".txt" or ext == ".md":
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        elif ext == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                text = json.dumps(data, indent=2)
        elif ext == ".csv":
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                text = "\n".join([", ".join(row) for row in reader])
        elif ext == ".html":
            with open(file_path, "r", encoding="utf-8") as f:
                soup = BeautifulSoup(f.read(), "html.parser")
                text = soup.get_text(separator="\n")
        elif ext == ".docx":
            doc = docx.Document(file_path)
            text = "\n".join([p.text for p in doc.paragraphs])
        elif ext == ".pdf":
            reader = pypdf.PdfReader(file_path)
            for page in reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
    
    # Simple clean
    text = "\n".join([line.strip() for line in text.split("\n") if line.strip()])
    return text

def load_documents(data_dir):
    docs = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            path = os.path.join(root, file)
            content = parse_file(path)
            if content:
                docs.append({"filename": file, "content": content})
    return docs

# ---------------------------------------------------------
# STEP 4: CHUNKING
# ---------------------------------------------------------
def chunk_text(text, size, overlap):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        start += size - overlap
    return chunks

# ---------------------------------------------------------
# STEP 5 & 6: EMBEDDINGS AND CHROMADB
# ---------------------------------------------------------
def build_vector_db(docs):
    print("Initializing ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
    # Recreate collection to ensure it's fresh if needed, but per instructions, reuse if exists.
    # However, for script 1, we normally build it. We'll check if it's empty.
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    
    if collection.count() > 0:
        print(f"ChromaDB already contains {collection.count()} chunks. Skipping rebuild.")
        return collection
        
    print("Loading SentenceTransformer...")
    # Keep embeddings on CPU to save memory for the LLM
    embedder = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    
    print("Chunking and Embedding documents...")
    ids = []
    documents = []
    metadatas = []
    
    idx = 0
    for doc in docs:
        chunks = chunk_text(doc["content"], CHUNK_SIZE, CHUNK_OVERLAP)
        for i, chunk in enumerate(chunks):
            ids.append(f"{doc['filename']}_chunk_{i}")
            documents.append(chunk)
            metadatas.append({"source": doc["filename"], "chunk_id": i})
            idx += 1
            
    print(f"Inserting {len(documents)} chunks into ChromaDB...")
    # Batch insert to avoid issues
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i+batch_size]
        batch_embeds = embedder.encode(batch_docs).tolist()
        collection.add(
            ids=ids[i:i+batch_size],
            documents=batch_docs,
            embeddings=batch_embeds,
            metadatas=metadatas[i:i+batch_size]
        )
        
    print("ChromaDB build complete.")
    
    # Free embedder memory
    del embedder
    gc.collect()
    
    return collection

# ---------------------------------------------------------
# STEP 7 & 8: RETRIEVAL AND LLM INFERENCE
# ---------------------------------------------------------
def generate_answer(question, context_chunks, model, tokenizer, device):
    context_str = "\n\n".join(context_chunks)
    
    system_prompt = RAG_SYSTEM_PROMPT
    
    user_prompt = f"Context:\n{context_str}\n\nQuestion: {question}"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(device)
    
    outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    
    # Extract just the response
    input_length = inputs.input_ids.shape[1]
    num_tokens = outputs[0][input_length:].shape[0]
    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    
    return response.strip().replace("**", ""), num_tokens

# ---------------------------------------------------------
# STEP 9: REMOVED (TESTING MOVED TO tests/test_01_rag.py)
# ---------------------------------------------------------

def run_baseline(device):
    import time
    import datetime
    start_time = time.time()
    
    # Load Documents
    print("\nReading documents from data/rag/...")
    docs = load_documents("data/rag")
    print(f"Found {len(docs)} documents.")
    
    # Build Chroma
    collection = build_vector_db(docs)
    
    build_time = time.time() - start_time
    print("\nRAG Database built successfully. Proceed to testing script or next step.")
    
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
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_file = f"analysis/rag_analysis_{run_id}.md"
    
    base_model_size = get_dir_size_mb(MODEL_ID)
    
    with open(analysis_file, "w", encoding="utf-8") as f:
        f.write("# RAG Database Build Analysis\n\n")
        f.write(f"- **Build Date**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- **Base Model**: `{MODEL_ID}` (Size: {base_model_size:.2f} MB)\n")
        f.write(f"- **Total Documents Processed**: {len(docs)}\n")
        f.write(f"- **Total Chunks Created**: {collection.count()}\n")
        f.write(f"- **Chunk Size Configuration**: {CHUNK_SIZE} tokens (Overlap: {CHUNK_OVERLAP})\n")
        f.write(f"- **Embedding Model**: `{EMBEDDING_MODEL}`\n")
        f.write(f"- **Total Build Time**: {build_time:.2f} seconds\n")
        
    print(f"Analysis saved to {analysis_file}")

if __name__ == "__main__":
    dev = print_system_check()
    run_baseline(dev)
