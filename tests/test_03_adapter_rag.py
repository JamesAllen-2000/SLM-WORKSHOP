"""
=============================================================================
 DEMO 3 - FINE-TUNED MODEL + RAG
=============================================================================
 RUN THIS AFTER: tests/test_02_adapter.py
 REQUIRES: chroma_db/ (built by 01_rag_baseline.py)

 WHAT CHANGED SINCE DEMO 2
   The model is identical. What is new is that before answering, we now
   SEARCH 592 chunks of real engineering documentation and paste the most
   relevant passages into the prompt.

 WHAT YOU ARE LOOKING FOR

   1. SPECIFICS - the answer now contains real asset tags, real readings and
      real history, because the documents are in front of the model.

   2. SOURCES - the demo prints which files it used. This is the single
      biggest practical advantage of RAG in an industrial setting: a
      technician can go and read the original record. An answer you can
      check is worth far more than an answer you must trust.

   3. THE RETRIEVAL DISTANCE - printed after every answer. SMALLER means a
      better match. Watch how it changes with how you phrase the question.

   4. IT IS SLOWER - roughly 90-110 seconds instead of 80. The retrieved
      documents add around 750 tokens the CPU must read before it can start
      writing. RAG buys accuracy with latency.

 NOW GO BACK AND ASK THE QUESTION THAT FAILED IN DEMOS 1 AND 2

      "What did the last technician find when they serviced P-104?"

   In demo 1 and 2 the model had to invent an answer. Here it can look it up.

 THE KEY IDEA OF THE WHOLE WORKSHOP
   Fine-tuning taught the model HOW to answer.
   RAG gives it the FACTS to answer with.
   You need both.
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
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from peft import PeftModel

from config import (
    LOCAL_MODEL_DIR,
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    get_embedder_path,
    RAG_SYSTEM_PROMPT,
    BASE_SYSTEM_PROMPT,
    EXACT_RAG_SYSTEM_PROMPT,
    EXACT_MATCH_THRESHOLD,
    STRONG_MATCH_THRESHOLD,
    NO_MATCH_THRESHOLD,
    TOP_K,
    SHOW_RETRIEVED_CONTEXT,
    CONTEXT_PREVIEW_CHARS,
    TEMPERATURE,
    TOP_P,
    MAX_TOKENS,
    REPEAT_PENALTY,
)

DEVICE = "cpu"

STANDARD_QUESTION = (
    "I am at a remote crude-transfer pumping station. Pump P-104 has stopped "
    "repeatedly and the local controller is showing a motor overload/thermal-"
    "protection alarm. I have no internet access. What are the common causes "
    "and what should I check first?"
)


# ---------------------------------------------------------------------------
# SECTION 1: LOAD MODEL, ADAPTER, AND KNOWLEDGE BASE
# ---------------------------------------------------------------------------
def get_latest_adapter():
    adapters = glob.glob("models/adapter_*")
    if not adapters:
        print("ERROR: No adapter found. Run 02_finetune_qlora.py first.")
        raise SystemExit(1)
    return max(adapters, key=os.path.getmtime)


def load_everything():
    adapter_dir = get_latest_adapter()

    print("=" * 60)
    print("DEMO 3: FINE-TUNED MODEL + RAG")
    print("=" * 60)
    print(f"Base model : {LOCAL_MODEL_DIR}")
    print(f"Adapter    : {adapter_dir}")
    print(f"Device     : {DEVICE.upper()}")

    print("\nOpening knowledge base...")
    try:
        collection = chromadb.PersistentClient(path=CHROMA_DB_PATH).get_collection(
            name=COLLECTION_NAME
        )
    except Exception:
        print(f"ERROR: No knowledge base at {CHROMA_DB_PATH}.")
        print("Run 01_rag_baseline.py first.")
        raise SystemExit(1)
    print(f"  {collection.count()} document chunks available")

    print(f"\nLoading embedding model from {get_embedder_path()} ...")
    embedder = SentenceTransformer(get_embedder_path(), device="cpu")

    print("\nLoading base model...")
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR)
    base_model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL_DIR,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).to(DEVICE)

    print("Applying adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_dir)

    print("  Ready.")
    print("=" * 60)
    return model, tokenizer, collection, embedder


# ---------------------------------------------------------------------------
# SECTION 2: RETRIEVE - search the documents
# ---------------------------------------------------------------------------
def retrieve(query, embedder, collection):
    """Find the TOP_K most relevant chunks. Smaller distance = better match."""
    query_vector = embedder.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )

    chunks = results["documents"][0] if results["documents"] else []
    metas = results["metadatas"][0] if results["metadatas"] else []
    distances = results["distances"][0] if results.get("distances") else []

    best_distance = distances[0] if distances else float("inf")
    sources = [m["source"] for m in metas]
    detail = [(metas[i]["source"], distances[i], chunks[i].strip())
              for i in range(len(chunks))]

    return chunks, sources, best_distance, detail


def show_retrieved_context(detail):
    """
    Print the actual document text that was handed to the model.

    Without this the retrieval step is invisible - the answer just appears and
    you have to take on trust that it came from your documents. Seeing the raw
    chunks first, with their real readings and dates in them, is what makes RAG
    click for people.
    """
    if not detail:
        print("\nNo documents matched closely enough - answering from the "
              "model's own knowledge.")
        return

    print("\n" + "=" * 60)
    print(f"RETRIEVED CONTEXT  ({len(detail)} chunks handed to the model)")
    print("=" * 60)

    for i, (source, dist, content) in enumerate(detail, 1):
        preview = content[:CONTEXT_PREVIEW_CHARS]
        if len(content) > CONTEXT_PREVIEW_CHARS:
            preview += f"... [+{len(content) - CONTEXT_PREVIEW_CHARS} more chars]"

        print(f"\n--- CHUNK {i} | distance {dist:.4f} | {source}")
        for line in preview.splitlines():
            print(f"    {line}")

    print("\n" + "=" * 60)


def choose_prompt(best_distance, chunks):
    """Trust the documents in proportion to how well they matched."""
    if not chunks or best_distance > NO_MATCH_THRESHOLD:
        return "NO_RAG", BASE_SYSTEM_PROMPT
    if best_distance <= EXACT_MATCH_THRESHOLD:
        return "EXACT", EXACT_RAG_SYSTEM_PROMPT
    if best_distance <= STRONG_MATCH_THRESHOLD:
        return "STRONG", RAG_SYSTEM_PROMPT
    return "MODERATE", RAG_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# SECTION 3: GENERATION
# ---------------------------------------------------------------------------
def log_interaction(query, answer, sources, latency, tokens, tps, mode, distance):
    os.makedirs("results", exist_ok=True)
    with open("results/test_03_adapter_rag_log.txt", "a", encoding="utf-8") as f:
        f.write(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}]\n")
        f.write(f"RAG MODE: {mode} | DISTANCE: {distance}\n")
        f.write(f"SPEED   : {latency:.2f}s | {tokens} tokens | {tps:.2f} t/s\n")
        f.write(f"QUERY   : {query}\n")
        f.write(f"SOURCES : {sources}\n")
        f.write(f"ANSWER  :\n{answer}\n")
        f.write("-" * 60 + "\n")


def ask(model, tokenizer, collection, embedder, question):
    start = time.time()

    chunks, sources, best_distance, detail = retrieve(question, embedder, collection)
    rag_mode, system_prompt = choose_prompt(best_distance, chunks)

    if rag_mode == "NO_RAG":
        chunks, sources, detail = [], [], []
        user_content = f"Question: {question}"
    else:
        context = "\n\n".join(chunks)
        user_content = f"Context:\n{context}\n\nQuestion: {question}"

    # Show what the model is about to read, before it answers.
    if SHOW_RETRIEVED_CONTEXT:
        show_retrieved_context(detail)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(DEVICE)

    print("\nAnswer: ", end="", flush=True)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_TOKENS,
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        repetition_penalty=REPEAT_PENALTY,
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
    if best_distance != float("inf"):
        print(f"Retrieval distance : {best_distance:.4f}  (smaller = better)")
    print(f"RAG mode           : {rag_mode}")
    if sources:
        print("Sources used:")
        for s in dict.fromkeys(sources):
            print(f"  * {s}")
    print(f"Prompt tokens read : {prompt_length}   <- documents made this big")
    print(f"Speed              : {latency:.2f}s | {tokens} tokens | {tps:.2f} t/s")
    print("-" * 60)

    log_interaction(question, answer, sources, latency, tokens, tps,
                    rag_mode, best_distance)
    return answer


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    model, tokenizer, collection, embedder = load_everything()

    # Nothing is asked automatically. The presenter types the question
    # live so the room watches it being asked. The suggested question is
    # printed here ready to copy and paste.
    print("\n" + "=" * 60)
    print("READY - type your question below, then press Enter.")
    print("Type 'exit' when you are done.")
    print("=" * 60)
    print("\nSuggested question to start with:\n")
    print(f"  {STANDARD_QUESTION}")
    print("\nThen try the question that failed in demos 1 and 2:")
    print("  What did the last technician find when they serviced P-104?")
    print("\nMore questions in chatbot_question_bank.md")
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

        ask(model, tokenizer, collection, embedder, user_input)

    print("\nSaved to results/test_03_adapter_rag_log.txt")
    print("Next: python tests/test_04_gguf.py")


if __name__ == "__main__":
    main()
