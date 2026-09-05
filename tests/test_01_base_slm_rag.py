"""
=============================================================================
 BONUS DEMO - BASE MODEL + RAG  (no fine-tuning)
=============================================================================
 THIS ONE IS OPTIONAL. It is not part of the main five-demo sequence.
 Skip it if you are short on time; run it if you want the complete picture.

 WHY IT EXISTS
   The main sequence changes one thing at a time:

     Demo 1  base                 - neither
     Demo 2  base + fine-tune     - behaviour only
     Demo 3  base + fine-tune+RAG - behaviour and facts
     Demo 4  quantized            - speed
     Demo 5  quantized + RAG      - everything

   That sequence never shows RAG WITHOUT fine-tuning. This file fills that
   gap, which lets you answer the obvious question from the room:

     "Do we actually need the fine-tuning, or is RAG doing all the work?"

 WHAT TO LOOK FOR
   The facts here will be just as good as demo 3 - RAG supplies those, and
   RAG does not care whether the model was fine-tuned. What differs is the
   SHAPE of the answer: expect looser structure and more waffle than demo 3.

   That comparison is the honest answer to the question above: RAG supplies
   the facts, fine-tuning supplies the discipline to present them well.
=============================================================================
"""
import os
import sys
import time
import datetime

# Allow importing config.py from the parent folder.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

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
# SECTION 1: LOAD MODEL AND KNOWLEDGE BASE  (no adapter this time)
# ---------------------------------------------------------------------------
def load_everything():
    print("=" * 60)
    print("BONUS DEMO: BASE MODEL + RAG (no fine-tuning)")
    print("=" * 60)
    print(f"Model  : {LOCAL_MODEL_DIR}")
    print(f"Device : {DEVICE.upper()}")

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
    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL_DIR,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).to(DEVICE)

    print("  Ready.")
    print("=" * 60)
    return model, tokenizer, collection, embedder


# ---------------------------------------------------------------------------
# SECTION 2: RETRIEVE
# ---------------------------------------------------------------------------
def retrieve(query, embedder, collection):
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
    return chunks, sources, best_distance


def choose_prompt(best_distance, chunks):
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
    with open("results/test_01_base_rag_log.txt", "a", encoding="utf-8") as f:
        f.write(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}]\n")
        f.write(f"RAG MODE: {mode} | DISTANCE: {distance}\n")
        f.write(f"SPEED   : {latency:.2f}s | {tokens} tokens | {tps:.2f} t/s\n")
        f.write(f"QUERY   : {query}\n")
        f.write(f"SOURCES : {sources}\n")
        f.write(f"ANSWER  :\n{answer}\n")
        f.write("-" * 60 + "\n")


def ask(model, tokenizer, collection, embedder, question):
    start = time.time()

    chunks, sources, best_distance = retrieve(question, embedder, collection)
    rag_mode, system_prompt = choose_prompt(best_distance, chunks)

    if rag_mode == "NO_RAG":
        chunks, sources = [], []
        user_content = f"Question: {question}"
    else:
        context = "\n\n".join(chunks)
        user_content = f"Context:\n{context}\n\nQuestion: {question}"

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

    print(f"\n[STANDARD QUESTION]\n{STANDARD_QUESTION}")
    ask(model, tokenizer, collection, embedder, STANDARD_QUESTION)

    print("\n" + "=" * 60)
    print("YOUR TURN - type a question, or 'exit' to finish.")
    print("Compare the SHAPE of these answers against demo 3.")
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

    print("\nSaved to results/test_01_base_rag_log.txt")


if __name__ == "__main__":
    main()
