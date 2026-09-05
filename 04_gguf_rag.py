"""
=============================================================================
 STEP 04 / DEMO 5 - QUANTIZED MODEL + RAG            [ THE FINAL PRODUCT ]
=============================================================================
 WHAT THIS DOES
   Everything the workshop has built, wired together:

     4-bit quantized fine-tuned model  (fast, small, offline)
                    +
     ChromaDB vector search over real engineering documents
                    =
     A field-service assistant that runs on a laptop with the wifi switched
     off, cites the documents it used, and answers in about 20 seconds.

 HOW A QUESTION FLOWS THROUGH THIS FILE

     your question
          |
          v  (1) EMBED       turn the question into a vector
          |
          v  (2) RETRIEVE    find the TOP_K closest document chunks
          |
          v  (3) GRADE       how good is the best match? -> pick a prompt
          |                     excellent -> EXACT prompt  (trust docs)
          |                     good      -> RAG prompt    (use docs)
          |                     poor      -> BASE prompt   (ignore docs)
          |
          v  (4) GENERATE    llama.cpp runs the 4-bit model on your CPU
          |
          v  (5) REPORT      answer + sources + speed, logged to results/

 WHY THIS IS THE INTERESTING DEMO
   Fine-tuning taught the model HOW to answer. RAG gives it the FACTS to
   answer with. Neither alone is enough: a fine-tuned model with no documents
   invents specifics, and documents with no trained behaviour produce
   unstructured walls of text.

 OUTPUT
   results/04_gguf_rag_log.txt   <- every question, answer, source and timing
=============================================================================
"""
import os
import sys
import glob
import time
import datetime

import chromadb
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

from config import (
    get_safe_model_name,
    get_embedder_path,
    trim_runaway,
    CHROMA_DB_PATH,
    COLLECTION_NAME,
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
    SHOW_RETRIEVED_CONTEXT,
    CONTEXT_PREVIEW_CHARS,
    DETERMINISTIC_MODE,
)

sys.stdout.reconfigure(encoding="utf-8")

# This workshop targets plain CPUs - no GPU, no NPU. Setting this to 0 (rather
# than -1, meaning "offload everything") keeps the backend label we print
# honest. A CPU-only llama-cpp-python build silently IGNORES n_gpu_layers=-1,
# so the previous version of this file reported "GPU" on every machine.
N_GPU_LAYERS = 0
BACKEND_LABEL = "CPU"


# ---------------------------------------------------------------------------
# SECTION 1: FIND THE QUANTIZED MODEL
# ---------------------------------------------------------------------------
def find_gguf_model():
    """Pick the most recently built 4-bit GGUF for the active model."""
    safe_name = get_safe_model_name().lower()
    matches = glob.glob(f"models/gguf/{safe_name}*-q4_k_m.gguf")

    if not matches:
        print("ERROR: No quantized model found in models/gguf/")
        print("Run 03_quantize_gguf.py first (or restore the shipped models/ folder).")
        raise SystemExit(1)

    return max(matches, key=os.path.getmtime)


# ---------------------------------------------------------------------------
# SECTION 2: LOAD THE MODEL AND THE KNOWLEDGE BASE
# ---------------------------------------------------------------------------
def load_everything():
    gguf_path = find_gguf_model()
    size_mb = os.path.getsize(gguf_path) / (1024 ** 2)

    print("=" * 60)
    print("DEMO 5: QUANTIZED MODEL + RAG  (the final product)")
    print("=" * 60)
    print(f"Model     : {gguf_path}")
    print(f"Size      : {size_mb:.0f} MB")
    print(f"Backend   : {BACKEND_LABEL} (n_gpu_layers={N_GPU_LAYERS})")
    print(f"Context   : {N_CTX} tokens")

    print("\nLoading quantized model...")
    start = time.time()
    llm = Llama(
        model_path=gguf_path,
        n_ctx=N_CTX,
        n_gpu_layers=N_GPU_LAYERS,
        verbose=False,
    )
    print(f"  Loaded in {time.time() - start:.1f}s")

    print("\nOpening knowledge base...")
    collection = chromadb.PersistentClient(path=CHROMA_DB_PATH).get_collection(
        name=COLLECTION_NAME
    )
    print(f"  {collection.count()} document chunks available")

    print(f"\nLoading embedding model from {get_embedder_path()} ...")
    embedder = SentenceTransformer(get_embedder_path(), device="cpu")

    print("=" * 60)
    return llm, collection, embedder


# ---------------------------------------------------------------------------
# SECTION 3: RETRIEVE - find relevant documents
# ---------------------------------------------------------------------------
def retrieve(query, embedder, collection):
    """
    Search the vector database and return de-duplicated, labelled chunks.

    Returns (context_blocks, source_filenames, best_distance).
    Remember: SMALLER distance = better match.
    """
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

    blocks, sources, seen = [], [], set()
    detail = []              # per-chunk info, for showing on screen
    for i, dist in enumerate(distances):
        if dist > NO_MATCH_THRESHOLD:
            continue

        content = chunks[i].strip()
        if content in seen:      # Overlapping chunks can return the same text
            continue             # twice - show the model only one copy.
        seen.add(content)

        source = metas[i]["source"]
        blocks.append(f"[SOURCE {len(blocks) + 1}]\nFile: {source}\n{content}")
        detail.append((source, dist, content))
        if source not in sources:
            sources.append(source)

    return blocks, sources, best_distance, detail


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


# ---------------------------------------------------------------------------
# SECTION 4: GRADE - how much should we trust the retrieved documents?
# ---------------------------------------------------------------------------
def choose_prompt(best_distance, blocks):
    """
    Pick a system prompt based on retrieval quality.

    Participants usually find this the most interesting part: RAG is not
    all-or-nothing. We change how strongly the model is told to stick to the
    documents, based on how good the match actually was.
    """
    if not blocks or best_distance > NO_MATCH_THRESHOLD:
        return "NO_RAG", BASE_SYSTEM_PROMPT
    if best_distance <= EXACT_MATCH_THRESHOLD:
        return "EXACT", EXACT_RAG_SYSTEM_PROMPT
    if best_distance <= STRONG_MATCH_THRESHOLD:
        return "STRONG", RAG_SYSTEM_PROMPT
    return "MODERATE", RAG_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# SECTION 5: GENERATE AND REPORT
# ---------------------------------------------------------------------------
def log_interaction(query, answer, sources, distance, mode, latency, tokens, tps):
    os.makedirs("results", exist_ok=True)
    with open("results/04_gguf_rag_log.txt", "a", encoding="utf-8") as f:
        f.write("-" * 60 + "\n")
        f.write(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}]\n")
        f.write(f"QUERY   : {query}\n")
        f.write(f"RAG MODE: {mode}\n")
        f.write(f"DISTANCE: {distance if distance != float('inf') else 'N/A'}\n")
        f.write(f"SPEED   : {latency:.2f}s | {tokens} tokens | {tps:.2f} t/s\n")
        f.write(f"BACKEND : {BACKEND_LABEL}\n")
        f.write(f"SOURCES : {sources}\n")
        f.write(f"ANSWER  :\n{answer}\n\n")


def process_query(query, llm, collection, embedder):
    start = time.time()

    blocks, sources, best_distance, detail = retrieve(query, embedder, collection)
    rag_mode, system_prompt = choose_prompt(best_distance, blocks)

    if rag_mode == "NO_RAG":
        blocks, sources, detail = [], [], []
        user_content = query
    else:
        context = "\n\n".join(blocks)
        user_content = f"RETRIEVED CONTEXT:\n{context}\n\nUSER QUESTION:\n{query}"

    # Show what the model is about to read, before it answers.
    if SHOW_RETRIEVED_CONTEXT:
        show_retrieved_context(detail)

    print("\nThinking...")

    try:
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0 if DETERMINISTIC_MODE else TEMPERATURE,
            top_p=TOP_P,
            max_tokens=MAX_TOKENS,
            repeat_penalty=REPEAT_PENALTY,
        )
        answer = response["choices"][0]["message"]["content"].strip()
        tokens = response["usage"]["completion_tokens"]
    except Exception as e:
        answer, tokens = f"Failed to generate: {e}", 0

    latency = time.time() - start
    tps = tokens / latency if latency > 0 else 0

    # A 1B model sometimes finishes the action plan and then writes it all out
    # again under an invented heading. Keep the first good copy, drop repeats.
    answer, trimmed_lines = trim_runaway(answer)

    print("\n" + answer + "\n")

    if trimmed_lines:
        print(f"[note] Trimmed {trimmed_lines} repeated lines. Small models "
              f"sometimes loop instead of stopping - see REPEAT_PENALTY in "
              f"config.py.")

    print("-" * 60)
    if best_distance != float("inf"):
        print(f"Retrieval distance : {best_distance:.4f}  (smaller = better match)")
    print(f"RAG mode           : {rag_mode}")
    print(f"Chunks used        : {len(blocks)}")
    if sources:
        print("Sources:")
        for s in sources:
            print(f"  * {s}")
    print(f"Backend            : {BACKEND_LABEL}")
    print(f"Speed              : {latency:.2f}s | {tokens} tokens | {tps:.2f} t/s")
    print("-" * 60)

    log_interaction(query, answer, sources, best_distance, rag_mode,
                    latency, tokens, tps)
    return answer


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
STANDARD_QUESTION = (
    "I am at a remote crude-transfer pumping station. Pump P-104 has stopped "
    "repeatedly and the local controller is showing a motor overload/thermal-"
    "protection alarm. I have no internet access. What are the common causes "
    "and what should I check first?"
)


def main():
    llm, collection, embedder = load_everything()

    # Nothing is asked automatically. The presenter types the question
    # live so the room watches it being asked. The suggested question is
    # printed here ready to copy and paste.
    print("\n" + "=" * 60)
    print("READY - type your question below, then press Enter.")
    print("Type 'exit' when you are done.")
    print("=" * 60)
    print("\nSuggested question to start with:\n")
    print(f"  {STANDARD_QUESTION}")
    print("\nThen try anything from chatbot_question_bank.md")
    print("\nTurn your wifi off and keep asking - nothing changes.")
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

        process_query(" ".join(user_input.split()), llm, collection, embedder)

    print("\nAll interactions saved to results/04_gguf_rag_log.txt")


if __name__ == "__main__":
    main()
