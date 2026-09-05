"""
=============================================================================
 STEP 01 - BUILD THE RAG KNOWLEDGE BASE               [ RUN BEFORE WORKSHOP ]
=============================================================================
 WHAT THIS DOES
   Turns the engineering documents in data/rag/ into a searchable vector
   database so the model can look things up instead of guessing.

   The pipeline in this file:

     data/rag/*.pdf .docx .html .json
              |
              v  (1) PARSE    - pull plain text out of each file format
              |
              v  (2) CHUNK    - cut text into 1024-character overlapping pieces
              |
              v  (3) EMBED    - convert each chunk into a 384-number vector
              |
              v  (4) STORE    - save vectors into ChromaDB on disk
              |
         chroma_db/   <- searched by every RAG demo in the workshop

 WHY VECTORS?
   Keyword search fails when the technician says "pump won't start" but the
   manual says "motor fails to energise". Embeddings capture MEANING, so
   similar ideas land close together in vector space even with no shared words.

 RUNTIME
   About 1 minute. Run it once before the workshop.

 OUTPUT
   chroma_db/                  <- the vector database
   analysis/rag_analysis_*.md  <- build statistics
=============================================================================
"""
import os
import csv
import gc
import sys
import json
import time
import datetime

import psutil
import chromadb
from sentence_transformers import SentenceTransformer

# Document parsers - one per file format found in data/rag/
import docx                    # .docx  (Word engineering records)
import pypdf                   # .pdf   (equipment manuals)
from bs4 import BeautifulSoup  # .html  (maintenance history pages)

from config import (
    LOCAL_MODEL_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    get_embedder_path,
)

DATA_DIR = "data/rag"


# ---------------------------------------------------------------------------
# SECTION 0: PRE-FLIGHT CHECK
# ---------------------------------------------------------------------------
def print_system_check():
    """Show the participant what hardware this is about to run on."""
    print("=" * 60)
    print("STEP 01: BUILD RAG KNOWLEDGE BASE")
    print("=" * 60)

    ram = psutil.virtual_memory()
    print(f"Total RAM      : {ram.total / (1024 ** 3):.2f} GB")
    print(f"Available RAM  : {ram.available / (1024 ** 3):.2f} GB")
    print(f"Embedding model: {get_embedder_path()}")
    print(f"Database path  : {CHROMA_DB_PATH}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# SECTION 1: PARSE - get plain text out of many different file formats
# ---------------------------------------------------------------------------
def parse_file(file_path):
    """
    Extract readable text from one document.

    Real field documentation is never one tidy format - here we handle the
    four that show up in data/rag/. Anything we cannot read is skipped with a
    warning rather than crashing the whole build.
    """
    ext = os.path.splitext(file_path)[1].lower()
    text = ""

    try:
        if ext in (".txt", ".md"):
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

        elif ext == ".json":
            # Service tickets. Re-serialise so the field names stay visible
            # as searchable text.
            with open(file_path, "r", encoding="utf-8") as f:
                text = json.dumps(json.load(f), indent=2)

        elif ext == ".csv":
            with open(file_path, "r", encoding="utf-8") as f:
                text = "\n".join(", ".join(row) for row in csv.reader(f))

        elif ext == ".html":
            # Strip all the HTML tags, keep only what a human would read.
            with open(file_path, "r", encoding="utf-8") as f:
                text = BeautifulSoup(f.read(), "html.parser").get_text(separator="\n")

        elif ext == ".docx":
            text = "\n".join(p.text for p in docx.Document(file_path).paragraphs)

        elif ext == ".pdf":
            for page in pypdf.PdfReader(file_path).pages:
                text += (page.extract_text() or "") + "\n"

    except Exception as e:
        print(f"  [SKIP] Could not parse {os.path.basename(file_path)}: {e}")
        return ""

    # Collapse blank lines and trim - PDFs especially are full of ragged
    # whitespace that would otherwise waste space inside our chunks.
    return "\n".join(line.strip() for line in text.split("\n") if line.strip())


def load_documents(data_dir):
    """Walk data/rag/ and parse every document we find."""
    docs = []
    for root, _, files in os.walk(data_dir):
        for file in sorted(files):
            content = parse_file(os.path.join(root, file))
            if content:
                docs.append({"filename": file, "content": content})
                print(f"  [OK]   {file} ({len(content):,} chars)")
    return docs


# ---------------------------------------------------------------------------
# SECTION 2: CHUNK - cut long documents into model-sized pieces
# ---------------------------------------------------------------------------
def chunk_text(text, size, overlap):
    """
    Slice text into overlapping windows.

    The overlap matters. Without it, a chunk boundary could fall in the middle
    of "the overload alarm is caused by | a blocked cooling fan" and neither
    half would answer the question. Overlapping means every sentence appears
    whole in at least one chunk.
    """
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


# ---------------------------------------------------------------------------
# SECTION 3 + 4: EMBED and STORE
# ---------------------------------------------------------------------------
def build_vector_db(docs, rebuild=False):
    """
    Convert every chunk into a vector and store it in ChromaDB.

    Set rebuild=True (CLI: --rebuild) to wipe and start over. You need that
    whenever you add or edit anything in data/rag/, because an existing
    database is otherwise reused as-is and your changes would be ignored.
    """
    print("\nOpening ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    if rebuild:
        try:
            client.delete_collection(name=COLLECTION_NAME)
            print("  Existing collection deleted (--rebuild).")
        except Exception:
            pass  # Nothing to delete on a first run - that is fine.

    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    if collection.count() > 0:
        print(f"  Database already holds {collection.count()} chunks - reusing it.")
        print("  (If you changed anything in data/rag/, re-run with: --rebuild)")
        return collection

    # --- Load the embedding model ------------------------------------------
    # Kept on CPU deliberately: it is tiny, and this leaves memory free for
    # the language model that runs later in the workshop.
    print(f"\nLoading embedding model from {get_embedder_path()} ...")
    embedder = SentenceTransformer(get_embedder_path(), device="cpu")

    # --- Chunk every document ----------------------------------------------
    print("\nChunking documents...")
    ids, documents, metadatas = [], [], []
    for doc in docs:
        for i, chunk in enumerate(chunk_text(doc["content"], CHUNK_SIZE, CHUNK_OVERLAP)):
            ids.append(f"{doc['filename']}_chunk_{i}")
            documents.append(chunk)
            # Metadata rides along with each vector so answers can cite the
            # source file they came from.
            metadatas.append({"source": doc["filename"], "chunk_id": i})

    print(f"  Produced {len(documents)} chunks from {len(docs)} documents.")

    # --- Embed and insert in batches ---------------------------------------
    # Batching keeps peak memory low and gives visible progress.
    print("\nEmbedding and inserting...")
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        collection.add(
            ids=ids[i:i + batch_size],
            documents=batch,
            embeddings=embedder.encode(batch).tolist(),
            metadatas=metadatas[i:i + batch_size],
        )
        print(f"  {min(i + batch_size, len(documents))}/{len(documents)} chunks stored")

    # Release the embedder - we are done with it in this script.
    del embedder
    gc.collect()

    print("  Vector database build complete.")
    return collection


# ---------------------------------------------------------------------------
# SECTION 5: WRITE THE ANALYSIS REPORT
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


def write_analysis(docs, collection, build_time):
    os.makedirs("analysis", exist_ok=True)
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_file = f"analysis/rag_analysis_{run_id}.md"

    with open(analysis_file, "w", encoding="utf-8") as f:
        f.write("# RAG Database Build Analysis\n\n")
        f.write(f"- **Build Date**: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write(f"- **Base Model**: `{LOCAL_MODEL_DIR}` "
                f"(Size: {get_dir_size_mb(LOCAL_MODEL_DIR):.2f} MB)\n")
        f.write(f"- **Embedding Model**: `{get_embedder_path()}`\n")
        f.write(f"- **Documents Processed**: {len(docs)}\n")
        f.write(f"- **Chunks Created**: {collection.count()}\n")
        f.write(f"- **Chunk Size**: {CHUNK_SIZE} characters (overlap {CHUNK_OVERLAP})\n")
        f.write(f"- **Database Size**: {get_dir_size_mb(CHROMA_DB_PATH):.2f} MB\n")
        f.write(f"- **Total Build Time**: {build_time:.2f} seconds\n")

    print(f"\nAnalysis saved to {analysis_file}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def run_build(rebuild=False):
    start_time = time.time()

    print(f"\nReading documents from {DATA_DIR}/ ...")
    docs = load_documents(DATA_DIR)

    if not docs:
        print(f"\nERROR: No readable documents found in {DATA_DIR}/")
        sys.exit(1)

    print(f"\nParsed {len(docs)} documents.")

    collection = build_vector_db(docs, rebuild=rebuild)
    build_time = time.time() - start_time

    write_analysis(docs, collection, build_time)

    print("\n" + "=" * 60)
    print(f"RAG knowledge base ready: {collection.count()} chunks "
          f"in {build_time:.1f}s")
    print("Next step: 02_finetune_qlora.py")
    print("=" * 60)


if __name__ == "__main__":
    print_system_check()
    run_build(rebuild="--rebuild" in sys.argv)
