"""
=============================================================================
 VERIFY SETUP - RUN THIS BEFORE YOU COME TO THE WORKSHOP
=============================================================================
 Checks that everything the workshop needs is installed and in place, then
 prints a single PASS/FAIL table.

     python verify_setup.py

 If every line says OK, you are ready. If anything says MISSING, the message
 tells you exactly which command fixes it.

 Run this twice:
   1. Right after `pip install -r requirements.txt`
   2. Again after you have run steps 01, 02 and 03

 The second run is the one that matters - it confirms your fine-tuned and
 quantized models actually exist.
=============================================================================
"""
import os
import sys
import glob
import platform

# ---------------------------------------------------------------------------
# Small helpers for the results table
# ---------------------------------------------------------------------------
RESULTS = []


def check(label, ok, detail="", fix="", critical=True):
    RESULTS.append({
        "label": label,
        "ok": ok,
        "detail": detail,
        "fix": fix,
        "critical": critical,
    })
    status = "OK     " if ok else ("MISSING" if critical else "SKIP   ")
    print(f"  [{status}] {label:<34} {detail}")
    if not ok and fix:
        print(f"            -> {fix}")


def section(title):
    print(f"\n{title}")
    print("-" * 60)


def dir_size_mb(path):
    total = 0
    if os.path.isdir(path):
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.exists(fp):
                    total += os.path.getsize(fp)
    return total / (1024 ** 2)


# ---------------------------------------------------------------------------
# SECTION 1: PYTHON AND HARDWARE
# ---------------------------------------------------------------------------
def check_environment():
    section("1. ENVIRONMENT")

    major, minor = sys.version_info[:2]
    # 3.10+ works; 3.13 is what the workshop is tested on.
    check(
        "Python version",
        major == 3 and minor >= 10,
        f"{platform.python_version()} (tested on 3.13)",
        "Install Python 3.13 and recreate your virtual environment.",
    )

    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        cores = psutil.cpu_count(logical=False) or psutil.cpu_count()
        check("RAM", ram_gb >= 7.5, f"{ram_gb:.1f} GB (8 GB minimum)",
              "You may hit memory errors during step 03.")
        check("CPU cores", True, f"{cores} physical cores", critical=False)
    except ImportError:
        check("psutil", False, "", "pip install -r requirements.txt")


# ---------------------------------------------------------------------------
# SECTION 2: PYTHON PACKAGES
# ---------------------------------------------------------------------------
def check_packages():
    section("2. PYTHON PACKAGES")

    packages = [
        ("torch", "PyTorch", "pip install -r requirements.txt"),
        ("transformers", "Transformers", "pip install -r requirements.txt"),
        ("peft", "PEFT (LoRA)", "pip install -r requirements.txt"),
        ("trl", "TRL (training)", "pip install -r requirements.txt"),
        ("bitsandbytes", "bitsandbytes (4-bit)", "pip install -r requirements.txt"),
        ("datasets", "Datasets", "pip install -r requirements.txt"),
        ("sentence_transformers", "SentenceTransformers", "pip install -r requirements.txt"),
        ("chromadb", "ChromaDB", "pip install -r requirements.txt"),
        ("pypdf", "pypdf (.pdf)", "pip install -r requirements.txt"),
        ("docx", "python-docx (.docx)", "pip install python-docx"),
        ("bs4", "BeautifulSoup (.html)", "pip install beautifulsoup4"),
    ]

    for module, label, fix in packages:
        try:
            mod = __import__(module)
            version = getattr(mod, "__version__", "")
            check(label, True, version)
        except Exception:
            check(label, False, "", fix)

    # llama-cpp-python gets its own message: this is the one that most often
    # fails to install, because PyPI ships no pre-built wheel for it.
    try:
        import llama_cpp
        check("llama-cpp-python", True, getattr(llama_cpp, "__version__", ""))
    except Exception:
        check(
            "llama-cpp-python", False, "",
            "pip install llama-cpp-python "
            "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu",
        )


# ---------------------------------------------------------------------------
# SECTION 3: MODELS (shipped to you, or built by steps 02 and 03)
# ---------------------------------------------------------------------------
def check_models():
    section("3. MODELS")

    try:
        from config import LOCAL_MODEL_DIR, EMBEDDING_MODEL_DIR, get_safe_model_name
    except Exception as e:
        check("config.py", False, str(e)[:40], "Run from the project root folder.")
        return

    # --- Base LLM ----------------------------------------------------------
    base_ok = os.path.isfile(os.path.join(LOCAL_MODEL_DIR, "config.json"))
    check(
        "Base model", base_ok,
        f"{LOCAL_MODEL_DIR} ({dir_size_mb(LOCAL_MODEL_DIR):.0f} MB)" if base_ok else "",
        "Copy the models/ folder from the workshop USB / shared drive.",
    )

    # --- Embedding model ---------------------------------------------------
    embed_ok = os.path.isdir(EMBEDDING_MODEL_DIR)
    check(
        "Embedding model", embed_ok,
        f"{EMBEDDING_MODEL_DIR} ({dir_size_mb(EMBEDDING_MODEL_DIR):.0f} MB)" if embed_ok else "",
        "Copy models/embedder/, or it will be downloaded on first use "
        "(needs internet).",
        critical=False,
    )

    # --- Fine-tuned adapter (step 02) --------------------------------------
    adapters = glob.glob("models/adapter_*")
    check(
        "Fine-tuned adapter", bool(adapters),
        f"{max(adapters, key=os.path.getmtime)}" if adapters else "",
        "Run: python 02_finetune_qlora.py   (takes 20-70 minutes)",
    )

    # --- Quantized GGUF (step 03) ------------------------------------------
    safe_name = get_safe_model_name().lower()
    ggufs = glob.glob(f"models/gguf/{safe_name}*-q4_k_m.gguf")
    if ggufs:
        newest = max(ggufs, key=os.path.getmtime)
        detail = f"{os.path.basename(newest)} " \
                 f"({os.path.getsize(newest) / (1024 ** 2):.0f} MB)"
    else:
        detail = ""
    check(
        "Quantized model (GGUF)", bool(ggufs), detail,
        "Run: python 03_quantize_gguf.py   (takes about 2 minutes)",
    )


# ---------------------------------------------------------------------------
# SECTION 4: KNOWLEDGE BASE (step 01)
# ---------------------------------------------------------------------------
def check_knowledge_base():
    section("4. RAG KNOWLEDGE BASE")

    try:
        from config import CHROMA_DB_PATH, COLLECTION_NAME
        import chromadb
    except Exception:
        check("ChromaDB", False, "", "pip install -r requirements.txt")
        return

    if not os.path.isdir(CHROMA_DB_PATH):
        check("Vector database", False, "",
              "Run: python 01_rag_baseline.py   (takes about 1 minute)")
        return

    try:
        count = chromadb.PersistentClient(path=CHROMA_DB_PATH).get_collection(
            name=COLLECTION_NAME
        ).count()
        check("Vector database", count > 0, f"{count} document chunks",
              "Run: python 01_rag_baseline.py --rebuild")
    except Exception:
        check("Vector database", False, "",
              "Run: python 01_rag_baseline.py --rebuild")


# ---------------------------------------------------------------------------
# SECTION 5: LLAMA.CPP BINARIES (needed by step 03)
# ---------------------------------------------------------------------------
def check_llama_cpp():
    section("5. LLAMA.CPP TOOLS")

    quantize = any(
        os.path.exists(p)
        for p in ("llama.cpp/llama-quantize.exe", "llama.cpp/llama-quantize",
                  "llama-quantize.exe")
    )
    check("llama-quantize", quantize, "",
          "Restore the llama.cpp/ folder from the workshop bundle.")

    convert = any(
        os.path.exists(p)
        for p in ("llama.cpp/convert_hf_to_gguf.py",
                  "llama_source/convert_hf_to_gguf.py")
    )
    check("convert_hf_to_gguf.py", convert, "",
          "Restore the llama_source/ folder from the workshop bundle.")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("WORKSHOP SETUP VERIFICATION")
    print("=" * 60)

    check_environment()
    check_packages()
    check_models()
    check_knowledge_base()
    check_llama_cpp()

    failures = [r for r in RESULTS if not r["ok"] and r["critical"]]
    warnings = [r for r in RESULTS if not r["ok"] and not r["critical"]]

    print("\n" + "=" * 60)
    if not failures:
        print("RESULT: READY FOR THE WORKSHOP")
        if warnings:
            print(f"        ({len(warnings)} optional item(s) missing - not blocking)")
        print("\nYou can start with:  python tests/test_00_base_slm.py")
    else:
        print(f"RESULT: {len(failures)} PROBLEM(S) TO FIX")
        print("\nFix these before the workshop:")
        for r in failures:
            print(f"  - {r['label']}")
            if r["fix"]:
                print(f"      {r['fix']}")
    print("=" * 60)

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
