"""
=============================================================================
 STEP 00 - DOWNLOAD THE MODELS                          [ ORGANISER ONLY ]
=============================================================================
 WHAT THIS DOES
   Downloads the two models the workshop needs from the HuggingFace Hub:

     1. The BASE LLM        - google/gemma-3-1b-it  (~2 GB)
     2. The EMBEDDING MODEL - all-MiniLM-L6-v2      (~90 MB)

   The embedding model is easy to forget. It is a separate, much smaller
   model whose only job is turning text into vectors so RAG can search.
   Without a local copy, every RAG script silently phones home to the
   internet on startup - which breaks the "fully offline edge device" story
   and saturates the venue wifi when everyone runs it at once.

 WHO RUNS THIS
   Nobody, during the workshop. Participants receive the models/ folder
   pre-populated, because downloading ~2 GB on shared wifi is far too slow.

   Organisers run this ONCE while preparing the distribution bundle.

 OUTPUT
   models/it/          <- base LLM      (folder name comes from config.py)
   models/embedder/    <- embedding model
=============================================================================
"""
import os
import sys
import shutil

from huggingface_hub import snapshot_download

from config import (
    ACTIVE_MODEL_ID,
    LOCAL_MODEL_DIR,
    EMBEDDING_MODEL_ID,
    EMBEDDING_MODEL_DIR,
)


# ---------------------------------------------------------------------------
# HELPER: explain a gated-repo failure in plain language
# ---------------------------------------------------------------------------
def _explain_gated_repo(repo_id):
    """
    Gemma is a GATED model. Google requires you to accept their licence and
    prove who you are before the weights can be downloaded. An unauthenticated
    request gets a bare '401 Unauthorized', which is not a helpful message,
    so we translate it here.

    Participants never hit this - they receive models/ pre-populated.
    """
    print("\n" + "!" * 70)
    print(f"CANNOT DOWNLOAD: {repo_id} is a GATED model.")
    print("!" * 70)
    print("\nThis is not a bug. Google requires a licence acceptance plus a")
    print("HuggingFace login before these weights can be downloaded.\n")
    print("Fix it in three steps:\n")
    print(f"  1. Open https://huggingface.co/{repo_id}")
    print("     Sign in and click 'Acknowledge license'. Approval is instant.\n")
    print("  2. Create an access token (role: read) at")
    print("     https://huggingface.co/settings/tokens\n")
    print("  3. Log in from this terminal, then re-run this script:")
    print("       hf auth login")
    print("     (on older huggingface_hub versions: huggingface-cli login)\n")
    print("     Or set the token as an environment variable instead:")
    print("       $env:HF_TOKEN = \"hf_your_token_here\"      # PowerShell\n")
    print("Alternative: use a model that is NOT gated. In config.py, switch")
    print("ACTIVE_MODEL_ID to one of the Qwen options - they need no login.")
    print("!" * 70 + "\n")


def _check_login():
    """Report whether we are authenticated, before we try to download."""
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        print("HuggingFace auth: token found in environment.")
        return True

    try:
        from huggingface_hub import get_token
        if get_token():
            print("HuggingFace auth: logged in.")
            return True
    except Exception:
        pass

    print("HuggingFace auth: NOT logged in.")
    print("  Fine for open models. Gated models (Gemma) will fail - "
          "run 'hf auth login' first.")
    return False


# ---------------------------------------------------------------------------
# HELPER: size reporting and copy filtering
# ---------------------------------------------------------------------------
def _dir_size_mb(path):
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp) and not os.path.islink(fp):
                total += os.path.getsize(fp)
    return total / (1024 ** 2)


def _make_ignore(patterns):
    """
    Build a shutil.copytree `ignore` callback from HuggingFace-style patterns.

    Handles both plain file globs ("*.h5") and whole-directory patterns
    ("onnx/*", which we treat as "skip the onnx folder entirely").
    """
    if not patterns:
        return None

    import fnmatch

    file_globs = [p for p in patterns if not p.endswith("/*")]
    skip_dirs = {p[:-2] for p in patterns if p.endswith("/*")}

    def _ignore(directory, names):
        ignored = set()
        for name in names:
            if name in skip_dirs:
                ignored.add(name)
                continue
            if any(fnmatch.fnmatch(name, g) for g in file_globs):
                ignored.add(name)
        return ignored

    return _ignore


# ---------------------------------------------------------------------------
# HELPER: download a repo, then copy it into our short local path
# ---------------------------------------------------------------------------
def _download_to(repo_id, target_dir, ignore_patterns=None):
    """
    Fetches `repo_id` and places it at `target_dir`.

    We deliberately download into the default HuggingFace cache FIRST and copy
    afterwards. HuggingFace uses very long temporary filenames while
    downloading; combined with a deep project path those blow past the Windows
    260-character MAX_PATH limit and the download dies halfway through.
    Downloading to the short cache path first sidesteps that entirely.
    """
    os.makedirs(target_dir, exist_ok=True)
    print(f"  Fetching {repo_id} ...")

    try:
        cache_path = snapshot_download(
            repo_id=repo_id,
            ignore_patterns=ignore_patterns,
        )
    except Exception as e:
        # GatedRepoError is the common case; a plain 401 means the same thing.
        if "GatedRepo" in type(e).__name__ or "401" in str(e) or "restricted" in str(e):
            _explain_gated_repo(repo_id)
            raise SystemExit(1)
        raise

    print(f"  Copying into {target_dir} ...")
    # IMPORTANT: ignore_patterns above only filters what gets DOWNLOADED. The
    # HuggingFace cache may still hold files from an earlier, unfiltered run,
    # and copytree would happily copy those too - silently inflating the
    # bundle (the embedding model goes from 88 MB to 932 MB this way).
    # So we apply the same patterns again when copying.
    shutil.copytree(
        cache_path,
        target_dir,
        dirs_exist_ok=True,
        ignore=_make_ignore(ignore_patterns),
    )
    print(f"  Done -> {target_dir} ({_dir_size_mb(target_dir):.0f} MB)\n")


# ---------------------------------------------------------------------------
# SECTION 1: THE BASE LANGUAGE MODEL
# ---------------------------------------------------------------------------
def download_base_model():
    print("[1/2] BASE LANGUAGE MODEL")
    _download_to(
        ACTIVE_MODEL_ID,
        LOCAL_MODEL_DIR,
        # Skip weight formats we never use (TensorFlow / Flax / CoreML).
        # These can easily double the download size for no benefit.
        ignore_patterns=["*.msgpack", "*.h5", "coreml/*"],
    )


# ---------------------------------------------------------------------------
# SECTION 2: THE EMBEDDING MODEL (needed for RAG search)
# ---------------------------------------------------------------------------
def download_embedding_model():
    print("[2/2] EMBEDDING MODEL (for RAG)")
    _download_to(
        EMBEDDING_MODEL_ID,
        EMBEDDING_MODEL_DIR,
        # This repo ships the same weights in six formats. Without this filter
        # the download is ~930 MB instead of ~90 MB - and every one of those
        # extra megabytes has to be copied onto every participant's laptop.
        # We keep only model.safetensors plus the tokenizer files.
        ignore_patterns=[
            "onnx/*",           # 476 MB
            "openvino/*",       # 109 MB
            "*.h5",             # TensorFlow
            "*.ot",             # Rust
            "pytorch_model.bin",  # duplicate of model.safetensors
            "*.msgpack",
        ],
    )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("STEP 00: DOWNLOAD MODELS  (organiser preparation step)")
    print("=" * 60)

    # Check this BEFORE downloading 2 GB, not after it fails.
    _check_login()
    print("-" * 60)

    # Skip work that is already done - re-running this script is cheap.
    if os.path.isfile(os.path.join(LOCAL_MODEL_DIR, "model.safetensors")):
        print(f"[1/2] BASE LANGUAGE MODEL - already present at "
              f"{LOCAL_MODEL_DIR}, skipping.\n")
    else:
        download_base_model()

    download_embedding_model()

    print("=" * 60)
    print("All models downloaded.")
    print(f"  LLM      : {LOCAL_MODEL_DIR}")
    print(f"  Embedder : {EMBEDDING_MODEL_DIR}")
    print("\nThese two folders are what you distribute to participants.")
    print("=" * 60)
