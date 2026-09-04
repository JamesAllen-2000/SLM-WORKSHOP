import os
import json
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

from config import ACTIVE_MODEL_ID, LOCAL_MODEL_DIR

# ---------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------

# ---------------------------------------------------------
# 1. DOWNLOAD
# ---------------------------------------------------------
def download_model():
    print("="*40)
    print("STEP 0: DOWNLOAD BASE MODEL")
    print("="*40)
    
    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
    print(f"Downloading {ACTIVE_MODEL_ID} to {LOCAL_MODEL_DIR}...")
    
    # Download to default HF cache first to avoid MAX_PATH issues with long temporary download filenames
    cache_path = snapshot_download(
        repo_id=ACTIVE_MODEL_ID,
        ignore_patterns=["*.msgpack", "*.h5", "coreml/*"] # ignore formats we don't need
    )
    
    print(f"Moving model files from cache ({cache_path}) to {LOCAL_MODEL_DIR}...")
    import shutil
    shutil.copytree(cache_path, LOCAL_MODEL_DIR, dirs_exist_ok=True)
    
    print("Download and copy complete!\n")

if __name__ == "__main__":
    download_model()
