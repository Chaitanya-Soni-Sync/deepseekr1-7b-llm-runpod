#!/usr/bin/env python3
"""
Pre-download script for DeepSeek R1 Distill Qwen 7B model.
Useful for building Docker images with pre-cached models.
"""

import os
import logging
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
CACHE_DIR = "/runpod-volume/hf_cache"
LOCAL_MODEL_PATH = "/runpod-volume/model"

# Set environment variables
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["HF_HUB_CACHE"] = CACHE_DIR

def check_disk_space():
    """Check available disk space"""
    try:
        import shutil
        total, used, free = shutil.disk_usage("/runpod-volume")
        free_gb = free // (1024**3)
        total_gb = total // (1024**3)
        logger.info(f"Disk space - Total: {total_gb}GB, Free: {free_gb}GB")
        
        if free_gb < 20:  # DeepSeek models are large
            logger.error(f"Insufficient disk space: {free_gb}GB available, need at least 20GB")
            return False
        return True
    except Exception as e:
        logger.warning(f"Could not check disk space: {e}")
        return True

def download_model():
    """Download and cache the model"""
    try:
        # Create directories
        Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
        Path(LOCAL_MODEL_PATH).mkdir(parents=True, exist_ok=True)
        
        # Check disk space
        if not check_disk_space():
            raise RuntimeError("Insufficient disk space")
        
        logger.info(f"Downloading model: {MODEL_NAME}")
        logger.info(f"Cache directory: {CACHE_DIR}")
        logger.info(f"Local model path: {LOCAL_MODEL_PATH}")
        
        # Download tokenizer
        logger.info("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            cache_dir=CACHE_DIR
        )
        
        # Download model
        logger.info("Downloading model (this may take a while)...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            cache_dir=CACHE_DIR,
            low_cpu_mem_usage=True
        )
        
        # Save locally
        logger.info(f"Saving model to: {LOCAL_MODEL_PATH}")
        model.save_pretrained(LOCAL_MODEL_PATH)
        tokenizer.save_pretrained(LOCAL_MODEL_PATH)
        
        logger.info("Model download and caching complete!")
        
        # Verify the download
        logger.info("Verifying download...")
        test_tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH, local_files_only=True)
        logger.info(f"Tokenizer vocab size: {test_tokenizer.vocab_size}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        return False

if __name__ == "__main__":
    success = download_model()
    exit(0 if success else 1)
