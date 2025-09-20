
import os
import io
import json
import shutil
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")


def get_standard_features():
    """Get the standard feature schema for our dataset."""
    from datasets import Features, Value
    return Features({
        'example_id': Value('int64'),
        'dataset_idx': Value('int64'),
        'model': Value('string'),
        'completion': Value('string'),
        'outline_generated': Value('string'),
        'reconstructed_text': Value('string'),  # Always string, empty if no reconstruction
        'embedding_id': Value('int64')
    })


