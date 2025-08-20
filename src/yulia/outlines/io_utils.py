import time
import requests
from typing import List, Dict, Tuple
from datasets import load_dataset
from yulia.outlines.config  import (
    DEEPINFRA_API_KEY,
    DEEPINFRA_API_URL,
    OUTLINE_PROMPT_RULES,
    OUTLINE_TEMPERATURE,
    OUTLINE_MAX_TOKENS,
    N_SAMPLES,  # Import N_SAMPLES from config
)

# ===== Dataset Loader =====
def load_sample(
    dataset_name: str, 
    split: str, 
    n: int = None, 
    start_idx: int = 0,
    seed: int = None
) -> List[Dict]:
    """
    Load items from a HuggingFace dataset.
    
    Args:
        dataset_name: Name of the dataset on HuggingFace
        split: Dataset split to use
        n: Number of items to load (None = all remaining items after start_idx)
        start_idx: Start from this index in the dataset
        seed: If provided, shuffle the dataset with this seed
    
    Returns:
        List of dicts with 'prompt', 'completion', and 'dataset_idx' fields
    """
    ds = load_dataset(dataset_name, split=split)
    
    if seed is not None:
        # If seed provided, shuffle but keep track of original indices
        ds = ds.add_column("dataset_idx", range(len(ds)))
        ds = ds.shuffle(seed=seed)
    else:
        # Otherwise use sequential indices
        ds = ds.add_column("dataset_idx", range(len(ds)))
    
    # Select range to process
    samples_to_load = n if n is not None else N_SAMPLES  # Use config N_SAMPLES as default
    ds = ds.select(range(start_idx, min(start_idx + samples_to_load, len(ds))))
    
    return [{
        "prompt": ex["prompt"],
        "completion": ex["completion"],
        "dataset_idx": ex["dataset_idx"]  # Original index in dataset
    } for ex in ds]

# ===== Outline Generation =====

def _api_call(
    model: str,
    system: str,
    user: str,
    temperature: float = OUTLINE_TEMPERATURE,
    max_tokens: int = OUTLINE_MAX_TOKENS,
    **kwargs
) -> dict:
    headers = {
        "Authorization": f"Bearer {DEEPINFRA_API_KEY or ''}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    payload.update(kwargs)
    
    r = requests.post(DEEPINFRA_API_URL, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()


def build_outline_prompt(completion_text: str) -> str:
    completion_text = (completion_text or "")
    return f"""

{OUTLINE_PROMPT_RULES}


Text:
{completion_text}

Now produce ONLY the outline in this exact format, returned in your message content (no reasoning fields):

Outline:
1. <main point>
   - <subpoint>
2. <main point>
   - <subpoint>
"""


def extract_outline_for_model(model: str, completion_text: str, retry: int = 2, sleep: float = 1.0) -> Tuple[str, str]:
    """
    Returns (clean_outline_text, outline_prompt_used)
    """
    outline_prompt = build_outline_prompt(completion_text)

    system = (
        "You are an outline extractor. "
        "Only return the final outline. "
        "Don't include your reasoning."
    )

    last_err = None
    for _ in range(retry + 1):
        try:
            data = _api_call(model, system, outline_prompt)
            msg = data["choices"][0]["message"]
            # Prefer 'content', but some models put text in 'reasoning_content'
            raw = (msg.get("content") or msg.get("reasoning_content") or "").strip()
            return raw, outline_prompt
        except Exception as e:
            last_err = e
        time.sleep(sleep)

    if last_err:
        print(f"[WARN] {model} outline extraction failed: {last_err}")
    return "", outline_prompt
