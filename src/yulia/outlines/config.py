import os

# DeepInfra
DEEPINFRA_API_URL = "https://api.deepinfra.com/v1/openai/chat/completions"
DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY")  

# Models to compare (DeepInfra model names)
MODELS = [
    "meta-llama/Llama-3.3-70B-Instruct"
]

# HuggingFace configuration
HF_DATASET = "nickypro/fineweb-llama3b-regen"
HF_SPLIT = "train"
HF_REPO_ID = os.getenv("HF_REPO_ID", "yulia/outlines-data")  # Default repo ID for saving results
HF_PRIVATE = True  # Whether to create private repo when saving results

N_SAMPLES = 10

# Results directory setup
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Include sample count in filenames
VERSION = "1.0"
METRICS_CSV = os.path.join(RESULTS_DIR, f"model_outline_metrics_{VERSION}_n{N_SAMPLES}.csv")
GENERATIONS_CSV = os.path.join(RESULTS_DIR, f"outlines_{VERSION}_n{N_SAMPLES}.csv")


OUTLINE_PROMPT_RULES = """
Return a short, high-level bullet-point outline of the main ideas from the text you are given.
Do NOT include any reasoning.

Rules:
- Make as many points as you see fit considering the text at hand
- At most 2 short subpoints per point
- Short phrases only (no lengthy sentences)
- Specific to this text (not generic).
"""


# Metric thresholds (for quick pass/fail)
THRESHOLDS = {
    "recall": 0.50,       # target minimum
    "conciseness": 0.32,  # target maximum
    "redundancy": 0.40    # target maximum
}

# API parameters for outline generation
OUTLINE_TEMPERATURE = 0.2
OUTLINE_MAX_TOKENS = 700

# Batch processing parameters
SONAR_BATCH_SIZE = 32      # Batch size for SONAR embedding generation (GPU memory dependent)
PROCESS_BATCH_SIZE = 100   # Batch size for processing samples (CPU memory dependent)
