import os
from dotenv import load_dotenv

load_dotenv()

DEEPINFRA_API_URL = "https://api.deepinfra.com/v1/openai/chat/completions"
DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY")  

MODELS = [
    "meta-llama/Llama-3.3-70B-Instruct"
]

# HuggingFace configuration
HF_DATASET = "nickypro/fineweb-llama3b-regen"
HF_SPLIT = "train"
HF_REPO_ID = os.getenv("HF_REPO_ID", "yulia-volkova/parascopes-outlines-data")  
HF_PRIVATE = True  

N_SAMPLES = 200

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


VERSION = "1.0"  
GENERATIONS_CSV = os.path.join(RESULTS_DIR, f"outlines_{VERSION}_n{N_SAMPLES}.csv")


OUTLINE_PROMPT_RULES = """
Return a short, high-level bullet-point outline of the main ideas from the text you are given.
Do NOT include any reasoning.

Rules:
- Make as 4-5 bullet points maximum
- Use numbers to enumerate the bullet points
- Aim to capture main ideas of the whole text in the bullet points
- At most 2 short subpoints per point
- Short phrases only (no lengthy sentences)
- Specific to this text (not generic).
"""



# THRESHOLDS = {
#     "recall": 0.50,       # target minimum
#     "conciseness": 0.32,  # target maximum
#     "redundancy": 0.40    # target maximum
# }

# API parameters for outline generation
OUTLINE_TEMPERATURE = 0.2
OUTLINE_MAX_TOKENS = 700


SONAR_BATCH_SIZE = 32      # Batch size for SONAR embedding generation (GPU memory dependent)
PROCESS_BATCH_SIZE = 100   # Batch size for processing samples (CPU memory dependent)
