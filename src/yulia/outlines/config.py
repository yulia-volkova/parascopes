import os

# DeepInfra
DEEPINFRA_API_URL = "https://api.deepinfra.com/v1/openai/chat/completions"
DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY")  

# Models to compare (DeepInfra model names)
MODELS = [
    "openai/gpt-oss-120b",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
    "Qwen/Qwen2-72B-Instruct"
]

HF_DATASET = "nickypro/fineweb-llama3b-regen"
HF_SPLIT   = "train"

N_SAMPLES = 10

# Results directory setup
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Include sample count in filenames
VERSION = "0.4"
METRICS_CSV = os.path.join(RESULTS_DIR, f"model_outline_metrics_{VERSION}_n{N_SAMPLES}.csv")
GENERATIONS_CSV = os.path.join(RESULTS_DIR, f"outlines_{VERSION}_n{N_SAMPLES}.csv")


OUTLINE_PROMPT_RULES = """
Return a concise, high-level bullet-point outline of the main ideas from the text provided.
Do NOT include any reasoning, commentary, or interpretation.

Rules:
- Preserve the logical flow and cover all key sections of the text.
- Make as many numbered main points as needed but capture only the most important ideas.
- Under each main point, include as many concise subpoints but keep them short.
- Skip specific examples and details.
- Avoid lengthy descriptive terms; use short, precise wording and keep only the most necessary details.
- Summarize the ideas when possible. 
- Focus on the core idea rather than repeating phrases from the text.
- If multiple subpoints convey the same idea, consolidate them.
- Ensure subpoints reflect the overall idea, not just the beginning or end of the corresponding section.
- Keep content specific to the provided text (no generic statements or added context).
"""


# Metric thresholds (for quick pass/fail)
THRESHOLDS = {
    "recall": 0.50,       # target minimum
    "conciseness": 0.32,  # target maximum
    "redundancy": 0.40    # target maximum
}

# API parameters for outline generation
OUTLINE_TEMPERATURE = 0.2  # Lower temperature for more focused outline generation
OUTLINE_MAX_TOKENS = 600   # Higher token limit for structured outlines