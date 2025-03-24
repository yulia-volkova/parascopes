# %%
import json
from tqdm import tqdm
from utils_gen import get_llama_completion, get_prompts_parallel
from utils_load_data import load_split_paragraphs

prompt_question = """---

The above is an extracted paragraph from a text. Please write a formatted string that explains:
- [text type] what type of text it is (e.g: user request, article paragraph, cooking instruction, bullet point list, section title, code block, etc...)
- [broad topic] What the main broad topic is
- [secondary topic] What the secondary topic is (if applicable)
- [specific topic] What the specific topic is

You should answer in the form: "A [text type] about [broad topic] and [secondary topic], specifically about [specific topic]"

Answer directly now:
"""

paragraphs = load_split_paragraphs(0)
prompts = []
for paragraph in paragraphs:
    prompts.append(prompt_question + paragraph)

# Get 5 examples
if __name__ == "__main__":
    print(len(prompts))
    print(json.dumps(get_prompts_parallel(prompts[:50:10]), indent=4))

# %%
# Function to process paragraphs in batches
def get_summary_strings_batched(paragraphs, batch_size=100):
    """
    Process paragraphs in batches to get summary strings using the LLM.

    Args:
        paragraphs: List of paragraph texts
        batch_size: Size of each batch to process

    Returns:
        List of summary strings for all paragraphs
    """
    all_results = []
    total_paragraphs = len(paragraphs)

    # Create prompts for all paragraphs
    all_prompts = []
    for paragraph in paragraphs:
        all_prompts.append(prompt_question + paragraph)

    # Process in batches
    num_batches = (total_paragraphs + batch_size - 1) // batch_size
    for i in tqdm(range(0, total_paragraphs, batch_size),
                  desc="Processing batches",
                  total=num_batches,
                  postfix=lambda b: f"batch {b//batch_size + 1}: paragraphs {b} to {min(b + batch_size, total_paragraphs)-1}"):
        batch_end = min(i + batch_size, total_paragraphs)

        batch_prompts = all_prompts[i:batch_end]
        batch_results = get_prompts_parallel(batch_prompts, max_workers=10)
        all_results.extend(batch_results)

    return all_results

# Example usage
if __name__ == "__main__":
    # Process all paragraphs in batches of 100
    all_summaries = get_summary_strings_batched(paragraphs, batch_size=100)

    # Save results to file
    with open("../data/paragraphs_0_summaries.json", "w") as f:
        json.dump(all_summaries, f, indent=2)

    print(f"Processed {len(all_summaries)} paragraphs and saved summaries to file")
