# %%
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Tuple
from openai import OpenAI
from utils_parallel import exponential_backoff, process_in_parallel


rubric = """
#### 0. Complexity
How complex is the text?
0: Trivial (e.g: just says "** Section **")
1. Simple (e.g: "** Section 1: Green Tea **")
2. Some detail (e.g: a short undetailed sentence or two about something)
3. Many details (e.g: a detailed paragraph with specific information)

#### 1. Coherence
How coherent is the response (Text 2) compared to the reference (Text 1)?
0: Completely incoherent (e.g., excessive repetition, nonsensical phrases, strange symbols).
1: Partially coherent, but repetitive or has formatting issues (e.g., repeated key phrases, awkward pauses).
2: Mostly coherent with minor errors (e.g., slight redundancy but logical progression).
3: Flawless flow (e.g., logical progression, clear transitions, no repetition).

NOTE: Coherence should NOT influence the scoring of the other rubrics. Similarly,
it should NOT take into account the structure, subject, entities, or details of
the texts.

#### 2. Structure
How similar is the structure of the response (Text 2) to the reference (Text 1)?
0: No alignment (e.g., one looks like a title, the other is a paragraph).
1: Partial overlap (e.g., one lists in a bulleted list, the other lists within a paragraph).
2: Highly similar structure (e.g., same structure, both are a title).

#### 3. Subject Match
How similar is the subject of the response (Text 2) to the reference (Text 1)?
-1: No subjects to compare.
0: Completelly unrelated subjects ("corporate law" vs "particle physics")
1: Vaguely similar field (e.g: "biology" vs "physics" are both sciences)
2: Related general domain or adjacent fields (e.g., "history" vs. "archaeology" or "alternative medicine" vs. "traditional remedies").
3: Same subject (e.g., both discuss "ancient mayans" or "the properties of the human body").
4: Identical focus (e.g., both analyze "ancient mayan architecture").

#### 4. Entities
How similar are the entities in the response (Text 2) to the reference (Text 1)?
-1: No entities to compare.
0: Completelly unrelated ("Norway" vs "smartphone")
1: Vaguely similar category (e.g: the same kinds of entities, e.g: countries, humans, cities)
2: Similar category (e.g., countries similar in name or heritage, similar profession people, similar types of objects).
3: Partial identical entities (e.g., both mention "Nigella sativa" but differ in others, or both mention two different names for the same entity, e.g "Major nutrients" vs "Macro-nutrients").
4: Almost all key entities match exactly (e.g., both list "Nigella sativa, thymoquinone, antioxidants").

#### 5. Details
How similar are the details in the response (Text 2) to the reference (Text 1)?
-1: Neither text has details to compare.
0: Details differ completely (e.g., Text 1 lists benefits; Text 2 is generic).
1: Minimal depth (e.g., both mention "anti-inflammatory properties" with no specifics).
2: Moderate depth (e.g., discuss benefits + 1-2 supporting facts).
3: Highly specific details (e.g., "40% reduction in inflammation").

#### 6. Terminology
How similar is the terminology in the response (Text 2) to the reference (Text 1)?
-1: No terminology to compare.
0: No shared terms (e.g., "bioactive compounds" vs. "natural stuff").
1: Some overlap (e.g., both mention at least some terms, such as "anti-inflammatory", or similar terms for the same level concept, such as "inflamation reduction").
2: Domain-specific alignment in style (e.g., "apoptosis inhibition" vs. "cell death prevention" = jargon mismatch).

#### 7. Tone
How similar is the tone of the response (Text 2) to the reference (Text 1)?
0: Mismatched (e.g., clinical vs. casual, or positive vs. neutral vs. negative. E.g: "This sucks" vs "This is good").
1: Consistent (e.g., both neutral clinical: "studies suggest benefits" vs. "research indicates").

#### 8. Identical
Is the response (Text 2) essentially identical to the reference (Text 1)?
0: Not identical.
1: Identical. (e.g: "this is good" vs "This is good.")
---

JSON output: {
    "reasoning": {complexity, coherence, structure, subject, entities, details, terminology, tone} - each with explanation string
    "scoring":  {Same keys as above} - each with number score
}

Reasoning should be concise, and explicitly state the "name" of the level (such
as for entities: "[reasoning about the texts], Thus: similar category and partial identical entities: thus out of 4, score 3").

If the specific category does not apply to many things (e.g: complexity is 0 or
1), and there is no specifics between either, then by default give full points.
"""


json_schema = {
    "type": "object",
    "properties": {
        "reasoning": {
            "complexity": {"type": "string"},
            "coherence": {"type": "string"},
            "structure": {"type": "string"},
            "subject": {"type": "string"},
            "entities": {"type": "string"},
            "details": {"type": "string"},
            "terminology": {"type": "string"},
            "tone": {"type": "string"},
            "identical": {"type": "string"}
        },
        "scoring": {
            "complexity": {"type": "integer", "minimum": 0, "maximum": 3},
            "coherence": {"type": "integer", "minimum": 0, "maximum": 3},
            "structure": {"type": "integer", "minimum": 0, "maximum": 2},
            "subject": {"type": "integer", "minimum": 0, "maximum": 4},
            "entities": {"type": "integer", "minimum": -1, "maximum": 4},
            "details": {"type": "integer", "minimum": -1, "maximum": 3},
            "terminology": {"type": "integer", "minimum": -1, "maximum": 2},
            "tone": {"type": "integer", "minimum": 0, "maximum": 1},
            "identical": {"type": "integer", "minimum": 0, "maximum": 1}
        },
    },
    "required": ["reasoning", "scoring"]
}


@exponential_backoff
def ruberic_compare(ref_text, comp_text):
    import openai
    import os

    # Create a prompt that instructs the evaluator using the above ruberic
    prompt = (
        f"Using the following rubric, compare the two texts below:\n\n"
        f"Rubric: {rubric}\n\n"
        f"Text 1: {ref_text}\n\n"
        f"Text 2: {comp_text}\n\n"
        "The output must be a valid JSON object and nothing else."
    )
    client = OpenAI(
        #api_key=os.getenv("OPENAI_API_KEY", "YOUR_API_KEY"),
        api_key=os.getenv("OPENROUTER_API_KEY", "YOUR_API_KEY"),
        base_url="https://openrouter.ai/api/v1"
    )
    response = client.chat.completions.create(
        #model="openai/o3-mini",
        model="openai/gpt-4o-mini",
        #model="gpt-4o-mini",
        # model="meta-llama/llama-3.3-70b-instruct",
        messages=[{
            "role": "user",
            "content": "You are an expert evaluator.\n\n" + prompt
        }],
        temperature=0.3,
        max_tokens=4000,
        response_format={"type": "json_object"}, # Enable JSON mode
        function_call={"name": "validate_json", "parameters": json_schema} # Pass schema via function call
    )
    # With JSON mode, the response is guaranteed to be valid JSON
    result = response.choices[0].message.content
    return result

def get_ruberic_comparison(ref_texts: List[str], comp_texts: List[str], label=None):
    results = []
    for index, (ref_text, comp_text) in enumerate(zip(ref_texts, comp_texts)):
        result = ruberic_compare(ref_text, comp_text)
        results.append(result)
        print({"index": index, "type": label, "reference": ref_text, "comparison": comp_text, "result": result})
    return results

def get_ruberic_parallel(ref_texts: List[str], comp_texts: List[str], label=None):
    items = list(zip(np.arange(len(ref_texts)), ref_texts, comp_texts))
    print(f"Processing {len(items)} comparisons in parallel")

    def get_rubric(items):
        index, ref_text, comp_text = items
        result = ruberic_compare(ref_text, comp_text)
        print({"index": index, label: label, "reference": ref_text, "comparison": comp_text, "result": result})
        return result

    # Process comparisons in parallel
    results = process_in_parallel(
        items,
        get_rubric,
        max_workers=20
    )
    return results

def load_texts(file_path: str) -> Tuple[List[str], Dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        texts = json.load(f)
    try:
        return texts["outputs"], texts
    except:
        return texts, {}

def get_ruberic_comparison_data(ref_file, compare_files):
    """
    Compute the cosine similarity between each corresponding text in the reference texts and the texts from
    the comparison files, then plot a violin plot of these similarities.

    Parameters:
        ref_file (str): Path to the reference texts JSON file.
        compare_files (dict): Dictionary mapping a label to a JSON file path.
        output_image (str): File path to save the resulting violin plot image.
        model_name (str): The SentenceTransformer model to use (default: "all-mpnet-base-v2").
    """
    print(f"Loading reference texts from: {ref_file}")
    ref_texts, _ = load_texts(ref_file)

    all_ruberic_scores = []
    all_labels = []

    for label, file_path in (pbar := tqdm(compare_files.items(), total=len(compare_files))):
        pbar.set_description(f"Processing {label}")
        comp_texts, comp_data = load_texts(file_path)
        assert len(comp_texts) == len(ref_texts)
        ref_texts  = np.array(ref_texts)
        comp_texts = np.array(comp_texts)

        # Filter out texts with a cheat fraction greater than 0.5
        indices = np.where(np.array(comp_data["cheat_fracs"]) <= 0.5)[0] if "cheat_fracs" in comp_data else np.arange(len(comp_texts))
        ruberic_scores = get_ruberic_parallel(ref_texts[indices], comp_texts[indices], label)

        all_ruberic_scores.extend(ruberic_scores)
        all_labels.extend([label] * len(ruberic_scores))

        with open(f"ruberic_scores/{label}.json", "w") as f:
            json.dump(ruberic_scores, f)

    # Build a DataFrame for plotting.
    df_plot = pd.DataFrame({
        "Ruberic Scores": all_ruberic_scores,
        "Comparison Type": all_labels
    })

    return df_plot

if __name__ == "__main__":

    cossim_plot_path = "cossim-plot.png"

    # Manually list the files.
    #ref_file = "comparison_texts/original_texts.json"
    ref_file = "comparison_texts/train_paragraphs_98.json"
    compare_files = {
        #"mlp": "comparison_texts/mlp_decoded_texts.json",
        #"linear": "comparison_texts/linear_decoded_texts.json",
        #"continued": "comparison_texts/parascope_continuation_texts.json",
        #"baseline": "comparison_texts/baseline_0_outputs.json",
        #"cheat-1": "comparison_texts/baseline_1_outputs.json",
        #"cheat-5": "comparison_texts/baseline_5_outputs.json",
        #"cheat-10": "comparison_texts/baseline_10_outputs.json",
        #"regenerated": "comparison_texts/regenerated_outputs.json",
        #"auto-decoded": "comparison_texts/original_decoded_texts.json",
        "mlp-train": "comparison_texts/mlp_train_decoded_texts.json",
        "linear-train": "comparison_texts/linear_train_decoded_texts.json",
    }

    df_plot = get_ruberic_comparison_data(ref_file, compare_files)
    # plot_ruberic_comparison(df_plot, cossim_plot_path)

