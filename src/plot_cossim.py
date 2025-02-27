# %%
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from utils_plot import load_rubric_results

def compute_cosine_similarity(ref_embeddings, comp_embeddings):
    """ Compute the cosine similarity between two arrays of embeddings. """
    dot_products = np.sum(ref_embeddings * comp_embeddings, axis=1)
    norm_ref = np.linalg.norm(ref_embeddings, axis=1)
    norm_comp = np.linalg.norm(comp_embeddings, axis=1)
    cosine_sim = dot_products / (norm_ref * norm_comp + 1e-8)
    cosine_sim = dot_products / (norm_ref * norm_comp + 1e-8)
    return cosine_sim

def load_texts(file_path):
    """
    Load a JSON file that contains a list of texts.

    Parameters:
        file_path (str): Path to the JSON file.

    Returns:
        List of texts.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        texts = json.load(f)
    try:
        return texts["outputs"], texts
    except:
        return texts, {}

def get_cosine_similarity_data(ref_file, compare_files, model_name="all-mpnet-base-v2"):
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

    print("Initializing SentenceTransformer...")
    model = SentenceTransformer(model_name)

    print("Computing embeddings for reference texts...")
    ref_embeddings = model.encode(ref_texts, convert_to_numpy=True)

    all_similarities = []
    all_labels = []

    for label, file_path in (pbar := tqdm(compare_files.items(), total=len(compare_files))):
        pbar.set_description(f"Processing {label}")
        comp_texts, comp_data = load_texts(file_path)
        if "cheat_fracs" in comp_data:
            indices = np.where(np.array(comp_data["cheat_fracs"]) <= 0.5)[0]
        else:
            indices = np.arange(len(comp_texts))

        # If lengths do not match, truncate to the minimum length.
        if len(comp_texts) != len(ref_texts):
            print(f"Warning: For '{label}', number of texts ({len(comp_texts)}) "
                  f"does not match reference ({len(ref_texts)}); truncating to minimum length.")
            min_len = min(len(comp_texts), len(ref_texts))
            comp_texts = comp_texts[:min_len]
            ref_emb = ref_embeddings[:min_len]
        else:
            ref_emb = ref_embeddings

        comp_embeddings = model.encode(comp_texts, convert_to_numpy=True)

        cosine_similarities = compute_cosine_similarity(ref_emb[indices], comp_embeddings[indices])

        all_similarities.extend(cosine_similarities.tolist())
        all_labels.extend([label] * len(cosine_similarities))

    # Build a DataFrame for plotting.
    df_plot = pd.DataFrame({
        "Cosine Similarity": all_similarities,
        "Comparison Type": all_labels
    })

    return df_plot

def plot_cosine_similarity_violin(df_plot, output_image):
    plt.figure(figsize=(10, 6))
    # Create custom color palette based on comparison types
    palette = {}
    for i, label in enumerate(df_plot["Comparison Type"].unique()):
        # Use RGB colors instead of HSLA since seaborn doesn't support HSLA
        hue = i * 150 % 360
        # Convert HSL to RGB (approximation)
        r = 0.6 + 0.3 * np.cos(np.radians(hue))
        g = 0.6 + 0.3 * np.cos(np.radians(hue - 120))
        b = 0.6 + 0.3 * np.cos(np.radians(hue + 120))
        palette[label] = (r, g, b)

    sns.violinplot(data=df_plot, x="Comparison Type", y="Cosine Similarity", hue="Comparison Type", palette=palette, scale="width", legend=False)
    plt.title("Cosine Similarity Violin Plot")
    plt.tight_layout()
    plt.ylim(top=1.0)
    plt.savefig(output_image, dpi=300)
    print(f"\nViolin plot saved to: {output_image}")
    plt.show()

    # print means + stdev
    print(df_plot.groupby("Comparison Type")["Cosine Similarity"].agg(["mean", "std"]))


if __name__ == "__main__":

    cossim_plot_path = "cossim-plot.png"

    # Manually list the files.
    ref_file = "comparison_texts/original_texts.json"
    compare_files = {
        "mlp": "comparison_texts/mlp_decoded_texts.json",
        "linear": "comparison_texts/linear_decoded_texts.json",
        "continued": "comparison_texts/parascope_continuation_texts.json",
        "baseline": "comparison_texts/baseline_0_outputs.json",
        "cheat-1": "comparison_texts/baseline_1_outputs.json",
        "cheat-5": "comparison_texts/baseline_5_outputs.json",
        "cheat-10": "comparison_texts/baseline_10_outputs.json",
        "regenerated": "comparison_texts/regenerated_outputs.json",
        "auto-decoded": "comparison_texts/original_decoded_texts.json",
    }

    df_plot = get_cosine_similarity_data(ref_file, compare_files)
    plot_cosine_similarity_violin(df_plot, cossim_plot_path)

# %%


import json

print("\n=== [Original Text] vs [Other Text] by Cosine Sim Buckets (0.1 increments) ===\n")

ref_file = "comparison_texts/original_texts.json"
compare_files = {
    "mlp": "comparison_texts/mlp_decoded_texts.json",
    "linear": "comparison_texts/linear_decoded_texts.json",
    "continued": "comparison_texts/parascope_continuation_texts.json",
    "baseline": "comparison_texts/baseline_0_outputs.json",
    "cheat-1": "comparison_texts/baseline_1_outputs.json",
    "cheat-5": "comparison_texts/baseline_5_outputs.json",
    "cheat-10": "comparison_texts/baseline_10_outputs.json",
    "regenerated": "comparison_texts/regenerated_outputs.json",
    "auto-decoded": "comparison_texts/original_decoded_texts.json",
}

# Load original texts.
with open(ref_file, "r") as f:
    orig_data = json.load(f)
original_texts = orig_data["outputs"] if isinstance(orig_data, dict) and "outputs" in orig_data else orig_data

# For each comparison method, show one example per cosine similarity bucket.
for method, file_path in compare_files.items():
    try:
        with open(file_path, "r") as f:
            cmp_data = json.load(f)
    except Exception as e:
        print(f"Could not load {file_path} for method '{method}': {e}")
        continue

    compared_texts = cmp_data["outputs"] if isinstance(cmp_data, dict) and "outputs" in cmp_data else cmp_data

    # Filter df_plot for the current method and reset index for alignment.
    df_method = df_plot[df_plot["Comparison Type"] == method].reset_index(drop=True)
    if df_method.empty:
        continue

    # Add a bucket column by rounding the cosine similarity.
    df_method["bucket"] = df_method["Cosine Similarity"].round(1)

    print("------------------------------------------------------------")
    print(f"Method: {method}")
    # Loop over cosine similarity buckets 0.0, 0.1, ..., 1.0
    for i in range(11):
        bucket = round(i / 10, 1)
        sub_df = df_method[df_method["bucket"] == bucket]
        if sub_df.empty:
            continue
        # Pick the first example in this bucket.
        row = sub_df.iloc[0]
        local_idx = row.name  # local index in df_method assumed to match file order
        sim_val = row["Cosine Similarity"]
        orig_sample = original_texts[local_idx] if local_idx < len(original_texts) else "[Index OOR]"
        cmp_sample = compared_texts[local_idx] if local_idx < len(compared_texts) else "[Index OOR]"

        import textwrap
        from termcolor import colored

        print(f"### EXAMPLE {local_idx} ### (SIM = {sim_val:.4f})")
        print(textwrap.fill(colored(f"ORIGINAL: {orig_sample[:200]}", 'blue'),
                            width=120,
                            initial_indent='',
                            subsequent_indent=' ' * 10))
        print(textwrap.fill(colored(f"{method.capitalize()}: {cmp_sample[:200]}", 'yellow'),
                            width=120,
                            initial_indent='',
                            subsequent_indent=' ' * 10))
        print()
    print("------------------------------------------------------------\n")


# %%
