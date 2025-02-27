# %%
from utils_plot import load_rubric_results
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch

torch.set_grad_enabled(False)

def get_cosine_similarity_data(data_list, model_name="all-mpnet-base-v2"):
    """
    Compute the cosine similarity between each corresponding text in the reference texts and the texts from
    the comparison files, then plot a violin plot of these similarities.

    Parameters:
        ref_file (str): Path to the reference texts JSON file.
        compare_files (dict): Dictionary mapping a label to a JSON file path.
        output_image (str): File path to save the resulting violin plot image.
        model_name (str): The SentenceTransformer model to use (default: "all-mpnet-base-v2").
    """
    print("Initializing SentenceTransformer...")
    model = SentenceTransformer(model_name)

    ref_texts = [datum['reference'] for datum in data_list.values()]
    comp_texts = [datum['comparison'] for datum in data_list.values()]

    print("Computing embeddings for reference texts...")
    ref_embeddings = model.encode(ref_texts)
    print("Computing embeddings for comparison texts...")
    comp_embeddings = model.encode(comp_texts)

    cossims = torch.nn.functional.cosine_similarity(torch.tensor(ref_embeddings), torch.tensor(comp_embeddings), dim=1)

    for cossim, datum in zip(cossims, data_list.values()):
        datum['cosine_similarity'] = cossim.item()

    return data_list


data_dicts = load_rubric_results(indices_intersection=False)

for data_type, data_dict in tqdm(data_dicts.items()):
    get_cosine_similarity_data(data_dict, model_name="all-mpnet-base-v2")


# %%

import json
new_data_dict = {}

for data_type, data_dict in data_dicts.items():
    new_data_dict[data_type] = data_dict
    for index, datum in new_data_dict[data_type].items():
        #print(datum['cosine_similarity'])
        if isinstance(datum['result'], str):
            datum['result'] = json.loads(datum['result'])
        del datum['result']['cosine_similarity']
        datum['result']['scoring']['cosine_similarity'] = datum['cosine_similarity']


# %%
for data_type, data_dict in new_data_dict.items():
    print(f"# {data_type}")
    for datum in data_dict.values():
        print(datum['cosine_similarity'])
        print({"ref": datum['reference']})
        print({"comp": datum['comparison']})
        print(datum['result'])
        break

file_path = "processed_rubrics/all_data_with_cossim.json"

with open(file_path, "w") as f:
    json.dump(new_data_dict, f)


# %%
