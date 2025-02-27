# %%
import pickle
import os
import gc
from time import time
import json
from tqdm import tqdm
import torch
import einops
import torch.nn as nn
from welford_torch import Welford
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

PARAGRAPHS_FILE = "/workspace/SPAR/gen-dataset/new_split_dataset.jsonl"

def load_res_data(index, groups_to_load=2):
    file_path = f"./tensors/res_data_{index:03d}.pt"
    data = torch.load(file_path, map_location="cpu", weights_only=False)
    data = torch.cat(data, dim=2)  # Concatenate list of tensors
    data = data.squeeze(0)
    data = data[1:, :, :]  # Remove first layer
    data = einops.rearrange(data, 'layers samples dim -> samples layers dim')
    data = einops.rearrange(data, 'samples (layersg g) dim -> samples layersg (g dim)', g=4)
    data = einops.rearrange(data[:, -groups_to_load:], 'samples layersg gdim -> samples (layersg gdim)')
    return data.float()

def load_embeds(index):
    file_path = f"./sonar_embeds/embeds_{index:03d}.pt"
    data = torch.load(file_path, map_location="cpu", weights_only=False)
    data = torch.cat(data, dim=0)
    return data.float()

def load_paragraphs(file_path):
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    splits = [x["split"][1:] for x in data]
    splits = [item for sublist in splits for item in sublist]
    return splits

embeds = load_embeds(0).cuda()
res_data = load_res_data(0).cuda()
paragraphs = load_paragraphs(PARAGRAPHS_FILE)

print("embeds shape", embeds.shape)
print("res data shape", res_data.shape)

D_RES = res_data.shape[-1]
D_SONAR = embeds.shape[-1]

# %% TEST OUTPUTS

from sonar.inference_pipelines.text import EmbeddingToTextModelPipeline
vec2text_model = EmbeddingToTextModelPipeline(decoder="text_sonar_basic_decoder", tokenizer="text_sonar_basic_encoder", device=DEVICE)

# Example usage with the vec2text_model
with torch.no_grad():
    for index in [1, 100, 1000]:
        orig_emb   = embeds[index].unsqueeze(dim=0).to(DEVICE)
        paragraph = paragraphs[index]
        decoded_text = vec2text_model.predict(orig_emb, target_lang="eng_Latn")
        print(f"### EXAMPLE {index} ###")
        print({"ORIG":     paragraph})
        print({"DECODED":  decoded_text[0]})

# %%
