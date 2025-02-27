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

embeds = load_embeds(0)
res_data = load_res_data(0)
paragraphs = load_paragraphs(PARAGRAPHS_FILE)

print("embeds shape", embeds.shape)
print("res data shape", res_data.shape)

D_RES = res_data.shape[-1]
D_SONAR = embeds.shape[-1]

# %%

# Try to load Welford statistics from pickle file, compute if not exists
try:
    with open('./welford_data/welford_stats_10.pkl', 'rb') as f:
        print("Loading existing welford data")
        welford_stats = pickle.load(f)
        welford_emb = welford_stats['welford_emb']
        welford_res = welford_stats['welford_res']

except FileNotFoundError:
    welford_emb = Welford()
    welford_res = Welford()

    for i in tqdm(range(10)):
        res_data = load_res_data(i).cuda()
        embeds = load_embeds(i).cuda()

        welford_res.add_all(res_data)
        welford_emb.add_all(embeds)
        del res_data, embeds
        gc.collect()
        torch.cuda.empty_cache()

    # Save Welford statistics for first 10 files using pickle

    os.makedirs('./welford_data', exist_ok=True)
    with open('./welford_data/welford_stats_10.pkl', 'wb') as f:
        pickle.dump({
            'welford_emb': welford_emb,
            'welford_res': welford_res
        }, f)

# %%

def normalize_res(res_data):
    """Normalize residual data using precomputed mean and variance"""
    return (res_data - welford_res.mean) / torch.sqrt(welford_res.var_s + 1e-8)

def normalize_emb(emb_data):
    """Normalize embedding data using precomputed mean and variance"""
    return (emb_data - welford_emb.mean) / torch.sqrt(welford_emb.var_s + 1e-8)

def restore_emb(normed_emb_data):
    """Restore normalized embedding data to original scale using precomputed mean and variance"""
    return normed_emb_data * torch.sqrt(welford_emb.var_s + 1e-8) + welford_emb.mean

# %%
# Train linear model
# Create model

class Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(D_RES, D_SONAR)

    def forward(self, x):
        return self.linear(x)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        d_hidden = 4092  # Define hidden layer dimension
        self.sequential = nn.Sequential(
            nn.Linear(D_RES, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, D_SONAR)
        )

    def forward(self, x):
        return self.sequential(x)

ResidualToEmbed = Linear

torch.set_grad_enabled(True)
model = ResidualToEmbed().to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training loop
batch_size = 256
num_epochs = 10
num_files = 1

for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0
    train_batches = 0
    # Process files in chunks to manage memory
    for file_idx in (pbar := tqdm(range(num_files), desc=f"Epoch {epoch+1}")):
        # Load and normalize data for current file
        res_data = load_res_data(file_idx).cuda()
        embeds = load_embeds(file_idx).cuda()

        normalized_res_data = normalize_res(res_data)
        normalized_embeds = normalize_emb(embeds)
        dataset = TensorDataset(normalized_res_data, normalized_embeds)

        # Split into train/val for this file
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Train on current file
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_batches += 1

            # Update tqdm bar with current loss
            pbar.set_postfix({"Loss": f"{train_loss/train_batches:.4f}"})

        # Clean up memory
        del res_data, embeds, normalized_res_data, normalized_embeds, dataset
        gc.collect()
        torch.cuda.empty_cache()

    # Validation (using last file's validation set)
    model.eval()
    val_loss = 0
    with torch.no_grad():
        res_data = load_res_data(file_idx+1).float().cuda()
        embeds = load_embeds(file_idx+1).float().cuda()

        normalized_res_data = normalize_res(res_data)
        normalized_embeds = normalize_emb(embeds)
        dataset = TensorDataset(normalized_res_data, normalized_embeds)
        test_loader = DataLoader(dataset, batch_size=batch_size)
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            outputs = model(x)
            val_loss += criterion(outputs, y).item()

    print(f"Epoch {epoch+1}: Train Loss: {train_loss/train_batches:.4f}, Val Loss: {val_loss/len(test_loader):.4f}")

# %%
# Test outputs
print("Testing some outputs")
from sonar.inference_pipelines.text import EmbeddingToTextModelPipeline
vec2text_model = EmbeddingToTextModelPipeline(decoder="text_sonar_basic_decoder", tokenizer="text_sonar_basic_encoder", device=DEVICE)

# Example usage with the vec2text_model
with torch.no_grad():
    for index in [1, 100, 1000]:
        orig_emb   = embeds[index].unsqueeze(dim=0).to(DEVICE)
        test_input = res_data[index].unsqueeze(dim=0).to(DEVICE)
        predicted_embed = restore_emb(model(test_input))
        decoded_text = vec2text_model.predict(
            torch.cat([orig_emb, predicted_embed], dim=0),
            target_lang="eng_Latn"
        )
        print(f"### EXAMPLE {index} ###")
        print({"ORIG": decoded_text[0]})
        print({"NEW":  decoded_text[1]})

# %%
