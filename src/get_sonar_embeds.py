# %%
import os
import json
import torch
from tqdm import tqdm
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
#from sonar.inference_pipelines.text import EmbeddingToTextModelPipeline

torch.set_grad_enabled(False)

# %%
DATASET_FILE = '/workspace/SPAR/gen-dataset/new_split_dataset.jsonl'
with open(DATASET_FILE, 'r') as file:
    dataset = [json.loads(line) for line in file]

print("Dataset loaded:", len(dataset))
print(dataset[0])

# Initialize the SONAR models
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t2vec_model = TextToEmbeddingModelPipeline(encoder="text_sonar_basic_encoder", tokenizer="text_sonar_basic_encoder", device=DEVICE)

# %%
FOLDER, F_NAME = "./sonar_embeds", "embeds"
os.makedirs(FOLDER, exist_ok=True)

# Determine how many files exist in ./tensors
existing_files = len([name for name in os.listdir(FOLDER) if name.startswith(F_NAME) and name.endswith(".pt")])
skip_count = existing_files * 1000

batch = []
for i, data in enumerate(tqdm(dataset)):
    if i < skip_count:
        continue
    if i > 0 and i % 1000 == 0:
        batch_index = i // 1000 - 1
        torch.save(batch, f"{FOLDER}/{F_NAME}_{batch_index:03d}.pt")
        batch = []

    # Use SONAR model to get embeddings
    texts = data["split"][1:]
    try:
        embeddings = t2vec_model.predict(texts, source_lang="eng_Latn")
    except:
        print(data["split"])
        raise Exception
    batch.append(embeddings)
batch_index += 1
torch.save(batch, f"{FOLDER}/{F_NAME}_{batch_index:03d}.pt")


# %%
