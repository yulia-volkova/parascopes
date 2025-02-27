import os
import json
import torch
from tqdm import tqdm
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
import gc

torch.set_grad_enabled(False)

# Load dataset
with open('/workspace/SPAR/gen-dataset/new_split_dataset.jsonl', 'r') as file:
    dataset = [json.loads(line) for line in file]

# Initialize SONAR model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t2vec_model = TextToEmbeddingModelPipeline(encoder="text_sonar_basic_encoder",
                                         tokenizer="text_sonar_basic_encoder",
                                         device=DEVICE)

# Setup dirs
FOLDER, F_NAME = "./sonar_embeds", "embeds"
os.makedirs(FOLDER, exist_ok=True)

# Calculate expected files
total_files = (len(dataset) + 999) // 1000

# Get existing files - Fixed pattern matching
existing_files = {int(name.split('_')[1].split('.')[0]) for name in os.listdir(FOLDER)
                 if name.startswith(F_NAME) and name.endswith(".pt")}

print(set(range(total_files)) - existing_files)

# Process missing files
for batch_index in tqdm(range(total_files)):
    if batch_index in existing_files:
        continue

    start_idx = batch_index * 1000
    end_idx = min(start_idx + 1000, len(dataset))
    batch = []
    gc.collect()
    torch.cuda.empty_cache()

    for data in dataset[start_idx:end_idx]:
        texts = data["split"][1:]
        try:
            embeddings = t2vec_model.predict(texts, source_lang="eng_Latn")
        except:
            print(data["split"])
            raise Exception
        batch.append(embeddings)

    torch.save(batch, f"{FOLDER}/{F_NAME}_{batch_index:03d}.pt")