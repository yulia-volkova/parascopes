import os
import json
import torch
from tqdm import tqdm
import gc

torch.set_grad_enabled(False)

# Load dataset from a single JSON file
with open('transferred_activation_output.json', 'r') as file:
    dataset = json.load(file)["outputs"]

# Initialize SONAR model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
t2vec_model = TextToEmbeddingModelPipeline(
    encoder="text_sonar_basic_encoder",
    tokenizer="text_sonar_basic_encoder",
    device=DEVICE
)

# Setup output directory for saving results
OUTPUT_FOLDER = "./comparison_embeds"
OUTPUT_FILENAME = "parascope_continuation_embeds.pt"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Clean up to free memory
gc.collect()
torch.cuda.empty_cache()

embedding_results = []
BATCH_SIZE = 32  # Adjust the batch size as needed

for i in tqdm(range(0, len(dataset), BATCH_SIZE), desc="Processing dataset batches"):
    batch = dataset[i : i + BATCH_SIZE]
    try:
        # Process the current batch of texts
        batch_embeddings = t2vec_model.predict(batch, source_lang="eng_Latn")
    except Exception as e:
        print(f"Error processing batch starting at index {i}.")
        for j, data in enumerate(batch, start=i):
            print(f"Error with data item at index {j}: {data.get('split', 'N/A')}")
        raise e
    embedding_results.extend(batch_embeddings)

output_path = os.path.join(OUTPUT_FOLDER, OUTPUT_FILENAME)
torch.save(embedding_results, output_path)
print(f"Saved embeddings to {output_path}")
