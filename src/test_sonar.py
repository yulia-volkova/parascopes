# %%
from time import time as t
t0 = t()
import json
import torch
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
from sonar.inference_pipelines.text import EmbeddingToTextModelPipeline

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICE)
torch.set_grad_enabled(False)
get_vram_used = lambda: torch.cuda.memory_allocated() / (1024 ** 2)  # Returns VRAM used in MB

time_taken = {}
time_taken["import modules"] = t() - t0

# %%
# Initialize the TextToEmbeddingModelPipeline
t0 = t()
v0 = get_vram_used()
t2vec_model    = TextToEmbeddingModelPipeline(encoder="text_sonar_basic_encoder", tokenizer="text_sonar_basic_encoder", device=DEVICE)
v1 = get_vram_used()
vec2text_model = EmbeddingToTextModelPipeline(decoder="text_sonar_basic_decoder", tokenizer="text_sonar_basic_encoder", device=DEVICE)
v2 = get_vram_used()

time_taken["load models"] = t() - t0

# %%
# TEST EXAMPLE
# 1. Get the SONAR embeddings
t0 = t()
sentences = ['My name is SONAR.', 'I can embed the sentences into vectorial space.']
embeddings = t2vec_model.predict(sentences, source_lang="eng_Latn")
print("Embeddings shape:", embeddings.shape)  # Should print something like: torch.Size([2, 1024])

# Step 2: Reconstruct text from SONAR embeddings
reconstructed = vec2text_model.predict(embeddings, target_lang="eng_Latn", max_seq_len=512)
print("Reconstructed text:", reconstructed)  # Should print the original sentences

time_taken["simple example"] = t() - t0

# %%
# REAL EXAMPLE
# Try with dataset examples
original = [data["prompt"], *data["completion"].split("\n\n")]

# 1 - get embeds
t0 = t()
embeddings = t2vec_model.predict(original, source_lang="eng_Latn")
noise = torch.randn_like(embeddings)
print(embeddings.norm(dim=-1))
noise = 0.7 * embeddings.norm(dim=-1, keepdim=True) * noise / noise.norm(dim=-1, keepdim=True)
new_embeddings = embeddings + noise

# 2 - reconstruct outputs
reconstructed = vec2text_model.predict(new_embeddings, target_lang="eng_Latn", max_seq_len=512)
#reconstructed = vec2text_model.predict(embeddings, target_lang="eng_Latn", max_seq_len=512)
#print("Reconstructed text:", reconstructed)  # Should print the original
time_taken["dataset example"] = t() - t0

print(json.dumps([(original[i], reconstructed[i]) for i in range(len(reconstructed))] , indent=3))
print(embeddings.shape)
print(f"Importing modules took {time_taken['import modules']:.2f}s")
print(f"Models loaded in {time_taken['load models']:.2f}s, using {v2 - v0:.1f}MB VRAM ({v1-v0:.1f}MB enc + {v2-v1:.1f}MB dec)")
print(f"Simple example took {time_taken['simple example']:.2f}s")
print(f"Dataset example took {time_taken['dataset example']:.2f}s")
print(f"VRAM needed for context: {get_vram_used() - v2:.1f}MB")

# %%
get_pairwise_cossim = lambda emb : torch.nn.functional.cosine_similarity(emb.unsqueeze(1), emb.unsqueeze(0), dim=-1)
pairwise_cossim = get_pairwise_cossim(embeddings)
mean_cossim = pairwise_cossim[~torch.eye(pairwise_cossim.size(0), dtype=bool)].mean()
print("Mean cosine similarity:", mean_cossim.item())
print(json.dumps([str(i)+": "+x[:50] for (i, x) in enumerate(original)], indent=4))

print(torch.nn.functional.cosine_similarity(embeddings, new_embeddings))

# import seaborn as sns
# import matplotlib.pyplot as plt
# sns.heatmap(pairwise_cossim.cpu())
# plt.show()

# %%
text = """SONAR is a model from August 2023, trained as a semantic text auto-encoder, converting text into semantic embed vectors, which can later be then decoded back into text. Additionally, the model is trained such that the semantic embed vectors are to some degree "universal" for different languages, and one can embed in French and decode in English."""
# text = """I tried it, and SONAR seems to work surprisingly well. For example, the above paragraph and this paragraphs, if each are encoded into a two 1024 dimensional vectors (one for each paragraph), the model returns the following decoded outputs:"""
emb = t2vec_model.predict([text], source_lang="eng_Latn")
out = vec2text_model.predict(emb, target_lang="eng_Latn", max_seq_len=512)
print(text)
print(out)


# %%
from taker import Model
m = Model("meta-llama/Llama-3.2-3B-Instruct")

# %%

en = lambda: print("Is enabled:", m.hooks.collects["layer_0_pre_decoder"].enabled)
format_prompt = lambda _text : f"""<|start_header_id|>user<|end_header_id|>
{_text}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

en()
m.hooks.enable_collect_hooks("pre_decoder", layers=[0])
m.hooks.collects["layer_0_pre_decoder"].concat_mode = True
en()
m.generate(format_prompt("How many live in Japan?"), 50)
res = m.hooks["pre_decoder"]["collect"]
print(res[0].shape)

# %%
vec = res[0][0]
W_E = m.map["embed"].weight

token_ids = []
import torch
with torch.no_grad():
    for tok in vec:
        tok_id = ((tok.unsqueeze(0) - W_E)**2).sum(dim=-1).argmin(dim=-1)  # Shape: [26, 128256]
        token_ids.append(tok_id)
print(m.tokenizer.decode(token_ids))



# %%
