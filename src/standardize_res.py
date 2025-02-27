# %%
import torch
import pickle
from pathlib import Path
import einops
from tqdm import tqdm

class OnlineStandardizer:
    def __init__(self, device='cuda'):
        self.n_samples_seen = 0
        self.mean = None
        self.var = None
        self.device = device

    @torch.no_grad()
    def partial_fit(self, X):
        X = torch.tensor(X).to(self.device)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        batch_size = X.shape[0]
        if batch_size == 0:
            return self

        batch_sum = X.sum(dim=0)
        batch_sq_sum = (X ** 2).sum(dim=0)

        if self.n_samples_seen == 0:
            self.mean = batch_sum / batch_size
            self.var = batch_sq_sum / batch_size - self.mean ** 2
        else:
            prev_sum = self.mean * self.n_samples_seen
            new_sum = prev_sum + batch_sum
            new_mean = new_sum / (self.n_samples_seen + batch_size)

            prev_sq_sum = (self.var + self.mean ** 2) * self.n_samples_seen
            new_sq_sum = prev_sq_sum + batch_sq_sum
            new_var = new_sq_sum / (self.n_samples_seen + batch_size) - new_mean ** 2

            self.mean = new_mean
            self.var = new_var

        self.n_samples_seen += batch_size
        return self

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)

def load_data(file_path):
    data = torch.load(file_path, map_location="cpu")
    data = torch.cat(data, dim=2)  # Concatenate list of tensors
    data = data[:, 1:, :, :]  # Remove first layer
    data = einops.rearrange(data, 'b layers samples dim -> (b samples) layers dim')
    data = einops.rearrange(data, 'samples (layersg g) dim -> samples layersg (g dim)', g=4)
    return data

def process_data(files, n_groups=14, batch_size=32):
    standardizers = [OnlineStandardizer() for _ in range(n_groups)]

    total_samples = 0
    for file_path in tqdm(files, desc="Processing files", unit="file"):
        try:
            with torch.no_grad():

                total_samples += data.shape[0]  # Update total samples after rearrangement

                for group_idx in range(n_groups):
                    group_data = data[:, group_idx, :].to("cuda")
                    for idx in range(0, len(group_data), batch_size):
                        batch = group_data[idx:idx + batch_size].float()
                        standardizers[group_idx].partial_fit(batch)
                    del group_data
                    torch.cuda.empty_cache()

                # Update the main tqdm progress bar with the total samples processed
                tqdm.write(f"Processed {total_samples} samples so far.")


                del data
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            continue

    return standardizers

def save_standardizers(standardizers, output_dir="standardizer_models"):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    for i, standardizer in enumerate(standardizers):
        standardizer.save(output_dir / f"standardizer_group_{i}.pkl")

def run(base_dir="./tensors", n_groups=14, batch_size=32):
    files = sorted(Path(base_dir).glob("res_data_*.pt"))
    standardizers = process_data(files, n_groups, batch_size)
    save_standardizers(standardizers)
    return standardizers

if __name__ == "__main__":
    run()
# %%
