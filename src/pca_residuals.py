# %%
import os
import pickle
from tqdm import tqdm
import torch
import cupy as cp
import numpy as np
from cuml import IncrementalPCA
from cuml.linear_model import LinearRegression
from pathlib import Path
import einops  # Make sure to import einops


class OnlineStandardizer:
    """Online standardization for per-feature mean/std with improved numerical stability."""
    def __init__(self, device='cuda'):
        self.n_samples_seen = 0
        self.mean = None
        self.var = None
        self.device = device

    @torch.no_grad()
    def partial_fit(self, X):
        """Updates running statistics with improved numerical stability."""
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

    @torch.no_grad()
    def transform(self, X):
        """Standardizes data using computed statistics with epsilon for numerical stability."""
        X = cp.asarray(X)
        epsilon = 1e-8
        return (X - self.mean) / cp.sqrt(self.var + epsilon)

    def save(self, file_path):
        """Saves the state of the OnlineStandardizer to a file."""
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file_path):
        """Loads the OnlineStandardizer state from a file."""
        with open(file_path, 'rb') as f:
            return pickle.load(f)


class PCAPipeline:
    """Improved PCA pipeline with automatic file handling and progress tracking."""
    def __init__(self,
                 base_dir="./tensors",
                 file_names="res_data_*.pt",
                 n_groups=14,
                 group_components=512,
                 final_components=1024,
                 batch_size=1024
            ):
        self.base_dir = Path(__file__).parent / base_dir
        self.file_names = file_names
        self.n_groups = n_groups
        self.group_components = group_components
        self.final_components = final_components
        self.batch_size = batch_size

        # Create output directories
        self.pca_dir = Path("pca_groups")
        self.transform_dir = Path("group_transforms")
        self.final_dir = Path("final_pca")

        for dir_path in [self.pca_dir, self.transform_dir, self.final_dir]:
            dir_path.mkdir(exist_ok=True, parents=True)

    def get_data_files(self):
        """Gets sorted list of input files matching pattern res_data_*.pt"""
        return sorted(self.base_dir.glob(self.file_names))

    @torch.no_grad
    def load_res_data(self, file_path, group_size=4):
        """Trains PCA on one group of 4 layers with improved error handling and progress tracking."""

        data = torch.load(file_path, map_location='cpu')
        # Concatenate the list of tensors along the sample dimension
        data = torch.cat(data, dim=2)  # Concatenate along the sample dimension
        dim_shape = data.shape[-1]

        # remove layer 0
        data = data[:, 1:, :, :]  # Use all layers except the first one
        data = einops.rearrange(data, 'b layers samples dim -> (b samples) layers dim')
        data = einops.rearrange(data, 'samples (layersg g) dim -> samples layersg (g dim)', g=group_size)

        # Check for the expected shape
        if data.shape[-1] != group_size*dim_shape:
            print(data.shape)
            raise ValueError(f"Error: data has wrong shape :( {data.shape}")

        return data
    def train_group_pca(self, group_idx):
        """Trains PCA on one group of 4 layers with improved error handling and progress tracking."""
        group_dir = self.pca_dir / f"group_{group_idx}"
        group_dir.mkdir(exist_ok=True)

        # Load standardizer if already saved, otherwise create a new one
        standardizer_path = group_dir / f"standardizer_group_{group_idx}.pkl"
        if standardizer_path.exists():
            standardizer = OnlineStandardizer.load(str(standardizer_path))
        else:
            standardizer = OnlineStandardizer()

        ipca = IncrementalPCA(n_components=self.group_components, batch_size=self.batch_size)

        files = self.get_data_files()
        print(f"\nTraining PCA for group {group_idx}")

        for file_path in tqdm(files, desc=f"Group {group_idx} Standardization"):
            try:
                with torch.no_grad():
                    data = self.load_res_data(file_path)
                    group_data = data[:, group_idx, :].to("cuda")  # Shape: [10000, 12000]
                    del data

                # Standardize the data
                for idx in range(0, len(group_data), self.batch_size):
                    batch = group_data[idx:idx + self.batch_size].float()
                    standardizer.partial_fit(batch)  # Use torch tensor instead of cupy array
            finally:
                torch.cuda.empty_cache()

        for file_path in tqdm(files, desc=f"Group {group_idx} PCA"):
            try:
                with torch.no_grad():
                    data = self.load_res_data(file_path)
                    group_data = data[:, group_idx, :].to("cuda")  # Shape: [10000, 12000]
                    del data
                for idx in range(0, len(group_data), self.batch_size):
                    batch = group_data[idx:idx + self.batch_size].float()
                    group_data_norm = standardizer.transform(batch)  # Use torch tensor instead of cupy array
                    ipca.partial_fit(group_data_norm)

            finally:
                torch.cuda.empty_cache()

        # Save models with error handling
        try:
            cp.save(str(group_dir / "pca_components.npy"), ipca.components_)
            cp.save(str(group_dir / "mean.npy"), standardizer.mean)
            cp.save(str(group_dir / "var.npy"), standardizer.var)
            standardizer.save(str(standardizer_path))  # Save the standardizer
        except Exception as e:
            print(f"Error saving models for group {group_idx}: {str(e)}")

    def train_groupt

    def transform_group(self, group_idx):
        """Transforms data using a group's PCA with improved error handling."""
        group_dir = self.pca_dir / f"group_{group_idx}"

        try:
            components = cp.load(str(group_dir / "pca_components.npy"))
            mean = cp.load(str(group_dir / "mean.npy"))
            std = cp.sqrt(cp.load(str(group_dir / "var.npy")) + 1e-8)
        except Exception as e:
            print(f"Error loading PCA models for group {group_idx}: {str(e)}")
            return

        print(f"\nTransforming data for group {group_idx}")
        for file_path in tqdm(self.get_data_files(), desc=f"Group {group_idx} Transform"):
            try:
                data = torch.load(file_path)
                group_layers = data[:, group_idx*4:(group_idx+1)*4, :]
                group_flat = group_layers.reshape(group_layers.shape[0], -1).to("cuda")

                transformed_chunks = []
                for idx in range(0, len(group_flat), self.batch_size):
                    batch = cp.asarray(group_flat[idx:idx + self.batch_size])
                    batch_norm = (batch - mean) / std
                    transformed = cp.dot(batch_norm, components.T)
                    transformed_chunks.append(cp.asnumpy(transformed))

                transformed_data = np.concatenate(transformed_chunks, axis=0)
                output_path = self.transform_dir / f"group_{group_idx}_{file_path.name}"
                torch.save(transformed_data, str(output_path))

            except Exception as e:
                print(f"Error transforming {file_path} for group {group_idx}: {str(e)}")
                continue

            finally:
                torch.cuda.empty_cache()

    def train_final_pca(self):
        """Trains final PCA on concatenated group features with improved memory management."""
        print("\nTraining final PCA")
        ipca = IncrementalPCA(n_components=self.final_components, batch_size=self.batch_size)

        base_files = [f for f in os.listdir(self.transform_dir) if f.startswith("group_0_")]
        for file_name in tqdm(base_files, desc="Final PCA"):
            try:
                all_groups = []
                for group_idx in range(self.n_groups):
                    group_file = self.transform_dir / f"group_{group_idx}_{file_name.split('_', 2)[-1]}"
                    if not group_file.exists():
                        print(f"Warning: Missing group file {group_file}")
                        continue
                    data = torch.load(str(group_file))
                    all_groups.append(cp.asarray(data))

                concatenated = cp.hstack(all_groups)
                for idx in range(0, len(concatenated), self.batch_size):
                    batch = concatenated[idx:idx + self.batch_size]
                    ipca.partial_fit(batch)

            except Exception as e:
                print(f"Error in final PCA for {file_name}: {str(e)}")
                continue

            finally:
                del all_groups
                torch.cuda.empty_cache()

        try:
            cp.save(str(self.final_dir / "components.npy"), ipca.components_)
        except Exception as e:
            print(f"Error saving final PCA components: {str(e)}")

    def train_linear_model(self, labels_path):
        """Trains linear regression with improved error handling and memory management."""
        try:
            labels = torch.load(labels_path).to("cuda")
        except Exception as e:
            print(f"Error loading labels from {labels_path}: {str(e)}")
            return

        print("\nTraining linear model")
        model = LinearRegression(fit_intercept=True, output_type="cupy")

        # Process data in chunks
        X_chunks = []
        y_chunks = []
        chunk_size = 1000  # Adjust based on available memory

        files = sorted(self.final_dir.glob("*.pt"))
        for i in tqdm(range(0, len(files), chunk_size), desc="Processing chunks"):
            chunk_files = files[i:i + chunk_size]
            try:
                X_chunk = []
                for file_path in chunk_files:
                    data = torch.load(file_path).to("cuda")
                    X_chunk.append(cp.asarray(data))

                X_chunks.append(cp.vstack(X_chunk))
                y_chunks.append(labels[i:i + chunk_size])

            except Exception as e:
                print(f"Error processing chunk {i}: {str(e)}")
                continue

            finally:
                torch.cuda.empty_cache()

        try:
            X = cp.vstack(X_chunks)
            y = cp.vstack(y_chunks)
            model.fit(X, y)
            cp.save(str(self.final_dir / "linear_model.npy"), model.coef_)

        except Exception as e:
            print(f"Error training linear model: {str(e)}")

        finally:
            del X_chunks, y_chunks
            torch.cuda.empty_cache()

def main():
    # Initialize pipeline with customizable parameters
    pipeline = PCAPipeline(
        base_dir="./tensors",
        n_groups=14,
        group_components=512,
        final_components=1024,
        batch_size=32
    )

    # Train group PCAs
    for group_idx in range(pipeline.n_groups):
        pipeline.train_group_pca(group_idx)

    # Transform using group PCAs
    for group_idx in range(pipeline.n_groups):
        pipeline.transform_group(group_idx)

    # Train final PCA
    pipeline.train_final_pca()

    # Train linear model
    pipeline.train_linear_model("./tensors")

if __name__ == "__main__":
    main()
# %%
