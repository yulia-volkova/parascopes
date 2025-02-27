# notebooks/sonar/minimal_pca_example.py
import torch
import cupy as cp
from cuml import IncrementalPCA
import time

def main():
    # Initialize IncrementalPCA

    # Fit PCA in batches with timing
    n_samples = 10000
    dim = 100
    n_components = 512
    while dim <= 25600:
        print(f"Fitting PCA with n_samples: {n_samples}, dimension: {dim}, n_components: {n_components}")
        ipca = IncrementalPCA(n_components=10, batch_size=1000)
        data_cp = cp.random.randn(n_samples, dim)  # Update data for new dimension
        start_time = time.time()  # Start timing
        for idx in range(0, len(data_cp), 32):  # Batch size of 32
            batch = data_cp[idx:idx + 32]
            ipca.partial_fit(batch)
        end_time = time.time()  # End timing
        print(f"Time taken for dimension {dim}: {end_time - start_time:.4f} seconds")
        dim *= 2  # Increase dimension by a factor of 2

    # Get PCA components
    components = ipca.components_
    print("PCA Components Shape:", components.shape)

if __name__ == "__main__":
    main()
