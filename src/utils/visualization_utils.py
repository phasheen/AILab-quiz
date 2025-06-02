import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid

def visualize_reconstructions(original_images, reconstructed_images, n=5, save_path=None):
    """Visualize original and reconstructed images (e.g., for VAE/Autoencoders)."""
    if original_images is None or reconstructed_images is None:
        print("Warning: Original or reconstructed images are None, cannot visualize.")
        return

    original_images = original_images[:n].cpu().clamp(0, 1)
    reconstructed_images = reconstructed_images[:n].cpu().clamp(0, 1)
    
    if original_images.ndim == 3: # Add channel dim if missing (e.g. [N, H, W] -> [N, 1, H, W])
        original_images = original_images.unsqueeze(1)
    if reconstructed_images.ndim == 3:
        reconstructed_images = reconstructed_images.unsqueeze(1)

    fig, axes = plt.subplots(2, n, figsize=(n * 2, 4))
    for i in range(n):
        if original_images.shape[0] > i:
            axes[0, i].imshow(original_images[i].permute(1, 2, 0).squeeze().numpy(), cmap='gray')
            axes[0, i].set_title("Original")
            axes[0, i].axis("off")
        else:
            axes[0, i].axis("off") # Handle cases with fewer than n images

        if reconstructed_images.shape[0] > i:
            axes[1, i].imshow(reconstructed_images[i].permute(1, 2, 0).squeeze().numpy(), cmap='gray')
            axes[1, i].set_title("Reconstructed")
            axes[1, i].axis("off")
        else:
            axes[1,i].axis("off")
            
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Reconstruction plot saved to {save_path}")
    plt.show()

def visualize_generated_images(images, n=16, title="Generated Images", save_path=None, nrow=4):
    """Visualize a grid of generated images (e.g., for GANs)."""
    if images is None:
        print("Warning: Images are None, cannot visualize.")
        return
        
    images = images[:n].cpu().clamp(0, 1)
    grid = make_grid(images, nrow=nrow, padding=2, normalize=False)
    
    plt.figure(figsize=(nrow*2, (n//nrow)*2) if nrow > 0 else (n*2, 2) )
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Generated images plot saved to {save_path}")
    plt.show()

def plot_latent_space_distribution(latent_vectors, labels=None, dim_indices=[0, 1], title="Latent Space Distribution", save_path=None):
    """Plot a 2D scatter of latent vectors, optionally colored by labels."""
    if latent_vectors is None:
        print("Warning: Latent vectors are None, cannot visualize.")
        return

    latent_vectors = latent_vectors.cpu().numpy()
    dim1, dim2 = dim_indices
    
    plt.figure(figsize=(8, 6))
    if labels is not None:
        labels = labels.cpu().numpy()
        scatter = plt.scatter(latent_vectors[:, dim1], latent_vectors[:, dim2], c=labels, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label='Class Label')
    else:
        plt.scatter(latent_vectors[:, dim1], latent_vectors[:, dim2], alpha=0.7)
    
    plt.xlabel(f"Latent Dimension {dim1}")
    plt.ylabel(f"Latent Dimension {dim2}")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Latent space plot saved to {save_path}")
    plt.show()

def plot_class_distribution_histograms(data, labels, num_classes, feature_names=None, title_prefix="Feature Distribution", save_path_prefix=None):
    """Plot histograms of feature values per class."""
    if data is None or labels is None:
        print("Warning: Data or labels are None, cannot visualize histograms.")
        return

    data = data.cpu().numpy()
    labels = labels.cpu().numpy()
    num_features = data.shape[1]

    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(num_features)]

    for i in range(num_features):
        plt.figure(figsize=(10, 6))
        for cls in range(num_classes):
            plt.hist(data[labels == cls, i], bins=30, alpha=0.5, label=f'Class {cls}', density=True)
        plt.title(f"{title_prefix}: {feature_names[i]}")
        plt.xlabel(feature_names[i])
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        if save_path_prefix:
            plt.savefig(f"{save_path_prefix}_feature_{i}_hist.png")
            print(f"Histogram for feature {i} saved.")
        plt.show() 