import torch
import torch.nn as nn
import torch.nn.functional as F

class VAEEncoder(nn.Module):
    def __init__(self, input_dim=28*28, hidden_dims=[512, 256], latent_dim=64):
        super().__init__()
        self.flatten = nn.Flatten()

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        
        self.encoder = nn.Sequential(*layers)
        
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

    def forward(self, x):
        x = self.flatten(x)
        x_encoded = self.encoder(x)
        mu = self.fc_mu(x_encoded)
        logvar = self.fc_logvar(x_encoded)
        return mu, logvar

class VAEDecoder(nn.Module):
    def __init__(self, latent_dim=64, hidden_dims=[256, 512], output_dim=28*28, image_dims=(1, 28, 28)):
        super().__init__()
        self.image_dims = image_dims
        layers = []
        prev_dim = latent_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Sigmoid()) # To ensure output is in [0, 1] for images
        
        self.decoder = nn.Sequential(*layers)
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=self.image_dims)


    def forward(self, z):
        x_reconstructed = self.decoder(z)
        x_reconstructed = self.unflatten(x_reconstructed)
        return x_reconstructed

class VAE(nn.Module):
    def __init__(self, input_dim=28*28, encoder_hidden_dims=[512, 256], latent_dim=64, decoder_hidden_dims=[256, 512], image_dims=(1,28,28)):
        super().__init__()
        self.encoder = VAEEncoder(input_dim=input_dim, hidden_dims=encoder_hidden_dims, latent_dim=latent_dim)
        self.decoder = VAEDecoder(latent_dim=latent_dim, hidden_dims=decoder_hidden_dims, output_dim=input_dim, image_dims=image_dims)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            # During evaluation, typically use the mean for deterministic output
            return mu
        
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decoder(z)
        return reconstruction, mu, logvar

def vae_loss_function(recon_x, x, mu, logvar, beta=1.0):
    # Ensure x is flattened to match recon_x if necessary, or that recon_x matches x's shape
    # Assuming x is [B, C, H, W] and recon_x is [B, C, H, W]
    # If x is [B, 784] and recon_x from decoder is [B, 784] before unflatten, adjust MSE target.
    # For image data, usually recon_x and x are compared in their image form.
    recon_loss = F.mse_loss(recon_x, x, reduction='sum') # Sum over all pixels, then mean over batch
    
    # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # Ensure dimensions are handled correctly for sum.
    # It's typically sum over latent dimensions, then mean over batch.
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1) 
    kld_loss = torch.mean(kld) # Mean over batch

    total_loss = (recon_loss / x.size(0)) + beta * kld_loss # Normalize recon_loss by batch_size
    return total_loss, (recon_loss / x.size(0)), kld_loss 