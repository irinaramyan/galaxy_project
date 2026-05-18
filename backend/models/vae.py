import torch.nn as nn
import torch

class GalaxyVAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.LeakyReLU(0.2), #112*112
            nn.Conv2d(32, 64, 4, 2, 1), nn.LeakyReLU(0.2), #56*56
            nn.Conv2d(64, 128, 4, 2, 1), nn.LeakyReLU(0.2), #28*28
            nn.Conv2d(128, 256, 4, 2, 1), nn.LeakyReLU(0.2), #14*14
            nn.Flatten()
        )

        self.fc_mu = nn.Linear(256*14*14, latent_dim)
        self.fc_var = nn.Linear(256*14*14, latent_dim)

        self.fc_decode = nn.Linear(latent_dim, 256*14*14)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid()
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        h = self.encoder(x)
        mu, log_var = self.fc_mu(h), self.fc_var(h)
        z = self.reparameterize(mu, log_var)
        out = self.decoder(self.fc_decode(z).view(-1, 256, 14, 14))
        return out, mu, log_var
    
def vae_loss(recon_x, x, mu, log_var, beta=1.0):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction="sum")
    kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + beta * kld