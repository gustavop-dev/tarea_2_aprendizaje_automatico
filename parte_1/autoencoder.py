import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=128, latent_dim=32):
        """
        Autoencoder para datos de imágenes (por ejemplo MNIST 28x28=784)
        
        Args:
            input_dim: Dimensión de entrada (por defecto 784 para MNIST)
            hidden_dim: Dimensión de la capa oculta
            latent_dim: Dimensión del espacio latente (cuello de botella)
        """
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, latent_dim),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # Para normalizar la salida entre 0 y 1
        )
    
    def forward(self, x):
        # Codificar
        latent = self.encoder(x)
        # Decodificar
        reconstructed = self.decoder(latent)
        return reconstructed
    
    def encode(self, x):
        """Obtener solo la representación latente"""
        return self.encoder(x)
    
    def decode(self, z):
        """Decodificar desde el espacio latente"""
        return self.decoder(z) 