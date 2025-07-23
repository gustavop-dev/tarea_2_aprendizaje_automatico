import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PatchEmbedding(nn.Module):
    """Convierte imagen en secuencia de patches embeddings"""
    
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # Proyección lineal de patches a embeddings
        self.projection = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
    def forward(self, x):
        # x: (batch_size, channels, height, width)
        x = self.projection(x)  # (batch_size, embed_dim, n_patches_sqrt, n_patches_sqrt)
        x = x.flatten(2)        # (batch_size, embed_dim, n_patches)
        x = x.transpose(1, 2)   # (batch_size, n_patches, embed_dim)
        return x

class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention mechanism"""
    
    def __init__(self, embed_dim=768, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.projection = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        
        # Generar Q, K, V
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, num_heads, seq_len, head_dim)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attention = (queries @ keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        
        # Aplicar attention a values
        out = attention @ values  # (batch_size, num_heads, seq_len, head_dim)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        
        return self.projection(out)

class TransformerBlock(nn.Module):
    """Bloque Transformer completo"""
    
    def __init__(self, embed_dim=768, num_heads=8, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Attention con conexión residual
        x = x + self.attention(self.norm1(x))
        # MLP con conexión residual
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformerEncoder(nn.Module):
    """Encoder basado en Vision Transformer"""
    
    def __init__(self, img_size=224, patch_size=16, in_channels=3, 
                 embed_dim=768, num_layers=6, num_heads=8, mlp_ratio=4, dropout=0.1):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.num_patches = self.patch_embed.n_patches
        
        # Positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        # Convertir imagen a patches
        x = self.patch_embed(x)  # (batch_size, num_patches, embed_dim)
        
        # Añadir positional encoding
        x = x + self.pos_embedding
        x = self.dropout(x)
        
        # Aplicar transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        x = self.norm(x)
        return x

class VisionTransformerDecoder(nn.Module):
    """Decoder para reconstruir imagen desde embeddings"""
    
    def __init__(self, embed_dim=768, patch_size=16, img_size=224, 
                 out_channels=3, num_layers=6, num_heads=8, mlp_ratio=4, dropout=0.1):
        super().__init__()
        
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Transformer blocks para el decoder
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Proyección a píxeles
        self.pixel_projection = nn.Linear(embed_dim, patch_size * patch_size * out_channels)
        
    def forward(self, x):
        # Aplicar transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Proyectar a píxeles
        x = self.pixel_projection(x)  # (batch_size, num_patches, patch_size^2 * channels)
        
        # Reconstruir imagen
        batch_size = x.shape[0]
        patches_per_side = int(math.sqrt(self.num_patches))
        
        x = x.reshape(batch_size, patches_per_side, patches_per_side, 
                     self.patch_size, self.patch_size, -1)
        x = x.permute(0, 5, 1, 3, 2, 4)  # (batch_size, channels, h_patches, patch_size, w_patches, patch_size)
        x = x.reshape(batch_size, -1, self.img_size, self.img_size)
        
        return torch.sigmoid(x)  # Normalizar entre 0 y 1

class VisionTransformerAutoencoder(nn.Module):
    """Autoencoder completo basado en Vision Transformer"""
    
    def __init__(self, img_size=224, patch_size=16, in_channels=3, 
                 embed_dim=768, num_layers=6, num_heads=8, mlp_ratio=4, 
                 dropout=0.1, latent_dim=256):
        super().__init__()
        
        # Encoder
        self.encoder = VisionTransformerEncoder(
            img_size, patch_size, in_channels, embed_dim, 
            num_layers, num_heads, mlp_ratio, dropout
        )
        
        # Bottleneck para espacio latente
        self.to_latent = nn.Linear(embed_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, embed_dim)
        
        # Decoder
        self.decoder = VisionTransformerDecoder(
            embed_dim, patch_size, img_size, in_channels,
            num_layers, num_heads, mlp_ratio, dropout
        )
        
    def encode(self, x):
        """Obtener representación latente"""
        x = self.encoder(x)
        # Usar promedio de todos los patches para representación global
        x = x.mean(dim=1)  # (batch_size, embed_dim)
        x = self.to_latent(x)
        return x
    
    def decode(self, z):
        """Decodificar desde representación latente"""
        z = self.from_latent(z)  # (batch_size, embed_dim)
        # Expandir a todos los patches
        z = z.unsqueeze(1).repeat(1, self.encoder.num_patches, 1)
        x = self.decoder(z)
        return x
        
    def forward(self, x):
        # Encoder
        encoded = self.encoder(x)
        
        # Bottleneck
        latent = encoded.mean(dim=1)  # Representación global
        latent = self.to_latent(latent)
        
        # Decoder 
        decoded_latent = self.from_latent(latent)
        decoded_latent = decoded_latent.unsqueeze(1).repeat(1, self.encoder.num_patches, 1)
        reconstructed = self.decoder(decoded_latent)
        
        return reconstructed

def create_vision_autoencoder(img_size=32, patch_size=4, channels=3):
    """
    Crear autoencoder optimizado para imágenes pequeñas (ej: CIFAR-10)
    """
    return VisionTransformerAutoencoder(
        img_size=img_size,
        patch_size=patch_size, 
        in_channels=channels,
        embed_dim=256,     # Más pequeño para imágenes pequeñas
        num_layers=4,      # Menos capas
        num_heads=8,
        latent_dim=128,
        dropout=0.1
    )

# Ejemplo de uso
if __name__ == "__main__":
    # Crear modelo para CIFAR-10 (32x32)
    model = create_vision_autoencoder(img_size=32, patch_size=4, channels=3)
    
    # Test forward pass
    x = torch.randn(2, 3, 32, 32)  # Batch de 2 imágenes CIFAR-10
    reconstructed = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {reconstructed.shape}")
    print(f"Parámetros del modelo: {sum(p.numel() for p in model.parameters()):,}") 