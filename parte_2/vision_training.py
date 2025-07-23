import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from vision_autoencoder import create_vision_autoencoder

def get_cifar10_dataloader(batch_size=32, train=True):
    """
    Obtener DataLoader de CIFAR-10 con transformaciones apropiadas
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalizar a [-1, 1]
    ])
    
    dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=train, 
        download=True, 
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=train, 
        num_workers=2
    )
    
    return dataloader

def train_vision_autoencoder_with_gradient_accumulation(
    model, 
    train_loader, 
    num_epochs=15, 
    learning_rate=1e-4,
    accumulation_steps=4,
    device='cpu',
    save_reconstructions=True
):
    """
    Entrenamiento del Vision Transformer Autoencoder con Gradient Accumulation
    
    Args:
        model: El modelo VisionTransformerAutoencoder
        train_loader: DataLoader con imágenes
        num_epochs: Número de épocas
        learning_rate: Tasa de aprendizaje (más baja para ViT)
        accumulation_steps: Número de pasos para acumular gradientes
        device: Dispositivo ('cpu' o 'cuda')
        save_reconstructions: Si guardar ejemplos de reconstrucciones
    """
    
    # Configurar modelo y optimizador
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)  # AdamW mejor para Transformers
    criterion = nn.MSELoss()
    
    # Scheduler para reducir learning rate
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Listas para almacenar las pérdidas
    train_losses = []
    
    print(f"Iniciando entrenamiento de Vision Transformer en {device}")
    print(f"Gradient Accumulation Steps: {accumulation_steps}")
    print(f"Parámetros del modelo: {sum(p.numel() for p in model.parameters()):,}")
    
    # Guardar algunas imágenes originales para comparación
    original_images = None
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        accumulated_loss = 0.0
        batch_count = 0
        
        # Limpiar gradientes al inicio de cada época
        optimizer.zero_grad()
        
        for batch_idx, (data, _) in enumerate(train_loader):
            inputs = data.to(device)
            
            # Guardar algunas imágenes originales para visualización
            if original_images is None and save_reconstructions:
                original_images = inputs[:8].cpu()  # Primeras 8 imágenes
            
            # Forward pass
            reconstructed = model(inputs)
            
            # Desnormalizar para calcular loss en rango [0,1]
            inputs_denorm = (inputs + 1) / 2  # De [-1,1] a [0,1] 
            reconstructed_denorm = (reconstructed + 1) / 2 if reconstructed.min() < 0 else reconstructed
            
            loss = criterion(reconstructed_denorm, inputs_denorm)
            
            # Backward pass con normalización por accumulation_steps
            loss = loss / accumulation_steps
            loss.backward()
            
            accumulated_loss += loss.item()
            
            # Actualizar pesos cada accumulation_steps batches
            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping para estabilidad
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                optimizer.zero_grad()
                
                avg_loss = accumulated_loss * accumulation_steps
                total_loss += avg_loss
                batch_count += 1
                accumulated_loss = 0.0
                
                if batch_count % 50 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f'Época [{epoch+1}/{num_epochs}], Batch [{batch_count}], '
                          f'Loss: {avg_loss:.6f}, LR: {current_lr:.6f}')
        
        # Actualizar pesos si quedan gradientes acumulados
        if (len(train_loader) % accumulation_steps) != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            avg_loss = accumulated_loss * accumulation_steps
            total_loss += avg_loss
            batch_count += 1
        
        # Actualizar learning rate
        scheduler.step()
        
        # Calcular pérdida promedio de la época
        epoch_loss = total_loss / batch_count if batch_count > 0 else 0.0
        train_losses.append(epoch_loss)
        
        print(f'Época [{epoch+1}/{num_epochs}] completada, Loss promedio: {epoch_loss:.6f}')
        
        # Guardar reconstrucciones cada 5 épocas
        if save_reconstructions and (epoch + 1) % 5 == 0:
            save_reconstruction_examples(model, original_images, device, epoch + 1)
    
    return train_losses

def save_reconstruction_examples(model, original_images, device, epoch):
    """
    Guardar ejemplos de reconstrucciones para visualización
    """
    model.eval()
    
    with torch.no_grad():
        inputs = original_images.to(device)
        reconstructed = model(inputs)
        
        # Desnormalizar para visualización
        inputs_vis = (inputs + 1) / 2  # De [-1,1] a [0,1]
        reconstructed_vis = (reconstructed + 1) / 2 if reconstructed.min() < 0 else reconstructed
        
        # Crear figura con originales y reconstrucciones
        fig, axes = plt.subplots(2, 8, figsize=(16, 4))
        
        for i in range(8):
            # Imagen original
            img_orig = inputs_vis[i].cpu().numpy().transpose(1, 2, 0)
            axes[0, i].imshow(np.clip(img_orig, 0, 1))
            axes[0, i].set_title(f'Original {i+1}')
            axes[0, i].axis('off')
            
            # Imagen reconstruida
            img_recon = reconstructed_vis[i].cpu().numpy().transpose(1, 2, 0)
            axes[1, i].imshow(np.clip(img_recon, 0, 1))
            axes[1, i].set_title(f'Reconstruida {i+1}')
            axes[1, i].axis('off')
        
        plt.suptitle(f'Reconstrucciones después de {epoch} épocas')
        plt.tight_layout()
        plt.savefig(f'reconstructions_epoch_{epoch}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
    model.train()

def plot_training_curve(losses):
    """Graficar la curva de entrenamiento"""
    plt.figure(figsize=(10, 6))
    plt.plot(losses, 'b-', linewidth=2)
    plt.title('Curva de Entrenamiento del Vision Transformer Autoencoder')
    plt.xlabel('Época')
    plt.ylabel('Loss (MSE)')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Escala logarítmica para mejor visualización
    plt.tight_layout()
    plt.savefig('vision_training_curve.png', dpi=300, bbox_inches='tight')
    plt.show()

def calculate_reconstruction_metrics(model, test_loader, device):
    """
    Calcular métricas de reconstrucción en el conjunto de test
    """
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for data, _ in test_loader:
            inputs = data.to(device)
            reconstructed = model(inputs)
            
            # Desnormalizar
            inputs_denorm = (inputs + 1) / 2
            reconstructed_denorm = (reconstructed + 1) / 2 if reconstructed.min() < 0 else reconstructed
            
            # MSE
            mse = nn.functional.mse_loss(reconstructed_denorm, inputs_denorm)
            total_mse += mse.item() * inputs.size(0)
            
            # MAE
            mae = nn.functional.l1_loss(reconstructed_denorm, inputs_denorm)
            total_mae += mae.item() * inputs.size(0)
            
            total_samples += inputs.size(0)
    
    avg_mse = total_mse / total_samples
    avg_mae = total_mae / total_samples
    
    print(f"\nMétricas de Reconstrucción:")
    print(f"MSE promedio: {avg_mse:.6f}")
    print(f"MAE promedio: {avg_mae:.6f}")
    print(f"RMSE promedio: {np.sqrt(avg_mse):.6f}")
    
    return avg_mse, avg_mae

def main():
    """Función principal para ejecutar el entrenamiento"""
    
    # Configurar dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")
    
    # Crear modelo Vision Transformer Autoencoder
    model = create_vision_autoencoder(img_size=32, patch_size=4, channels=3)
    print(f"Modelo Vision Transformer creado")
    
    # Crear dataloaders
    print("Descargando CIFAR-10...")
    train_loader = get_cifar10_dataloader(batch_size=16, train=True)  # Batch más pequeño para ViT
    test_loader = get_cifar10_dataloader(batch_size=16, train=False)
    
    print(f"Train dataset: {len(train_loader)} batches")
    print(f"Test dataset: {len(test_loader)} batches")
    
    # Entrenar modelo
    losses = train_vision_autoencoder_with_gradient_accumulation(
        model=model,
        train_loader=train_loader,
        num_epochs=15,  # Menos épocas, ViT aprende más rápido
        learning_rate=1e-4,  # Learning rate más bajo para ViT
        accumulation_steps=8,  # Más accumulation para simular batch más grande
        device=device,
        save_reconstructions=True
    )
    
    # Calcular métricas en test set
    print("\nEvaluando en conjunto de test...")
    test_mse, test_mae = calculate_reconstruction_metrics(model, test_loader, device)
    
    # Graficar resultados
    plot_training_curve(losses)
    
    # Guardar modelo
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_losses': losses,
        'test_mse': test_mse,
        'test_mae': test_mae,
        'model_config': {
            'img_size': 32,
            'patch_size': 4,
            'channels': 3,
            'embed_dim': 256,
            'num_layers': 4,
            'num_heads': 8,
            'latent_dim': 128
        }
    }, 'vision_autoencoder_model.pth')
    
    print("\nModelo Vision Transformer guardado como 'vision_autoencoder_model.pth'")
    
    return model, losses

if __name__ == "__main__":
    model, losses = main() 