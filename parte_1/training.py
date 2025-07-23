import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from autoencoder import Autoencoder

def train_autoencoder_with_gradient_accumulation(
    model, 
    train_loader, 
    num_epochs=10, 
    learning_rate=1e-3,
    accumulation_steps=4,
    device='cpu'
):
    """
    Entrenamiento del Autoencoder con Gradient Accumulation
    
    Args:
        model: El modelo Autoencoder
        train_loader: DataLoader con los datos de entrenamiento
        num_epochs: Número de épocas
        learning_rate: Tasa de aprendizaje
        accumulation_steps: Número de pasos para acumular gradientes
        device: Dispositivo ('cpu' o 'cuda')
    """
    
    # Configurar modelo y optimizador
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Listas para almacenar las pérdidas
    train_losses = []
    
    print(f"Iniciando entrenamiento en {device}")
    print(f"Gradient Accumulation Steps: {accumulation_steps}")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        accumulated_loss = 0.0
        batch_count = 0
        
        # Limpiar gradientes al inicio de cada época
        optimizer.zero_grad()
        
        for batch_idx, data in enumerate(train_loader):
            # Si los datos vienen con etiquetas, tomar solo las imágenes
            if isinstance(data, (list, tuple)):
                inputs = data[0]
            else:
                inputs = data
            
            inputs = inputs.to(device)
            
            # Aplanar las imágenes si es necesario (para MNIST: 28x28 -> 784)
            if len(inputs.shape) > 2:
                inputs = inputs.view(inputs.size(0), -1)
            
            # Forward pass
            reconstructed = model(inputs)
            loss = criterion(reconstructed, inputs)
            
            # Backward pass con normalización por accumulation_steps
            loss = loss / accumulation_steps
            loss.backward()
            
            accumulated_loss += loss.item()
            
            # Actualizar pesos cada accumulation_steps batches
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                
                avg_loss = accumulated_loss * accumulation_steps
                total_loss += avg_loss
                batch_count += 1
                accumulated_loss = 0.0
                
                if batch_count % 10 == 0:
                    print(f'Época [{epoch+1}/{num_epochs}], Batch [{batch_count}], Loss: {avg_loss:.6f}')
        
        # Actualizar pesos si quedan gradientes acumulados
        if (len(train_loader) % accumulation_steps) != 0:
            optimizer.step()
            optimizer.zero_grad()
            avg_loss = accumulated_loss * accumulation_steps
            total_loss += avg_loss
            batch_count += 1
        
        # Calcular pérdida promedio de la época
        epoch_loss = total_loss / batch_count if batch_count > 0 else 0.0
        train_losses.append(epoch_loss)
        
        print(f'Época [{epoch+1}/{num_epochs}] completada, Loss promedio: {epoch_loss:.6f}')
    
    return train_losses

def create_sample_dataset():
    """
    Crear un dataset de ejemplo con datos sintéticos para probar el autoencoder
    """
    from torch.utils.data import TensorDataset
    
    # Generar datos sintéticos (imágenes de 28x28 = 784 dimensiones)
    np.random.seed(42)
    n_samples = 1000
    data = np.random.rand(n_samples, 784).astype(np.float32)
    
    # Añadir algunos patrones para que el autoencoder tenga algo que aprender
    for i in range(n_samples):
        if i % 3 == 0:
            data[i, :200] = 0.8  # Patrón 1
        elif i % 3 == 1:
            data[i, 200:400] = 0.8  # Patrón 2
        else:
            data[i, 400:600] = 0.8  # Patrón 3
    
    # Crear dataset y dataloader
    dataset = TensorDataset(torch.from_numpy(data))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    return dataloader

def plot_training_curve(losses):
    """Graficar la curva de entrenamiento"""
    plt.figure(figsize=(10, 6))
    plt.plot(losses, 'b-', linewidth=2)
    plt.title('Curva de Entrenamiento del Autoencoder')
    plt.xlabel('Época')
    plt.ylabel('Loss (MSE)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_curve.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Función principal para ejecutar el entrenamiento"""
    
    # Configurar dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")
    
    # Crear modelo
    model = Autoencoder(input_dim=784, hidden_dim=256, latent_dim=64)
    print(f"Modelo creado con {sum(p.numel() for p in model.parameters())} parámetros")
    
    # Crear dataset de ejemplo
    train_loader = create_sample_dataset()
    print(f"Dataset creado con {len(train_loader)} batches")
    
    # Entrenar modelo
    losses = train_autoencoder_with_gradient_accumulation(
        model=model,
        train_loader=train_loader,
        num_epochs=20,
        learning_rate=1e-3,
        accumulation_steps=4,
        device=device
    )
    
    # Graficar resultados
    plot_training_curve(losses)
    
    # Guardar modelo
    torch.save(model.state_dict(), 'autoencoder_model.pth')
    print("Modelo guardado como 'autoencoder_model.pth'")
    
    return model, losses

if __name__ == "__main__":
    model, losses = main() 