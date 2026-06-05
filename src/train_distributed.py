"""
Entrenamiento distribuido de MNIST con Ray Train.
Autor: Persona 3 - Desarrollador Principal
Propósito: Demostrar paralelización de entrenamiento de modelo ML
"""

# ============================================================================
# 1. IMPORTS Y CONFIGURACIÓN INICIAL
# ============================================================================

import os
import tempfile
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

import ray
from ray import train, tune
from ray.train import Checkpoint, CheckpointConfig
from ray.train.torch import TorchTrainer
from ray.train.torch import TorchConfig
from ray.air.config import ScalingConfig

# Configuración básica (puede sobrescribirse desde config.py)
NUM_WORKERS = 2           # Número de workers paralelos (cambiar a 1, 2, 4)
EPOCHS = 5                # Épocas de entrenamiento
BATCH_SIZE = 64           # Tamaño de batch por worker
LEARNING_RATE = 0.001


# ============================================================================
# 2. DEFINICIÓN DEL MODELO (Red Neuronal Simple)
# ============================================================================

class SimpleMNISTModel(nn.Module):
    """
    Modelo CNN simple para clasificación de dígitos MNIST (10 clases).
    Arquitectura liviana para que el ejemplo sea rápido.
    """
    def __init__(self):
        super(SimpleMNISTModel, self).__init__()
        # Capa convolucional 1: entrada 1 canal (grises) -> 32 filtros
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        # Capa convolucional 2: 32 -> 64 filtros
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # Capa fully connected: después de dos maxpool (28x28 -> 7x7)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        # Capa de salida: 10 clases
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Aplicar convolución + ReLU + MaxPool (reduce dimensión a la mitad)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  # 28x28 -> 14x14
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)  # 14x14 -> 7x7
        # Aplanar para las capas densas
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# ============================================================================
# 3. FUNCIÓN DE CARGA DE DATOS (Distribuida)
# ============================================================================

def load_data(data_dir: str = "./data"):
    """
    Carga MNIST y lo particiona para entrenamiento distribuido.
    Usa DistributedSampler para que cada worker vea una parte única de los datos.
    """
    # Transformación: convertir a tensor y normalizar a [0,1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Descargar datasets (se sincroniza automáticamente en todos los workers)
    train_dataset = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )

    return train_dataset, test_dataset


# ============================================================================
# 4. FUNCIÓN DE ENTRENAMIENTO POR WORKER (CORAZÓN DISTRIBUIDO)
# ============================================================================

def train_func_per_worker(config: Dict[str, Any]):
    """
    Esta función se ejecuta en CADA worker de Ray.
    Cada worker tiene su propia GPU/CPU y una partición de datos.
    
    Args:
        config: Diccionario con hiperparámetros (epochs, lr, batch_size)
    """
    # Obtener configuración
    epochs = config.get("epochs", EPOCHS)
    lr = config.get("lr", LEARNING_RATE)
    batch_size = config.get("batch_size", BATCH_SIZE)
    
    # ===== 4.1 Configurar dispositivo =====
    # Ray Train expone la variable de entorno CUDA_VISIBLE_DEVICES automáticamente
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ===== 4.2 Crear modelo y moverlo al dispositivo =====
    model = SimpleMNISTModel()
    model = model.to(device)
    
    # ===== 4.3 Configurar optimizador =====
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # ===== 4.4 Cargar datos de forma DISTRIBUIDA =====
    # Obtener el dataset completo
    train_dataset, _ = load_data()
    
    # Crear DistributedSampler: cada worker toma un subset NO SUPERPUESTO
    # `train.get_context().get_world_size()` = número total de workers
    # `train.get_context().get_rank()` = índice de este worker (0,1,2,3)
    sampler = DistributedSampler(
        train_dataset,
        num_replicas=train.get_context().get_world_size(),
        rank=train.get_context().get_rank(),
        shuffle=True
    )
    
    # DataLoader con el sampler distribuido
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,          # <-- CLAVE: partición única por worker
        num_workers=0             # Evitar sub-workers (para simplicidad)
    )
    
    # ===== 4.5 Bucle de entrenamiento =====
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Iterar sobre los batches asignados a este worker
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            loss = F.nll_loss(output, target)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        # Calcular pérdida promedio en este epoch (solo este worker)
        avg_loss = total_loss / num_batches
        
        # ===== 4.6 Reportar métricas a Ray (para agregación automática) =====
        # Ray automáticamente promedia las pérdidas de todos los workers
        train.report(
            metrics={"loss": avg_loss, "epoch": epoch},
            checkpoint=Checkpoint.from_dict(
                {"model_state_dict": model.state_dict()}
            )
        )
        
        print(f"[Worker {train.get_context().get_rank()}] "
              f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")


# ============================================================================
# 5. FUNCIÓN PRINCIPAL: ORQUESTAR EL ENTRENAMIENTO DISTRIBUIDO
# ============================================================================

def main():
    """
    Orquesta el entrenamiento distribuido usando Ray Train.
    """
    print("=" * 60)
    print("ENTRENAMIENTO DISTRIBUIDO CON RAY TRAIN")
    print("=" * 60)
    
    # ===== 5.1 Inicializar Ray (local o conectado a cluster) =====
    # `ignore_reinit_error=True` permite ejecutar múltiples veces en notebooks
    ray.init(ignore_reinit_error=True, address="auto")
    print(f"Ray inicializado. Nodos disponibles: {ray.cluster_resources()}")
    
    # ===== 5.2 Configurar el entrenador distribuido =====
    # ScalingConfig: define cuántos workers usar
    scaling_config = ScalingConfig(
        num_workers=NUM_WORKERS,       # Workers en paralelo
        use_gpu=torch.cuda.is_available(),  # Usar GPU si está disponible
        trainer_resources={"CPU": 1},       # Recursos para el driver
    )
    
    # Configuración de Torch (necesaria para que Ray maneje correctamente PyTorch)
    torch_config = TorchConfig(backend="gloo" if not torch.cuda.is_available() else "nccl")
    
    # Configuración de hiperparámetros
    train_config = {
        "epochs": EPOCHS,
        "lr": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
    }
    
    # Crear el TorchTrainer
    trainer = TorchTrainer(
        train_loop_per_worker=train_func_per_worker,  # Función que ejecuta cada worker
        train_loop_config=train_config,               # Parámetros para esa función
        scaling_config=scaling_config,                # Escalamiento
        torch_config=torch_config,                    # Configuración PyTorch
        checkpoint_config=CheckpointConfig(
            num_to_keep=1,          # Guardar solo el último checkpoint
            checkpoint_score_attribute="loss",
            checkpoint_score_order="min"
        )
    )
    
    # ===== 5.3 Ejecutar entrenamiento =====
    print(f"\nIniciando entrenamiento con {NUM_WORKERS} workers...")
    result = trainer.fit()
    
    # ===== 5.4 Mostrar resultados =====
    print("\n" + "=" * 60)
    print("ENTRENAMIENTO COMPLETADO")
    print("=" * 60)
    print(f"Mejor pérdida: {result.metrics['loss']:.4f}")
    print(f"Checkpoint final guardado en: {result.checkpoint}")
    
    # ===== 5.5 Apagar Ray (opcional, libera recursos) =====
    # ray.shutdown()  # Descomentar si no se necesita Ray después
    
    return result


# ============================================================================
# 6. PUNTO DE ENTRADA
# ============================================================================

if __name__ == "__main__":
    main()