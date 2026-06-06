# ==========================================================
# ENTRENAMIENTO DISTRIBUIDO CON RAY TRAIN Y PYTORCH
# DATASET: MNIST
# ==========================================================

# Framework de computación distribuida
import ray

# Librerías de PyTorch
import torch
import torch.optim as optim

# Dataset MNIST y transformaciones
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Componentes de Ray Train
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer
from ray import train

# Modelo definido en model.py
from model import NeuralNet


# ==========================================================
# FUNCIÓN DE ENTRENAMIENTO
# Esta función será ejecutada por cada worker de Ray
# ==========================================================
def train_func(config):

    print("Worker iniciado")

    # Convierte imágenes a tensores
    transform = transforms.ToTensor()

    print("Cargando dataset MNIST...")

    # Carga el dataset MNIST
    dataset = datasets.MNIST(
        root="datasets",
        train=True,
        download=True,
        transform=transform
    )

    print(f"Dataset cargado: {len(dataset)} imágenes")

    # DataLoader para procesar datos en lotes
    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True
    )

    # Crear modelo
    model = NeuralNet()

    # Función de pérdida para clasificación
    criterion = torch.nn.CrossEntropyLoss()

    # Optimizador Adam
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    # Número de épocas
    epochs = 3

    print("Iniciando entrenamiento...")

    # Ciclo principal de entrenamiento
    for epoch in range(epochs):

        total_loss = 0.0

        # Recorrer todos los lotes
        for images, labels in dataloader:

            # Reiniciar gradientes
            optimizer.zero_grad()

            # Predicción
            outputs = model(images)

            # Calcular pérdida
            loss = criterion(outputs, labels)

            # Backpropagation
            loss.backward()

            # Actualizar pesos
            optimizer.step()

            # Acumular pérdida
            total_loss += loss.item()

        # Mostrar progreso
        print(
            f"Worker terminado - Epoch {epoch + 1}/{epochs} "
            f"Loss: {total_loss:.4f}"
        )

        # Reportar métricas a Ray
        train.report({
            "epoch": epoch,
            "loss": total_loss
        })


# ==========================================================
# PROGRAMA PRINCIPAL
# ==========================================================
if __name__ == "__main__":

    print("Iniciando Ray...")

    # Inicializar Ray
    ray.init()

    # Configuración del entrenador distribuido
    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        scaling_config=ScalingConfig(
            num_workers=2,   # Usa 2 para laptop
            use_gpu=False
        )
    )

    print("Ejecutando entrenamiento distribuido...")

    # Ejecutar entrenamiento
    result = trainer.fit()

    print("\n===================================")
    print("ENTRENAMIENTO FINALIZADO")
    print("===================================")

    print(result)

    # Cerrar Ray
    ray.shutdown()

    print("Ray finalizado correctamente.")