# ==========================================================
# ENTRENAMIENTO DISTRIBUIDO CON RAY TRAIN Y PYTORCH
# DATASET: MNIST
# ==========================================================
 
import os
import time
import argparse
import logging
 
import numpy as np
import ray
import ray.data
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer, prepare_model
from ray import train
 
from model import NeuralNet
 
logging.getLogger("ray").setLevel(logging.WARNING)
 
 
# ==========================================================
# FUNCIÓN DE ENTRENAMIENTO
# ==========================================================
def train_func(config):
    batch_size = config.get("batch_size", 64)
    lr         = config.get("lr", 0.001)
    epochs     = config.get("epochs", 3)
 
    # 1. Obtener el shard del dataset que corresponde a este worker
    train_data_shard = train.get_dataset_shard("train")
 
    # 2. Crear modelo y prepararlo para DDP (también mueve al device correcto)
    model = NeuralNet()
    model = prepare_model(model)
 
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
 
    # Detectar device (CPU o GPU asignada por Ray)
    device = next(model.parameters()).device
 
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
 
        # 3. Ray Data devuelve dicts, NO tuplas.
        #    from_torch con MNIST produce columnas "item" (imagen) y "label".
        for batch in train_data_shard.iter_torch_batches(
            batch_size=batch_size
        ):
            images = batch["image"].float().to(device)  # [B, 1, 28, 28]
            labels = batch["label"].long().to(device)   # [B]
 
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
 
            total_loss  += loss.item()
            num_batches += 1
 
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        train.report({"epoch": epoch, "loss": avg_loss})
        print(f"[Worker] Epoch {epoch+1}/{epochs}  loss={avg_loss:.4f}")
 
 
# ==========================================================
# PROGRAMA PRINCIPAL
# ==========================================================
if __name__ == "__main__":
 
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers",    type=int,   default=4)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=0.001)
    parser.add_argument("--epochs",     type=int,   default=3)
    args = parser.parse_args()
 
    print(f"Iniciando Ray con {args.workers} workers...")
    ray.init(address="local", ignore_reinit_error=True)
 
    # -------------------------------------------------------
    # Preparar dataset MNIST como Ray Dataset
    # from_torch() convierte un torch Dataset a Ray Dataset;
    # produce columnas {"item": tensor, "label": int}
    # -------------------------------------------------------
    print("Preparando dataset MNIST...")
    transform     = transforms.ToTensor()
    train_dataset = datasets.MNIST(
        root="datasets", train=True, download=True, transform=transform
    )
 
    # Ray Data no sabe convertir tensores PyTorch a Arrow directamente.
    # Solución: convertir a numpy primero y construir el dataset con from_items.
    # Cada elemento es un dict {"image": ndarray float32, "label": int}.
    print("Convirtiendo MNIST a formato Ray Data (numpy)...")
    data_list = [
        {"image": img.numpy().astype("float32"), "label": int(lbl)}
        for img, lbl in train_dataset
    ]
    ray_train_ds = ray.data.from_items(data_list)
    print(f"Dataset listo: {len(data_list)} imágenes")
 
    # -------------------------------------------------------
    # Configurar y lanzar el entrenador distribuido
    # -------------------------------------------------------
    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        datasets={"train": ray_train_ds},
        scaling_config=ScalingConfig(
            num_workers=args.workers,
            use_gpu=False,
        ),
        train_loop_config={
            "batch_size": args.batch_size,
            "lr":         args.lr,
            "epochs":     args.epochs,
        },
    )
 
    print("Iniciando entrenamiento distribuido...")
    start_time = time.time()
    result     = trainer.fit()
    elapsed    = time.time() - start_time
 
    print("\n" + "="*50)
    print("ENTRENAMIENTO FINALIZADO")
    print("="*50)
    print(f"Workers:       {args.workers}")
    print(f"Tiempo total:  {elapsed:.2f} segundos")
    print(f"Métricas finales: {result.metrics}")
    print("="*50)
 
    ray.shutdown()
    print("Ray finalizado correctamente.")