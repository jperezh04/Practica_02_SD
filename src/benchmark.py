# ==========================================================
# BENCHMARK — Mide escalabilidad con 1, 2, 3 y 4 workers
#
# Para cada configuración de workers ejecuta el entrenamiento
# completo y registra:
#   - Tiempo total
#   - Tiempo por época
#   - Loss final
#   - Speedup respecto a 1 worker
#
# Guarda resultados en results/benchmark.csv
# Genera gráficas en results/
#
# Uso:
#   python benchmark.py
#   python benchmark.py --epochs 5 --batch_size 128
# ==========================================================

import argparse
import csv
import os
import time
import logging
import numpy as np
import ray
from torchvision import datasets, transforms

from coordinator import distributed_train, prepare_shards

logging.getLogger("ray").setLevel(logging.WARNING)

RESULTS_DIR = "results"


def run_benchmark(epochs: int, batch_size: int, lr: float):

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Preparar dataset una sola vez
    print("Cargando MNIST para benchmark...")
    transform = transforms.ToTensor()
    mnist     = datasets.MNIST(root="datasets", train=True,
                               download=True, transform=transform)
    full_data = [
        {"image": img.numpy().astype("float32"), "label": int(lbl)}
        for img, lbl in mnist
    ]
    print(f"Dataset: {len(full_data)} imágenes\n")

    worker_configs = [1, 2, 3, 4]
    summary        = []   # lista de dicts para el CSV

    for num_workers in worker_configs:
        print("=" * 50)
        print(f"Ejecutando con {num_workers} worker(s)...")
        print("=" * 50)

        total_time, epoch_times, node_loss_log = distributed_train(
            num_nodes  = num_workers,
            epochs     = epochs,
            batch_size = batch_size,
            lr         = lr,
        )

        # Loss final: promedio de todos los nodos en la última época
        final_losses = [node_loss_log[i][-1] for i in range(num_workers)]
        avg_final_loss = np.mean(final_losses)

        summary.append({
            "workers":        num_workers,
            "total_time_s":   round(total_time, 3),
            "avg_epoch_time": round(np.mean(epoch_times), 3),
            "final_loss":     round(avg_final_loss, 4),
        })

        print(f"\n→ {num_workers} worker(s): {total_time:.2f}s | loss={avg_final_loss:.4f}\n")

    # Calcular speedup usando tiempo de 1 worker como baseline
    baseline_time = summary[0]["total_time_s"]
    for row in summary:
        row["speedup"] = round(baseline_time / row["total_time_s"], 3)

    # -------------------------------------------------------
    # Guardar CSV
    # -------------------------------------------------------
    csv_path = os.path.join(RESULTS_DIR, "benchmark.csv")
    fieldnames = ["workers", "total_time_s", "avg_epoch_time", "final_loss", "speedup"]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)

    print(f"\nResultados guardados en: {csv_path}")

    # -------------------------------------------------------
    # Imprimir tabla resumen
    # -------------------------------------------------------
    print("\n" + "=" * 60)
    print("RESUMEN DE BENCHMARK")
    print("=" * 60)
    print(f"{'Workers':>8} | {'Tiempo(s)':>10} | {'T/Época(s)':>10} | {'Loss':>8} | {'Speedup':>8}")
    print("-" * 60)
    for row in summary:
        print(
            f"{row['workers']:>8} | "
            f"{row['total_time_s']:>10.2f} | "
            f"{row['avg_epoch_time']:>10.2f} | "
            f"{row['final_loss']:>8.4f} | "
            f"{row['speedup']:>8.2f}x"
        )
    print("=" * 60)

    return summary


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=3)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=0.001)
    args = parser.parse_args()

    ray.init(address="local", ignore_reinit_error=True)

    run_benchmark(
        epochs     = args.epochs,
        batch_size = args.batch_size,
        lr         = args.lr,
    )

    ray.shutdown()
    print("\nBenchmark finalizado.")