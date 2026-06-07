# ==========================================================
# COORDINADOR — Entrenamiento distribuido simulado con Ray
#
# Arquitectura:
#
#   ┌─────────────┐
#   │ Coordinator │  ← este proceso
#   └──────┬──────┘
#          │  ray.remote calls
#    ┌─────┴──────────────────┐
#    ▼        ▼       ▼       ▼
# Nodo 0  Nodo 1  Nodo 2  Nodo 3
#
# Flujo por época:
#   1. Cada nodo entrena su shard  → gradientes locales
#   2. Coordinador recolecta grads → AllReduce (promedio)
#   3. Coordinador envía grads agregados a cada nodo
#   4. Cada nodo aplica optimizer.step() con esos grads
#
# Esto replica el protocolo Ring-AllReduce usado en
# sistemas reales (Horovod, NCCL, etc.)
# ==========================================================

import time
import argparse
import logging
import numpy as np
import ray
from torchvision import datasets, transforms

from node_worker import NodeWorker

logging.getLogger("ray").setLevel(logging.WARNING)


# ==========================================================
# ALL-REDUCE: agrega gradientes de todos los nodos
# ==========================================================
def allreduce(gradients_per_node: list) -> list:
    """
    Promedia los gradientes de todos los nodos.
    
    gradients_per_node: lista de listas de ndarrays
        → un elemento por nodo, cada elemento = lista de grads por capa
    
    En producción esto sería Ring-AllReduce (O(n) en ancho de banda).
    Aquí usamos reducción centralizada para simplicidad.
    """
    num_layers = len(gradients_per_node[0])
    averaged   = []
    for layer_idx in range(num_layers):
        layer_grads = np.stack(
            [gradients_per_node[node][layer_idx] for node in range(len(gradients_per_node))]
        )
        averaged.append(layer_grads.mean(axis=0))
    return averaged


# ==========================================================
# PREPARAR DATASET Y DIVIDIRLO EN SHARDS
# ==========================================================
def prepare_shards(num_nodes: int):
    print("Descargando/cargando MNIST...")
    transform     = transforms.ToTensor()
    train_dataset = datasets.MNIST(
        root="datasets", train=True, download=True, transform=transform
    )

    print(f"Convirtiendo {len(train_dataset)} imágenes a numpy...")
    full_data = [
        {"image": img.numpy().astype("float32"), "label": int(lbl)}
        for img, lbl in train_dataset
    ]

    # Dividir en shards iguales (uno por nodo)
    shard_size = len(full_data) // num_nodes
    shards = []
    for i in range(num_nodes):
        start = i * shard_size
        end   = start + shard_size if i < num_nodes - 1 else len(full_data)
        shards.append(full_data[start:end])

    print(f"Shards creados: {num_nodes} nodos × ~{shard_size} muestras")
    return shards


# ==========================================================
# ENTRENAMIENTO DISTRIBUIDO COMPLETO
# ==========================================================
def distributed_train(num_nodes: int, epochs: int, batch_size: int, lr: float):

    shards = prepare_shards(num_nodes)

    # -------------------------------------------------------
    # Lanzar los nodos como Ray Actors (procesos independientes)
    # -------------------------------------------------------
    print(f"\nLanzando {num_nodes} nodos Ray...")
    nodes = [NodeWorker.remote(node_id=i, num_nodes=num_nodes) for i in range(num_nodes)]

    # Setup: enviar shard a cada nodo
    setup_futures = [nodes[i].setup.remote(shards[i], lr=lr) for i in range(num_nodes)]
    for msg in ray.get(setup_futures):
        print(f"  {msg}")

    # Sincronizar pesos iniciales (todos parten del mismo modelo)
    initial_weights = ray.get(nodes[0].get_weights.remote())
    ray.get([node.set_weights.remote(initial_weights) for node in nodes])
    print("Pesos iniciales sincronizados en todos los nodos.\n")

    # -------------------------------------------------------
    # Bucle de entrenamiento distribuido
    # -------------------------------------------------------
    epoch_times    = []   # tiempo total por época
    node_loss_log  = {i: [] for i in range(num_nodes)}

    print("=" * 60)
    print(f"INICIANDO ENTRENAMIENTO DISTRIBUIDO")
    print(f"Nodos: {num_nodes}  |  Épocas: {epochs}  |  Batch: {batch_size}  |  LR: {lr}")
    print("=" * 60)

    train_start = time.time()

    for epoch in range(epochs):
        t_epoch = time.time()

        # PASO 1: todos los nodos entrenan su shard en paralelo
        train_futures = [
            node.train_epoch.remote(epoch=epoch, batch_size=batch_size)
            for node in nodes
        ]
        epoch_results = ray.get(train_futures)   # [{epoch, loss, time}, ...]

        # ---- Métricas de esta época ----
        epoch_elapsed = time.time() - t_epoch
        epoch_times.append(epoch_elapsed)

        losses = [r["loss"] for r in epoch_results]
        times  = [r["time"] for r in epoch_results]
        global_loss = np.mean(losses)

        for i, r in enumerate(epoch_results):
            node_loss_log[i].append(r["loss"])

        print(f"\nÉpoca {epoch + 1}/{epochs}  (total: {epoch_elapsed:.2f}s)")
        print(f"  Loss global (promedio nodos): {global_loss:.4f}")
        for i, r in enumerate(epoch_results):
            print(f"  Nodo {i}: loss={r['loss']:.4f}  tiempo_local={r['time']:.2f}s")

    total_time = time.time() - train_start
    return total_time, epoch_times, node_loss_log


# ==========================================================
# BASELINE (1 nodo) para calcular speedup
# ==========================================================
def baseline_train(epochs: int, batch_size: int, lr: float):
    print("\n" + "=" * 60)
    print("BASELINE: entrenando con 1 solo nodo (para calcular speedup)")
    print("=" * 60)

    shards = prepare_shards(1)

    node = NodeWorker.remote(node_id=0, num_nodes=1)
    ray.get(node.setup.remote(shards[0], lr=lr))

    t0 = time.time()
    for epoch in range(epochs):
        ray.get(node.train_epoch.remote(epoch=epoch, batch_size=batch_size))
    return time.time() - t0


# ==========================================================
# REPORTE FINAL
# ==========================================================
def print_report(num_nodes, epochs, total_time, epoch_times, node_loss_log, baseline_time):
    print("\n" + "=" * 60)
    print("REPORTE FINAL DE ENTRENAMIENTO DISTRIBUIDO")
    print("=" * 60)

    print(f"\n{'CONFIGURACIÓN':}")
    print(f"  Nodos simulados : {num_nodes}")
    print(f"  Épocas          : {epochs}")

    print(f"\n{'TIEMPO POR NODO (tiempo local de entrenamiento, época final)':}")
    for node_id, losses in node_loss_log.items():
        print(f"  Nodo {node_id}: loss final = {losses[-1]:.4f}")

    print(f"\n{'TIEMPO POR ÉPOCA (coordinador)':}")
    for i, t in enumerate(epoch_times):
        print(f"  Época {i+1}: {t:.2f}s")

    print(f"\n{'MÉTRICAS GLOBALES':}")
    print(f"  Tiempo total (distribuido, {num_nodes} nodos): {total_time:.2f}s")
    print(f"  Tiempo total (baseline, 1 nodo)            : {baseline_time:.2f}s")
    speedup = baseline_time / total_time
    print(f"  Speedup                                    : {speedup:.2f}x")

    print(f"\n{'LOSS POR ÉPOCA POR NODO':}")
    header = "  Época  |" + "".join(f"  Nodo {i}  |" for i in range(num_nodes))
    print(header)
    print("  " + "-" * (len(header) - 2))
    for ep in range(epochs):
        row = f"    {ep+1:2d}   |"
        for node_id in range(num_nodes):
            row += f"  {node_loss_log[node_id][ep]:.4f}  |"
        print(row)

    print("\n" + "=" * 60)


# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Entrenamiento distribuido simulado con Ray")
    parser.add_argument("--nodes",      type=int,   default=4,     help="Número de nodos simulados")
    parser.add_argument("--epochs",     type=int,   default=3,     help="Número de épocas")
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=0.001)
    parser.add_argument("--no-baseline", action="store_true",      help="Omitir benchmark de 1 nodo")
    args = parser.parse_args()

    ray.init(address="local", ignore_reinit_error=True)

    # Entrenamiento distribuido con N nodos
    total_time, epoch_times, node_loss_log = distributed_train(
        num_nodes  = args.nodes,
        epochs     = args.epochs,
        batch_size = args.batch_size,
        lr         = args.lr,
    )

    # Baseline para calcular speedup
    if args.no_baseline:
        baseline_time = total_time  # speedup = 1.0 si se omite
    else:
        baseline_time = baseline_train(
            epochs     = args.epochs,
            batch_size = args.batch_size,
            lr         = args.lr,
        )

    print_report(
        num_nodes     = args.nodes,
        epochs        = args.epochs,
        total_time    = total_time,
        epoch_times   = epoch_times,
        node_loss_log = node_loss_log,
        baseline_time = baseline_time,
    )

    ray.shutdown()
    print("\nRay finalizado correctamente.")