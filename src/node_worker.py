# ==========================================================
# NODE WORKER — Simula un nodo distribuido con Ray Actors
#
# Cada instancia de NodeWorker es un proceso Ray independiente
# que mantiene:
#   - Su propio shard del dataset MNIST
#   - Su propia copia del modelo
#   - Su historial de métricas por época
#
# La coordinación (AllReduce de gradientes) la maneja
# el coordinador en coordinator.py
# ==========================================================

import time
import ray
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

from model import NeuralNet


@ray.remote
class NodeWorker:
    """
    Simula un nodo en un cluster distribuido.

    En un cluster real, cada NodeWorker correría en una
    máquina distinta. Aquí todos corren en localhost pero
    en procesos separados de Python, lo que replica el
    modelo de actores distribuidos.
    """

    def __init__(self, node_id: int, num_nodes: int):
        self.node_id   = node_id
        self.num_nodes = num_nodes

        self.model     = NeuralNet()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None          # se inicializa en setup()
        self.data      = []            # shard local del dataset
        self.metrics   = []            # historial: [{epoch, loss, time}]

    # ----------------------------------------------------------
    # SETUP: recibe el shard que le corresponde a este nodo
    # ----------------------------------------------------------
    def setup(self, data_shard: list, lr: float = 0.001):
        """
        data_shard: lista de dicts {"image": ndarray, "label": int}
        """
        self.data      = data_shard
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        return f"[Nodo {self.node_id}] Setup completo — {len(self.data)} muestras"

    # ----------------------------------------------------------
    # TRAIN ONE EPOCH: entrena una época y devuelve métricas
    # ----------------------------------------------------------
    def train_epoch(self, epoch: int, batch_size: int = 64):
        self.model.train()
        total_loss  = 0.0
        num_batches = 0
        t0          = time.time()

        # Shuffle local del shard
        indices = np.random.permutation(len(self.data))

        for start in range(0, len(self.data), batch_size):
            idx    = indices[start : start + batch_size]
            batch  = [self.data[i] for i in idx]

            images = torch.tensor(
                np.stack([b["image"] for b in batch]), dtype=torch.float32
            )                                      # [B, 1, 28, 28]
            labels = torch.tensor(
                [b["label"] for b in batch], dtype=torch.long
            )                                      # [B]

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss    = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss  += loss.item()
            num_batches += 1

        elapsed  = time.time() - t0
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        record = {"epoch": epoch, "loss": avg_loss, "time": elapsed}
        self.metrics.append(record)
        return record

    # ----------------------------------------------------------
    # GET GRADIENTS: extrae los gradientes actuales del modelo
    # Devuelve lista de ndarrays (uno por parámetro)
    # ----------------------------------------------------------
    def get_gradients(self):
        grads = []
        for param in self.model.parameters():
            if param.grad is not None:
                grads.append(param.grad.numpy().copy())
            else:
                grads.append(np.zeros(param.shape))
        return grads

    # ----------------------------------------------------------
    # SET GRADIENTS + STEP: recibe gradientes agregados y aplica
    # ----------------------------------------------------------
    def apply_gradients(self, averaged_grads: list):
        """
        Recibe los gradientes promediados por el coordinador
        (resultado del AllReduce) y ejecuta el paso del optimizador.
        """
        for param, grad in zip(self.model.parameters(), averaged_grads):
            param.grad = torch.tensor(grad, dtype=torch.float32)
        self.optimizer.step()

    # ----------------------------------------------------------
    # SYNC WEIGHTS: recibe pesos del coordinador y los carga
    # (para mantener todos los nodos en el mismo estado inicial)
    # ----------------------------------------------------------
    def set_weights(self, weights: list):
        for param, w in zip(self.model.parameters(), weights):
            param.data = torch.tensor(w, dtype=torch.float32)

    def get_weights(self):
        return [p.data.numpy().copy() for p in self.model.parameters()]

    # ----------------------------------------------------------
    # MÉTRICAS
    # ----------------------------------------------------------
    def get_metrics(self):
        return {"node_id": self.node_id, "history": self.metrics}

    def get_node_id(self):
        return self.node_id