# ==========================================================
# TUNE SEARCH — Hyperparameter Tuning Distribuido con Ray Tune
#
# Busca la mejor combinación de:
#   - learning rate
#   - batch size
#   - neuronas en capa oculta
#
# Estrategia: ASHA Scheduler
#   Lanza muchos trials en paralelo y mata temprano los que
#   no muestran mejora → más eficiente que grid search.
#
# Uso:
#   python tune_search.py
#   python tune_search.py --trials 20 --epochs 5
# ==========================================================

import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler

logging.getLogger("ray").setLevel(logging.WARNING)


# ==========================================================
# MODELO FLEXIBLE — acepta tamaño de capa oculta como parámetro
# ==========================================================
class NeuralNetTunable(nn.Module):
    def __init__(self, hidden: int = 128):
        super().__init__()
        self.fc1  = nn.Linear(28 * 28, hidden)
        self.relu = nn.ReLU()
        self.fc2  = nn.Linear(hidden, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# ==========================================================
# FUNCIÓN DE ENTRENAMIENTO PARA UN TRIAL
# Ray Tune llama esta función una vez por combinación de HPs
# ==========================================================
def train_trial(config, data=None):
    """
    config: dict con lr, batch_size, hidden (inyectado por Tune)
    data:   dataset completo (pasado via tune.with_parameters)
    """
    lr         = config["lr"]
    batch_size = config["batch_size"]
    hidden     = config["hidden"]
    epochs     = config.get("epochs", 3)

    model     = NeuralNetTunable(hidden=hidden)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss  = 0.0
        num_batches = 0

        indices = np.random.permutation(len(data))

        for start in range(0, len(data), batch_size):
            idx    = indices[start : start + batch_size]
            batch  = [data[i] for i in idx]

            images = torch.tensor(
                np.stack([b["image"] for b in batch]), dtype=torch.float32
            )
            labels = torch.tensor(
                [b["label"] for b in batch], dtype=torch.long
            )

            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss  += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # Reportar métrica a Tune — ASHA usa esto para decidir
        # si continuar o terminar este trial
        tune.report({"loss": avg_loss, "epoch": epoch})


# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--trials",  type=int, default=12,
                        help="Número de combinaciones a probar")
    parser.add_argument("--epochs",  type=int, default=3,
                        help="Épocas máximas por trial")
    parser.add_argument("--workers", type=int, default=4,
                        help="Trials en paralelo")
    args = parser.parse_args()

    ray.init(
        address="local",
        ignore_reinit_error=True,
        dashboard_host="0.0.0.0",
        dashboard_port=8265,
    )

    # -------------------------------------------------------
    # Cargar dataset una sola vez y compartirlo entre trials
    # -------------------------------------------------------
    print("Cargando MNIST...")
    transform = transforms.ToTensor()
    mnist     = datasets.MNIST(root="datasets", train=True,
                               download=True, transform=transform)
    data = [
        {"image": img.numpy().astype("float32"), "label": int(lbl)}
        for img, lbl in mnist
    ]
    print(f"Dataset listo: {len(data)} imágenes\n")

    # -------------------------------------------------------
    # Espacio de búsqueda de hiperparámetros
    # -------------------------------------------------------
    search_space = {
        "lr":         tune.loguniform(1e-4, 1e-1),   # escala logarítmica
        "batch_size": tune.choice([32, 64, 128]),
        "hidden":     tune.choice([64, 128, 256]),
        "epochs":     args.epochs,
    }

    # -------------------------------------------------------
    # ASHA Scheduler — termina trials malos temprano
    #
    # grace_period: mínimo de épocas antes de poder terminar
    # reduction_factor: en cada ronda elimina 1/factor trials
    # -------------------------------------------------------
    scheduler = ASHAScheduler(
        metric           = "loss",
        mode             = "min",
        max_t            = args.epochs,
        grace_period     = 1,
        reduction_factor = 2,
    )

    # -------------------------------------------------------
    # Lanzar búsqueda
    # -------------------------------------------------------
    print("=" * 60)
    print(f"Iniciando Ray Tune: {args.trials} trials × {args.epochs} épocas")
    print(f"Trials en paralelo: {args.workers}")
    print(f"Dashboard Ray:      http://localhost:8265")
    print("=" * 60 + "\n")

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_trial, data=data),
            resources={"cpu": 1},
        ),
        param_space    = search_space,
        tune_config    = tune.TuneConfig(
            num_samples      = args.trials,
            scheduler        = scheduler,
            max_concurrent_trials = args.workers,
        ),
    )

    results = tuner.fit()

    # -------------------------------------------------------
    # Mostrar resultados
    # -------------------------------------------------------
    best = results.get_best_result(metric="loss", mode="min")

    print("\n" + "=" * 60)
    print("RESULTADOS DE HYPERPARAMETER TUNING")
    print("=" * 60)

    print("\nTop 5 configuraciones:")
    df = results.get_dataframe()
    df_sorted = df[["config/lr", "config/batch_size", "config/hidden", "loss"]]\
                  .sort_values("loss").head(5)
    print(df_sorted.to_string(index=False))

    print(f"\n{'MEJOR CONFIGURACIÓN':}")
    print(f"  learning_rate : {best.config['lr']:.6f}")
    print(f"  batch_size    : {best.config['batch_size']}")
    print(f"  hidden_units  : {best.config['hidden']}")
    print(f"  loss final    : {best.metrics['loss']:.4f}")
    print("=" * 60)

    ray.shutdown()
    print("\nRay Tune finalizado.")