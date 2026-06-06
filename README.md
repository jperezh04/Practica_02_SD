
Proyecto: SDTeoRAY_Practica2
=================================

Descripción
-----------
Este proyecto entrena una red neuronal simple sobre el dataset MNIST usando PyTorch y Ray Train.

Qué hace
---------
- Descarga y carga el dataset MNIST en `datasets/`.
- Define un modelo en `src/model.py` (clase `NeuralNet`).
- Ejecuta el bucle de entrenamiento en `src/train.py`, utilizando Ray Train para ejecutar la función de entrenamiento en múltiples workers.

Cómo funciona (resumen técnico)
------------------------------
1. `src/train.py` inicializa Ray con `ray.init()` y crea un `TorchTrainer` que ejecuta `train_func` en cada worker.
2. En `train_func` se prepara el `DataLoader` de PyTorch y se llama a `prepare_data_loader` para integrarlo con Ray.
3. El modelo (`NeuralNet`) se envuelve con `prepare_model` para que cada worker tenga la copia correcta.
4. Se usa `torch.optim.Adam` y `torch.nn.CrossEntropyLoss` para optimizar la red.
5. Durante cada época se computa la pérdida y se usa `train.report()` para reportar métricas a Ray.
6. Al finalizar, el proceso principal llama a `trainer.fit()` y muestra las métricas agregadas.

Por qué se hizo así
-------------------
- Ray Train: facilita ejecutar entrenamientos distribuidos (multi-worker) sin manejar manualmente procesos o sincronización.
- PyTorch: biblioteca estándar para modelos y optimización; `torchvision` se usa para el dataset MNIST.
- Separación de responsabilidades: `src/model.py` contiene la definición del modelo y `src/train.py` controla el flujo de entrenamiento. Esto mejora la mantenibilidad.

Instrucciones rápidas
---------------------
Desde la raíz del proyecto, con Python 3.11 disponible via `py -3.11`:

1. Instalar dependencias:

```cmd
py -3.11 -m pip install -r requirements.txt
```

2. Ejecutar el entrenamiento:

```cmd
py -3.11 src\train.py
```
