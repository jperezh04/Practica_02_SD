# ==========================================================
# MODELO DE RED NEURONAL PARA CLASIFICACIÓN DE DÍGITOS MNIST
# ==========================================================

# Importa el módulo de redes neuronales de PyTorch
import torch.nn as nn


# ----------------------------------------------------------
# Clase que define la arquitectura de la red neuronal
# ----------------------------------------------------------
class NeuralNet(nn.Module):

    # Constructor del modelo
    def __init__(self):

        # Inicializa la clase base nn.Module
        super().__init__()

        # --------------------------------------------------
        # CAPA 1 (Entrada -> Capa Oculta)
        #
        # Entrada:
        #   28 x 28 = 784 píxeles por imagen
        #
        # Salida:
        #   128 neuronas ocultas
        # --------------------------------------------------
        self.fc1 = nn.Linear(28 * 28, 128)

        # Función de activación ReLU
        # Reemplaza valores negativos por 0
        self.relu = nn.ReLU()

        # --------------------------------------------------
        # CAPA 2 (Capa Oculta -> Salida)
        #
        # Entrada:
        #   128 neuronas
        #
        # Salida:
        #   10 neuronas
        #
        # Cada neurona representa una clase:
        # 0,1,2,3,4,5,6,7,8,9
        # --------------------------------------------------
        self.fc2 = nn.Linear(128, 10)

    # ------------------------------------------------------
    # Forward Pass
    #
    # Define cómo fluyen los datos a través de la red
    # ------------------------------------------------------
    def forward(self, x):

        # Convierte cada imagen de:
        #
        # [batch_size, 1, 28, 28]
        #
        # a:
        #
        # [batch_size, 784]
        #
        # -1 permite que PyTorch calcule automáticamente
        # el tamaño del batch.
        x = x.view(-1, 28 * 28)

        # Primera transformación lineal
        x = self.fc1(x)

        # Aplicación de ReLU
        x = self.relu(x)

        # Capa de salida
        x = self.fc2(x)

        # Se devuelve el resultado sin Softmax.
        #
        # Esto es correcto porque durante el entrenamiento
        # se utiliza CrossEntropyLoss(), la cual aplica
        # internamente LogSoftmax.
        return x