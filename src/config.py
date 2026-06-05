"""
Configuración centralizada para el entrenamiento distribuido.
"""

# Configuración del cluster
NUM_WORKERS = 2          # Cambiar a 1, 2, 4 para pruebas de escalabilidad
USE_GPU = False          # Forzar CPU para pruebas consistentes

# Hiperparámetros
EPOCHS = 5
BATCH_SIZE = 64
LEARNING_RATE = 0.001

# Datos
DATA_DIR = "./data"