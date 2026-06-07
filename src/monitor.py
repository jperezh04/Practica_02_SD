import psutil
import time
import ray
from prometheus_client import Gauge, Counter, Histogram, start_http_server

# ── Métricas Prometheus ──────────────────────────────────────────────────────
cpu_usage       = Gauge('training_cpu_percent',       'Uso de CPU durante entrenamiento')
ram_usage       = Gauge('training_ram_mb',            'Uso de RAM en MB')
ram_percent     = Gauge('training_ram_percent',       'Uso de RAM en porcentaje')
epoch_duration  = Histogram('epoch_duration_seconds', 'Duracion de cada epoca', ['worker_id'])
images_total    = Counter('images_processed_total',   'Total de imagenes procesadas', ['worker_id'])
active_workers  = Gauge('ray_active_workers',         'Workers activos en Ray')
training_loss   = Gauge('training_loss',              'Loss actual del entrenamiento', ['worker_id'])


def start_monitor(port=8000):
    """Inicia el servidor HTTP de Prometheus y el loop de monitoreo de sistema."""
    start_http_server(port)
    print(f"[Monitor] Prometheus corriendo en http://localhost:{port}/metrics")


def record_epoch(worker_id: str, duration: float, loss: float, num_images: int):
    """Llamar al final de cada epoca desde train.py."""
    epoch_duration.labels(worker_id=worker_id).observe(duration)
    images_total.labels(worker_id=worker_id).inc(num_images)
    training_loss.labels(worker_id=worker_id).set(loss)


def monitor_loop(interval: float = 1.0, max_seconds: float = 300):
    """Loop que recolecta CPU/RAM y workers activos de Ray."""
    print(f"[Monitor] Recolectando metricas cada {interval}s (max {max_seconds}s)")
    start = time.time()
    try:
        while (time.time() - start) < max_seconds:
            cpu_usage.set(psutil.cpu_percent(interval=None))
            mem = psutil.virtual_memory()
            ram_usage.set(mem.used / 1024 / 1024)
            ram_percent.set(mem.percent)

            try:
                stats = ray.cluster_resources()
                workers = stats.get("CPU", 0)
                active_workers.set(workers)
            except Exception:
                pass

            time.sleep(interval)
    except KeyboardInterrupt:
        print("[Monitor] Detenido.")


if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)
    start_monitor(port=8000)
    monitor_loop(interval=1.0, max_seconds=600)