"""
Verifica que Ray pueda levantar workers correctamente.
Ejecutar: python tests/test_workers.py
"""

import ray

@ray.remote
def hello_worker():
    import os
    return f"Hola desde worker PID={os.getpid()}"

def test_ray_workers():
    ray.init(ignore_reinit_error=True)
    
    # Lanzar 4 workers remotos
    futures = [hello_worker.remote() for _ in range(4)]
    results = ray.get(futures)
    
    print("Workers disponibles:")
    for r in results:
        print(f"  - {r}")
    
    print(f"\n✓ Ray funcionando correctamente con {len(results)} workers")
    ray.shutdown()

if __name__ == "__main__":
    test_ray_workers()