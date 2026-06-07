# ==========================================================
# PLOT RESULTS — Genera gráficas para el informe
#
# Lee los CSV generados por benchmark.py y monitor.py
# y produce 4 gráficas en results/:
#
#   1. speedup.png       — Speedup vs número de workers
#   2. tiempo.png        — Tiempo total vs workers
#   3. loss.png          — Loss final vs workers
#   4. recursos.png      — CPU % y RAM MB a lo largo del tiempo
#
# Uso:
#   python plot_results.py
# ==========================================================
 
import os
import csv
import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
 
RESULTS_DIR = "results"
BENCH_CSV   = os.path.join(RESULTS_DIR, "benchmark.csv")
MONITOR_CSV = os.path.join(RESULTS_DIR, "monitor_log.csv")
 
 
# ==========================================================
# HELPERS
# ==========================================================
def load_csv(path):
    if not os.path.exists(path):
        return None
    with open(path, newline="") as f:
        return list(csv.DictReader(f))
 
 
def save(fig, name):
    path = os.path.join(RESULTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Guardada: {path}")
    plt.close(fig)
 
 
# Estilo general
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "#f8f9fa",
    "axes.grid":        True,
    "grid.color":       "white",
    "grid.linewidth":   1.2,
    "font.family":      "DejaVu Sans",
    "font.size":        11,
})
COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]
 
 
# ==========================================================
# GRÁFICA 1 — Speedup vs Workers
# ==========================================================
def plot_speedup(rows):
    workers = [int(r["workers"])  for r in rows]
    speedup = [float(r["speedup"]) for r in rows]
 
    fig, ax = plt.subplots(figsize=(7, 4.5))
 
    ax.plot(workers, speedup, "o-", color=COLORS[0],
            linewidth=2.5, markersize=8, label="Speedup real")
    ax.plot(workers, workers, "--", color="#9E9E9E",
            linewidth=1.5, label="Speedup ideal (lineal)")
 
    for x, y in zip(workers, speedup):
        ax.annotate(f"{y:.2f}x", xy=(x, y),
                    xytext=(0, 10), textcoords="offset points",
                    ha="center", fontsize=10, color=COLORS[0])
 
    ax.set_xlabel("Número de workers")
    ax.set_ylabel("Speedup")
    ax.set_title("Speedup vs Número de Workers", fontweight="bold")
    ax.set_xticks(workers)
    ax.legend()
    save(fig, "speedup.png")
 
 
# ==========================================================
# GRÁFICA 2 — Tiempo total vs Workers
# ==========================================================
def plot_tiempo(rows):
    workers = [int(r["workers"])     for r in rows]
    tiempos = [float(r["total_time_s"]) for r in rows]
    epocas  = [float(r["avg_epoch_time"]) for r in rows]
 
    fig, ax = plt.subplots(figsize=(7, 4.5))
 
    x     = np.arange(len(workers))
    width = 0.35
 
    bars1 = ax.bar(x - width/2, tiempos, width, label="Tiempo total",
                   color=COLORS[1], alpha=0.85)
    bars2 = ax.bar(x + width/2, epocas,  width, label="Tiempo/época",
                   color=COLORS[2], alpha=0.85)
 
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{bar.get_height():.1f}s", ha="center", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{bar.get_height():.1f}s", ha="center", fontsize=9)
 
    ax.set_xlabel("Número de workers")
    ax.set_ylabel("Segundos")
    ax.set_title("Tiempo de Entrenamiento vs Workers", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{w} worker{'s' if w > 1 else ''}" for w in workers])
    ax.legend()
    save(fig, "tiempo.png")
 
 
# ==========================================================
# GRÁFICA 3 — Loss final vs Workers
# ==========================================================
def plot_loss(rows):
    workers = [int(r["workers"])      for r in rows]
    losses  = [float(r["final_loss"]) for r in rows]
 
    fig, ax = plt.subplots(figsize=(7, 4.5))
 
    bars = ax.bar(workers, losses, color=COLORS[3], alpha=0.85, width=0.5)
 
    for bar, val in zip(bars, losses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", fontsize=10)
 
    ax.set_xlabel("Número de workers")
    ax.set_ylabel("Loss (CrossEntropy)")
    ax.set_title("Loss Final vs Número de Workers", fontweight="bold")
    ax.set_xticks(workers)
    ax.set_ylim(0, max(losses) * 1.2)
    save(fig, "loss.png")
 
 
# ==========================================================
# GRÁFICA 4 — CPU y RAM a lo largo del tiempo
# ==========================================================
def plot_recursos(rows):
    times   = [float(r["time_s"])  for r in rows]
    cpu     = [float(r["cpu_pct"]) for r in rows]
    ram_mb  = [float(r["ram_mb"])  for r in rows]
 
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
 
    ax1.plot(times, cpu, color=COLORS[0], linewidth=1.5)
    ax1.fill_between(times, cpu, alpha=0.2, color=COLORS[0])
    ax1.set_ylabel("CPU (%)")
    ax1.set_title("Uso de CPU durante el entrenamiento", fontweight="bold")
    ax1.set_ylim(0, 105)
 
    ax2.plot(times, ram_mb, color=COLORS[3], linewidth=1.5)
    ax2.fill_between(times, ram_mb, alpha=0.2, color=COLORS[3])
    ax2.set_ylabel("RAM (MB)")
    ax2.set_xlabel("Tiempo (s)")
    ax2.set_title("Uso de RAM durante el entrenamiento", fontweight="bold")
 
    fig.tight_layout(pad=2)
    save(fig, "recursos.png")
 
 
# ==========================================================
# GRÁFICA 5 — Panel resumen (todas en una)
# ==========================================================
def plot_panel(bench_rows, monitor_rows):
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("Resumen de Rendimiento — Entrenamiento Distribuido con Ray",
                 fontsize=14, fontweight="bold", y=0.98)
 
    gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)
 
    workers = [int(r["workers"])       for r in bench_rows]
    speedup = [float(r["speedup"])     for r in bench_rows]
    tiempos = [float(r["total_time_s"]) for r in bench_rows]
    losses  = [float(r["final_loss"])  for r in bench_rows]
 
    # — Speedup
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(workers, speedup, "o-", color=COLORS[0], linewidth=2.5, markersize=7)
    ax1.plot(workers, workers, "--", color="#9E9E9E", linewidth=1.2)
    ax1.set_title("Speedup vs Workers")
    ax1.set_xlabel("Workers"); ax1.set_ylabel("Speedup")
    ax1.set_xticks(workers)
 
    # — Tiempo
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(workers, tiempos, color=COLORS[1], alpha=0.85, width=0.5)
    ax2.set_title("Tiempo Total vs Workers")
    ax2.set_xlabel("Workers"); ax2.set_ylabel("Segundos")
    ax2.set_xticks(workers)
 
    # — Loss
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.bar(workers, losses, color=COLORS[3], alpha=0.85, width=0.5)
    ax3.set_title("Loss Final vs Workers")
    ax3.set_xlabel("Workers"); ax3.set_ylabel("Loss")
    ax3.set_xticks(workers)
 
    # — CPU (si hay datos de monitor)
    ax4 = fig.add_subplot(gs[1, 1])
    if monitor_rows:
        times = [float(r["time_s"])  for r in monitor_rows]
        cpu   = [float(r["cpu_pct"]) for r in monitor_rows]
        ax4.plot(times, cpu, color=COLORS[2], linewidth=1.5)
        ax4.fill_between(times, cpu, alpha=0.2, color=COLORS[2])
        ax4.set_title("CPU durante entrenamiento")
        ax4.set_xlabel("Tiempo (s)"); ax4.set_ylabel("CPU (%)")
        ax4.set_ylim(0, 105)
    else:
        ax4.text(0.5, 0.5, "Ejecuta monitor.py\npara ver CPU/RAM",
                 ha="center", va="center", transform=ax4.transAxes,
                 fontsize=11, color="gray")
        ax4.set_title("CPU durante entrenamiento")
 
    save(fig, "panel_resumen.png")
 
 
# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":
 
    os.makedirs(RESULTS_DIR, exist_ok=True)
 
    bench_rows   = load_csv(BENCH_CSV)
    monitor_rows = load_csv(MONITOR_CSV)
 
    if bench_rows is None:
        print(f"ERROR: No se encontró {BENCH_CSV}")
        print("Ejecuta primero: python benchmark.py")
        sys.exit(1)
 
    print("Generando gráficas...\n")
 
    plot_speedup(bench_rows)
    plot_tiempo(bench_rows)
    plot_loss(bench_rows)
 
    if monitor_rows:
        plot_recursos(monitor_rows)
    else:
        print("  (Sin datos de monitor — ejecuta monitor.py para la gráfica de recursos)")
 
    plot_panel(bench_rows, monitor_rows)
 
    print(f"\nTodas las gráficas guardadas en: {RESULTS_DIR}/")
    print("Archivos: speedup.png, tiempo.png, loss.png, recursos.png, panel_resumen.png")