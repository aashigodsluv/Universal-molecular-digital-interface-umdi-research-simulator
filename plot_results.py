import json
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
R = ROOT / "results"
F = ROOT / "figures"
F.mkdir(exist_ok=True)

def rows(name):
    return json.loads((R / f"{name}.json").read_text(encoding="utf-8"))

def save(name):
    plt.savefig(F / name, dpi=180, bbox_inches="tight")
    plt.close()

# 1. Latency vs channels
x = rows("latency_throughput_vs_channels")
plt.figure()
plt.plot([r["channels"] for r in x], [r["mean_latency_ms"] for r in x], marker="o")
plt.xlabel("Molecular channels")
plt.ylabel("Mean latency (ms)")
plt.title("Latency vs Number of Molecular Channels")
plt.grid(True, alpha=.25)
save("fig_latency_vs_channels.png")

# 2. Throughput vs parallelism
plt.figure()
plt.plot([r["channels"] for r in x], [r["throughput_bytes_per_s"] for r in x], marker="o")
plt.xlabel("Molecular channels")
plt.ylabel("Throughput (bytes/s)")
plt.title("Throughput vs Parallelism")
plt.grid(True, alpha=.25)
save("fig_throughput_vs_parallelism.png")

# 3. Channel utilization vs parallelism
plt.figure()
plt.plot(
    [r["channels"] for r in x],
    [r.get("mean_channel_utilization", 0) * 100 for r in x],
    marker="o"
)
plt.xlabel("Molecular channels")
plt.ylabel("Mean channel utilization (%)")
plt.title("Channel Utilization vs Parallelism")
plt.grid(True, alpha=.25)
save("fig_channel_utilization.png")

# 4. Scheduler comparison
x = rows("fifo_vs_molecular_aware")
plt.figure()
plt.bar([r["scheduler"] for r in x], [r["mean_latency_ms"] for r in x])
plt.ylabel("Mean latency (ms)")
plt.title("FIFO vs Molecular-Aware Scheduling")
plt.grid(True, axis="y", alpha=.25)
save("fig_scheduler_comparison.png")

# 5. Cache capacity vs hit rate
x = rows("cache_vs_repeated_access")
hit_rates = [
    (r["cache_hits"] / r["requests"] * 100) if r.get("requests", 0) else 0
    for r in x
]
plt.figure()
plt.plot([r["cache_size"] for r in x], hit_rates, marker="o")
plt.xlabel("Cache size (blocks)")
plt.ylabel("Cache hit rate (%)")
plt.title("Cache Hit Rate vs Cache Capacity")
plt.grid(True, alpha=.25)
save("fig_cache_hits.png")

# 6. Error probability vs reconstruction success
x = rows("error_vs_reconstruction")
plt.figure()
plt.plot(
    [r["error_probability"] * 100 for r in x],
    [r["reconstruction_success_rate"] * 100 for r in x],
    marker="o"
)
plt.xlabel("Injected error probability (%)")
plt.ylabel("Successful reconstruction rate (%)")
plt.title("Error Rate vs Reconstruction Success")
plt.grid(True, alpha=.25)
save("fig_error_recovery.png")

# 7. Queue depth vs latency
x = rows("queue_depth_vs_latency")
plt.figure()
plt.plot([r["queue_depth"] for r in x], [r["mean_latency_ms"] for r in x], marker="o")
plt.xlabel("Queued requests")
plt.ylabel("Mean latency (ms)")
plt.title("Queue Depth vs Average Latency")
plt.grid(True, alpha=.25)
save("fig_queue_latency.png")

# 8. Endurance
x = rows("write_repetition_endurance")
failure_rates = [
    r.get(
        "write_failure_rate",
        (r.get("failed", 0) / r["rewrites"]) if r.get("rewrites", 0) else 0
    ) * 100
    for r in x
]
plt.figure()
plt.plot([r["rewrites"] for r in x], failure_rates, marker="o")
plt.xlabel("Write repetitions")
plt.ylabel("Write failure rate (%)")
plt.title("Write Repetition and Simulated Endurance")
plt.grid(True, alpha=.25)
save("fig_endurance.png")

# 9-11. Prefetch experiments
x = rows("prefetch_vs_no_prefetch")
labels = [
    "Off" if not r.get("prefetch_enabled", False)
    else f'Depth {r.get("prefetch_depth", 0)}'
    for r in x
]
throughput = [r.get("throughput_bytes_per_s", 0) / 1_000_000 for r in x]
latency = [r.get("mean_latency_ms", 0) for r in x]
prefetch_hit_rates = [
    (r.get("cache_hits", 0) / r.get("requests", 1)) * 100
    if r.get("requests", 0) else 0
    for r in x
]

plt.figure()
plt.plot(labels, throughput, marker="o")
plt.xlabel("Prefetch configuration")
plt.ylabel("Throughput (MB/s)")
plt.title("Prefetch Depth vs Throughput")
plt.grid(True, alpha=.25)
save("fig_prefetch_throughput.png")

plt.figure()
plt.plot(labels, latency, marker="o")
plt.xlabel("Prefetch configuration")
plt.ylabel("Mean latency (ms)")
plt.title("Prefetch Depth vs Mean Latency")
plt.grid(True, alpha=.25)
save("fig_prefetch_latency.png")

plt.figure()
plt.plot(labels, prefetch_hit_rates, marker="o")
plt.xlabel("Prefetch configuration")
plt.ylabel("Cache hit rate (%)")
plt.title("Prefetch Depth vs Cache Hit Rate")
plt.grid(True, alpha=.25)
save("fig_prefetch_cache_hits.png")

print("FIGURES GENERATED")
for p in sorted(F.glob("fig_*.png")):
    print(p.name)
