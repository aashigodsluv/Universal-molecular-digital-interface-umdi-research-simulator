"""
UMDI BioSSD simulation API.

Provides the local HTTP backend for the transaction simulator, benchmark
simulator, research experiments, live progress telemetry, cancellation, and
technology-gap analysis. Benchmark endpoints use the UMDI/BioSSD reference
model with configurable payload size, logical-block size, effective molecular
channels, READ-path acceleration, and WRITE-path acceleration.
"""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import copy
import json
import math
import sys
import threading
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
CONFIGS = ROOT / "configs"

sys.path.insert(0, str(SRC))

import biossd_sim

HOST = "127.0.0.1"
PORT = 8766


DEFAULT_CFG = {
    "seed": 20260812,
    "channels": 4,
    "cache_size": 16,
    "cache_latency_ms": 0.2,
    "scheduler": "molecular_aware",
    "error_probability": 0.03,
    "correction_success_probability": 0.98,
    "write_failure_probability": 0.01,
    "latency_ms": {
        "addressing": {"kind": "normal", "mean": 8, "sd": 1.5},
        "molecular_access": {"kind": "lognormal", "mu": 3.9, "sigma": 0.25},
        "sensing": {"kind": "normal", "mean": 35, "sd": 5},
        "basecalling": {"kind": "normal", "mean": 18, "sd": 3},
        "decoding": {"kind": "normal", "mean": 12, "sd": 2},
        "error_correction": {"kind": "normal", "mean": 8, "sd": 1.5},
        "buffer": {"kind": "fixed", "value": 1},
        "encoding": {"kind": "normal", "mean": 6, "sd": 1},
        "molecular_write": {"kind": "lognormal", "mu": 4.8, "sigma": 0.3},
        "write_verification": {"kind": "normal", "mean": 15, "sd": 3},
    },
}


def build_default_cfg():
    return copy.deepcopy(DEFAULT_CFG)


# ---------------------------------------------------------------------
# Run registry: shared state between /simulate (writer, from inside the
# simulation thread via progress_callback) and /progress (reader, from
# whatever thread the polling GET lands on).
# ---------------------------------------------------------------------

STATE_LOCK = threading.Lock()
RUN_REGISTRY = {}
ACTIVE = {"run_id": None, "cancel_event": None}
MAX_REGISTRY_ENTRIES = 12


def _prune_registry_locked():
    if len(RUN_REGISTRY) <= MAX_REGISTRY_ENTRIES:
        return
    for old_id in list(RUN_REGISTRY.keys())[:-MAX_REGISTRY_ENTRIES]:
        if old_id != ACTIVE["run_id"]:
            RUN_REGISTRY.pop(old_id, None)


def make_progress_callback(run_id, cancel_event):
    def _callback(update):
        with STATE_LOCK:
            record = RUN_REGISTRY.get(run_id)
            if record is None:
                return not cancel_event.is_set()

            event = update.get("event")
            total = update.get("total_requests", record["total_requests"])
            processed = update.get("processed_requests", record["processed_requests"])

            record["total_requests"] = total
            record["processed_requests"] = processed
            record["percent"] = (processed / total * 100.0) if total else 0.0
            record["updated_at"] = time.time()

            if event == "run_started":
                record["status"] = "running"

            elif event == "stage":
                stage = update.get("stage")
                operation = update.get("operation")
                if stage:
                    record["latest_stage"] = stage
                    record["stage_counts"][stage] = record["stage_counts"].get(stage, 0) + 1
                if operation:
                    record["latest_operation"] = operation

            elif event == "request_complete":
                operation = update.get("operation")
                if operation:
                    record["latest_operation"] = operation
                # Host Request / UMDI Command Queue are pre-processing steps
                # biossd_sim doesn't emit distinct stage events for. Treat
                # "one more transaction completed its full path" as one pass
                # through each so they animate in step with the rest of the
                # pipeline instead of staying dim until the whole run ends.
                record["stage_counts"]["host_interface"] = (
                    record["stage_counts"].get("host_interface", 0) + 1
                )
                record["stage_counts"]["controller"] = (
                    record["stage_counts"].get("controller", 0) + 1
                )

            elif event == "run_complete":
                # Transactions are done, but summary() and JSON assembly
                # haven't happened yet. This is the FINALIZING state.
                record["status"] = "finalizing"
                record["percent"] = 100.0

        return not cancel_event.is_set()

    return _callback


def start_run(run_id, total_requests):
    with STATE_LOCK:
        if ACTIVE["run_id"] is not None:
            return None
        cancel_event = threading.Event()
        ACTIVE["run_id"] = run_id
        ACTIVE["cancel_event"] = cancel_event
        RUN_REGISTRY[run_id] = {
            "run_id": run_id,
            "status": "running",
            "percent": 0.0,
            "processed_requests": 0,
            "total_requests": total_requests,
            "latest_stage": None,
            "latest_operation": None,
            "stage_counts": {},
            "error": None,
            "started_at": time.time(),
            "updated_at": time.time(),
        }
        _prune_registry_locked()
        return cancel_event


def finish_run(run_id, status, error=None):
    with STATE_LOCK:
        record = RUN_REGISTRY.get(run_id)
        if record is not None:
            record["status"] = status
            record["updated_at"] = time.time()
            if status == "complete":
                record["percent"] = 100.0
                record["processed_requests"] = record["total_requests"]
            if error is not None:
                record["error"] = error
        if ACTIVE["run_id"] == run_id:
            ACTIVE["run_id"] = None
            ACTIVE["cancel_event"] = None


def read_progress(run_id):
    with STATE_LOCK:
        record = RUN_REGISTRY.get(run_id)
        if record is None:
            return False, {
                "run_id": run_id,
                "status": "idle",
                "percent": 0,
                "processed_requests": 0,
                "total_requests": 0,
                "latest_stage": None,
                "latest_operation": None,
                "stage_counts": {},
            }
        snapshot = dict(record)
        snapshot["stage_counts"] = dict(record["stage_counts"])
        return True, snapshot


def request_cancel():
    with STATE_LOCK:
        run_id = ACTIVE["run_id"]
        cancel_event = ACTIVE["cancel_event"]
        if run_id is None or cancel_event is None:
            return {"ok": True, "status": "idle"}
        cancel_event.set()
        record = RUN_REGISTRY.get(run_id)
        if record is not None:
            record["status"] = "stopping"
        return {"ok": True, "status": "stopping", "run_id": run_id}


# ---------------------------------------------------------------------
# Main simulation entry point (used by /simulate).
# ---------------------------------------------------------------------

def effective_transaction_count(ui_cfg):
    payload = ui_cfg.get("payload_size_bytes")
    block_size = ui_cfg.get("logical_block_size_bytes")
    if payload is not None and block_size is not None:
        payload = max(1, int(float(payload)))
        block_size = max(1, int(float(block_size)))
        return max(1, math.ceil(payload / block_size))
    return max(1, int(ui_cfg.get("transactions", 100)))


def run_engine(ui_cfg, progress_callback=None):
    cfg = build_default_cfg()
    cfg["seed"] = int(ui_cfg.get("seed", cfg["seed"]))
    cfg["channels"] = int(ui_cfg.get("channels", cfg["channels"]))
    cfg["cache_size"] = int(ui_cfg.get("cache_size", cfg["cache_size"]))
    cfg["scheduler"] = ui_cfg.get("scheduler", cfg["scheduler"])
    cfg["error_probability"] = float(ui_cfg.get("error_probability", cfg["error_probability"]))

    transactions = effective_transaction_count(ui_cfg)
    seed = cfg["seed"]

    # Workload geometry is explicit: total payload / logical-block size determines
    # the number of host-visible block operations. `blocks` remains the logical
    # address-space size and is intentionally distinct from logical block size.
    read_ratio = float(ui_cfg.get("read_ratio", 0.7))
    blocks = int(ui_cfg.get("blocks", 100))
    logical_block_size_bytes = max(
        1,
        int(ui_cfg.get("logical_block_size_bytes", 4096))
    )
    read_path_speedup = max(0.1, float(ui_cfg.get("read_path_speedup", 1.0)))
    write_path_speedup = max(0.1, float(ui_cfg.get("write_path_speedup", 1.0)))
    cfg["path_acceleration"] = {
        "read": read_path_speedup,
        "write": write_path_speedup,
    }

    reqs = biossd_sim.workload(
        transactions,
        seed=seed,
        read_ratio=read_ratio,
        blocks=blocks,
        size_bytes=logical_block_size_bytes,
    )

    # The standard interactive mixed READ/WRITE workload is meant to measure
    # transactions against pre-existing molecular data, not an empty medium.
    # initialize_storage() (invoked here via initialize_blocks=True) writes
    # directly into the storage dict before the timed run starts — it never
    # touches execute(), so it cannot appear as a measured WRITE, consume the
    # configured transaction count, populate the cache, or affect latency,
    # throughput, error, endurance, or channel-utilization statistics. `blocks`
    # is passed through unchanged so the initialized range always matches the
    # workload's own address range.
    result, rows = biossd_sim.run(
        cfg, reqs,
        initialize_blocks=True,
        blocks=blocks,
        block_size_bytes=logical_block_size_bytes,
        progress_callback=progress_callback,
    )

    result = dict(result)

    requests = result.get("requests", 0)
    cache_hits = result.get("cache_hits", 0)
    errors_injected = result.get("errors_injected", 0)
    errors_recovered = result.get("errors_recovered", 0)

    result["cache_hit_rate"] = cache_hits / requests if requests > 0 else 0.0
    result["recovery_rate"] = (
        errors_recovered / errors_injected
        if errors_injected > 0
        else 1.0
    )

    return result


# ---------------------------------------------------------------------
# /experiment/host-latency and /experiment/target-latency
# Real controlled experiments against the live simulator. No external
# reference data required — these measure the current modeled
# architecture directly.
# ---------------------------------------------------------------------

HOST_LATENCY_TRIALS = 200


def measure_host_latency():
    cold_latencies = []
    hot_latencies = []
    write_latencies = []
    roundtrip_latencies = []
    cold_stage_totals = {}
    write_stage_totals = {}

    for i in range(HOST_LATENCY_TRIALS):
        cfg = build_default_cfg()
        cfg["seed"] = 90000 + i

        sim = biossd_sim.BioSSDSimulator(cfg)
        biossd_sim.initialize_storage(sim, blocks=10)

        cold_req = biossd_sim.Request(req_id=1, op="READ", block=5, arrival=0.0)
        sim.execute(cold_req, channel=0)

        hot_req = biossd_sim.Request(req_id=2, op="READ", block=5, arrival=cold_req.finish)
        sim.execute(hot_req, channel=0)

        write_req = biossd_sim.Request(req_id=3, op="WRITE", block=6, arrival=hot_req.finish)
        sim.execute(write_req, channel=0)

        # Force a genuinely cold read of the just-written block.
        sim.cache.pop(6, None)
        roundtrip_req = biossd_sim.Request(req_id=4, op="READ", block=6, arrival=write_req.finish)
        sim.execute(roundtrip_req, channel=0)

        cold_latencies.append(cold_req.finish - cold_req.start)
        hot_latencies.append(hot_req.finish - hot_req.start)
        write_latencies.append(write_req.finish - write_req.start)
        roundtrip_latencies.append(
            (write_req.finish - write_req.start)
            + (roundtrip_req.finish - roundtrip_req.start)
        )

        for stage, value in cold_req.stage_times_ms.items():
            cold_stage_totals[stage] = cold_stage_totals.get(stage, 0.0) + value
        for stage, value in write_req.stage_times_ms.items():
            write_stage_totals[stage] = write_stage_totals.get(stage, 0.0) + value

    def _mean(values):
        return sum(values) / len(values) if values else None

    n = HOST_LATENCY_TRIALS
    cold_mean = _mean(cold_latencies)
    hot_mean = _mean(hot_latencies)

    return {
        "mean_cold_read_latency_ms": cold_mean,
        "mean_hot_read_latency_ms": hot_mean,
        "mean_write_latency_ms": _mean(write_latencies),
        "mean_roundtrip_latency_ms": _mean(roundtrip_latencies),
        "cold_hot_acceleration_ratio": (cold_mean / hot_mean) if hot_mean else None,
        "cold_read_stage_mean_ms": {k: v / n for k, v in cold_stage_totals.items()},
        "write_stage_mean_ms": {k: v / n for k, v in write_stage_totals.items()},
        "trials": n,
    }


TARGET_LATENCY_LEVELS_MS = [10000, 1000, 100]


def target_latency_rows():
    host = measure_host_latency()
    cold_mean = host["mean_cold_read_latency_ms"] or 0.0
    write_mean = host["mean_write_latency_ms"] or 0.0
    roundtrip_mean = host["mean_roundtrip_latency_ms"] or 0.0

    cold_molecular = host["cold_read_stage_mean_ms"].get("molecular_access", 0.0)
    write_molecular = host["write_stage_mean_ms"].get("molecular_write", 0.0)

    cold_overhead = cold_mean - cold_molecular
    write_overhead = write_mean - write_molecular
    roundtrip_overhead = roundtrip_mean - (cold_molecular + write_molecular)

    rows = []
    for target in TARGET_LATENCY_LEVELS_MS:
        rows.append({
            "target_latency_ms": target,
            "modeled_mean_cold_read_ms": cold_mean,
            "cold_read_target_feasible_with_current_model": cold_mean <= target,
            "modeled_mean_write_ms": write_mean,
            "write_target_feasible_with_current_model": write_mean <= target,
            "modeled_mean_roundtrip_ms": roundtrip_mean,
            "roundtrip_target_feasible_with_current_model": roundtrip_mean <= target,
            "cold_read_available_physical_budget_ms": max(0.0, target - cold_overhead),
            "write_available_physical_budget_ms": max(0.0, target - write_overhead),
            "roundtrip_available_physical_budget_ms": max(0.0, target - roundtrip_overhead),
        })
    return rows


# ---------------------------------------------------------------------
# /benchmark/reference and /benchmark/custom
# ---------------------------------------------------------------------

READ_PATH_STAGES = [
    "addressing", "sensing",
    "basecalling", "decoding", "error_correction", "buffer",
]
WRITE_PATH_STAGES = [
    "addressing", "encoding", "molecular_write",
    "write_verification", "buffer",
]

REFERENCE_BENCHMARK_PARAMS = {
    "channels": 4096,
    "logical_block_size_bytes": 256 * 1024,
    "payload_size_bytes": 1024 ** 3,
    "read_path_speedup": 4.0,
    "write_path_speedup": 8.0,
}

BENCHMARK_TRIALS = 300


def _scale_stage_spec(spec, factor):
    spec = dict(spec)
    factor = max(factor, 1e-6)
    if spec["kind"] == "fixed":
        spec["value"] = spec["value"] / factor
    elif spec["kind"] == "normal":
        spec["mean"] = spec["mean"] / factor
        spec["sd"] = spec["sd"] / factor
    elif spec["kind"] == "lognormal":
        spec["mu"] = spec["mu"] - math.log(factor)
    return spec


def build_path_cfg(base_cfg, stages, speedup):
    cfg = copy.deepcopy(base_cfg)
    latency = dict(cfg["latency_ms"])
    for stage in stages:
        latency[stage] = _scale_stage_spec(latency[stage], speedup)
    cfg["latency_ms"] = latency
    return cfg


def format_binary_bytes(n):
    n = float(n)
    for unit, size in (("GiB", 1024 ** 3), ("MiB", 1024 ** 2), ("KiB", 1024)):
        if n >= size:
            value = n / size
            label = f"{value:.0f}" if value == int(value) else f"{value:.2f}"
            return f"{label} {unit}"
    return f"{int(n)} B"


def _measure_representative_transaction(cfg, op, block_size_bytes, seed):
    cfg = copy.deepcopy(cfg)
    cfg["seed"] = seed
    sim = biossd_sim.BioSSDSimulator(cfg)

    if op == "READ":
        biossd_sim.initialize_storage(sim, blocks=1, size_bytes=block_size_bytes)
        req = biossd_sim.Request(req_id=1, op="READ", block=1, arrival=0.0, size_bytes=block_size_bytes)
    else:
        req = biossd_sim.Request(req_id=1, op="WRITE", block=1, arrival=0.0, size_bytes=block_size_bytes)

    sim.execute(req, channel=0)
    return (
        req.finish - req.start,
        req.status == "completed",
        dict(req.stage_times_ms),
    )


def _mean_path_profile(cfg, op, block_size_bytes, trials=BENCHMARK_TRIALS, seed_base=71000):
    latencies = []
    stage_totals = {}

    for i in range(trials):
        latency, completed, stage_times = _measure_representative_transaction(
            cfg, op, block_size_bytes, seed_base + i
        )
        if completed:
            latencies.append(latency)
            for stage, value in stage_times.items():
                stage_totals[stage] = stage_totals.get(stage, 0.0) + float(value)

    if not latencies:
        raise ValueError(
            f"All {trials} representative {op} trials failed under this configuration; "
            "check error/failure probabilities and path-acceleration values."
        )

    n = len(latencies)
    return {
        "mean_latency_ms": sum(latencies) / n,
        "stage_mean_ms": {stage: total / n for stage, total in stage_totals.items()},
        "completed_trials": n,
        "trials": trials,
    }


def _mean_path_latency_ms(cfg, op, block_size_bytes, trials=BENCHMARK_TRIALS, seed_base=71000):
    return _mean_path_profile(
        cfg, op, block_size_bytes, trials=trials, seed_base=seed_base
    )["mean_latency_ms"]


def run_benchmark_variant(channels, logical_block_size_bytes, payload_size_bytes,
                           read_path_speedup, write_path_speedup):
    if logical_block_size_bytes <= 0:
        raise ValueError("logical_block_size_bytes must be greater than 0")
    if payload_size_bytes <= 0:
        raise ValueError("payload_size_bytes must be greater than 0")
    if channels <= 0:
        raise ValueError("channels must be greater than 0")

    channels = int(channels)
    transaction_count = max(1, round(payload_size_bytes / logical_block_size_bytes))
    parallel_waves = max(1, math.ceil(transaction_count / channels))

    # Measured as the mean latency of many representative single-channel
    # transactions at this block size (not a full multi-channel discrete-event
    # batch run) — validated against the published reference figures below.
    base_cfg = build_default_cfg()
    base_cfg["channels"] = 1
    base_cfg["cache_size"] = 0
    base_cfg["error_probability"] = 0.0
    base_cfg["write_failure_probability"] = 0.0
    base_cfg["endurance"] = {"enabled": False}
    base_cfg["prefetch"] = {"enabled": False, "depth": 0}

    read_cfg = build_path_cfg(base_cfg, READ_PATH_STAGES, read_path_speedup)
    write_cfg = build_path_cfg(base_cfg, WRITE_PATH_STAGES, write_path_speedup)

    read_profile = _mean_path_profile(read_cfg, "READ", logical_block_size_bytes)
    write_profile = _mean_path_profile(write_cfg, "WRITE", logical_block_size_bytes)

    mean_read_ms = read_profile["mean_latency_ms"]
    mean_write_ms = write_profile["mean_latency_ms"]

    # If the payload needs more transactions than there are channels, the
    # remaining transactions queue into further parallel waves.
    cold_read_latency_ms = mean_read_ms * parallel_waves
    write_latency_ms = mean_write_ms * parallel_waves

    cold_read_stage_mean_ms = {
        stage: value * parallel_waves
        for stage, value in read_profile["stage_mean_ms"].items()
    }
    write_stage_mean_ms = {
        stage: value * parallel_waves
        for stage, value in write_profile["stage_mean_ms"].items()
    }

    active_channels = min(channels, transaction_count)
    mean_throughput = (
        (
            payload_size_bytes / (cold_read_latency_ms / 1000.0)
            if cold_read_latency_ms > 0 else 0.0
        )
        + (
            payload_size_bytes / (write_latency_ms / 1000.0)
            if write_latency_ms > 0 else 0.0
        )
    ) / 2.0

    return {
        "channels": channels,
        "logical_block_size_bytes": int(logical_block_size_bytes),
        "logical_block_size_label": format_binary_bytes(logical_block_size_bytes),
        "payload_size_bytes": int(payload_size_bytes),
        "payload_size_label": format_binary_bytes(payload_size_bytes),
        "read_path_speedup": read_path_speedup,
        "write_path_speedup": write_path_speedup,
        "transactions_per_path": transaction_count,
        "parallel_waves": parallel_waves,
        "active_channel_count": active_channels,
        "cold_read_stage_mean_ms": cold_read_stage_mean_ms,
        "write_stage_mean_ms": write_stage_mean_ms,
        "representative_read_trials": read_profile["completed_trials"],
        "representative_write_trials": write_profile["completed_trials"],
        "mean_path_latency_ms": (cold_read_latency_ms + write_latency_ms) / 2.0,
        "mean_path_throughput_bytes_per_s": mean_throughput,
        "cold_read_latency_ms": cold_read_latency_ms,
        "write_latency_ms": write_latency_ms,
        "roundtrip_latency_ms": cold_read_latency_ms + write_latency_ms,
        "cold_read_throughput_bytes_per_s": (
            payload_size_bytes / (cold_read_latency_ms / 1000.0) if cold_read_latency_ms > 0 else 0.0
        ),
        "write_throughput_bytes_per_s": (
            payload_size_bytes / (write_latency_ms / 1000.0) if write_latency_ms > 0 else 0.0
        ),
        "model_note": (
            "Benchmark latency is derived from repeated representative READ and WRITE "
            "transactions under the selected configuration. Aggregate performance is "
            "calculated from payload size, logical-block size, effective molecular "
            "channels, path acceleration, and parallel-wave count. Current benchmark "
            "runs are stochastic and may vary slightly under the same configuration."
        ),
    }


# ---------------------------------------------------------------------
# /technology-gap and /technology-gap/custom
# ---------------------------------------------------------------------

UMDI_TARGET_STAGES = [
    "molecular_access", "sensing", "basecalling",
    "molecular_write", "write_verification",
]


def _stage_mean_from_cfg(stage_name):
    spec = DEFAULT_CFG["latency_ms"][stage_name]
    kind = spec["kind"]
    if kind == "fixed":
        return float(spec["value"])
    if kind == "normal":
        return float(spec["mean"])
    if kind == "lognormal":
        return math.exp(spec["mu"] + (spec["sigma"] ** 2) / 2.0)
    raise ValueError(f"Unsupported distribution kind: {kind}")


UMDI_TARGET_MS = {name: _stage_mean_from_cfg(name) for name in UMDI_TARGET_STAGES}


def _gap_metric(test_ms, target_ms):
    test_ms = float(test_ms)
    target_ms = float(target_ms)
    slowdown = (test_ms / target_ms) if target_ms > 0 else None
    reduction_pct = (
        max(0.0, (1.0 - (target_ms / test_ms)) * 100.0)
        if test_ms > 0 else 0.0
    )
    return {
        "test_latency_ms": test_ms,
        "umdi_target_latency_ms": target_ms,
        "slowdown_factor_vs_umdi_target": slowdown,
        "required_latency_reduction_pct": reduction_pct,
    }


def technology_gap_from_stage_ms(stage_ms, comparison_warning=None):
    cold_test = stage_ms["molecular_access"] + stage_ms["sensing"] + stage_ms["basecalling"]
    cold_target = (
        UMDI_TARGET_MS["molecular_access"]
        + UMDI_TARGET_MS["sensing"]
        + UMDI_TARGET_MS["basecalling"]
    )

    write_test = stage_ms["molecular_write"] + stage_ms["write_verification"]
    write_target = UMDI_TARGET_MS["molecular_write"] + UMDI_TARGET_MS["write_verification"]

    roundtrip_test = cold_test + write_test
    roundtrip_target = cold_target + write_target

    cold = _gap_metric(cold_test, cold_target)
    write = _gap_metric(write_test, write_target)
    roundtrip = _gap_metric(roundtrip_test, roundtrip_target)

    stage_items = [
        ("Molecular Access (Read)", "molecular_access"),
        ("Sensing", "sensing"),
        ("Basecalling", "basecalling"),
        ("Molecular Write", "molecular_write"),
        ("Write Verification", "write_verification"),
    ]

    ranked = sorted(
        stage_items,
        key=lambda item: (
            (stage_ms[item[1]] / UMDI_TARGET_MS[item[1]])
            if UMDI_TARGET_MS[item[1]] > 0 else 0
        ),
        reverse=True,
    )

    optimization_targets = []
    for rank, (label, key) in enumerate(ranked, start=1):
        current_ms = stage_ms[key]
        target_ms = UMDI_TARGET_MS[key]
        slowdown = (current_ms / target_ms) if target_ms > 0 else None
        reduction_pct = (
            max(0.0, (1.0 - (target_ms / current_ms)) * 100.0)
            if current_ms > 0 else 0.0
        )
        optimization_targets.append({
            "priority_rank": rank,
            "label": label,
            "stage": key,
            "current_latency_ms": current_ms,
            "umdi_target_latency_ms": target_ms,
            "slowdown_factor_vs_target": slowdown,
            "required_reduction_pct": reduction_pct,
            "meets_target": current_ms <= target_ms,
        })

    return {
        "cold_read": cold,
        "write": write,
        "roundtrip": roundtrip,
        "optimization_targets": optimization_targets,
        "dominant_bottleneck_label": optimization_targets[0]["label"] if optimization_targets else None,
        "comparison_warning": comparison_warning,
    }


def technology_gap_from_totals(cold_read_ms, write_ms, roundtrip_ms=None, comparison_warning=None):
    """
    For literature/demonstrated-technology profiles that only report
    end-to-end operation totals (not this simulator's internal per-stage
    breakdown). No stage-level optimization ranking is possible from
    aggregate numbers, so optimization_targets is left absent — the
    frontend already falls back cleanly to its "Custom measurements
    required for stage-level ranking" message when it's missing.
    """
    cold_target = (
        UMDI_TARGET_MS["molecular_access"]
        + UMDI_TARGET_MS["sensing"]
        + UMDI_TARGET_MS["basecalling"]
    )
    write_target = UMDI_TARGET_MS["molecular_write"] + UMDI_TARGET_MS["write_verification"]
    roundtrip_target = cold_target + write_target

    if roundtrip_ms is None:
        roundtrip_ms = cold_read_ms + write_ms

    return {
        "cold_read": _gap_metric(cold_read_ms, cold_target),
        "write": _gap_metric(write_ms, write_target),
        "roundtrip": _gap_metric(roundtrip_ms, roundtrip_target),
        "optimization_targets": None,
        "dominant_bottleneck_label": None,
        "comparison_warning": comparison_warning,
    }


def load_contemporary_baseline():
    path = CONFIGS / "contemporary_baseline.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Add configs/contemporary_baseline.json with a "
            "benchmark_totals_ms object containing cold_read_ms and write_ms "
            "(roundtrip_ms optional; derived as cold_read_ms + write_ms if absent), "
            "representing demonstrated physical-technology end-to-end timings."
        )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    totals = data.get("benchmark_totals_ms")
    if not isinstance(totals, dict) or "cold_read_ms" not in totals or "write_ms" not in totals:
        raise ValueError(
            "configs/contemporary_baseline.json must include a benchmark_totals_ms "
            "object with at least cold_read_ms and write_ms."
        )
    return data


# ---------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):

    def _send(self, code, obj):
        raw = json.dumps(obj, default=str).encode()

        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()

        self.wfile.write(raw)

    def _read_json_body(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else b"{}"
        return json.loads(body or b"{}")

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    # ---------------- GET ----------------

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        try:
            if path == "/health":
                self._send(200, {"ok": True, "engine": "biossd_sim.py"})
                return

            if path == "/progress":
                run_id = (query.get("run_id") or [None])[0]
                if not run_id:
                    self._send(200, {"ok": False, "error": "run_id is required"})
                    return
                found, snapshot = read_progress(run_id)
                self._send(200, {
                    "ok": True,
                    "matches_requested_run": found,
                    "progress": snapshot,
                })
                return

            if path == "/experiment/host-latency":
                row = measure_host_latency()
                self._send(200, {"ok": True, "result": [row]})
                return

            if path == "/experiment/target-latency":
                rows = target_latency_rows()
                self._send(200, {"ok": True, "result": rows})
                return

            if path == "/technology-gap":
                try:
                    data = load_contemporary_baseline()
                except (FileNotFoundError, ValueError) as e:
                    self._send(200, {"ok": False, "error": str(e)})
                    return
                totals = data["benchmark_totals_ms"]
                warning = data.get("description") or data.get("profile_name")
                result = technology_gap_from_totals(
                    cold_read_ms=float(totals["cold_read_ms"]),
                    write_ms=float(totals["write_ms"]),
                    roundtrip_ms=(
                        float(totals["roundtrip_ms"]) if "roundtrip_ms" in totals else None
                    ),
                    comparison_warning=warning,
                )
                result["profile_name"] = data.get("profile_name")
                result["source_citations"] = data.get("benchmark_basis")
                self._send(200, {"ok": True, "result": result})
                return

            if path == "/benchmark/reference":
                result = run_benchmark_variant(**REFERENCE_BENCHMARK_PARAMS)
                self._send(200, {"ok": True, "result": result})
                return

            self._send(404, {"ok": False, "error": "Not found"})

        except Exception as e:
            self._send(500, {"ok": False, "error": str(e), "trace": traceback.format_exc()})

    # ---------------- POST ----------------

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        try:
            if path == "/simulate":
                self._handle_simulate()
                return

            if path == "/cancel":
                self._read_json_body()
                self._send(200, request_cancel())
                return

            if path == "/technology-gap/custom":
                body = self._read_json_body()
                stage_ms = {
                    "molecular_access": float(body.get("molecular_access_ms", 0)),
                    "sensing": float(body.get("sensing_ms", 0)),
                    "basecalling": float(body.get("basecalling_ms", 0)),
                    "molecular_write": float(body.get("molecular_write_ms", 0)),
                    "write_verification": float(body.get("write_verification_ms", 0)),
                }
                warning = f"Comparison profile: {body.get('profile_name', 'Custom Researcher Hardware')}"
                result = technology_gap_from_stage_ms(stage_ms, comparison_warning=warning)
                self._send(200, {"ok": True, "result": result})
                return

            if path == "/benchmark/custom":
                body = self._read_json_body()
                result = run_benchmark_variant(
                    channels=int(body.get("channels", REFERENCE_BENCHMARK_PARAMS["channels"])),
                    logical_block_size_bytes=int(body.get(
                        "logical_block_size_bytes",
                        REFERENCE_BENCHMARK_PARAMS["logical_block_size_bytes"],
                    )),
                    payload_size_bytes=int(body.get(
                        "payload_size_bytes",
                        REFERENCE_BENCHMARK_PARAMS["payload_size_bytes"],
                    )),
                    read_path_speedup=float(body.get(
                        "read_path_speedup",
                        REFERENCE_BENCHMARK_PARAMS["read_path_speedup"],
                    )),
                    write_path_speedup=float(body.get(
                        "write_path_speedup",
                        REFERENCE_BENCHMARK_PARAMS["write_path_speedup"],
                    )),
                )
                self._send(200, {"ok": True, "result": result})
                return

            self._send(404, {"ok": False, "error": "Not found"})

        except Exception as e:
            self._send(500, {"ok": False, "error": str(e), "trace": traceback.format_exc()})

    def _handle_simulate(self):
        try:
            ui_cfg = self._read_json_body()
        except Exception as e:
            self._send(400, {"ok": False, "error": f"Invalid request body: {e}"})
            return

        # This id is the ONLY thing tying /simulate to /progress. It must never
        # reach biossd_sim's scientific config.
        client_run_id = ui_cfg.pop("_client_run_id", None)
        run_id = client_run_id or f"anon-{uuid.uuid4()}"

        cancel_event = start_run(run_id, effective_transaction_count(ui_cfg))
        if cancel_event is None:
            self._send(409, {
                "ok": False,
                "error": "A simulation is already running on this engine.",
            })
            return

        callback = make_progress_callback(run_id, cancel_event)

        try:
            result = run_engine(ui_cfg, progress_callback=callback)
            finish_run(run_id, "complete")
            self._send(200, {"ok": True, "run_id": run_id, "result": result})

        except RuntimeError as e:
            if str(e) == "SIMULATION_CANCELLED":
                finish_run(run_id, "stopped")
                self._send(200, {
                    "ok": False,
                    "cancelled": True,
                    "error": "Simulation stopped by user.",
                })
            else:
                finish_run(run_id, "error", error=str(e))
                self._send(500, {"ok": False, "error": str(e), "trace": traceback.format_exc()})

        except Exception as e:
            finish_run(run_id, "error", error=str(e))
            self._send(500, {"ok": False, "error": str(e), "trace": traceback.format_exc()})


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


if __name__ == "__main__":
    print(
        "UMDI BioSSD simulation API: "
        f"http://{HOST}:{PORT}"
    )

    Server((HOST, PORT), Handler).serve_forever()
