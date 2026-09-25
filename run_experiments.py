import csv
import json
import math
import heapq
import statistics

import matplotlib.pyplot as plt
from pathlib import Path

import src.biossd_sim as biossd_sim

from src.biossd_sim import (
    BioSSDSimulator,
    READ_STAGES,
    WRITE_STAGES,
    Request,
    initialize_storage,
    run,
    workload,
)


ROOT = Path(__file__).parent

BASE = json.loads(
    (ROOT / "configs" / "baseline.json").read_text()
)

OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

CONTEMPORARY_BASELINE_PATH = (
    ROOT / "configs" / "contemporary_baseline.json"
)


# ============================================================
# RESEARCHER-CONFIGURABLE EXPERIMENT SWEEPS
# ============================================================

CHANNEL_VALUES = [
    1, 2, 4, 8, 16, 32, 64, 120
]

SCHEDULERS = [
    "fifo",
    "molecular_aware"
]

CACHE_VALUES = [
    0, 4, 8, 16, 32, 64, 128
]

ERROR_VALUES = [
    0.00,
    0.01,
    0.03,
    0.05,
    0.10,
    0.20
]

QUEUE_VALUES = [
    50,
    100,
    250,
    500,
    1000,
    2000,
    4000
]

REWRITE_VALUES = [
    1,
    10,
    50,
    100,
    250,
    500,
    1000
]

PREFETCH_DEPTH_VALUES = [
    0,
    1,
    2,
    4
]

HOST_LATENCY_TRIALS = 250

TARGET_LATENCY_MS = [
    10000,
    5000,
    2000,
    1000,
    500,
    100
]

# Experiment 8E: user payload size × logical block size scaling.
PAYLOAD_SIZE_VALUES = [
    5,
    1024,
    4096,
    16 * 1024,
    64 * 1024,
    256 * 1024,
    1024 * 1024,
    5 * 1024 * 1024,
    10 * 1024 * 1024,
]

LOGICAL_BLOCK_SIZE_VALUES = [
    512,
    1024,
    4096,
    16 * 1024,
    64 * 1024,
]


# Large-file optimization search used to derive the paper's optimized
# UMDI target model. The search remains computational and does not imply
# that current molecular hardware already achieves the selected values.
OPTIMIZATION_PAYLOAD_BYTES = 1024 * 1024 * 1024  # 1 GiB (1,073,741,824 bytes)

OPTIMIZATION_CHANNEL_VALUES = [
    4,
    8,
    16,
    32,
    64,
    128,
]

OPTIMIZATION_BLOCK_SIZE_VALUES = [
    4 * 1024,
    16 * 1024,
    64 * 1024,
    256 * 1024,
    1024 * 1024,
]

OPTIMIZATION_WRITE_SPEEDUP_VALUES = [
    1,
    2,
    4,
    8,
    16,
    32,
    64,
]


SSD_CLASS_TARGET_BYTES_PER_S = 500_000_000

SSD_TARGET_CHANNEL_VALUES = [
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
]

SSD_TARGET_BLOCK_SIZE_VALUES = [
    64 * 1024,
    256 * 1024,
    1024 * 1024,
    4 * 1024 * 1024,
    16 * 1024 * 1024,
]

SSD_TARGET_READ_SPEEDUP_VALUES = [
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
]

SSD_TARGET_WRITE_SPEEDUP_VALUES = [
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
]


# ============================================================
# HELPERS
# ============================================================

def clone():
    return json.loads(
        json.dumps(BASE)
    )


def save(name, data):
    json_path = OUT / f"{name}.json"
    csv_path = OUT / f"{name}.csv"

    json_path.write_text(
        json.dumps(
            data,
            indent=2
        )
    )

    if not data:
        return

    normalized = []

    for row in data:
        clean = {}

        for key, value in row.items():
            if isinstance(value, (dict, list)):
                clean[key] = json.dumps(value)
            else:
                clean[key] = value

        normalized.append(clean)

    fieldnames = []
    seen = set()

    for row in normalized:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with open(
        csv_path,
        "w",
        newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(normalized)


def error_recovery_rate(summary):
    injected = summary.get(
        "errors_injected",
        0
    )

    recovered = summary.get(
        "errors_recovered",
        0
    )

    if injected <= 0:
        return None

    return recovered / injected



def percentile(values, fraction):
    if not values:
        return None

    ordered = sorted(values)
    index = max(
        0,
        math.ceil(
            fraction * len(ordered)
        ) - 1
    )
    return ordered[index]


def stats(values):
    values = [
        float(value)
        for value in values
        if value is not None
    ]

    if not values:
        return {
            "count": 0,
            "mean_ms": None,
            "median_ms": None,
            "p95_ms": None,
            "min_ms": None,
            "max_ms": None,
        }

    return {
        "count": len(values),
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "p95_ms": percentile(values, 0.95),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def stage_means(rows):
    totals = {}
    counts = {}

    for row in rows:
        for stage, value in row.get(
            "stage_times_ms",
            {}
        ).items():
            totals[stage] = totals.get(stage, 0.0) + float(value)
            counts[stage] = counts.get(stage, 0) + 1

    return {
        stage: totals[stage] / counts[stage]
        for stage in totals
    }


def completed_only(rows):
    return [
        row
        for row in rows
        if row["status"] == "completed"
    ]


def execute_single(simulator, request):
    rows = simulator.run([request])

    if not rows:
        raise RuntimeError(
            "Simulator returned no row for request"
        )

    return rows[-1]


def clean_latency_cfg():
    cfg = clone()

    cfg["prefetch"] = {
        "enabled": False,
        "depth": 0
    }

    cfg["error_probability"] = 0.0
    cfg["write_failure_probability"] = 0.0

    cfg["endurance"] = {
        **cfg.get("endurance", {}),
        "enabled": False
    }

    return cfg


def experiment8_host_visible_latency():
    """
    8A: forced cold molecular READ
    8B: hot electronic READ of the same block
    8C: verified molecular WRITE
    8D: WRITE -> forced cold READ round trip
    """

    cfg = clean_latency_cfg()

    cold_rows = []
    hot_rows = []
    write_rows = []
    roundtrip_rows = []

    for trial in range(HOST_LATENCY_TRIALS):
        block = (trial % 100) + 1

        read_cfg = json.loads(json.dumps(cfg))
        read_cfg["seed"] = int(cfg["seed"]) + (trial * 10) + 1

        sim = BioSSDSimulator(read_cfg)
        initialize_storage(sim, blocks=100)
        sim.cache.clear()

        cold = execute_single(
            sim,
            Request(
                req_id=1,
                op="READ",
                block=block,
                arrival=0.0,
                size_bytes=4096,
                locality=block % 16,
            )
        )
        cold_rows.append(cold)

        hot = execute_single(
            sim,
            Request(
                req_id=2,
                op="READ",
                block=block,
                arrival=sim.now,
                size_bytes=4096,
                locality=block % 16,
            )
        )
        hot_rows.append(hot)

        write_cfg = json.loads(json.dumps(cfg))
        write_cfg["seed"] = int(cfg["seed"]) + (trial * 10) + 2

        sim_write = BioSSDSimulator(write_cfg)
        write_block = 1000 + trial

        write = execute_single(
            sim_write,
            Request(
                req_id=1,
                op="WRITE",
                block=write_block,
                arrival=0.0,
                size_bytes=4096,
                locality=write_block % 16,
            )
        )
        write_rows.append(write)

        rt_cfg = json.loads(json.dumps(cfg))
        rt_cfg["seed"] = int(cfg["seed"]) + (trial * 10) + 3

        sim_rt = BioSSDSimulator(rt_cfg)
        rt_block = 2000 + trial

        write_rt = execute_single(
            sim_rt,
            Request(
                req_id=1,
                op="WRITE",
                block=rt_block,
                arrival=0.0,
                size_bytes=4096,
                locality=rt_block % 16,
            )
        )

        # Force the READ to cross the molecular boundary.
        sim_rt.cache.pop(rt_block, None)

        read_rt = execute_single(
            sim_rt,
            Request(
                req_id=2,
                op="READ",
                block=rt_block,
                arrival=sim_rt.now,
                size_bytes=4096,
                locality=rt_block % 16,
            )
        )

        roundtrip_rows.append({
            "trial": trial + 1,
            "write_status": write_rt["status"],
            "read_status": read_rt["status"],
            "write_latency_ms": write_rt["latency_ms"],
            "cold_read_latency_ms": read_rt["latency_ms"],
            "roundtrip_latency_ms": (
                write_rt["latency_ms"]
                + read_rt["latency_ms"]
            ),
            "write_stage_times_ms":
                write_rt["stage_times_ms"],
            "read_stage_times_ms":
                read_rt["stage_times_ms"],
        })

    cold_completed = completed_only(cold_rows)
    hot_completed = completed_only(hot_rows)
    write_completed = completed_only(write_rows)

    successful_roundtrips = [
        row
        for row in roundtrip_rows
        if (
            row["write_status"] == "completed"
            and row["read_status"] == "completed"
        )
    ]

    cold_stats = stats([
        row["latency_ms"]
        for row in cold_completed
    ])

    hot_stats = stats([
        row["latency_ms"]
        for row in hot_completed
    ])

    write_stats = stats([
        row["latency_ms"]
        for row in write_completed
    ])

    roundtrip_stats = stats([
        row["roundtrip_latency_ms"]
        for row in successful_roundtrips
    ])

    acceleration = (
        cold_stats["mean_ms"]
        / hot_stats["mean_ms"]
        if hot_stats["mean_ms"] not in (None, 0)
        else None
    )

    result = {
        "trials": HOST_LATENCY_TRIALS,

        "cold_read_success_rate":
            len(cold_completed) / HOST_LATENCY_TRIALS,

        "hot_read_success_rate":
            len(hot_completed) / HOST_LATENCY_TRIALS,

        "write_success_rate":
            len(write_completed) / HOST_LATENCY_TRIALS,

        "roundtrip_success_rate":
            len(successful_roundtrips) / HOST_LATENCY_TRIALS,

        "mean_cold_read_latency_ms":
            cold_stats["mean_ms"],
        "median_cold_read_latency_ms":
            cold_stats["median_ms"],
        "p95_cold_read_latency_ms":
            cold_stats["p95_ms"],
        "min_cold_read_latency_ms":
            cold_stats["min_ms"],
        "max_cold_read_latency_ms":
            cold_stats["max_ms"],

        "mean_hot_read_latency_ms":
            hot_stats["mean_ms"],
        "median_hot_read_latency_ms":
            hot_stats["median_ms"],
        "p95_hot_read_latency_ms":
            hot_stats["p95_ms"],
        "min_hot_read_latency_ms":
            hot_stats["min_ms"],
        "max_hot_read_latency_ms":
            hot_stats["max_ms"],

        "mean_write_latency_ms":
            write_stats["mean_ms"],
        "median_write_latency_ms":
            write_stats["median_ms"],
        "p95_write_latency_ms":
            write_stats["p95_ms"],
        "min_write_latency_ms":
            write_stats["min_ms"],
        "max_write_latency_ms":
            write_stats["max_ms"],

        "mean_roundtrip_latency_ms":
            roundtrip_stats["mean_ms"],
        "median_roundtrip_latency_ms":
            roundtrip_stats["median_ms"],
        "p95_roundtrip_latency_ms":
            roundtrip_stats["p95_ms"],
        "min_roundtrip_latency_ms":
            roundtrip_stats["min_ms"],
        "max_roundtrip_latency_ms":
            roundtrip_stats["max_ms"],

        "cold_hot_acceleration_ratio":
            acceleration,

        "cold_read_stage_mean_ms":
            stage_means(cold_completed),

        "hot_read_stage_mean_ms":
            stage_means(hot_completed),

        "write_stage_mean_ms":
            stage_means(write_completed),

        "hot_cache_hits":
            sum(
                1
                for row in hot_completed
                if row.get("cache_hit")
            ),
    }

    return (
        [result],
        cold_rows,
        hot_rows,
        write_rows,
        roundtrip_rows,
    )



def human_bytes(value):
    """Format byte counts using IEC binary prefixes for binary-sized parameters."""
    value = int(value)
    if value < 1024:
        return f"{value} B"
    if value < 1024 ** 2:
        return f"{value / 1024:g} KiB"
    if value < 1024 ** 3:
        return f"{value / (1024 ** 2):g} MiB"
    if value < 1024 ** 4:
        return f"{value / (1024 ** 3):g} GiB"
    return f"{value / (1024 ** 4):g} TiB"


def split_payload_into_blocks(
    payload_size_bytes,
    logical_block_size_bytes,
):
    payload_size_bytes = max(1, int(payload_size_bytes))
    logical_block_size_bytes = max(
        1,
        int(logical_block_size_bytes)
    )

    full_blocks, remainder = divmod(
        payload_size_bytes,
        logical_block_size_bytes
    )

    sizes = [
        logical_block_size_bytes
        for _ in range(full_blocks)
    ]

    if remainder:
        sizes.append(remainder)

    return sizes or [payload_size_bytes]


def build_file_requests(
    operation,
    block_sizes,
    start_block,
):
    return [
        Request(
            req_id=index + 1,
            op=operation,
            block=start_block + index,
            arrival=0.0,
            size_bytes=size_bytes,
            locality=(start_block + index) % 16,
        )
        for index, size_bytes in enumerate(
            block_sizes
        )
    ]


def initialize_file_storage(
    simulator,
    block_sizes,
    start_block,
):
    for index, size_bytes in enumerate(
        block_sizes
    ):
        block = start_block + index
        simulator.storage[block] = {
            "addr": simulator.addr(block),
            "data": f"DATA-{block}",
            "size_bytes": size_bytes,
            "rewrites": 0,
            "state": "verified",
        }


def file_completion_latency_ms(rows):
    if not rows:
        return 0.0
    return max(
        float(row["finish"])
        for row in rows
    ) - min(
        float(row["arrival"])
        for row in rows
    )


def run_file_operation(
    cfg,
    operation,
    payload_size_bytes,
    logical_block_size_bytes,
    start_block,
    hot=False,
):
    """
    Efficient isolated payload simulation.

    All logical blocks are available to the controller at time zero.
    Each block is assigned to the earliest available molecular channel,
    preserving the configured channel count without repeatedly sorting a
    large general-purpose queue. This is appropriate for the isolated
    file-scaling experiment because every request has the same operation
    class and no competing background workload.
    """
    block_sizes = split_payload_into_blocks(
        payload_size_bytes,
        logical_block_size_bytes
    )

    simulator = BioSSDSimulator(
        json.loads(
            json.dumps(cfg)
        )
    )

    channel_heap = [
        (0.0, channel)
        for channel in range(
            int(cfg["channels"])
        )
    ]

    heapq.heapify(channel_heap)

    molecular_reads = 0
    molecular_writes = 0

    for index, size_bytes in enumerate(
        block_sizes
    ):
        available_at, channel = heapq.heappop(
            channel_heap
        )

        request = Request(
            req_id=index + 1,
            op=operation,
            block=start_block + index,
            arrival=0.0,
            size_bytes=size_bytes,
            locality=(start_block + index) % 16,
        )

        if hot:
            service_ms = float(
                cfg["cache_latency_ms"]
            )
        else:
            service_ms = 0.0

            stages = (
                READ_STAGES
                if operation == "READ"
                else WRITE_STAGES
            )

            for stage in stages:
                value = simulator.sample(
                    cfg["latency_ms"][stage]
                )

                value *= (
                    simulator.stage_payload_factor(
                        request,
                        stage
                    )
                )

                service_ms += value

            if operation == "READ":
                molecular_reads += 1
            else:
                molecular_writes += 1

        finish = available_at + service_ms

        heapq.heappush(
            channel_heap,
            (finish, channel)
        )

    latency_ms = max(
        time
        for time, _channel
        in channel_heap
    )

    throughput = (
        payload_size_bytes
        / (latency_ms / 1000.0)
        if latency_ms > 0
        else 0.0
    )

    return {
        "latency_ms": latency_ms,
        "throughput_bytes_per_s": throughput,
        "success_rate": 1.0,
        "logical_blocks": len(block_sizes),
        "molecular_reads": molecular_reads,
        "molecular_writes": molecular_writes,
    }


def run_file_roundtrip(
    cfg,
    payload_size_bytes,
    logical_block_size_bytes,
    start_block,
):
    """
    Full file WRITE followed by a forced cold molecular READ.

    The round-trip value is the sum of independently simulated complete
    WRITE and cold-READ file latencies.
    """
    write = run_file_operation(
        cfg,
        "WRITE",
        payload_size_bytes,
        logical_block_size_bytes,
        start_block,
        hot=False,
    )

    cold = run_file_operation(
        cfg,
        "READ",
        payload_size_bytes,
        logical_block_size_bytes,
        start_block + 100000,
        hot=False,
    )

    return {
        "write_latency_ms":
            write["latency_ms"],
        "cold_read_latency_ms":
            cold["latency_ms"],
        "roundtrip_latency_ms":
            write["latency_ms"]
            + cold["latency_ms"],
        "logical_blocks":
            write["logical_blocks"],
    }


def experiment8e_payload_block_scaling():
    """
    Simulate real payload-size effects from 5 bytes through 10 MiB
    across multiple logical-block granularities.

    This is the first UMDI experiment in which request.size_bytes
    directly changes molecular processing time.
    """
    cfg = clean_latency_cfg()

    cfg["payload_scaling"] = {
        **cfg.get("payload_scaling", {}),
        "enabled": True,
        "reference_block_size_bytes": 4096,
    }

    results = []
    condition = 0

    for block_size in LOGICAL_BLOCK_SIZE_VALUES:
        for payload_size in PAYLOAD_SIZE_VALUES:
            condition += 1
            base_block = (
                1000000
                + condition * 50000
            )

            cold = run_file_operation(
                cfg,
                "READ",
                payload_size,
                block_size,
                base_block,
                hot=False,
            )

            hot = run_file_operation(
                cfg,
                "READ",
                payload_size,
                block_size,
                base_block + 10000,
                hot=True,
            )

            write = run_file_operation(
                cfg,
                "WRITE",
                payload_size,
                block_size,
                base_block + 20000,
                hot=False,
            )

            roundtrip = run_file_roundtrip(
                cfg,
                payload_size,
                block_size,
                base_block + 30000,
            )

            results.append({
                "payload_size_bytes":
                    payload_size,
                "payload_size_label":
                    human_bytes(payload_size),
                "logical_block_size_bytes":
                    block_size,
                "logical_block_size_label":
                    human_bytes(block_size),
                "logical_blocks":
                    cold["logical_blocks"],
                "cold_read_latency_ms":
                    cold["latency_ms"],
                "hot_read_latency_ms":
                    hot["latency_ms"],
                "write_latency_ms":
                    write["latency_ms"],
                "roundtrip_latency_ms":
                    roundtrip[
                        "roundtrip_latency_ms"
                    ],
                "cold_read_throughput_bytes_per_s":
                    cold[
                        "throughput_bytes_per_s"
                    ],
                "hot_read_throughput_bytes_per_s":
                    hot[
                        "throughput_bytes_per_s"
                    ],
                "write_throughput_bytes_per_s":
                    write[
                        "throughput_bytes_per_s"
                    ],
                "cold_read_success_rate":
                    cold["success_rate"],
                "hot_read_success_rate":
                    hot["success_rate"],
                "write_success_rate":
                    write["success_rate"],
                "cold_molecular_reads":
                    cold["molecular_reads"],
                "hot_molecular_reads":
                    hot["molecular_reads"],
                "molecular_writes":
                    write["molecular_writes"],
            })

    return results




def scaled_write_cfg(
    base_cfg,
    write_speedup,
):
    """
    Return a copy of the baseline target configuration with only the
    molecular-write stage accelerated.

    A write_speedup of 8 means the molecular_write stage latency is divided
    by 8. Controller stages are left untouched.
    """
    cfg = json.loads(
        json.dumps(base_cfg)
    )

    speedup = max(
        1.0,
        float(write_speedup)
    )

    spec = cfg[
        "latency_ms"
    ][
        "molecular_write"
    ]

    if spec["kind"] == "normal":
        spec["mean"] = (
            float(spec["mean"])
            / speedup
        )
        spec["sd"] = (
            float(spec.get("sd", 0.0))
            / speedup
        )

    elif spec["kind"] == "lognormal":
        # Scaling a lognormal random variable by 1/speedup is equivalent
        # to subtracting ln(speedup) from mu.
        spec["mu"] = (
            float(spec["mu"])
            - math.log(speedup)
        )

    elif spec["kind"] == "fixed":
        spec["value"] = (
            float(spec["value"])
            / speedup
        )

    elif spec["kind"] == "uniform":
        spec["low"] = (
            float(spec["low"])
            / speedup
        )
        spec["high"] = (
            float(spec["high"])
            / speedup
        )

    return cfg


def expected_stage_latency_ms(
    cfg,
    stage,
    size_bytes,
):
    """
    Deterministic expected latency for optimization search.

    Uses the mean of the configured distribution and the same payload-scaling
    rule as the simulator. This avoids simulating hundreds of thousands of
    individual blocks merely to rank hardware target configurations.
    """
    spec = cfg["latency_ms"][stage]
    kind = spec["kind"]

    if kind == "fixed":
        value = float(spec["value"])

    elif kind == "normal":
        value = float(spec["mean"])

    elif kind == "lognormal":
        mu = float(spec["mu"])
        sigma = float(spec["sigma"])
        value = math.exp(
            mu + (sigma * sigma) / 2.0
        )

    elif kind == "uniform":
        value = (
            float(spec["low"])
            + float(spec["high"])
        ) / 2.0

    else:
        raise ValueError(
            f"Unsupported latency distribution: {kind}"
        )

    if stage in biossd_sim.PAYLOAD_SCALED_STAGES:
        reference = int(
            cfg.get(
                "payload_scaling",
                {}
            ).get(
                "reference_block_size_bytes",
                4096
            )
        )

        value *= (
            float(size_bytes)
            / float(reference)
        )

    return value


def analytical_file_latency_ms(
    cfg,
    operation,
    payload_size_bytes,
    logical_block_size_bytes,
):
    block_sizes = split_payload_into_blocks(
        payload_size_bytes,
        logical_block_size_bytes
    )

    stages = (
        READ_STAGES
        if operation == "READ"
        else WRITE_STAGES
    )

    channels = max(
        1,
        int(cfg["channels"])
    )

    # Full blocks are identical. Use expected service time and distribute
    # them evenly across channels. The final partial block, if present, is
    # handled separately.
    full_size = int(
        logical_block_size_bytes
    )

    full_blocks, remainder = divmod(
        int(payload_size_bytes),
        full_size
    )

    full_service_ms = sum(
        expected_stage_latency_ms(
            cfg,
            stage,
            full_size
        )
        for stage in stages
    )

    per_channel_full = (
        full_blocks // channels
    )

    extra_full = (
        full_blocks % channels
    )

    channel_times = [
        per_channel_full
        * full_service_ms
        for _ in range(channels)
    ]

    for index in range(extra_full):
        channel_times[index] += (
            full_service_ms
        )

    if remainder:
        partial_service_ms = sum(
            expected_stage_latency_ms(
                cfg,
                stage,
                remainder
            )
            for stage in stages
        )

        fastest_channel = min(
            range(channels),
            key=lambda index:
                channel_times[index]
        )

        channel_times[
            fastest_channel
        ] += partial_service_ms

    latency_ms = max(
        channel_times
        or [0.0]
    )

    return {
        "latency_ms": latency_ms,
        "logical_blocks": len(block_sizes),
        "throughput_bytes_per_s": (
            payload_size_bytes
            / (latency_ms / 1000.0)
            if latency_ms > 0
            else 0.0
        ),
    }


def experiment8f_large_file_optimization():
    """
    Search UMDI target configurations for a 1 GiB payload.

    Variables:
      - molecular channel count
      - logical block size
      - molecular-write stage speedup

    Uses deterministic expected stage latency for ranking. The rest of the
    experiment suite retains stochastic simulation.
    """
    base_cfg = clean_latency_cfg()

    results = []

    for channels in OPTIMIZATION_CHANNEL_VALUES:
        for block_size in OPTIMIZATION_BLOCK_SIZE_VALUES:
            for write_speedup in OPTIMIZATION_WRITE_SPEEDUP_VALUES:

                cfg = scaled_write_cfg(
                    base_cfg,
                    write_speedup
                )

                cfg["channels"] = channels

                write = analytical_file_latency_ms(
                    cfg,
                    "WRITE",
                    OPTIMIZATION_PAYLOAD_BYTES,
                    block_size,
                )

                cold = analytical_file_latency_ms(
                    cfg,
                    "READ",
                    OPTIMIZATION_PAYLOAD_BYTES,
                    block_size,
                )

                results.append({
                    "payload_size_bytes":
                        OPTIMIZATION_PAYLOAD_BYTES,
                    "payload_size_label":
                        human_bytes(
                            OPTIMIZATION_PAYLOAD_BYTES
                        ),
                    "channels":
                        channels,
                    "logical_block_size_bytes":
                        block_size,
                    "logical_block_size_label":
                        human_bytes(
                            block_size
                        ),
                    "molecular_write_speedup":
                        write_speedup,
                    "logical_blocks":
                        write["logical_blocks"],
                    "write_latency_ms":
                        write["latency_ms"],
                    "cold_read_latency_ms":
                        cold["latency_ms"],
                    "roundtrip_latency_ms":
                        write["latency_ms"]
                        + cold["latency_ms"],
                    "write_throughput_bytes_per_s":
                        write[
                            "throughput_bytes_per_s"
                        ],
                    "cold_read_throughput_bytes_per_s":
                        cold[
                            "throughput_bytes_per_s"
                        ],
                })

    results.sort(
        key=lambda row:
            row["write_latency_ms"]
    )

    for rank, row in enumerate(
        results,
        start=1
    ):
        row["write_rank"] = rank

    best = results[0]

    return results, {
        "profile_name":
            "Optimized UMDI Large-File Target Model",
        "profile_type":
            "computational_target",
        "optimization_payload_bytes":
            OPTIMIZATION_PAYLOAD_BYTES,
        "optimization_payload_label":
            human_bytes(
                OPTIMIZATION_PAYLOAD_BYTES
            ),
        "objective":
            "Minimum modeled 1 GiB WRITE latency",
        "channels":
            best["channels"],
        "logical_block_size_bytes":
            best[
                "logical_block_size_bytes"
            ],
        "logical_block_size_label":
            best[
                "logical_block_size_label"
            ],
        "molecular_write_speedup":
            best[
                "molecular_write_speedup"
            ],
        "logical_blocks":
            best["logical_blocks"],
        "modeled_write_latency_ms":
            best["write_latency_ms"],
        "modeled_cold_read_latency_ms":
            best["cold_read_latency_ms"],
        "modeled_roundtrip_latency_ms":
            best["roundtrip_latency_ms"],
        "write_throughput_bytes_per_s":
            best[
                "write_throughput_bytes_per_s"
            ],
        "cold_read_throughput_bytes_per_s":
            best[
                "cold_read_throughput_bytes_per_s"
            ],
        "interpretation":
            "This profile is the fastest configuration found within the "
            "defined computational search space. It is an engineering target, "
            "not a claim of present physical molecular-storage performance.",
    }



def write_cfg_with_stage_speedups(
    base_cfg,
    speedups,
):
    cfg = json.loads(
        json.dumps(base_cfg)
    )

    for stage, speedup in speedups.items():
        speedup = max(
            1.0,
            float(speedup)
        )

        spec = cfg["latency_ms"][stage]

        if spec["kind"] == "normal":
            spec["mean"] = (
                float(spec["mean"])
                / speedup
            )
            spec["sd"] = (
                float(spec.get("sd", 0.0))
                / speedup
            )

        elif spec["kind"] == "lognormal":
            spec["mu"] = (
                float(spec["mu"])
                - math.log(speedup)
            )

        elif spec["kind"] == "fixed":
            spec["value"] = (
                float(spec["value"])
                / speedup
            )

        elif spec["kind"] == "uniform":
            spec["low"] = (
                float(spec["low"])
                / speedup
            )
            spec["high"] = (
                float(spec["high"])
                / speedup
            )

    return cfg


def unified_ssd_target_cfg(
    base_cfg,
    read_speedup,
    write_speedup,
):
    """
    Apply independent speedups to payload-sensitive READ and WRITE paths.

    READ-side accelerated stages:
      sensing, basecalling, decoding, error_correction, buffer

    WRITE-side accelerated stages:
      encoding, molecular_write, write_verification, buffer

    Addressing and molecular_access remain controller/per-block costs in the
    current scaling model.
    """
    stage_speedups = {
        "sensing":
            read_speedup,
        "basecalling":
            read_speedup,
        "decoding":
            read_speedup,
        "error_correction":
            read_speedup,
        "encoding":
            write_speedup,
        "molecular_write":
            write_speedup,
        "write_verification":
            write_speedup,
        "buffer":
            max(
                read_speedup,
                write_speedup
            ),
    }

    return write_cfg_with_stage_speedups(
        base_cfg,
        stage_speedups
    )


def experiment8g_ssd_class_unified_target():
    """
    Determine what UMDI target configuration is required for a modeled
    1 GiB payload to approach 500 MB/s for cold READ and WRITE, while also
    completing a full WRITE→cold-READ round trip within the combined
    SSD-class latency budget.
    """
    base_cfg = clean_latency_cfg()

    target_single_latency_ms = (
        OPTIMIZATION_PAYLOAD_BYTES
        / SSD_CLASS_TARGET_BYTES_PER_S
        * 1000.0
    )

    target_roundtrip_latency_ms = (
        2.0
        * target_single_latency_ms
    )

    rows = []

    for channels in SSD_TARGET_CHANNEL_VALUES:
        for block_size in SSD_TARGET_BLOCK_SIZE_VALUES:
            for read_speedup in SSD_TARGET_READ_SPEEDUP_VALUES:
                for write_speedup in SSD_TARGET_WRITE_SPEEDUP_VALUES:

                    cfg = unified_ssd_target_cfg(
                        base_cfg,
                        read_speedup,
                        write_speedup
                    )

                    cfg["channels"] = channels

                    read = analytical_file_latency_ms(
                        cfg,
                        "READ",
                        OPTIMIZATION_PAYLOAD_BYTES,
                        block_size,
                    )

                    write = analytical_file_latency_ms(
                        cfg,
                        "WRITE",
                        OPTIMIZATION_PAYLOAD_BYTES,
                        block_size,
                    )

                    roundtrip_ms = (
                        read["latency_ms"]
                        + write["latency_ms"]
                    )

                    read_bps = (
                        OPTIMIZATION_PAYLOAD_BYTES
                        / (
                            read["latency_ms"]
                            / 1000.0
                        )
                        if read["latency_ms"] > 0
                        else 0.0
                    )

                    write_bps = (
                        OPTIMIZATION_PAYLOAD_BYTES
                        / (
                            write["latency_ms"]
                            / 1000.0
                        )
                        if write["latency_ms"] > 0
                        else 0.0
                    )

                    meets_read = (
                        read["latency_ms"]
                        <= target_single_latency_ms
                    )

                    meets_write = (
                        write["latency_ms"]
                        <= target_single_latency_ms
                    )

                    meets_roundtrip = (
                        roundtrip_ms
                        <= target_roundtrip_latency_ms
                    )

                    rows.append({
                        "payload_size_bytes":
                            OPTIMIZATION_PAYLOAD_BYTES,
                        "payload_size_label":
                            human_bytes(
                                OPTIMIZATION_PAYLOAD_BYTES
                            ),
                        "channels":
                            channels,
                        "logical_block_size_bytes":
                            block_size,
                        "logical_block_size_label":
                            human_bytes(
                                block_size
                            ),
                        "read_path_speedup":
                            read_speedup,
                        "write_path_speedup":
                            write_speedup,
                        "logical_blocks":
                            read["logical_blocks"],
                        "cold_read_latency_ms":
                            read["latency_ms"],
                        "write_latency_ms":
                            write["latency_ms"],
                        "roundtrip_latency_ms":
                            roundtrip_ms,
                        "cold_read_throughput_bytes_per_s":
                            read_bps,
                        "write_throughput_bytes_per_s":
                            write_bps,
                        "meets_read_500_MBps_target":
                            meets_read,
                        "meets_write_500_MBps_target":
                            meets_write,
                        "meets_roundtrip_target":
                            meets_roundtrip,
                        "meets_all_targets":
                            (
                                meets_read
                                and meets_write
                                and meets_roundtrip
                            ),
                    })

    feasible = [
        row
        for row in rows
        if row["meets_all_targets"]
    ]

    # Balanced cost heuristic:
    # favor fewer channels and lower path speedups simultaneously.
    def balance_cost(row):
        return (
            row["channels"]
            * (
                row["read_path_speedup"]
                + row["write_path_speedup"]
            )
        )

    balanced = min(
        feasible,
        key=lambda row: (
            balance_cost(row),
            max(
                row["read_path_speedup"],
                row["write_path_speedup"]
            ),
            row["channels"],
            row["roundtrip_latency_ms"],
        ),
        default=None
    )

    # Frontier: least combined speedup at each channel count.
    frontier = []

    for channels in SSD_TARGET_CHANNEL_VALUES:
        candidates = [
            row
            for row in feasible
            if row["channels"] == channels
        ]

        if not candidates:
            continue

        best = min(
            candidates,
            key=lambda row: (
                row["read_path_speedup"]
                + row["write_path_speedup"],
                max(
                    row["read_path_speedup"],
                    row["write_path_speedup"]
                ),
                row["roundtrip_latency_ms"],
            )
        )

        frontier.append(
            dict(best)
        )

    summary = {
        "target":
            "500 MB/s modeled 1 GiB cold READ and WRITE with SSD-class round trip",
        "target_throughput_bytes_per_s":
            SSD_CLASS_TARGET_BYTES_PER_S,
        "target_1gb_read_latency_ms":
            target_single_latency_ms,
        "target_1gb_write_latency_ms":
            target_single_latency_ms,
        "target_1gb_roundtrip_latency_ms":
            target_roundtrip_latency_ms,
        "balanced_target_configuration":
            balanced,
        "frontier":
            frontier,
        "interpretation":
            "The configuration quantifies the channel count, logical-block "
            "size and independent READ/WRITE path acceleration required for "
            "the modeled UMDI architecture to approach 500 MB/s in both "
            "directions. It is a computational engineering target, not a "
            "claim of present physical molecular-storage performance.",
    }

    return rows, summary


def experiment9_target_latency_budget(
    experiment8_summary
):
    """
    Target budgets are engineering requirements, not claims
    of achieved physical BioSSD performance.
    """

    row = experiment8_summary[0]

    cold_stages = row[
        "cold_read_stage_mean_ms"
    ]

    write_stages = row[
        "write_stage_mean_ms"
    ]

    cold_controller_stages = [
        "addressing",
        "decoding",
        "error_correction",
        "buffer",
    ]

    write_controller_stages = [
        "addressing",
        "encoding",
        "buffer",
    ]

    cold_physical_stages = [
        "molecular_access",
        "sensing",
        "basecalling",
    ]

    write_physical_stages = [
        "molecular_write",
        "write_verification",
    ]

    cold_controller_ms = sum(
        cold_stages.get(stage, 0.0)
        for stage in cold_controller_stages
    )

    cold_physical_ms = sum(
        cold_stages.get(stage, 0.0)
        for stage in cold_physical_stages
    )

    write_controller_ms = sum(
        write_stages.get(stage, 0.0)
        for stage in write_controller_stages
    )

    write_physical_ms = sum(
        write_stages.get(stage, 0.0)
        for stage in write_physical_stages
    )

    roundtrip_controller_ms = (
        cold_controller_ms
        + write_controller_ms
    )

    roundtrip_physical_ms = (
        cold_physical_ms
        + write_physical_ms
    )

    results = []

    for target_ms in TARGET_LATENCY_MS:
        cold_budget = target_ms - cold_controller_ms
        write_budget = target_ms - write_controller_ms
        roundtrip_budget = (
            target_ms - roundtrip_controller_ms
        )

        results.append({
            "target_latency_ms": target_ms,

            "modeled_mean_cold_read_ms":
                row["mean_cold_read_latency_ms"],

            "cold_read_controller_overhead_ms":
                cold_controller_ms,

            "cold_read_current_physical_ms":
                cold_physical_ms,

            "cold_read_available_physical_budget_ms":
                cold_budget,

            "cold_read_required_physical_scale":
                (
                    cold_budget / cold_physical_ms
                    if cold_physical_ms > 0
                    else None
                ),

            "cold_read_target_feasible_with_current_model":
                row["mean_cold_read_latency_ms"]
                <= target_ms,

            "modeled_mean_write_ms":
                row["mean_write_latency_ms"],

            "write_controller_overhead_ms":
                write_controller_ms,

            "write_current_physical_ms":
                write_physical_ms,

            "write_available_physical_budget_ms":
                write_budget,

            "write_required_physical_scale":
                (
                    write_budget / write_physical_ms
                    if write_physical_ms > 0
                    else None
                ),

            "write_target_feasible_with_current_model":
                row["mean_write_latency_ms"]
                <= target_ms,

            "modeled_mean_roundtrip_ms":
                row["mean_roundtrip_latency_ms"],

            "roundtrip_controller_overhead_ms":
                roundtrip_controller_ms,

            "roundtrip_current_physical_ms":
                roundtrip_physical_ms,

            "roundtrip_available_physical_budget_ms":
                roundtrip_budget,

            "roundtrip_required_physical_scale":
                (
                    roundtrip_budget / roundtrip_physical_ms
                    if roundtrip_physical_ms > 0
                    else None
                ),

            "roundtrip_target_feasible_with_current_model":
                row["mean_roundtrip_latency_ms"]
                <= target_ms,
        })

    return results



def load_contemporary_baseline():
    """
    Load the literature-grounded contemporary comparison profile.

    This profile is deliberately identified as a composite reference rather
    than a single physical device.
    """
    return json.loads(
        CONTEMPORARY_BASELINE_PATH.read_text(
            encoding="utf-8"
        )
    )


def gap_metrics(test_ms, target_ms):
    test_ms = float(test_ms)
    target_ms = float(target_ms)

    absolute_gap_ms = test_ms - target_ms

    slowdown_factor = (
        test_ms / target_ms
        if target_ms > 0
        else None
    )

    required_reduction_pct = (
        max(
            0.0,
            (test_ms - target_ms)
            / test_ms
            * 100.0
        )
        if test_ms > 0
        else 0.0
    )

    return {
        "test_latency_ms": test_ms,
        "umdi_target_latency_ms": target_ms,
        "absolute_gap_ms": absolute_gap_ms,
        "slowdown_factor_vs_umdi_target": slowdown_factor,
        "required_latency_reduction_pct": required_reduction_pct,
        "meets_umdi_target": test_ms <= target_ms,
    }




def hardware_optimization_targets(host_summary, custom_physical_stages):
    target = host_summary[0]
    target_stage_map = {
        "molecular_access": float(target["cold_read_stage_mean_ms"].get("molecular_access", 0.0)),
        "sensing": float(target["cold_read_stage_mean_ms"].get("sensing", 0.0)),
        "basecalling": float(target["cold_read_stage_mean_ms"].get("basecalling", 0.0)),
        "molecular_write": float(target["write_stage_mean_ms"].get("molecular_write", 0.0)),
        "write_verification": float(target["write_stage_mean_ms"].get("write_verification", 0.0)),
    }
    current_stage_map = {
        "molecular_access": float(custom_physical_stages.get("molecular_access_ms", 0.0)),
        "sensing": float(custom_physical_stages.get("sensing_ms", 0.0)),
        "basecalling": float(custom_physical_stages.get("basecalling_ms", 0.0)),
        "molecular_write": float(custom_physical_stages.get("molecular_write_ms", 0.0)),
        "write_verification": float(custom_physical_stages.get("write_verification_ms", 0.0)),
    }
    labels = {
        "molecular_access": "Molecular access",
        "sensing": "Sensing",
        "basecalling": "Basecalling",
        "molecular_write": "Molecular write",
        "write_verification": "Write verification",
    }
    rows=[]
    for stage,current_ms in current_stage_map.items():
        target_ms=target_stage_map[stage]
        gap=current_ms-target_ms
        slowdown=(current_ms/target_ms) if target_ms>0 else None
        reduction=(max(0.0,(current_ms-target_ms)/current_ms*100.0) if current_ms>0 else 0.0)
        rows.append({
            "stage":stage,
            "label":labels[stage],
            "current_latency_ms":current_ms,
            "umdi_target_latency_ms":target_ms,
            "absolute_gap_ms":gap,
            "slowdown_factor_vs_target":slowdown,
            "required_reduction_pct":reduction,
            "meets_target":current_ms<=target_ms,
        })
    rows.sort(key=lambda r:(max(0.0,r["absolute_gap_ms"]), r["slowdown_factor_vs_target"] or 0.0), reverse=True)
    for i,row in enumerate(rows,1): row["priority_rank"]=i
    dominant=next((r for r in rows if not r["meets_target"]),None)
    return {
        "dominant_bottleneck": dominant["stage"] if dominant else None,
        "dominant_bottleneck_label": dominant["label"] if dominant else "None - all supplied stages meet the UMDI target",
        "optimization_targets": rows,
        "interpretation_note":"Stage-level performance targets derived from the UMDI baseline; they identify where latency must be reduced but do not prescribe a specific physical implementation.",
    }

def experiment10_technology_gap(
    host_summary,
    profile=None,
    custom_physical_stages=None,
):
    """
    Experiment 10: Technology-gap analysis.

    Modes:
      1. Literature reference profile containing documented end-to-end
         comparison timings.
      2. Researcher-supplied physical-stage measurements.

    The UMDI target is always taken from Experiment 8 under baseline.json.
    """

    target = host_summary[0]

    target_cold_ms = float(
        target["mean_cold_read_latency_ms"]
    )

    target_write_ms = float(
        target["mean_write_latency_ms"]
    )

    target_roundtrip_ms = float(
        target["mean_roundtrip_latency_ms"]
    )

    if custom_physical_stages is not None:
        cold_controller_ms = sum(
            target["cold_read_stage_mean_ms"].get(
                stage,
                0.0
            )
            for stage in [
                "addressing",
                "decoding",
                "error_correction",
                "buffer",
            ]
        )

        write_controller_ms = sum(
            target["write_stage_mean_ms"].get(
                stage,
                0.0
            )
            for stage in [
                "addressing",
                "encoding",
                "buffer",
            ]
        )

        molecular_access_ms = float(
            custom_physical_stages.get(
                "molecular_access_ms",
                0.0
            )
        )

        sensing_ms = float(
            custom_physical_stages.get(
                "sensing_ms",
                0.0
            )
        )

        basecalling_ms = float(
            custom_physical_stages.get(
                "basecalling_ms",
                0.0
            )
        )

        molecular_write_ms = float(
            custom_physical_stages.get(
                "molecular_write_ms",
                0.0
            )
        )

        write_verification_ms = float(
            custom_physical_stages.get(
                "write_verification_ms",
                0.0
            )
        )

        cold_physical_ms = (
            molecular_access_ms
            + sensing_ms
            + basecalling_ms
        )

        write_physical_ms = (
            molecular_write_ms
            + write_verification_ms
        )

        test_cold_ms = (
            cold_controller_ms
            + cold_physical_ms
        )

        test_write_ms = (
            write_controller_ms
            + write_physical_ms
        )

        test_roundtrip_ms = (
            test_cold_ms
            + test_write_ms
        )

        optimization = hardware_optimization_targets(
            host_summary,
            custom_physical_stages
        )

        return [{
            "profile_name": custom_physical_stages.get(
                "profile_name",
                "Custom Researcher Hardware"
            ),
            "profile_type": "custom_measured_stages",
            "comparison_warning": (
                "Calculated from researcher-supplied physical-stage values "
                "plus the UMDI baseline controller overhead."
            ),
            "input_physical_stages_ms": {
                "molecular_access": molecular_access_ms,
                "sensing": sensing_ms,
                "basecalling": basecalling_ms,
                "molecular_write": molecular_write_ms,
                "write_verification": write_verification_ms,
            },
            "controller_overhead_ms": {
                "cold_read": cold_controller_ms,
                "write": write_controller_ms,
            },
            "cold_read": gap_metrics(
                test_cold_ms,
                target_cold_ms
            ),
            "write": gap_metrics(
                test_write_ms,
                target_write_ms
            ),
            "roundtrip": gap_metrics(
                test_roundtrip_ms,
                target_roundtrip_ms
            ),
            "dominant_bottleneck": optimization["dominant_bottleneck"],
            "dominant_bottleneck_label": optimization["dominant_bottleneck_label"],
            "optimization_targets": optimization["optimization_targets"],
            "optimization_note": optimization["interpretation_note"],
        }]

    if profile is None:
        profile = load_contemporary_baseline()

    benchmarks = profile[
        "benchmark_totals_ms"
    ]

    return [{
        "profile_name": profile["profile_name"],
        "profile_type": profile["profile_type"],
        "comparison_warning": profile["description"],
        "cold_read": gap_metrics(
            benchmarks["cold_read_ms"],
            target_cold_ms
        ),
        "write": gap_metrics(
            benchmarks["write_ms"],
            target_write_ms
        ),
        "roundtrip": gap_metrics(
            benchmarks["roundtrip_ms"],
            target_roundtrip_ms
        ),
        "benchmark_basis": profile.get(
            "benchmark_basis",
            {}
        ),
        "notes": profile.get(
            "notes",
            []
        ),
    }]




# ============================================================
# PUBLICATION FIGURES FOR CORE HOST-VISIBLE PERFORMANCE
# ============================================================

def save_bar_figure(labels, values, title, ylabel, filename):
    """
    Save a clean publication figure using matplotlib defaults.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(
        labels,
        values
    )

    ax.set_title(title)
    ax.set_ylabel(ylabel)

    ax.tick_params(
        axis="x",
        rotation=20
    )

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    fig.tight_layout()

    fig.savefig(
        FIGURES / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


def plot_host_visible_figures(
    host_summary,
    target_results,
    technology_gap_results,
    payload_block_results,
    large_file_optimization_results,
    ssd_target_summary
):
    """
    Generate the Experiment 8-10 publication figures directly from
    the results produced in the same simulator run.
    """

    host = host_summary[0]

    # --------------------------------------------------------
    # 1. Host-visible transaction latency
    # --------------------------------------------------------
    save_bar_figure(
        labels=[
            "Hot READ",
            "Cold READ",
            "WRITE",
            "WRITE→Cold READ",
        ],
        values=[
            host["mean_hot_read_latency_ms"],
            host["mean_cold_read_latency_ms"],
            host["mean_write_latency_ms"],
            host["mean_roundtrip_latency_ms"],
        ],
        title="Host-Visible UMDI BioSSD Transaction Latency",
        ylabel="Mean latency (ms)",
        filename="host_visible_latency.png",
    )

    # --------------------------------------------------------
    # 2. Cold READ stage breakdown
    # --------------------------------------------------------
    cold_stages = host[
        "cold_read_stage_mean_ms"
    ]

    cold_order = [
        "addressing",
        "molecular_access",
        "sensing",
        "basecalling",
        "decoding",
        "error_correction",
        "buffer",
    ]

    save_bar_figure(
        labels=[
            stage.replace("_", " ").title()
            for stage in cold_order
        ],
        values=[
            cold_stages.get(stage, 0.0)
            for stage in cold_order
        ],
        title="Cold Molecular READ Stage-Latency Breakdown",
        ylabel="Mean stage latency (ms)",
        filename="cold_read_stage_latency.png",
    )

    # --------------------------------------------------------
    # 3. WRITE stage breakdown
    # --------------------------------------------------------
    write_stages = host[
        "write_stage_mean_ms"
    ]

    write_order = [
        "addressing",
        "encoding",
        "molecular_write",
        "write_verification",
        "buffer",
    ]

    save_bar_figure(
        labels=[
            stage.replace("_", " ").title()
            for stage in write_order
        ],
        values=[
            write_stages.get(stage, 0.0)
            for stage in write_order
        ],
        title="Molecular WRITE Stage-Latency Breakdown",
        ylabel="Mean stage latency (ms)",
        filename="write_stage_latency.png",
    )

    # --------------------------------------------------------
    # 4. Target-latency feasibility
    # --------------------------------------------------------
    targets = [
        row["target_latency_ms"]
        for row in target_results
    ]

    cold_values = [
        row["modeled_mean_cold_read_ms"]
        for row in target_results
    ]

    write_values = [
        row["modeled_mean_write_ms"]
        for row in target_results
    ]

    roundtrip_values = [
        row["modeled_mean_roundtrip_ms"]
        for row in target_results
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        targets,
        cold_values,
        marker="o",
        label="Cold READ"
    )

    ax.plot(
        targets,
        write_values,
        marker="o",
        label="WRITE"
    )

    ax.plot(
        targets,
        roundtrip_values,
        marker="o",
        label="WRITE→Cold READ"
    )

    ax.plot(
        targets,
        targets,
        linestyle="--",
        label="Target boundary"
    )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_title(
        "UMDI Target-Latency Feasibility"
    )

    ax.set_xlabel(
        "Host-visible latency target (ms)"
    )

    ax.set_ylabel(
        "Modeled mean transaction latency (ms)"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIGURES / "target_latency_analysis.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    # --------------------------------------------------------
    # 5. Contemporary technology gap
    # --------------------------------------------------------
    technology = technology_gap_results[0]

    categories = [
        "Cold READ",
        "WRITE",
        "WRITE→READ",
    ]

    contemporary_values = [
        technology["cold_read"]["test_latency_ms"],
        technology["write"]["test_latency_ms"],
        technology["roundtrip"]["test_latency_ms"],
    ]

    target_values = [
        technology["cold_read"]["umdi_target_latency_ms"],
        technology["write"]["umdi_target_latency_ms"],
        technology["roundtrip"]["umdi_target_latency_ms"],
    ]

    x = list(range(len(categories)))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(
        [value - width / 2 for value in x],
        contemporary_values,
        width,
        label="Contemporary reference"
    )

    ax.bar(
        [value + width / 2 for value in x],
        target_values,
        width,
        label="UMDI target"
    )

    ax.set_yscale("log")

    ax.set_xticks(x)
    ax.set_xticklabels(categories)

    ax.set_title(
        "Technology Gap Between Demonstrated Molecular Storage and UMDI Target"
    )

    ax.set_ylabel(
        "Latency (ms, logarithmic scale)"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIGURES / "technology_gap_analysis.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


    # --------------------------------------------------------
    # 6. Payload-size scaling by logical block size
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for block_size in LOGICAL_BLOCK_SIZE_VALUES:
        subset = [
            row
            for row in payload_block_results
            if row[
                "logical_block_size_bytes"
            ] == block_size
        ]

        subset.sort(
            key=lambda row:
                row["payload_size_bytes"]
        )

        ax.plot(
            [
                row["payload_size_bytes"]
                for row in subset
            ],
            [
                row["roundtrip_latency_ms"]
                for row in subset
            ],
            marker="o",
            label=human_bytes(block_size)
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(
        "Payload Size vs WRITE→Cold-READ Latency"
    )
    ax.set_xlabel(
        "User payload size (bytes, logarithmic scale)"
    )
    ax.set_ylabel(
        "Host-visible round-trip latency (ms, logarithmic scale)"
    )
    ax.legend(title="Logical block size")
    fig.tight_layout()
    fig.savefig(
        FIGURES
        / "payload_size_vs_roundtrip_latency.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)

    # --------------------------------------------------------
    # 7. Block-size effect for a 10 MiB payload
    # --------------------------------------------------------
    ten_mb = 10 * 1024 * 1024

    subset = [
        row
        for row in payload_block_results
        if row[
            "payload_size_bytes"
        ] == ten_mb
    ]

    subset.sort(
        key=lambda row:
            row["logical_block_size_bytes"]
    )

    labels = [
        row["logical_block_size_label"]
        for row in subset
    ]

    x = list(range(len(labels)))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.bar(
        [value - width for value in x],
        [
            row["cold_read_latency_ms"]
            for row in subset
        ],
        width,
        label="Cold READ"
    )

    ax.bar(
        x,
        [
            row["write_latency_ms"]
            for row in subset
        ],
        width,
        label="WRITE"
    )

    ax.bar(
        [value + width for value in x],
        [
            row["roundtrip_latency_ms"]
            for row in subset
        ],
        width,
        label="WRITE→Cold READ"
    )

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(
        "Logical Block Size Effect for a 10 MiB Payload"
    )
    ax.set_xlabel("Logical block size")
    ax.set_ylabel(
        "Host-visible latency (ms, logarithmic scale)"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        FIGURES / "block_size_effect_10mb.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)



    # --------------------------------------------------------
    # 8. Large-file optimization: best WRITE latency by channel count
    # --------------------------------------------------------
    best_by_channels = []

    for channels in OPTIMIZATION_CHANNEL_VALUES:
        candidates = [
            row
            for row in large_file_optimization_results
            if row["channels"] == channels
        ]

        best_by_channels.append(
            min(
                candidates,
                key=lambda row:
                    row["write_latency_ms"]
            )
        )

    fig, ax = plt.subplots(
        figsize=(9, 5.5)
    )

    ax.plot(
        [
            row["channels"]
            for row in best_by_channels
        ],
        [
            row["write_latency_ms"] / 1000.0
            for row in best_by_channels
        ],
        marker="o"
    )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")

    ax.set_title(
        "Optimized 1 GiB UMDI WRITE Latency by Molecular Channel Count"
    )

    ax.set_xlabel(
        "Molecular channels"
    )

    ax.set_ylabel(
        "Best modeled WRITE latency (s, logarithmic scale)"
    )

    fig.tight_layout()

    fig.savefig(
        FIGURES / "optimized_1gb_write_latency.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    # --------------------------------------------------------
    # 9. Unified 500 MB/s READ/WRITE target frontier
    # --------------------------------------------------------
    frontier = ssd_target_summary[
        "frontier"
    ]

    if frontier:
        fig, ax = plt.subplots(
            figsize=(9, 5.5)
        )

        ax.plot(
            [
                row["channels"]
                for row in frontier
            ],
            [
                row["read_path_speedup"]
                for row in frontier
            ],
            marker="o",
            label="READ-path speedup"
        )

        ax.plot(
            [
                row["channels"]
                for row in frontier
            ],
            [
                row["write_path_speedup"]
                for row in frontier
            ],
            marker="o",
            label="WRITE-path speedup"
        )

        ax.set_xscale(
            "log",
            base=2
        )

        ax.set_yscale("log")

        ax.set_title(
            "UMDI Requirements for 500 MB/s READ and WRITE"
        )

        ax.set_xlabel(
            "Molecular channels"
        )

        ax.set_ylabel(
            "Required payload-sensitive path speedup"
        )

        ax.legend()

        fig.tight_layout()

        fig.savefig(
            FIGURES / "ssd_class_500MBps_unified_frontier.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close(fig)


# ============================================================
# MAIN EXPERIMENT SUITE
# ============================================================

def main():

    experiments = {}


    # ========================================================
    # 1. CHANNEL SCALING
    # ========================================================

    results = []

    for channels in CHANNEL_VALUES:

        cfg = clone()
        cfg["channels"] = channels

        requests = workload(
            n=1000,
            seed=100,
            read_ratio=0.70,
            blocks=100
        )

        summary, _ = run(
            cfg,
            requests,
            initialize_blocks=True,
            blocks=100
        )

        results.append({
            "channels": channels,
            **summary
        })

    experiments[
        "latency_throughput_vs_channels"
    ] = results


    # ========================================================
    # 2. FIFO VS MOLECULAR-AWARE SCHEDULING
    # ========================================================

    results = []

    for scheduler in SCHEDULERS:

        cfg = clone()
        cfg["scheduler"] = scheduler

        requests = workload(
            n=2000,
            seed=101,
            read_ratio=0.70,
            blocks=100
        )

        summary, _ = run(
            cfg,
            requests,
            initialize_blocks=True,
            blocks=100
        )

        results.append({
            "scheduler": scheduler,
            **summary
        })

    experiments[
        "fifo_vs_molecular_aware"
    ] = results


    # ========================================================
    # 3. CACHE VS REPEATED ACCESS
    # ========================================================

    results = []

    for cache_size in CACHE_VALUES:

        cfg = clone()
        cfg["cache_size"] = cache_size

        requests = workload(
            n=1500,
            seed=102,
            read_ratio=0.70,
            blocks=100
        )

        for i, request in enumerate(requests):
            if i % 3 == 0:
                request.block = 20
                request.locality = 20 % 16

        summary, _ = run(
            cfg,
            requests,
            initialize_blocks=True,
            blocks=100
        )

        results.append({
            "cache_size": cache_size,
            **summary
        })

    experiments[
        "cache_vs_repeated_access"
    ] = results


    # ========================================================
    # 4. ERROR PROBABILITY VS RECONSTRUCTION
    # ========================================================

    results = []

    for error_probability in ERROR_VALUES:

        cfg = clone()

        # Disable cache so every READ reaches the molecular layer.
        cfg["cache_size"] = 0

        cfg[
            "error_probability"
        ] = error_probability

        requests = workload(
            n=2000,
            seed=103,
            read_ratio=1.0,
            blocks=100
        )

        summary, _ = run(
            cfg,
            requests,
            initialize_blocks=True,
            blocks=100
        )

        results.append({
            "error_probability":
                error_probability,

            "reconstruction_success_rate":
                summary[
                    "transaction_success_rate"
                ],

            "error_recovery_rate":
                error_recovery_rate(
                    summary
                ),

            **summary
        })

    experiments[
        "error_vs_reconstruction"
    ] = results


    # ========================================================
    # 5. QUEUE DEPTH VS LATENCY
    # ========================================================

    results = []

    for queue_depth in QUEUE_VALUES:

        cfg = clone()

        requests = workload(
            n=queue_depth,
            seed=104,
            read_ratio=0.70,
            blocks=100,
            simultaneous=True
        )

        summary, _ = run(
            cfg,
            requests,
            initialize_blocks=True,
            blocks=100
        )

        results.append({
            "queue_depth": queue_depth,
            **summary
        })

    experiments[
        "queue_depth_vs_latency"
    ] = results


    # ========================================================
    # 6. WRITE REPETITION / ENDURANCE
    # ========================================================

    results = []

    for rewrites in REWRITE_VALUES:

        cfg = clone()

        cfg["cache_size"] = 0

        # Isolate wear-induced failures.
        cfg[
            "write_failure_probability"
        ] = 0.0

        requests = [
            Request(
                req_id=i + 1,
                op="WRITE",
                block=20,
                arrival=0.0,
                size_bytes=4096,
                locality=4
            )
            for i in range(rewrites)
        ]

        summary, rows = run(
            cfg,
            requests
        )

        completed_writes = sum(
            1
            for row in rows
            if row["status"] == "completed"
        )

        failed_writes = sum(
            1
            for row in rows
            if row["status"] == "failed"
        )

        endurance_failures = sum(
            1
            for row in rows
            if row.get(
                "endurance_failure",
                False
            )
        )

        results.append({
            "rewrites": rewrites,

            "completed_writes":
                completed_writes,

            "failed_writes":
                failed_writes,

            "write_failure_rate": (
                failed_writes / rewrites
                if rewrites
                else 0.0
            ),

            "endurance_failures":
                endurance_failures,

            **summary
        })

    experiments[
        "write_repetition_endurance"
    ] = results


    # ========================================================
    # 7. PREFETCHING
    # ========================================================
    #
    # Tests OFF vs different sequential prefetch depths.
    # Same workload and seed for every condition.
    #
    # A read-heavy workload with initialized storage is used
    # so prefetching can potentially reduce later molecular reads.
    # ========================================================

    results = []

    for depth in PREFETCH_DEPTH_VALUES:

        cfg = clone()

        if depth == 0:
            cfg["prefetch"] = {
                "enabled": False,
                "depth": 0
            }
        else:
            cfg["prefetch"] = {
                "enabled": True,
                "depth": depth
            }

        requests = workload(
            n=2000,
            seed=105,
            read_ratio=0.90,
            blocks=100,
            locality_strength=2
        )

        summary, _ = run(
            cfg,
            requests,
            initialize_blocks=True,
            blocks=100
        )

        results.append({
            "prefetch_enabled":
                depth > 0,

            "prefetch_depth":
                depth,

            **summary
        })

    experiments[
        "prefetch_vs_no_prefetch"
    ] = results


    # ========================================================
    # 8. HOST-VISIBLE READ / WRITE / ROUND-TRIP LATENCY
    # ========================================================

    (
        host_summary,
        cold_rows,
        hot_rows,
        write_rows,
        roundtrip_rows,
    ) = experiment8_host_visible_latency()

    experiments[
        "host_visible_read_write_latency"
    ] = host_summary

    save(
        "host_visible_cold_read_trials",
        cold_rows
    )

    save(
        "host_visible_hot_read_trials",
        hot_rows
    )

    save(
        "host_visible_write_trials",
        write_rows
    )

    save(
        "host_visible_roundtrip_trials",
        roundtrip_rows
    )



    # ========================================================
    # 8E. PAYLOAD-SIZE × LOGICAL-BLOCK-SIZE SCALING
    # ========================================================

    payload_block_results = (
        experiment8e_payload_block_scaling()
    )

    experiments[
        "payload_block_size_scaling"
    ] = payload_block_results



    # ========================================================
    # 8F. LARGE-FILE OPTIMIZATION SEARCH
    # ========================================================

    (
        large_file_optimization_results,
        optimized_model
    ) = experiment8f_large_file_optimization()

    experiments[
        "large_file_optimization"
    ] = large_file_optimization_results



    # ========================================================
    # 8G. UNIFIED SSD-CLASS READ / WRITE / ROUND-TRIP TARGET
    # ========================================================

    (
        ssd_target_results,
        ssd_target_summary
    ) = experiment8g_ssd_class_unified_target()

    experiments[
        "ssd_class_unified_target"
    ] = ssd_target_results


    # ========================================================
    # 9. TARGET-LATENCY / BOTTLENECK BUDGET ANALYSIS
    # ========================================================

    target_results = (
        experiment9_target_latency_budget(
            host_summary
        )
    )

    experiments[
        "target_latency_bottleneck_analysis"
    ] = target_results



    # ========================================================
    # 10. TECHNOLOGY-GAP ANALYSIS
    # ========================================================

    technology_gap_results = (
        experiment10_technology_gap(
            host_summary
        )
    )

    experiments[
        "technology_gap_analysis"
    ] = technology_gap_results


    # ========================================================
    # SAVE ALL DATASETS
    # ========================================================

    for name, data in experiments.items():
        save(
            name,
            data
        )


    (
        OUT / "optimized_umdi_model.json"
    ).write_text(
        json.dumps(
            optimized_model,
            indent=2
        ),
        encoding="utf-8"
    )


    (
        OUT / "ssd_class_unified_target_summary.json"
    ).write_text(
        json.dumps(
            ssd_target_summary,
            indent=2
        ),
        encoding="utf-8"
    )

    # Generate publication figures from the same experimental outputs.
    plot_host_visible_figures(
        host_summary,
        target_results,
        technology_gap_results,
        payload_block_results,
        large_file_optimization_results,
        ssd_target_summary
    )


    # ========================================================
    # TERMINAL REPORT
    # ========================================================

    print()
    print("=" * 60)

    print(
        "UMDI BioSSD EXPERIMENT SUITE COMPLETED"
    )

    print("=" * 60)

    for name, data in experiments.items():
        print(
            f"{name}: "
            f"{len(data)} experimental points"
        )

    print()

    print(
        f"Channel sweep: {CHANNEL_VALUES}"
    )

    print(
        f"Cache sweep: {CACHE_VALUES}"
    )

    print(
        f"Error sweep: {ERROR_VALUES}"
    )

    print(
        f"Queue sweep: {QUEUE_VALUES}"
    )

    print(
        f"Endurance sweep: {REWRITE_VALUES}"
    )

    print(
        f"Prefetch depths: {PREFETCH_DEPTH_VALUES}"
    )

    print(
        f"Host latency trials: {HOST_LATENCY_TRIALS}"
    )

    print(
        f"Target latency budgets (ms): {TARGET_LATENCY_MS}"
    )

    print(
        "Payload sweep: "
        + ", ".join(
            human_bytes(value)
            for value in PAYLOAD_SIZE_VALUES
        )
    )

    print(
        "Logical block sizes: "
        + ", ".join(
            human_bytes(value)
            for value in LOGICAL_BLOCK_SIZE_VALUES
        )
    )


    print(
        "Technology-gap profile: "
        "Contemporary Demonstrated-Technology Composite"
    )


    print(
        "Large-file optimization search: "
        f"{len(large_file_optimization_results)} configurations"
    )

    print(
        "Optimized 1 GiB target model: "
        f"{optimized_model['channels']} channels, "
        f"{optimized_model['logical_block_size_label']} blocks, "
        f"{optimized_model['molecular_write_speedup']}x molecular-write speedup"
    )

    print(
        "Optimized modeled 1 GiB WRITE: "
        f"{optimized_model['modeled_write_latency_ms'] / 1000.0:.3f} s"
    )


    print(
        "SSD-class unified READ / WRITE target: "
        "500 MB/s"
    )

    balanced = ssd_target_summary.get(
        "balanced_target_configuration"
    )

    if balanced:
        print(
            "Balanced SSD-class target configuration: "
            f"{balanced['channels']} channels, "
            f"{balanced['logical_block_size_label']} blocks, "
            f"{balanced['read_path_speedup']}x READ-path speedup, "
            f"{balanced['write_path_speedup']}x WRITE-path speedup"
        )

        print(
            "Balanced modeled 1 GiB latencies: "
            f"READ {balanced['cold_read_latency_ms'] / 1000.0:.3f} s, "
            f"WRITE {balanced['write_latency_ms'] / 1000.0:.3f} s, "
            f"ROUND-TRIP {balanced['roundtrip_latency_ms'] / 1000.0:.3f} s"
        )

    print()

    host = host_summary[0]

    print("CORE HOST-VISIBLE LATENCY RESULTS")
    print("-" * 60)

    print(
        "Cold READ mean: "
        f"{host['mean_cold_read_latency_ms']:.3f} ms"
    )

    print(
        "Hot READ mean: "
        f"{host['mean_hot_read_latency_ms']:.3f} ms"
    )

    print(
        "WRITE mean: "
        f"{host['mean_write_latency_ms']:.3f} ms"
    )

    print(
        "WRITE->READ round-trip mean: "
        f"{host['mean_roundtrip_latency_ms']:.3f} ms"
    )

    print(
        "Cold/Hot acceleration ratio: "
        f"{host['cold_hot_acceleration_ratio']:.2f}x"
    )

    print()

    print(
        f"Results written to: {OUT}"
    )

    print(
        f"Figures written to: {FIGURES}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()