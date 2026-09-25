import math
import random
from collections import OrderedDict, deque
from dataclasses import asdict, dataclass, field

READ_STAGES = [
    "addressing",
    "molecular_access",
    "sensing",
    "basecalling",
    "decoding",
    "error_correction",
    "buffer",
]

WRITE_STAGES = [
    "addressing",
    "encoding",
    "molecular_write",
    "write_verification",
    "buffer",
]

# Payload-sensitive stages scale with bytes relative to the original
# 4096-byte request reference. Addressing and molecular access remain
# per-logical-block operations.
PAYLOAD_SCALED_STAGES = {
    "sensing",
    "basecalling",
    "decoding",
    "error_correction",
    "buffer",
    "encoding",
    "molecular_write",
    "write_verification",
}



@dataclass
class Request:
    req_id: int
    op: str
    block: int
    arrival: float
    size_bytes: int = 4096
    locality: int = 0
    start: float = 0.0
    finish: float = 0.0
    status: str = "queued"
    cache_hit: bool = False
    error_injected: bool = False
    error_recovered: bool = False
    rewrite_cycle: int = 0
    endurance_failure: bool = False
    channel: int = -1
    service_time_ms: float = 0.0
    queue_wait_ms: float = 0.0
    request_class: str = ""
    stage_times_ms: dict = field(default_factory=dict)


class BioSSDSimulator:
    def __init__(self, cfg, progress_callback=None):
        self.cfg = cfg
        self.progress_callback = progress_callback
        self.total_requests = 0
        self.processed_requests = 0
        self.rng = random.Random(cfg["seed"])
        self.now = 0.0
        self.queue = deque()
        self.storage = {}
        self.cache = OrderedDict()

        channels = int(cfg["channels"])
        if channels < 1:
            raise ValueError("channels must be at least 1")

        self.busy = [0.0] * channels
        self.completed = []
        self.failed = []

        # Per-channel resource telemetry.
        self.channel_busy_time_ms = [0.0] * channels
        self.channel_requests = [0] * channels
        self.channel_completed = [0] * channels
        self.channel_failed = [0] * channels
        self.channel_bytes_completed = [0] * channels

        # Queue-depth telemetry sampled on each event-loop iteration.
        self.queue_depth_samples = []
        self.peak_queue_depth = 0

        self.m = {
            "cache_hits": 0,
            "molecular_reads": 0,
            "molecular_writes": 0,
            "errors_injected": 0,
            "errors_recovered": 0,
            "endurance_failures": 0,
            "maximum_rewrite_cycles": 0,
            "total_rewrite_cycles": 0,
            "prefetch_attempts": 0,
            "prefetch_completed": 0,
            "prefetch_errors": 0,
            "prefetch_errors_recovered": 0,
            "prefetch_time_ms": 0.0,
        }

    def sample(self, spec):
        kind = spec["kind"]

        if kind == "fixed":
            return float(spec["value"])

        if kind == "normal":
            return max(
                0.01,
                self.rng.gauss(spec["mean"], spec["sd"])
            )

        if kind == "lognormal":
            return self.rng.lognormvariate(
                spec["mu"],
                spec["sigma"]
            )

        if kind == "uniform":
            return self.rng.uniform(
                spec["low"],
                spec["high"]
            )

        raise ValueError(f"Unknown distribution: {kind}")

    def addr(self, block):
        return f"M-{(block * 7919) % 100000:05d}"

    def cache_get(self, block):
        if block in self.cache:
            value = self.cache.pop(block)
            self.cache[block] = value
            self.m["cache_hits"] += 1
            return value

        return None

    def cache_put(self, block, value):
        if self.cfg["cache_size"] <= 0:
            return

        self.cache.pop(block, None)
        self.cache[block] = value

        while len(self.cache) > self.cfg["cache_size"]:
            self.cache.popitem(last=False)

    def order(self):
        requests = list(self.queue)
        self.queue.clear()

        if self.cfg["scheduler"] == "molecular_aware":
            requests.sort(
                key=lambda r: (
                    r.locality,
                    0 if r.op == "READ" else 1
                )
            )

        self.queue.extend(requests)

    # ---------------------------------------------------------
    # ENDURANCE MODEL
    # ---------------------------------------------------------

    def endurance_probability(self, block):
        """
        Additional write-failure probability caused by accumulated
        rewrite wear.

        Expected config:
        "endurance": {
            "enabled": true,
            "max_rewrites": 1000,
            "degradation_start": 100,
            "failure_probability_at_limit": 0.50,
            "degradation_exponent": 2.0
        }

        This is a computational assumption, not a measured physical
        endurance law for DNA or another molecular medium.
        """

        endurance = self.cfg.get("endurance", {})

        if not endurance.get("enabled", False):
            return 0.0

        cycles = int(
            self.storage.get(block, {}).get("rewrites", 0)
        )

        max_rewrites = int(
            endurance.get("max_rewrites", 1000)
        )

        degradation_start = int(
            endurance.get("degradation_start", 0)
        )

        failure_at_limit = float(
            endurance.get(
                "failure_probability_at_limit",
                0.50
            )
        )

        exponent = float(
            endurance.get("degradation_exponent", 2.0)
        )

        if max_rewrites <= 0:
            raise ValueError(
                "endurance.max_rewrites must be greater than 0"
            )

        degradation_start = max(0, degradation_start)

        if degradation_start >= max_rewrites:
            degradation_start = max_rewrites - 1

        failure_at_limit = min(
            1.0,
            max(0.0, failure_at_limit)
        )

        if exponent <= 0:
            exponent = 1.0

        if cycles < degradation_start:
            return 0.0

        wear_window = max_rewrites - degradation_start

        progress = (
            cycles - degradation_start
        ) / wear_window

        progress = min(
            1.0,
            max(0.0, progress)
        )

        return min(
            1.0,
            max(
                0.0,
                failure_at_limit
                * (progress ** exponent)
            )
        )

    def record_channel_result(self, request, channel, completed):
        service_time = max(
            0.0,
            request.finish - request.start
        )

        request.channel = channel
        request.service_time_ms = service_time

        self.channel_busy_time_ms[channel] += service_time
        self.channel_requests[channel] += 1

        if completed:
            self.channel_completed[channel] += 1
            self.channel_bytes_completed[channel] += request.size_bytes
        else:
            self.channel_failed[channel] += 1

    # ---------------------------------------------------------
    # PREFETCH MODEL
    # ---------------------------------------------------------

    def prefetch_enabled(self):
        return bool(
            self.cfg.get("prefetch", {}).get("enabled", False)
        )

    def prefetch_depth(self):
        return max(
            0,
            int(self.cfg.get("prefetch", {}).get("depth", 1))
        )

    def prefetch_after_read(self, source_block):
        """
        Sequential speculative prefetching. After a successful molecular
        READ, fetch the next N logical blocks into cache. Prefetched blocks
        incur the same modeled molecular-read stages. A failed speculative
        read does not fail the host request.
        """
        if not self.prefetch_enabled():
            return 0.0
        if self.cfg["cache_size"] <= 0:
            return 0.0

        depth = self.prefetch_depth()
        if depth <= 0:
            return 0.0

        total = 0.0
        for offset in range(1, depth + 1):
            block = source_block + offset
            if block not in self.storage or block in self.cache:
                continue

            self.m["prefetch_attempts"] += 1
            self.m["molecular_reads"] += 1
            elapsed = 0.0
            prefetched_size = max(
                1,
                int(self.storage.get(block, {}).get("size_bytes", 4096))
            )
            for stage in READ_STAGES:
                stage_time = self.sample(self.cfg["latency_ms"][stage])
                if stage in PAYLOAD_SCALED_STAGES:
                    stage_time *= (
                        float(prefetched_size)
                        / float(self.reference_block_size_bytes())
                    )
                stage_time /= self.path_acceleration_factor("READ", stage)
                elapsed += stage_time
            total += elapsed

            failed = False
            if self.rng.random() < self.cfg["error_probability"]:
                self.m["prefetch_errors"] += 1
                if self.rng.random() < self.cfg["correction_success_probability"]:
                    self.m["prefetch_errors_recovered"] += 1
                else:
                    failed = True

            if not failed:
                self.cache_put(block, self.storage[block]["data"])
                self.m["prefetch_completed"] += 1

        self.m["prefetch_time_ms"] += total
        return total

    def reference_block_size_bytes(self):
        scaling = self.cfg.get("payload_scaling", {})
        return max(
            1,
            int(
                scaling.get(
                    "reference_block_size_bytes",
                    4096
                )
            )
        )

    def payload_scaling_enabled(self):
        return bool(
            self.cfg.get(
                "payload_scaling",
                {}
            ).get(
                "enabled",
                True
            )
        )

    def stage_payload_factor(self, request, stage):
        """
        Scale payload-sensitive work linearly with bytes relative to the
        4096-byte reference request. This is a modeling assumption, not
        a measured molecular law.
        """
        if not self.payload_scaling_enabled():
            return 1.0

        if stage not in PAYLOAD_SCALED_STAGES:
            return 1.0

        return (
            float(request.size_bytes)
            / float(self.reference_block_size_bytes())
        )

    def _emit_progress(self, event, **payload):
        if self.progress_callback is None:
            return
        try:
            keep_running = self.progress_callback({
                "event": event,
                "simulated_time_ms": self.now,
                "processed_requests": self.processed_requests,
                "total_requests": self.total_requests,
                **payload,
            })
        except Exception:
            # Telemetry failures must never alter simulation results.
            return
        if keep_running is False:
            # Cooperative cancellation at a safe simulator checkpoint.
            raise RuntimeError("SIMULATION_CANCELLED")

    def path_acceleration_factor(self, operation, stage):
        path = self.cfg.get("path_acceleration", {})
        if operation == "READ" and stage in READ_STAGES:
            return max(0.1, float(path.get("read", 1.0)))
        if operation == "WRITE" and stage in WRITE_STAGES:
            return max(0.1, float(path.get("write", 1.0)))
        return 1.0

    def _record_stage(self, request, stage):
        self._emit_progress(
            "stage",
            stage=stage,
            operation=request.op,
            request_id=request.req_id,
        )
        value = self.sample(
            self.cfg["latency_ms"][stage]
        )

        value *= self.stage_payload_factor(
            request,
            stage
        )
        value /= self.path_acceleration_factor(
            request.op,
            stage
        )

        request.stage_times_ms[stage] = (
            request.stage_times_ms.get(stage, 0.0)
            + value
        )
        return value

    def execute(self, request, channel):
        cfg = self.cfg

        request.start = max(
            self.now,
            self.busy[channel],
            request.arrival
        )

        request.queue_wait_ms = max(
            0.0,
            request.start - request.arrival
        )

        request.status = "running"
        elapsed = 0.0

        if request.op == "READ":
            cached = self.cache_get(request.block)

            if cached is not None:
                request.cache_hit = True
                request.request_class = "hot_read"
                request.stage_times_ms["cache"] = float(
                    cfg["cache_latency_ms"]
                )
                request.finish = (
                    request.start
                    + cfg["cache_latency_ms"]
                )
                request.status = "completed"
                self.completed.append(request)
                self.busy[channel] = request.finish
                self.record_channel_result(
                    request,
                    channel,
                    completed=True
                )
                return

            request.request_class = "cold_read"
            self.m["molecular_reads"] += 1

            for stage in READ_STAGES:
                elapsed += self._record_stage(
                    request,
                    stage
                )

            if (
                self.rng.random()
                < cfg["error_probability"]
            ):
                request.error_injected = True
                self.m["errors_injected"] += 1

                if (
                    self.rng.random()
                    < cfg["correction_success_probability"]
                ):
                    request.error_recovered = True
                    self.m["errors_recovered"] += 1
                else:
                    request.status = "failed"
                    request.finish = (
                        request.start + elapsed
                    )
                    self.failed.append(request)
                    self.busy[channel] = request.finish
                    self.record_channel_result(
                        request,
                        channel,
                        completed=False
                    )
                    return

            if request.block not in self.storage:
                request.status = "failed"
                request.finish = (
                    request.start + elapsed
                )
                self.failed.append(request)
                self.busy[channel] = request.finish
                self.record_channel_result(
                    request,
                    channel,
                    completed=False
                )
                return

            self.cache_put(
                request.block,
                self.storage[request.block]["data"]
            )

            prefetch_elapsed = self.prefetch_after_read(request.block)
            if prefetch_elapsed > 0:
                request.stage_times_ms["prefetch"] = prefetch_elapsed
                elapsed += prefetch_elapsed

        else:
            request.request_class = "write"
            self.m["molecular_writes"] += 1

            for stage in WRITE_STAGES:
                elapsed += self._record_stage(
                    request,
                    stage
                )

            current_cycles = int(
                self.storage.get(
                    request.block,
                    {}
                ).get("rewrites", 0)
            )

            request.rewrite_cycle = current_cycles + 1

            ordinary_failure_probability = float(
                cfg.get(
                    "write_failure_probability",
                    0.0
                )
            )

            ordinary_failure_probability = min(
                1.0,
                max(
                    0.0,
                    ordinary_failure_probability
                )
            )

            wear_failure_probability = (
                self.endurance_probability(
                    request.block
                )
            )

            # Treat ordinary failure and wear failure as independent
            # mechanisms rather than simply taking the larger value.
            combined_probability = (
                1.0
                - (
                    1.0 - ordinary_failure_probability
                )
                * (
                    1.0 - wear_failure_probability
                )
            )

            if (
                self.rng.random()
                < combined_probability
            ):
                request.status = "failed"
                request.finish = (
                    request.start + elapsed
                )

                if wear_failure_probability > 0.0:
                    request.endurance_failure = True
                    self.m["endurance_failures"] += 1

                self.failed.append(request)
                self.busy[channel] = request.finish
                self.record_channel_result(
                    request,
                    channel,
                    completed=False
                )
                return

            new_cycles = current_cycles + 1

            self.storage[request.block] = {
                "addr": self.addr(request.block),
                "data": f"DATA-{request.block}",
                "size_bytes": request.size_bytes,
                "rewrites": new_cycles,
                "state": "verified",
            }

            self.m["total_rewrite_cycles"] += 1

            self.m["maximum_rewrite_cycles"] = max(
                self.m["maximum_rewrite_cycles"],
                new_cycles
            )

            self.cache_put(
                request.block,
                f"DATA-{request.block}"
            )

        request.finish = request.start + elapsed
        request.status = "completed"
        self.completed.append(request)
        self.busy[channel] = request.finish
        self.record_channel_result(
            request,
            channel,
            completed=True
        )

    def run(self, requests):
        self.total_requests = len(requests)
        self.processed_requests = 0
        self._emit_progress("run_started")
        self.queue = deque(requests)

        while (
            self.queue
            or any(
                x > self.now
                for x in self.busy
            )
        ):
            current_depth = len(self.queue)
            self.queue_depth_samples.append(current_depth)
            self.peak_queue_depth = max(
                self.peak_queue_depth,
                current_depth
            )

            self.order()

            free_channels = [
                i
                for i, busy_until in enumerate(
                    self.busy
                )
                if busy_until <= self.now
            ]

            while free_channels and self.queue:
                request = self.queue.popleft()
                self.execute(
                    request,
                    free_channels.pop(0)
                )
                self.processed_requests += 1
                self._emit_progress(
                    "request_complete",
                    operation=request.op,
                    request_id=request.req_id,
                    status=request.status,
                )

            future = [
                x
                for x in self.busy
                if x > self.now
            ]

            if future:
                self.now = min(future)
            elif self.queue:
                self.now += 0.001
            else:
                break

        rows = []

        for request in sorted(
            self.completed + self.failed,
            key=lambda x: x.req_id
        ):
            row = asdict(request)
            row["latency_ms"] = (
                request.finish - request.arrival
            )
            rows.append(row)

        self._emit_progress("run_complete")
        return rows


def workload(
    n,
    seed=1,
    read_ratio=0.7,
    blocks=100,
    simultaneous=False,
    locality_strength=4,
    size_bytes=4096,
):
    """Generate a reproducible workload."""

    rng = random.Random(seed)
    requests = []
    centers = [10, 30, 50, 70, 90]

    for i in range(n):
        operation = (
            "READ"
            if rng.random() < read_ratio
            else "WRITE"
        )

        center = rng.choice(centers)

        block = max(
            1,
            min(
                blocks,
                int(
                    rng.gauss(
                        center,
                        locality_strength
                    )
                )
            )
        )

        arrival = (
            0.0
            if simultaneous
            else i * 0.1
        )

        requests.append(
            Request(
                req_id=i + 1,
                op=operation,
                block=block,
                arrival=arrival,
                size_bytes=max(1, int(size_bytes)),
                locality=block % 16,
            )
        )

    return requests


def initialize_storage(simulator, blocks=100, size_bytes=4096):
    """Create valid molecular data blocks before read experiments."""

    for block in range(1, blocks + 1):
        simulator.storage[block] = {
            "addr": simulator.addr(block),
            "data": f"DATA-{block}",
            "size_bytes": max(1, int(size_bytes)),
            "rewrites": 0,
            "state": "verified",
        }


def summary(rows, simulator):
    completed = [
        row
        for row in rows
        if row["status"] == "completed"
    ]

    latencies = [
        row["latency_ms"]
        for row in completed
    ]

    total_time = max(
        [
            row["finish"]
            for row in rows
        ]
        or [0]
    )

    bytes_completed = sum(
        row["size_bytes"]
        for row in completed
    )

    requests = len(rows)
    errors_injected = simulator.m["errors_injected"]
    errors_recovered = simulator.m["errors_recovered"]

    if total_time > 0:
        channel_utilization = [
            min(1.0, busy_time / total_time)
            for busy_time in simulator.channel_busy_time_ms
        ]
    else:
        channel_utilization = [
            0.0
            for _ in simulator.channel_busy_time_ms
        ]

    mean_channel_utilization = (
        sum(channel_utilization) / len(channel_utilization)
        if channel_utilization
        else 0.0
    )

    active_channel_count = sum(
        1
        for count in simulator.channel_requests
        if count > 0
    )

    mean_sampled_queue_depth = (
        sum(simulator.queue_depth_samples)
        / len(simulator.queue_depth_samples)
        if simulator.queue_depth_samples
        else 0.0
    )

    completed_queue_waits = [
        row.get("queue_wait_ms", 0.0)
        for row in completed
    ]

    cold_read_count = sum(
        1 for row in completed
        if row.get("request_class") == "cold_read"
    )

    hot_read_count = sum(
        1 for row in completed
        if row.get("request_class") == "hot_read"
    )

    write_count = sum(
        1 for row in completed
        if row.get("request_class") == "write"
    )

    return {
        "requests": requests,
        "completed": len(completed),
        "failed": requests - len(completed),
        "transaction_success_rate": (
            len(completed) / requests
            if requests
            else 0.0
        ),
        "mean_latency_ms": (
            sum(latencies) / len(latencies)
            if latencies
            else None
        ),
        "p95_latency_ms": (
            sorted(latencies)[
                max(
                    0,
                    math.ceil(
                        0.95 * len(latencies)
                    ) - 1
                )
            ]
            if latencies
            else None
        ),
        "throughput_bytes_per_s": (
            bytes_completed
            / (total_time / 1000)
            if total_time
            else 0
        ),
        "cache_hits": simulator.m["cache_hits"],
        "molecular_reads": simulator.m["molecular_reads"],
        "molecular_writes": simulator.m["molecular_writes"],
        "errors_injected": errors_injected,
        "errors_recovered": errors_recovered,
        "error_recovery_rate": (
            errors_recovered / errors_injected
            if errors_injected
            else None
        ),
        "endurance_failures": simulator.m[
            "endurance_failures"
        ],
        "maximum_rewrite_cycles": simulator.m[
            "maximum_rewrite_cycles"
        ],
        "total_rewrite_cycles": simulator.m[
            "total_rewrite_cycles"
        ],
        "prefetch_enabled": simulator.prefetch_enabled(),
        "prefetch_depth": simulator.prefetch_depth(),
        "prefetch_attempts": simulator.m["prefetch_attempts"],
        "prefetch_completed": simulator.m["prefetch_completed"],
        "prefetch_errors": simulator.m["prefetch_errors"],
        "prefetch_errors_recovered": simulator.m["prefetch_errors_recovered"],
        "prefetch_time_ms": simulator.m["prefetch_time_ms"],

        "simulation_time_ms": total_time,
        "cold_read_count": cold_read_count,
        "hot_read_count": hot_read_count,
        "write_count": write_count,
        "mean_queue_wait_ms": (
            sum(completed_queue_waits) / len(completed_queue_waits)
            if completed_queue_waits
            else 0.0
        ),

        # Channel/resource telemetry
        "active_channel_count": active_channel_count,
        "mean_channel_utilization": mean_channel_utilization,
        "max_channel_utilization": (
            max(channel_utilization)
            if channel_utilization
            else 0.0
        ),
        "min_channel_utilization": (
            min(channel_utilization)
            if channel_utilization
            else 0.0
        ),
        "channel_utilization": channel_utilization,
        "channel_busy_time_ms": simulator.channel_busy_time_ms,
        "channel_requests": simulator.channel_requests,
        "channel_completed": simulator.channel_completed,
        "channel_failed": simulator.channel_failed,
        "channel_bytes_completed": simulator.channel_bytes_completed,
        "peak_queue_depth": simulator.peak_queue_depth,
        "mean_sampled_queue_depth": mean_sampled_queue_depth,
    }


def run(
    cfg,
    requests,
    initialize_blocks=False,
    blocks=100,
    block_size_bytes=4096,
    progress_callback=None,
):
    """Execute one simulation using supplied configuration and workload."""

    simulator = BioSSDSimulator(cfg, progress_callback=progress_callback)

    if initialize_blocks:
        initialize_storage(
            simulator,
            blocks,
            size_bytes=block_size_bytes
        )

    rows = simulator.run(requests)

    return (
        summary(rows, simulator),
        rows,
    )


def run_with_simulator(
    cfg,
    requests,
    initialize_blocks=False,
    blocks=100,
):
    """Execute a simulation and return the live simulator state."""

    simulator = BioSSDSimulator(cfg)

    if initialize_blocks:
        initialize_storage(
            simulator,
            blocks
        )

    rows = simulator.run(requests)

    return (
        summary(rows, simulator),
        rows,
        simulator,
    )
