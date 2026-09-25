# BioSSD Simulator Unit and Live-Progress Validation Audit

**Audit date:** September 2, 2026

## Verified benchmark byte counts

- Payload: **1,073,741,824 bytes = 1 GiB**
- Logical block: **262,144 bytes = 256 KiB**
- Logical blocks: **4,096**
- READ latency: **1.234970651 s**
- WRITE latency: **1.200826256 s**
- Round trip: **2.435796907 s**
- READ throughput: **869.447239880 MB/s** (decimal MB/s)
- WRITE throughput: **894.169176165 MB/s** (decimal MB/s)

## Terminology corrections

- Binary payload/block sizes use IEC prefixes: **KiB, MiB, GiB**.
- Throughput remains decimal **MB/s**, converted using **1 MB = 1,000,000 bytes**.
- The benchmark UI displays **1 GiB** and **256 KiB**.
- The benchmark source JSON labels the exact implemented byte counts correctly.
- `run_experiments.py` `human_bytes()` emits IEC binary units.
- `plot_results.py` throughput conversion matches decimal MB/s labels.

## Live execution-state validation

- `src/biossd_sim.py` emits real stage events.
- `frontend_api.py` exposes `/progress` through `ThreadingHTTPServer`.
- `frontend/index.html` polls `/progress` and drives pipeline state from backend telemetry.
- The old indeterminate compact-progress animation was removed so the real percentage is visible.
- Navigation hover explanations and Scaling/global-run status integration are retained.

## Result-integrity note

No benchmark latency or throughput result was changed during this audit. Only unit nomenclature and conversion consistency were corrected.
