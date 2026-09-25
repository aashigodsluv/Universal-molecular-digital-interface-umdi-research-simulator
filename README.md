# Universal Molecular Digital Interface (UMDI) Research Simulator

A computational research simulator for the Universal Molecular Digital Interface (UMDI) architecture and BioSSD, designed to evaluate whether modeled molecular-storage configurations can approach SSD-class operational performance.

UMDI is the molecular-to-digital storage architecture. BioSSD is a device implementation of that architecture. This simulator is the computational environment used to evaluate UMDI/BioSSD behavior, performance assumptions, and SSD-class operating targets.

## What the simulator models

- discrete-event READ/WRITE workloads
- queueing and controller scheduling
- molecular-channel concurrency
- electronic cache and sequential prefetch
- molecular READ/WRITE pathway timing
- error injection, correction, and recovery
- rewrite/endurance assumptions
- logical-block and payload scaling
- host-visible latency and throughput
- channel utilization and queue depth
- SSD-class target analysis
- technology-gap analysis
- standardized large-payload benchmark reproduction

The simulator models architectural and operational behavior. It does not simulate molecular chemistry or atomic dynamics.

## Two simulation modes

### Transaction Simulator

Models BioSSD under mixed READ/WRITE activity, including queueing, caching, scheduling, errors, prefetching, channel utilization, and transaction-level telemetry. It is used to inspect how the architecture behaves during host-visible storage activity and to evaluate that behavior against SSD-class operational targets.

### Benchmark Simulator

Evaluates large-payload BioSSD performance against SSD-class targets using payload size, logical-block size, effective molecular channels, READ-path acceleration, and WRITE-path acceleration. It can reproduce the published optimized reference configuration or test a researcher-defined configuration.

Both modes share analysis views for Overview, READ/WRITE, Targets, Technology Gap, Pipeline, and Channel Scaling.

## Published benchmark reference

Reference configuration:

- Payload: **1 GiB**
- Logical block size: **256 KiB**
- Effective molecular channels: **4,096**
- READ-path acceleration: **4×**
- WRITE-path acceleration: **8×**

Published modeled result:

- READ latency: **1.235 s**
- WRITE latency: **1.201 s**
- WRITE-to-READ round trip: **2.436 s**
- READ throughput: **869 MB/s**
- WRITE throughput: **894 MB/s**

Benchmark runs are stochastic. The published result is retained as the fixed research reference while fresh runs may vary slightly.

## Repository structure

```text
configs/                 Reproducible model and technology-reference configuration
figures/                 Research figures generated from experiment outputs
frontend/                Interactive browser interface
results/                 Exported research experiment outputs
src/                     Core BioSSD simulation engine
frontend_api.py           Local HTTP API connecting the browser UI to Python
run_experiments.py        Batch research experiment runner
plot_results.py           Figure-generation script
RUN_UMDI_EXPERIMENTS.bat  Windows batch-experiment launcher
START_INTERACTIVE_UMDI.bat Windows interactive-simulator launcher
```

## Requirements

- Python 3.11+ recommended
- `matplotlib`
- A modern web browser

Install the Python dependency:

```bash
python -m pip install -r requirements.txt
```

## Run the interactive simulator

### Windows

Double-click:

```text
START_INTERACTIVE_UMDI.bat
```

This starts `frontend_api.py` at:

```text
http://127.0.0.1:8766
```

and opens `frontend/index.html` in your browser.

Keep the Python engine terminal open while using the simulator.

### Manual launch

From the repository root:

```bash
python frontend_api.py
```

Then open:

```text
frontend/index.html
```

## Run the full experiment suite

```bash
python run_experiments.py
```

On Windows you can instead double-click:

```text
RUN_UMDI_EXPERIMENTS.bat
```

The experiment suite exports reproducible CSV/JSON outputs to `results/` and generates research figures in `figures/`.

## Reproducibility

The repository intentionally includes the current `results/` and `figures/` used by the research workflow. These provide inspectable reference outputs while `run_experiments.py` allows them to be regenerated from the supplied configuration and simulation engine.

`configs/baseline.json` contains the baseline simulator assumptions. `configs/contemporary_baseline.json` provides the demonstrated-technology comparison profile used by Technology Gap analysis.


## Scientific scope

Model parameters are computational assumptions unless explicitly tied to experimental measurements or literature-derived values. The simulator is intended for architecture evaluation, sensitivity testing, reproducibility, and researcher-defined configuration comparison rather than as a claim of an existing physical SSD-speed molecular device.

## License

This software is released under the **PolyForm Noncommercial License 1.0.0**. See `LICENSE`.

## Citation

Citation metadata is provided in `CITATION.cff`. The reserved Zenodo DOI for this software release is **10.5281/zenodo.22912687**.

## Author

Abraham Ikongshul Ashindortiang  
BioSSD / UMDI Research
