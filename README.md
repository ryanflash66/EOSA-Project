# Enfield Outage & Load Forecasting Research Toolkit

Integrated research codebase for the City of Enfield power resiliency project. The toolkit has two pillars:

- **Automated outage intelligence** – a Playwright-driven service that discovers and polls Duke Energy outage feeds, filters events to a study region, and exposes exports plus an operational FastAPI surface.
- **Demand forecasting & microgrid analysis** – reproducible scripts and notebooks for short‑horizon load prediction, PV+storage dispatch studies, and sensitivity analysis that support planning and design.

The audience is graduate‑level researchers and collaborators who need to reproduce experiments, extend the code, or plug the outputs into downstream analytics.

---

## Research Problem & Motivation
- Provide **near real‑time situational awareness** of Duke Energy outages for Enfield and neighboring jurisdictions to inform emergency response and infrastructure planning.
- Produce **data‑driven load forecasts** and **microgrid what‑if analyses** to size distributed PV + storage that can sustain critical services during grid disturbances.
- Deliver a **reproducible, inspectable codebase** so that results can be audited, re‑run with new data, or adapted to adjacent territories.

### Approach & Key Contributions
- **Automated endpoint discovery:** Playwright observes the live outage map, captures authenticated request templates (headers, params, jurisdiction token), and caches them for replay.
- **Robust polling pipeline:** httpx + APScheduler drive recurring requests; spatial filters (bbox/radius/viewport) prune events; detail enrichment merges per‑event metadata; structured JSON logs support traceability.
- **Normalized exports + API:** Events are canonicalised to CSV/GeoJSON and served through a FastAPI “ops plane” for dashboards or alerts.
- **Forecasting feature stack:** Enhanced calendar/lag/rolling features for 15‑minute and hourly horizons, with Gradient Boosting, HistGradientBoosting, Ridge, SARIMAX baselines, and reproducible train/test splits.
- **Microgrid feasibility tools:** Deterministic greedy heuristic and CVXPY optimisation demonstrate how PV/battery sizing affects unmet load for critical facilities; optional NREL PVWatts data integration.

---

## Methodology

### Outage Collector Service (`Data Scraping/outage-collector`)
1. **Configuration resolution** (`collector/config.py`): CLI → `.env` → presets precedence; presets define jurisdiction, bbox, radius, viewport polygon, and optional extra query parameters. Defaults target Raleigh (DEC) but any preset in `presets/places.yml` can be used.
2. **Endpoint discovery** (`collector/discovery.py`): Playwright visits Duke outage map entry points, scores candidate XHR/fetch requests, sanitises headers, and saves a `DiscoveryTemplate` cache (overrideable with `--discovery-override-url`).
3. **Polling loop** (`collector/poller.py`): httpx replays the template, enforces jurisdiction param, retries/refreshes discovery on 401/shape errors if enabled, caps events per cycle, and records raw payloads.
4. **Spatial filtering** (`collector/filters.py`): stage‑A bbox policy (centroid/vertex/intersects), optional viewport polygon clamp, and radius pruning relative to preset centre; bbox padding supported.
5. **Normalization & enrichment** (`collector/normalizer.py`, `collector/enrichment.py`): harmonises IDs/coordinates/causes/timestamps into `NormalizedEvent` objects; optional detail fetch per event merges crew status, ETR, and refined polygons.
6. **Exports & ops plane** (`collector/storage.py`, `collector/service.py`, `collector/scheduler.py`): CSV/GeoJSON writers; FastAPI endpoints for `/healthz`, `/stats`, `/events/latest`, `/config`; APScheduler launches poll job immediately and at `poll_interval_s`.
7. **Logging & observability** (`collector/logging_utils.py`): JSON‑structured logs with redacted config snapshots for auditability.

Assumptions: Duke’s public outage map remains reachable; jurisdiction tokens stay stable; Playwright can launch Chromium in headless mode on the host. If the map HTML or API schema changes, discovery heuristics may need retuning.

### Load Forecasting & Microgrid Analysis (`Prediction Models`, `Data Scraping/power-usage`)
- **Data**: Cleaned 15‑minute whole‑site energy series (`model_training/content/data/15_minute_timeseries_data_cleaned_ready.csv`) with pre‑split train/test CSVs at both 15‑minute and hourly resolution. Large Excel source also available.
- **Feature engineering** (`model_training/feature_engineering.py`): calendar features; extensive target lags; rolling stats; optional lagged sub‑meter aggregates; granularity‑aware window sizes.
- **Enhanced training pipeline** (`model_training/run_enhanced_training.py`): trains Gradient Boosting, HistGradientBoosting, Ridge; persists models and predictions under `content/data/models` and `content/data/analysis_outputs`; evaluates MAE/RMSE/MAPE/R2/Bias.
- **Exploratory script** (`Enfieild_outage.py`): end‑to‑end cleaning, aggregation, model training (including naive baseline), error breakdowns by period, and plot generation to `Prediction Models/plots`.
- **Time‑series baselines** (`model_training/sarimax.py`): grid search over SARIMAX orders with state‑budget guardrails.
- **Microgrid heuristics** (`power-usage/Power_usage_summary.py`): greedy dispatch across PV/battery/load sweeps with deterministic seeds; writes sensitivity table.
- **Convex dispatch demo** (`power-usage/nrel_api_demo.py`): CVXPY optimisation minimizing unmet load over 48h; optional PVWatts fetch via `NREL_API_KEY`; saves `microgrid_results.csv` and plots.

---

## Repository Layout

| Path | Purpose |
| --- | --- |
| `Data Scraping/outage-collector/collector/` | Outage service source (discovery, poller, filters, enrichment, exports, FastAPI). |
| `Data Scraping/outage-collector/presets/places.yml` | Place definitions (jurisdiction, bbox, polygon, radius, optional query params). |
| `Data Scraping/outage-collector/run_collector.ps1` | Helper launcher wrapping `python -m collector` with common flags. |
| `Data Scraping/power-usage/` | Microgrid sensitivity (`Power_usage_summary.py`) and CVXPY/NREL dispatch demo (`nrel_api_demo.py`). |
| `Prediction Models/model_training/` | Feature engineering, enhanced training pipeline, SARIMAX utility, cleaned datasets, and train/test splits. |
| `Prediction Models/plots/` | Saved evaluation figures from `Enfieild_outage.py`. |
| `Prediction Models/Enfieild_outage.py` | Monolithic exploratory training/evaluation script generating plots and metrics. |
| `requirements.txt` | Unified dependency pins (Python 3.11+). |

---

## Installation & Environment
1. **Create a Python 3.11+ virtual environment**:
   ```bash
   python -m venv .venv
   # Windows
   .\.venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
   ```
2. **Install dependencies** (covers outage collector, forecasting, and CVXPY demos):
   ```bash
   pip install -r requirements.txt
   ```
3. **Install Playwright browsers** (needed only for the outage collector discovery step):
   ```bash
   python -m playwright install
   ```
4. **Dev tools (optional)**: inside `Data Scraping/outage-collector`, install extras for formatting, linting, and tests:
   ```bash
   pip install .[dev]
   ```
5. **System prerequisites**: Playwright headless Chromium requires system libraries on Linux; CVXPY will default to open-source solvers (OSQP, Clarabel) included in requirements.

---

## Usage

### Outage Collector Service
- **Run with defaults (Raleigh preset, CSV+GeoJSON, port 8000):**
  ```bash
  cd "Data Scraping/outage-collector"
  python -m collector
  ```
- (Optional, Windows-only convenience) `run_collector.ps1` wraps the same command with defaults.
- **Custom run (example: Charlotte, geojson only, 60 s polling, tighter viewport):**
  ```bash
  python -m collector \
    --place charlotte \
    --format geojson \
    --poll-interval-s 60 \
    --viewport-only \
    --padding-km 2.0 \
    --log-level INFO
  ```
- **Key CLI/env options** (see `collector/config.py` for full list):
  - `--place <name>` or `--presets-file <path>` to select a preset.
  - `--format {csv,geojson,both}`; outputs land in `data/` by default.
  - `--output-dir <path>` to redirect exports.
  - `--bbox-policy {centroid,vertex,intersects}`, `--radius-km <float>`, `--viewport-only`, `--padding-km <float>` to control spatial filters.
  - `--poll-interval-s`, `--max-events-per-cycle`, `--rebootstrap-on-401` for resilience.
  - `--discovery-url` or `--discovery-override-url` to seed/bypass Playwright.
  - `.env` keys mirror CLI (e.g., `PLACE`, `FORMAT`, `POLL_INTERVAL_S`, `JURISDICTION`, `QUERY_PARAMS` via presets).
- **API surface (FastAPI, served on port 8000):**
  - `/healthz` – readiness plus last poll summary.
  - `/stats` – latest poll stats and export paths.
  - `/events/latest?limit=100` – recent normalized events.
  - `/config` – redacted runtime config and discovery template.
- **Exports & state**: CSV (`outage_events.csv`) and GeoJSON (`outage_events.geojson`) are written under the configured output directory; the service retains the most recent events, stats, and errors in memory for the API.

### Microgrid & Load Analysis
- **Greedy sensitivity sweep**:
  ```bash
  python "Data Scraping/power-usage/Power_usage_summary.py"
  ```
  Writes `sensitivity_table.csv` and prints PV/battery/load %served table (deterministic RNG seed).

- **Convex dispatch with optional NREL PVWatts**:
  ```bash
  # Optional: set an API key for real PV traces
  # Windows (cmd):    set NREL_API_KEY=<your_api_key>
  # macOS/Linux (sh): export NREL_API_KEY=<your_api_key>
  python "Data Scraping/power-usage/nrel_api_demo.py"
  ```
  Produces `microgrid_results.csv` plus plots; falls back to a synthetic PV curve if no API key.

### Load Forecasting Pipelines
- **Enhanced, reproducible training (uses pre-split CSVs in `model_training/content/data`)**:
  ```bash
  cd "Prediction Models/model_training"
  python run_enhanced_training.py
  ```
  Outputs:
  - Trained models: `content/data/models/{hourly,15min}/*.joblib`
  - Predictions: `content/data/analysis_outputs/preds_{hourly,15min}/*.csv`
  - Metrics JSON: `content/data/analysis_outputs/enhanced_metrics.json`

- **Exploratory end-to-end script** (plots + metrics):
  ```bash
  python "Prediction Models/Enfieild_outage.py"
  ```
  Regenerates figures in `Prediction Models/plots` and prints MAE/RMSE/MAPE/sMAPE/CVRMSE/NMBE/R2 for 15-minute and hourly horizons.

- **SARIMAX baseline** (choose seasonal grids):
  ```bash
  python "Prediction Models/model_training/sarimax.py" --train content/data/train_15min.csv --test content/data/test_15min.csv --seasonal-period 96
  ```
  Saves `pred_SARIMAX.csv` alongside the test file.

---

## Experiments & Reproducibility
- **Determinism**: Random seeds fixed where applicable (e.g., scikit‑learn models `random_state=42`; sensitivity sweep RNG seed=1). CVXPY solves are deterministic given the solver.
- **Data splits**: Use the provided `train_15min.csv`, `test_15min.csv`, `train_hourly.csv`, `test_hourly.csv` to mirror published results. The enhanced pipeline reads these splits automatically.
- **Reproducing outage exports**: Ensure Playwright browsers are installed, run the collector with the same preset/filters, and retain the emitted CSV/GeoJSON plus the structured logs for audit.
- **Metrics regeneration**: Rerun `run_enhanced_training.py`; metrics are stored in `analysis_outputs/enhanced_metrics.json`. Plots from `Enfieild_outage.py` are recreated under `Prediction Models/plots`.

---

## Datasets
- **Cleaned 15‑minute series**: `Prediction Models/model_training/content/data/15_minute_timeseries_data_cleaned_ready.csv` (and Excel analogue) – whole‑site energy plus sub‑meters.
- **Train/test splits**: `train_15min.csv`, `test_15min.csv`, `train_hourly.csv`, `test_hourly.csv` derived from the cleaned series (timestamps as index).
- **Generated artifacts**: `hourly_from_15min.csv` and `hourly_from_15min_electricity.csv` are aggregated versions; model predictions and metrics are produced under `content/data/analysis_outputs`.
- **Outage exports**: Written to `Data Scraping/outage-collector/data/` (or the configured `--output-dir`).
- **Licensing/PII**: Source data may contain operational details; share externally only with project approval. No public redistribution license is asserted in this repo.

---

## Results Artifacts
- **Plots**: All evaluation figures live in `Prediction Models/plots` (actual vs predicted, MAE/RMSE by period, heatmaps, rolling MAE, residuals).
- **Microgrid**: `microgrid_results.csv` summarises dispatch, SoC, curtailment, and unmet load for the CVXPY scenario.
- **Outage collector**: Latest poll stats and export paths are available via `/stats`; normalized events via `/events/latest`.

---

## Testing & Quality
- Outage collector: `pytest` from `Data Scraping/outage-collector` (tests cover config resolution, discovery heuristics, spatial filters, polling, storage, API).
  ```bash
  cd "Data Scraping/outage-collector"
  pytest
  ```
- Formatting/linting (optional dev extras): `ruff check .` and `black .` within the outage collector package.

---

## Notes & Limitations
- Discovery relies on Duke’s public outage map structure; significant front‑end/API changes may require updating request heuristics or presets.
- Playwright requires a GUI-capable environment for non‑headless runs; headless mode is default (`COLLECTOR_PLAYWRIGHT_HEADLESS=1`).
- Outage events are filtered using coarse geometry; false negatives/positives are possible at jurisdiction boundaries.
- Forecasting models assume stationarity in feature relationships; domain drift or meter reconfiguration will degrade performance and require retraining.
- Microgrid simulations use simplified device constraints and do not model inverter/thermal limits or tariff economics.

---

## Citation
If you use this toolkit in academic work, please cite it. Replace the fields as appropriate:

```bibtex
@misc{enfield_outage_toolkit_2025,
  title   = {Enfield Outage and Load Forecasting Research Toolkit},
  author  = {City of Enfield Power Project Team},
  year    = {2025},
  note    = {Version 0.1.0. Available at the project repository},
}
```

---

For questions or collaboration, open an issue or contact the project maintainers through the Enfield Power Project team.
