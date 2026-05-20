# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

**Maria** is a ground-based millimeter/submillimeter telescope simulator. It models atmospheric turbulence, CMB signals, sky maps, instrument properties, and detector noise to produce realistic Time-Ordered Data (TOD) for experiments like AtLAST and CMB-S4.

## Commands

```bash
# Install with development dependencies
pip install -e ".[dev]"
pre-commit install

# Run all tests
pytest -s -vv

# Skip tests requiring internet/data downloads
pytest -m "not internet"

# Lint and format
ruff check .
ruff format .
```

Line length is 125 characters. Style is enforced via `ruff` (rules E, I). Google-style docstrings.

## Architecture

The core abstraction is `Simulation` (`maria/sim/simulation.py`), which composes four mixin classes:
- `AtmosphereMixin` — turbulent atmospheric emission layers
- `CMBMixin` — CMB signal via HEALPix maps
- `MapMixin` — sky map sampling at detector positions
- `NoiseMixin` — white + pink (1/f) detector noise

### Simulation Pipeline

1. **Initialization**: Parse YAML configs → create `Instrument`, `Plan`, `Site` → build `Observation` objects
2. **Run** (`sim.run()` → `run_obs()`): For each observation, simulate all signal components and combine them
3. **Output**: A `TOD` object containing per-detector time series for each signal component plus metadata

### Key Classes

| Class | Location | Role |
|---|---|---|
| `Simulation` | `maria/sim/simulation.py` | Top-level orchestrator |
| `Instrument` | `maria/instrument/` | Detector arrays + bands |
| `Array` / `ArrayList` | `maria/array/` | Detector geometry (hexagonal packing, offsets) |
| `Band` | `maria/band/` | Frequency channels with noise/efficiency properties |
| `Plan` / `PlanList` | `maria/plan/` | Pointing patterns (daisy, lissajous, boustrophedon, …) |
| `Site` | `maria/site/` | Observatory location + weather model |
| `Observation` | `maria/sim/observation.py` | Combines instrument + plan + site for one scan |
| `TOD` | `maria/tod/` | Output container: data dict + coordinates + metadata |
| `Atmosphere` | `maria/atmosphere/` | Sparse GP turbulence model (2D/3D), Atacama weather stats |
| `ProjectionMap` / `HEALPixMap` | `maria/map/` | Flat-sky and spherical sky maps with WCS/FITS support |
| `Coordinates` | `maria/coords/` | Pointing frames (Az/El, RA/Dec) with JAX-accelerated transforms |
| `Quantity` | `maria/units/` | Units-aware array (K_RJ, K_CMB, pW, …) |

### Configuration System

All components are driven by YAML configs. Configs live alongside source:
- `maria/instrument/configs/` — instrument definitions
- `maria/band/configs/` — frequency channel properties
- `maria/plan/plans/` — pointing pattern definitions
- `maria/site/sites/` — observatory locations
- `maria/sim/params.yml` — simulation parameter schema

Kwargs passed to `Simulation()` override YAML config values. Most string arguments accept either a pre-defined config name or a dict.

### Typical Usage

```python
import maria
sim = maria.Simulation(
    instrument="mustang2",
    plans="ten_second_zenith_stare",
    site="alma",
    atmosphere="2d",
    cmb=True,
    noise=True,
)
tod = sim.run()
```

### Notable Implementation Details

- **JAX** is used heavily for numerical computation; some functions are JIT-compiled. Coordinate transforms in `maria/coords/` use JAX.
- **Dask** arrays are used for large TOD datasets.
- **Atmospheric model**: Sparse-precision auto-regressive Gaussian process; can be slow for large arrays. Progress bars indicate long operations.
- **Caching**: External data (atmosphere stats, CMB maps, Planck data) cached via `maria.io.set_cache_dir()`.
- **`dtype`** parameter (default `float32`) controls precision across the pipeline.
- **Custom errors**: `InvalidSimulationParameterError`, `PointingError`, `FrequencyOutOfBoundsError`, `ConfigurationError`, `ShapeError`.

### Testing

Tests mirror the source structure under `maria/tests/`. The `internet` marker gates tests that download external data — skip them with `-m "not internet"` in offline environments.
