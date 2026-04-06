# Installation

## Python version

Use **Python 3.10 or newer**. The package is tested with standard CPython on Windows and Linux-style paths in CI-friendly setups; macOS should behave the same if dependencies build cleanly.

## Virtual environment

Create an isolated environment so dependency versions stay predictable:

```bash
python -m venv .venv
```

Activate it (Windows PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

Activate it (bash):

```bash
source .venv/bin/activate
```

## Editable install

From the repository root (the directory that contains `pyproject.toml`):

```bash
pip install -U pip
pip install -e .
```

With development dependencies (Jupyter, Matplotlib, pytest):

```bash
pip install -e ".[dev]"
```

## Notebooks

After the editable install, start Jupyter from the repo root or any subdirectory:

```bash
jupyter notebook
```

Open files under `examples/notebooks/`. Kernel should resolve the local `quagua` package via the editable install.

## Offline or locked networks

If `pip` cannot reach PyPI, vendor wheels or use an internal index, then run the same `pip install -e ".[dev]"` against that index. Loaders that fetch CSVs from the public internet (see [datasets.md](datasets.md)) need network access or local file substitutes.
