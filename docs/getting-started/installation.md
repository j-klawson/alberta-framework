# Installation

## Requirements

- Python 3.13 or higher
- JAX 0.4 or higher

## Basic Installation

Install the core package:

```bash
pip install alberta-framework
```

## Optional Dependencies

The framework has several optional dependency groups for different use cases.

### Development

For running tests and linting:

```bash
pip install alberta-framework[dev]
```

Includes: pytest, pytest-cov, ruff, mypy

### Gymnasium Integration

For using RL environments as experience streams:

```bash
pip install alberta-framework[gymnasium]
```

Includes: gymnasium>=0.29.0

### Analysis Tools

For publication-quality experiments and visualization:

```bash
pip install alberta-framework[analysis]
```

Includes: matplotlib, scipy, joblib, tqdm

### Documentation

For building documentation locally:

```bash
pip install alberta-framework[docs]
```

Includes: mkdocs, mkdocs-material, mkdocstrings

### All Dependencies

Install everything:

```bash
pip install alberta-framework[dev,gymnasium,analysis,docs]
```

## Development Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/keith/alberta-framework.git
cd alberta-framework
pip install -e ".[dev,gymnasium,analysis,docs]"
```

## Verify Installation

```python
import alberta_framework
print(alberta_framework.__version__)
```

Run the test suite:

```bash
pytest tests/ -v
```
