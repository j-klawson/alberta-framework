# Contributing

Thank you for your interest in contributing to the Alberta Framework.

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/keith/alberta-framework.git
cd alberta-framework
```

2. Install in development mode:
```bash
pip install -e ".[dev,gymnasium,analysis,docs]"
```

3. Verify the setup:
```bash
pytest tests/ -v
```

## Code Standards

### Style

- Follow PEP 8 with 100 character line limit
- Use ruff for linting: `ruff check src/ tests/`
- Use mypy for type checking: `mypy src/`

### Docstrings

All public functions and classes must have NumPy-style docstrings:

```python
def example_function(param1: int, param2: str = "default") -> bool:
    """Short description of the function.

    Longer description if needed, explaining the behavior
    in more detail.

    Parameters
    ----------
    param1 : int
        Description of param1.
    param2 : str, optional
        Description of param2. Default is "default".

    Returns
    -------
    bool
        Description of return value.

    Examples
    --------
    >>> example_function(42)
    True

    References
    ----------
    .. [1] Author (Year). "Paper Title"
    """
    ...
```

### Type Hints

- All function signatures must have type hints
- Use modern Python 3.13 syntax (e.g., `list[int]` not `List[int]`)
- Use `jax.Array` for JAX arrays

### Testing

- Add tests for new functionality in `tests/`
- Aim for high coverage of core algorithms
- Use pytest fixtures from `conftest.py`

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with clear commit messages
3. Ensure all tests pass: `pytest tests/ -v`
4. Ensure linting passes: `ruff check src/ tests/`
5. Update documentation if needed
6. Submit a pull request with a clear description

## Design Principles

When contributing, keep these principles in mind:

1. **Temporal Uniformity**: All components update at every time step
2. **Immutable State**: Use NamedTuples, never mutate state
3. **Functional Style**: Pure functions enable JAX transformations
4. **Composition**: Prefer composition over inheritance

## Documentation

To build documentation locally:

```bash
pip install -e ".[docs]"
mkdocs serve
```

Then visit `http://localhost:8000`.

## Questions?

Open an issue on [GitHub](https://github.com/j-klawson/alberta-framework/issues) for questions or discussion.
