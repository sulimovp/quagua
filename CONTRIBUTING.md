# Contributing

We welcome issues and pull requests that fix bugs, improve documentation, or add small, well-scoped tests.

## Before you change code

- Match the existing style: type hints where already used, 4-space indent, no unnecessary dependencies.
- Keep prose in docs and log messages direct and factual; avoid marketing language.
- Docstrings and CLI-style messages in this repository stay English unless the project adopts another convention.

## Running tests

```bash
pip install -e ".[dev]"
pytest -q
```

## Licensing

By contributing, you agree your contributions are under the same license as the project (Apache-2.0), unless you explicitly state otherwise in the pull request for a trivial fix (typo) that is not copyrightable.
