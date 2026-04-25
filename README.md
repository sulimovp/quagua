# QuaGua

QuaGua is a small Python library for **encode-then-mix** pipelines on tabular data: you apply an encoder (often expanding or scrambling features), then a mixer that combines dimensions. It also ships an **evaluation harness** that measures downstream task performance (accuracy, F1, AUC) and simple **reconstruction-style attacks** (linear models, random forest, MLP) so you can compare utility and attack success in one place.

This is engineering tooling for prototyping and evaluation, not a certified privacy product. It does **not** implement differential privacy or formal guarantees. Use it to run controlled experiments; treat reported numbers as dataset- and model-dependent.

## Install

Requires Python 3.10+.

```bash
git clone <your-fork-or-mirror-url>  # original: git@github.com:sulimovp/quagua.git
cd quagua
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux / macOS
pip install -e ".[dev]"
```

The base install pulls in NumPy, pandas, scikit-learn, and SciPy. The `dev` extra adds pytest for running the test suite.

## Quick start

```python
import numpy as np
from quagua.encoders import (
    RandomOneToManyEncoder,
    ClassicalXORMixer,
    EncodeThenMixPipeline,
)

rng = np.random.default_rng(42)
X = rng.integers(0, 3, size=(100, 5)).astype(float)

encoder = RandomOneToManyEncoder(n_encodings_per_value=3, encoding_dim=4)
mixer = ClassicalXORMixer()
pipe = EncodeThenMixPipeline(encoder, mixer)

X_out = pipe.fit_transform(X)
print(X.shape, "->", X_out.shape)
```

For API details, optional quantum-inspired encoders, and the evaluation framework, see [docs/overview.md](docs/overview.md) and [docs/evaluation.md](docs/evaluation.md).

## Documentation

| Document | Contents |
|----------|----------|
| [docs/overview.md](docs/overview.md) | Scope, components, what is optional |
| [docs/installation.md](docs/installation.md) | Environments, editable install |
| [docs/evaluation.md](docs/evaluation.md) | `PrivacyAITestFramework` and metrics |
| [docs/datasets.md](docs/datasets.md) | Built-in loaders (Titanic, Adult, etc.) |
| [docs/threat_model.md](docs/threat_model.md) | Non-goals and how to read attack scores |

## Tests

```bash
pytest -q
```

## License

Apache License 2.0. See [LICENSE](LICENSE).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
