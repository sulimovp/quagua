# Overview

QuaGua targets **tabular** machine learning: rows are records, columns are features, and the task is usually classification or regression after encoding.

## Encode-then-mix

Many pipelines in this repository follow two steps:

1. **Encoder** — maps the original feature matrix to another representation. Some encoders are **irreversible** in the sense that you do not store an invertible map back to raw inputs; others are classical baselines (e.g. random projection).
2. **Mixer** — operates on the encoded matrix and combines information across dimensions (for example XOR-style mixing on discrete values, or additive mixing on normalized floats).

The class `EncodeThenMixPipeline` wires `encoder.fit` / `encoder.transform` and then `mixer.fit` / `mixer.transform` in that order.

## Optional quantum-inspired encoders

Modules under `quagua.encoders` include encoders whose names refer to quantum ideas (superposition, phase, walks, etc.). They are **numerical encoders** implemented with NumPy: they do not require a quantum computer and are not hardware-validated quantum protocols. Treat them as optional experimental transforms alongside classical and irreversible encoders.

## Evaluation harness

`PrivacyAITestFramework` (in `quagua.evaluation`) trains standard sklearn models on encoded data for **utility**, and trains attack models that try to predict original features or sensitive attributes from encoded representations for a **coarse privacy signal**. The implementation prints summaries and returns structured results for further analysis.

Numbers depend on the dataset, train/test split, random seeds, and model choices. Compare methods under the same protocol rather than relying on a single absolute score.

## Repository layout

- `quagua/` — installable package
- `docs/` — prose documentation
- `tests/` — pytest suite
