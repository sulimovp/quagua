# Evaluation

The main entry point is `quagua.evaluation.PrivacyAITestFramework`.

## Utility

For classification-style setups, the framework can fit several sklearn models on **encoded** train features and report accuracy, F1, and ROC AUC on a held-out test set. This answers whether the encoding still supports the downstream task you care about.

## Attack models

The same encoded representations can be fed to **attack** regressors or classifiers that try to recover original columns or sensitive fields. The code uses linear models, tree ensembles, and small MLPs depending on the code path you call. Lower reconstruction quality or attack AUC (in context) suggests the encoding hides that signal more effectively **for that attacker model and split** — not a universal guarantee.

## How to run

Load data with `quagua.data_utils`, build an encoder or pipeline, then call `PrivacyAITestFramework` methods in your own script. The exact method names and return dictionaries are defined in `quagua/evaluation/framework.py` — always match the version you installed.

## Interpreting output

- **Expansion factor** — ratio of encoded width to original feature count. High expansion can help mixing but costs storage and compute.
- **Correlation and R² style metrics** — reported where the implementation computes reconstruction-related scores. Constant or degenerate features are handled with guards so runs do not crash; read any WARNING lines in the log.
- **Multiple runs** — where the framework repeats experiments, use variance across runs as a sanity check, not only the mean.

## Reproducibility

Set NumPy and sklearn random seeds in your driver script or notebook before fitting encoders that draw random parameters. Some encoders resample per `transform` call; document your protocol if you publish numbers.
