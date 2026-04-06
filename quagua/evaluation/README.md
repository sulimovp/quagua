# Evaluation Module

This module contains the evaluation framework for testing privacy-preserving encoding methods.

## Components

- **`framework.py`**: Main evaluation framework
  - `PrivacyAITestFramework`: Comprehensive framework for evaluating encoding methods

## Usage

```python
from quagua.evaluation import PrivacyAITestFramework

# Initialize framework
framework = PrivacyAITestFramework()

# Evaluate a single encoder
results = framework.evaluate_encoding(
    encoder=encoder,
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
    encoder_name="My Encoder"
)

# Run comparison of multiple encoders
all_results = framework.run_comparison(
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
    encoders=my_encoders
)

# Visualize results
framework.plot_results(all_results)

# Statistical comparison
framework.compare_methods_statistically(all_results, selected_methods)
```

## Metrics

The framework evaluates:
- **Utility**: ML model accuracy, F1 score, ROC AUC
- **Privacy**: Reconstruction error, correlation, R² score
- **Efficiency**: Dimension expansion, computation time
