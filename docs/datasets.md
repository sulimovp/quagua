# Datasets

Helpers live in `quagua.data_utils`.

## Titanic

`load_titanic_data(include_continuous=True)` downloads the CSV used in many tutorials from a public GitHub raw URL (Data Science Dojo mirror). It returns `(X, y, feature_names)` as NumPy arrays and a name list after basic imputation and categorical encoding.

If your environment blocks outbound HTTPS, download the file manually and adapt the loader or pass a local path in a fork.

## Adult Census

`load_adult_census_data()` loads the UCI Adult-style census classification problem via the standard public URL used in the original helper. Check corporate policy before pulling external data from work machines.

## Credit card fraud

`load_credit_card_data()` expects to fetch from the UCI repository URL embedded in the code. The dataset is imbalanced; stratified splits are recommended when you evaluate classifiers.

## Generic entry point

`load_dataset(dataset_name)` dispatches on string names defined in that function. Prefer explicit loaders when you document an experiment so readers know which URL or file you used.

## Legal and compliance

These loaders are for **research and prototyping**. Production use requires your own data contracts, retention policy, and regional rules (GDPR, sector-specific constraints, etc.). QuaGua does not store data for you.
