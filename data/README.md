# Logistic Regression Datasets

All CSV files in this directory are normalized for the experiment loader: numeric
feature columns plus a binary label column named `y` or `target`.

## Generated Synthetic Datasets

These files were generated with `data/generate_logistic_data.py`:

- `synthetic_logistic_small_5d.csv`: small balanced sanity-check dataset.
- `synthetic_logistic_sparse_20d.csv`: 20-dimensional dataset with only a few
  informative coordinates, useful for lasso sparsity tests.
- `synthetic_logistic_noisy_20d.csv`: same sparse structure with 10% label
  flips, useful for robustness checks.
- `synthetic_logistic_imbalanced_10d.csv`: shifted intercept and mild noise,
  useful for class-imbalance behavior.

Regenerate them from the repository root:

```bash
python data/generate_logistic_data.py --n-samples 240 --dimension 5 --beta 2.0 -1.5 1.0 -0.5 0.25 --intercept 0.0 --seed 10 --flip-prob 0.0 --output data/synthetic_logistic_small_5d.csv
python data/generate_logistic_data.py --n-samples 1000 --dimension 20 --beta 2.5 -2.0 1.5 -1.0 0.75 0.5 0 0 0 0 0 0 0 0 0 0 0 0 0 0 --intercept 0.0 --seed 20 --flip-prob 0.0 --output data/synthetic_logistic_sparse_20d.csv
python data/generate_logistic_data.py --n-samples 1000 --dimension 20 --beta 2.5 -2.0 1.5 -1.0 0.75 0.5 0 0 0 0 0 0 0 0 0 0 0 0 0 0 --intercept 0.0 --seed 21 --flip-prob 0.1 --output data/synthetic_logistic_noisy_20d.csv
python data/generate_logistic_data.py --n-samples 800 --dimension 10 --beta 1.5 -1.25 1.0 -0.75 0.5 0.25 0 0 0 0 --intercept -1.2 --seed 30 --flip-prob 0.03 --output data/synthetic_logistic_imbalanced_10d.csv
```

## Downloaded Public Datasets

These files are downloaded from the UCI Machine Learning Repository and converted
to the same CSV schema:

- `iris.csv`: binary Setosa vs non-Setosa Iris.
- `breast_cancer_wdbc.csv`: Wisconsin Diagnostic Breast Cancer.
- `banknote_authentication.csv`: Banknote Authentication.
- `spambase.csv`: Spambase spam classification.

Download and normalize them from the repository root:

```bash
python data/download_binary_datasets.py
```
