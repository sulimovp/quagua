#!/usr/bin/env python3
"""
Build LaTeX tables for sex-attribute reconstruction from saved evaluation pickles.

Expects ``results/evaluation_results.pkl`` at the repository root (produce it by
running the evaluation notebook and saving the same structure). Writes ``.tex``
files under ``paper_tables/``.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score

# Repository root (parent of scripts/)
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from quagua.data_utils import load_adult_census_data

# Load saved results
results_file = repo_root / "results" / "evaluation_results.pkl"
if not results_file.exists():
    print(f"Error: {results_file} not found. Please run the notebook first.")
    sys.exit(1)

import pickle
with open(results_file, 'rb') as f:
    all_results = pickle.load(f)

# Define directories
sds_dir = project_root / 'sds2026'
sds_dir.mkdir(exist_ok=True)

# Load Adult data to get feature names
X_temp, y_temp, feature_names = load_adult_census_data()

# Find sex index
try:
    sex_idx = feature_names.index('sex')
except ValueError:
    print("Error: 'sex' feature not found")
    sys.exit(1)

# Process both methods
adult_results = all_results.get('Adult', {})
methods_to_check = ['Random Walk', 'Fast Privacy']

print("="*80)
print("GENERATING SEX RECONSTRUCTION TABLES FOR BOTH METHODS")
print("="*80)

for method_name in methods_to_check:
    if method_name not in adult_results:
        print(f"\nWarning: {method_name} not found in results. Skipping.")
        continue
    
    result = adult_results[method_name]
    X_original = result['X_test']
    X_reconstructed = result.get('reconstructed_data', None)
    
    if X_reconstructed is None:
        print(f"\nWarning: No reconstructed data for {method_name}. Skipping.")
        continue
    
    print(f"\n{method_name}:")
    print("-" * 80)
    
    original_sex = X_original[:, sex_idx]
    reconstructed_sex = X_reconstructed[:, sex_idx]
    
    # For binary features, round to nearest 0 or 1
    reconstructed_sex_binary = np.round(np.clip(reconstructed_sex, 0, 1)).astype(int)
    original_sex_int = original_sex.astype(int) if original_sex.dtype != int else original_sex
    
    r2_sex = r2_score(original_sex, reconstructed_sex)
    acc_sex = accuracy_score(original_sex_int, reconstructed_sex_binary)
    
    print(f"R² = {r2_sex:.4f}, Binary Accuracy = {acc_sex:.4f}")
    
    # Sample 10 rows
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(
        len(original_sex), min(10, len(original_sex)), replace=False
    )
    sex_table_data = []
    for idx in sample_indices:
        orig = int(original_sex_int[idx])
        recon_raw = reconstructed_sex[idx]
        recon_bin = int(reconstructed_sex_binary[idx])
        correct = "Yes" if orig == recon_bin else "No"
        sex_table_data.append((orig, f"{recon_raw:.3f}", recon_bin, correct))
    
    # Generate LaTeX table
    method_safe = method_name.replace(' ', '_').lower()
    table_filename = f"sex_reconstruction_table_{method_safe}.tex"
    
    with open(sds_dir / table_filename, "w") as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write(f"\\caption{{Reconstruction attack demonstration: Sex (binary feature) from Adult Census dataset using {method_name} encoder.}}\n")
        f.write("\\label{tab:sex_reconstruction}\n")
        f.write("\\begin{tabular}{cccc}\n")
        f.write("\\hline\n")
        f.write("Original & Reconstructed (raw) & Reconstructed (binary) & Correct? \\\\\n")
        f.write("\\hline\n")
        for orig, recon_raw, recon_bin, correct in sex_table_data:
            f.write(f"{orig} & {recon_raw} & {recon_bin} & {correct} \\\\\n")
        f.write("\\hline\n")
        f.write(f"\\multicolumn{{4}}{{c}}{{R² = {r2_sex:.4f}, Binary Accuracy = {acc_sex:.4f}}} \\\\\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"LaTeX table saved to {sds_dir}/{table_filename}")
    print(f"  R² = {r2_sex:.4f}, Binary Accuracy = {acc_sex:.4f}")

print("\n" + "="*80)
print("COMPARISON:")
print("="*80)
for method_name in methods_to_check:
    if method_name in adult_results:
        result = adult_results[method_name]
        X_original = result['X_test']
        X_reconstructed = result.get('reconstructed_data', None)
        if X_reconstructed is not None:
            original_sex = X_original[:, sex_idx]
            reconstructed_sex = X_reconstructed[:, sex_idx]
            reconstructed_sex_binary = np.round(np.clip(reconstructed_sex, 0, 1)).astype(int)
            original_sex_int = original_sex.astype(int) if original_sex.dtype != int else original_sex
            r2_sex = r2_score(original_sex, reconstructed_sex)
            acc_sex = accuracy_score(original_sex_int, reconstructed_sex_binary)
            print(f"{method_name}: R² = {r2_sex:.4f}, Binary Accuracy = {acc_sex:.4f}")
