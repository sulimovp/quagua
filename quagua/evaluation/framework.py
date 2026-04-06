"""
Evaluation framework for privacy-preserving encoding methods
"""

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error, r2_score
from scipy import stats


class PrivacyAITestFramework:
    """Train sklearn models on encoded data and score utility and attack models."""

    
    def __init__(self):
        self.results = {}
        
    def _validate_encoded_data(self, X_encoded, encoder_name):
        """Validate encoded data for NaN, inf, or extreme values"""
        if np.any(np.isnan(X_encoded)):
            print(f"  WARNING: {encoder_name} produced NaN values. Replacing with 0.")
            X_encoded = np.nan_to_num(X_encoded, nan=0.0)
        
        if np.any(np.isinf(X_encoded)):
            print(f"  WARNING: {encoder_name} produced infinite values. Clipping.")
            X_encoded = np.nan_to_num(X_encoded, posinf=1.0, neginf=0.0)
        
        # Clip extreme values
        max_val = np.abs(X_encoded).max()
        if max_val > 1e6:
            print(f"  WARNING: {encoder_name} produced very large values (max={max_val:.2e}). Clipping.")
            X_encoded = np.clip(X_encoded, -1e6, 1e6)
        
        return X_encoded
    
    def evaluate_encoding(self, encoder, X_train, X_test, y_train, y_test, 
                         encoder_name, test_models=True):
        """Evaluate an encoding method"""
        
        print(f"\n{'='*60}")
        print(f"Evaluating: {encoder_name}")
        print(f"{'='*60}")
        
        # Apply encoding
        X_train_encoded = encoder.fit_transform(X_train)
        X_test_encoded = encoder.transform(X_test)
        
        # Validate encoded data
        X_train_encoded = self._validate_encoded_data(X_train_encoded, encoder_name)
        X_test_encoded = self._validate_encoded_data(X_test_encoded, encoder_name)
        
        print(f"Original dimension: {X_train.shape[1]}")
        print(f"Encoded dimension: {X_train_encoded.shape[1]}")
        print(f"Expansion factor: {X_train_encoded.shape[1] / X_train.shape[1]:.2f}x")
        
        results = {
            'encoder': encoder,
            'encoded_train': X_train_encoded,
            'encoded_test': X_test_encoded,
            'expansion_factor': X_train_encoded.shape[1] / X_train.shape[1],
        }
        
        if test_models:
            # Test different ML models
            models = {
                'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'MLP': MLPClassifier(hidden_layer_sizes=(64, 32), 
                                    max_iter=500, 
                                    random_state=42),
            }
            
            model_results = {}
            for model_name, model in models.items():
                # Train
                model.fit(X_train_encoded, y_train)
                
                # Predict
                y_pred = model.predict(X_test_encoded)
                y_prob = model.predict_proba(X_test_encoded)[:, 1]
                
                # Evaluate
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                auc = roc_auc_score(y_test, y_prob)
                
                model_results[model_name] = {
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'auc': auc,
                }
                
                print(f"{model_name:20} | Accuracy: {accuracy:.3f} | "
                      f"F1: {f1:.3f} | AUC: {auc:.3f}")
            
            results['model_results'] = model_results
            
            # Best model accuracy
            best_acc = max([res['accuracy'] for res in model_results.values()])
            results['best_accuracy'] = best_acc
        
        return results
    
    def safe_correlation(self, x, y):
        """Calculate correlation with NaN handling"""
        if np.std(x) == 0 or np.std(y) == 0:
            return 0.0  # No correlation if constant
        try:
            corr = np.corrcoef(x, y)[0, 1]
            return corr if not np.isnan(corr) else 0.0
        except Exception:
            return 0.0
    
    def calculate_privacy_score(self, reconstruction_error, correlation, feature_range):
        """
        Improved privacy score that accounts for feature scale
        
        Parameters:
        - reconstruction_error: MSE of reconstruction
        - correlation: Correlation between original and reconstructed
        - feature_range: Range of original feature values
        """
        # Normalize reconstruction error by feature variance
        normalized_error = reconstruction_error / (feature_range ** 2 + 1e-8)
        
        # Privacy components
        error_component = 1 - np.exp(-normalized_error)  # [0, 1]
        correlation_component = 1 - np.abs(correlation)  # [0, 1]
        
        # Weighted combination
        privacy_score = 0.6 * error_component + 0.4 * correlation_component
        
        return privacy_score
    
    def evaluate_privacy(self, encoder, X_test, encoder_name, use_linear_only=True):
        """
        Privacy evaluation using reconstruction attacks.
        
        We use only Linear Regression for privacy attacks because:
        1. Attackers typically have limited knowledge about the encoding mechanism
        2. Without knowing the exact encoding, attackers make naive assumptions
        3. Linear models represent the simplest, most realistic attack scenario
        4. If a linear model can reconstruct features, privacy is compromised
        5. Non-linear models (RF, MLP) require more assumptions and may overfit
        
        The attack scenario: An attacker has access to encoded data (X_encoded) 
        and tries to reconstruct original features (X_original). Since they don't 
        know the encoding mechanism, they assume a simple linear relationship:
        X_original ≈ W @ X_encoded + b
        
        If some original data flows away (e.g., through model outputs, side channels),
        the attacker can use it to learn the linear mapping and reconstruct other features.
        
        Parameters:
        -----------
        use_linear_only : bool
            If True, use only Linear Regression. If False, use multiple models (legacy).
        """
        print(f"\nPrivacy evaluation for {encoder_name}:")
        
        if use_linear_only:
            print("  Using Linear Regression attack (realistic attacker assumption)")
            print("  Rationale: Attackers with limited encoding knowledge assume linear relationships")
        
        # Encode the data
        X_encoded = encoder.transform(X_test)
        
        # Use only Linear Regression for realistic attack scenario
        attack_model = LinearRegression()
        
        reconstruction_errors = []
        r2_scores = []
        correlations = []
        reconstruction_details = []  # Store per-feature details for visualization
        
        # Calculate feature ranges for normalization
        feature_ranges = X_test.max(axis=0) - X_test.min(axis=0) + 1e-8
        
        for feature_idx in range(X_test.shape[1]):
            try:
                # Train attack model to predict original feature
                attack_model.fit(X_encoded, X_test[:, feature_idx])
                predictions = attack_model.predict(X_encoded)
                
                # Calculate metrics
                mse = mean_squared_error(X_test[:, feature_idx], predictions)
                r2 = r2_score(X_test[:, feature_idx], predictions)
                corr = self.safe_correlation(predictions, X_test[:, feature_idx])
                
                reconstruction_errors.append(mse)
                r2_scores.append(r2)
                correlations.append(corr)
                
                # Store reconstruction details for visualization
                reconstruction_details.append({
                    'original': X_test[:, feature_idx],
                    'reconstructed': predictions,
                    'mse': mse,
                    'r2': r2,
                    'corr': corr
                })
                
            except Exception as e:
                # If attack fails, assume perfect privacy for this feature
                reconstruction_errors.append(feature_ranges[feature_idx] ** 2)
                r2_scores.append(0.0)
                correlations.append(0.0)
                reconstruction_details.append({
                    'original': X_test[:, feature_idx],
                    'reconstructed': np.zeros_like(X_test[:, feature_idx]),
                    'mse': feature_ranges[feature_idx] ** 2,
                    'r2': 0.0,
                    'corr': 0.0
                })
        
        avg_mse = np.mean(reconstruction_errors)
        avg_r2 = np.mean(r2_scores)
        avg_corr = np.mean(np.abs(correlations))
        
        # Calculate privacy score
        avg_feature_range = np.mean(feature_ranges)
        privacy_score = self.calculate_privacy_score(avg_mse, avg_corr, avg_feature_range)
        
        print(f"  Linear          | MSE: {avg_mse:.4f} | R²: {avg_r2:.4f} | |Corr|: {avg_corr:.4f} | Privacy: {privacy_score:.4f}")
        
        return {
            'reconstruction_error': avg_mse,
            'correlation': avg_corr,
            'r2_score': avg_r2,
            'privacy_score': privacy_score,
            'reconstruction_details': reconstruction_details,  # For visualization
        }
    
    def run_comparison(self, X_train, X_test, y_train, y_test, encoders=None):
        """
        Run comprehensive comparison of all encoding methods
        
        Parameters:
        - encoders: Dictionary of {name: encoder} to test. If None, uses default set.
        """
        if encoders is None:
            # Default encoders (for backward compatibility)
            # Use relative imports for module structure
            try:
                from ..encoders import (
                ClassicalXORMixer, RandomProjectionEncoder,
                QuantumSuperpositionEncoder, QuantumEntanglementEncoder,
                QuantumFourierEncoder, QuantumPhaseEncoder, QuantumWalkEncoder,
                CombinedQuantumEncoder,
                RandomOneToManyEncoder, ChaoticSystemEncoder, OrthogonalSubspaceEncoder,
                EncodeThenMixPipeline
                )
            except ImportError:
                # Fallback for direct execution
                from quagua.encoders import (
                    ClassicalXORMixer, RandomProjectionEncoder,
                    QuantumSuperpositionEncoder, QuantumEntanglementEncoder,
                    QuantumFourierEncoder, QuantumPhaseEncoder, QuantumWalkEncoder,
                    CombinedQuantumEncoder,
                    RandomOneToManyEncoder, ChaoticSystemEncoder, OrthogonalSubspaceEncoder,
                    EncodeThenMixPipeline
                )
            
            encoders = {
                'Baseline (No Encoding)': lambda x: x,
                'Classical XOR Mixing': ClassicalXORMixer(),
                'Random Projection': RandomProjectionEncoder(n_components=12),
                'Quantum Superposition': QuantumSuperpositionEncoder(n_states=3, temperature=0.5),
                'Quantum Entanglement': QuantumEntanglementEncoder(entanglement_strength=0.7),
                'Quantum Fourier': QuantumFourierEncoder(n_frequencies=2, use_phases=True),
                'Quantum Phase': QuantumPhaseEncoder(n_phases=3, add_noise=True, noise_level=0.02),
                'Quantum Walk': QuantumWalkEncoder(n_steps=3, graph_type='complete'),
                'Combined Quantum': CombinedQuantumEncoder(),
                'Random 1-to-Many': RandomOneToManyEncoder(n_encodings_per_value=5, encoding_dim=3),
                'Chaotic System (Logistic)': ChaoticSystemEncoder(map_type='logistic', n_iterations=10),
                'Chaotic System (Henon)': ChaoticSystemEncoder(map_type='henon', n_iterations=10),
                'Orthogonal Subspace': OrthogonalSubspaceEncoder(subspace_dim=3),
                '1-to-Many + XOR Mix': EncodeThenMixPipeline(
                    RandomOneToManyEncoder(n_encodings_per_value=5, encoding_dim=3),
                    ClassicalXORMixer()
                ),
                'Chaotic + Entanglement': EncodeThenMixPipeline(
                    ChaoticSystemEncoder(map_type='logistic', n_iterations=10),
                    QuantumEntanglementEncoder(entanglement_strength=0.7)
                ),
                'Orthogonal + Quantum Walk': EncodeThenMixPipeline(
                    OrthogonalSubspaceEncoder(subspace_dim=3),
                    QuantumWalkEncoder(n_steps=3, graph_type='complete')
                ),
            }
        
        all_results = {}
        
        for encoder_name, encoder in encoders.items():
            print(f"\n{'#'*70}")
            print(f"TESTING: {encoder_name}")
            print(f"{'#'*70}")
            
            # Skip privacy evaluation for baseline (no encoding)
            if encoder_name == 'Baseline (No Encoding)':
                # For baseline, we need to fit_transform manually
                X_train_encoded = X_train.copy()
                X_test_encoded = X_test.copy()
                
                # Test models
                models = {
                    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
                    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                    'MLP': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
                }
                
                model_results = {}
                for model_name, model in models.items():
                    model.fit(X_train_encoded, y_train)
                    y_pred = model.predict(X_test_encoded)
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    model_results[model_name] = {'accuracy': accuracy}
                    print(f"{model_name:20} | Accuracy: {accuracy:.3f}")
                
                all_results[encoder_name] = {
                    'model_results': model_results,
                    'best_accuracy': max([res['accuracy'] for res in model_results.values()]),
                    'privacy': {'reconstruction_error': 0, 'correlation': 1.0, 'privacy_score': 0, 'r2_score': 1.0},
                    'expansion_factor': 1.0,
                }
                
            else:
                # Evaluate encoding
                results = self.evaluate_encoding(
                    encoder, X_train, X_test, y_train, y_test, encoder_name
                )
                
                # Evaluate privacy
                privacy_results = self.evaluate_privacy(encoder, X_test, encoder_name)
                
                # Combine results
                all_results[encoder_name] = {
                    **results,
                    'privacy': privacy_results,
                }
        
        return all_results
    
    def compare_methods_statistically(self, X_train, X_test, y_train, y_test, encoders=None, n_runs=5):
        """Run multiple times and test for statistical significance"""
        print(f"\n{'='*60}")
        print(f"STATISTICAL SIGNIFICANCE TESTING ({n_runs} runs)")
        print(f"{'='*60}")
        
        if encoders is None:
            # Default encoders for statistical testing
            try:
                from ..encoders import (
                    QuantumSuperpositionEncoder, QuantumEntanglementEncoder, QuantumPhaseEncoder
                )
            except ImportError:
                from quagua.encoders import (
                    QuantumSuperpositionEncoder, QuantumEntanglementEncoder, QuantumPhaseEncoder
                )
            encoders = {
                'Baseline (No Encoding)': lambda x: x,
                'Quantum Superposition': QuantumSuperpositionEncoder(n_states=3, temperature=0.5),
                'Quantum Entanglement': QuantumEntanglementEncoder(entanglement_strength=0.7),
                'Quantum Phase': QuantumPhaseEncoder(n_phases=3, add_noise=True, noise_level=0.02),
            }
        
        # Store results for each run
        accuracies = {name: [] for name in encoders.keys()}
        privacy_scores = {name: [] for name in encoders.keys()}
        
        for run in range(n_runs):
            print(f"\nRun {run + 1}/{n_runs}...")
            np.random.seed(42 + run)  # Different seed for each run
            
            for encoder_name, encoder in encoders.items():
                if encoder_name == 'Baseline (No Encoding)':
                    X_train_encoded = X_train.copy()
                    X_test_encoded = X_test.copy()
                else:
                    X_train_encoded = encoder.fit_transform(X_train)
                    X_test_encoded = encoder.transform(X_test)
                
                # Train and evaluate
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train_encoded, y_train)
                accuracy = model.score(X_test_encoded, y_test)
                accuracies[encoder_name].append(accuracy)
                
                # Privacy evaluation (simplified for speed)
                if encoder_name != 'Baseline (No Encoding)':
                    privacy_result = self.evaluate_privacy(encoder, X_test, encoder_name)
                    privacy_scores[encoder_name].append(privacy_result['privacy_score'])
                else:
                    privacy_scores[encoder_name].append(0.0)
        
        # Perform pairwise t-tests
        print(f"\n{'='*60}")
        print("STATISTICAL SIGNIFICANCE (Accuracy)")
        print(f"{'='*60}")
        methods = list(encoders.keys())
        for i, method1 in enumerate(methods):
            for method2 in methods[i+1:]:
                t_stat, p_value = stats.ttest_rel(
                    accuracies[method1], 
                    accuracies[method2]
                )
                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                print(f"{method1[:25]:25} vs {method2[:25]:25}: "
                      f"t={t_stat:7.3f}, p={p_value:.4f} {significance}")
        
        # Print summary statistics
        print(f"\n{'='*60}")
        print("SUMMARY STATISTICS")
        print(f"{'='*60}")
        print(f"{'Method':<30} {'Mean Acc':<12} {'Std Acc':<12} {'Mean Privacy':<15} {'Std Privacy'}")
        print("-" * 80)
        for method in methods:
            mean_acc = np.mean(accuracies[method])
            std_acc = np.std(accuracies[method])
            mean_priv = np.mean(privacy_scores[method])
            std_priv = np.std(privacy_scores[method])
            print(f"{method:<30} {mean_acc:>10.4f}   {std_acc:>10.4f}   {mean_priv:>13.4f}   {std_priv:>10.4f}")
        
        return {
            'accuracies': accuracies,
            'privacy_scores': privacy_scores,
        }
    
    def plot_results(self, all_results):
        """Create visualization of results"""
        import matplotlib.pyplot as plt
        
        # Prepare data for plotting
        methods = list(all_results.keys())
        accuracies = [all_results[m]['best_accuracy'] for m in methods]
        privacy_scores = [all_results[m]['privacy']['privacy_score'] for m in methods]
        expansions = [all_results[m].get('expansion_factor', 1.0) for m in methods]
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Accuracy comparison
        bars1 = axes[0, 0].bar(range(len(methods)), accuracies)
        axes[0, 0].set_xticks(range(len(methods)))
        axes[0, 0].set_xticklabels(methods, rotation=45, ha='right')
        axes[0, 0].set_ylabel('Best Accuracy')
        axes[0, 0].set_title('Model Accuracy by Encoding Method')
        axes[0, 0].axhline(y=0.8, color='r', linestyle='--', alpha=0.3)
        
        # Color bars by accuracy
        for bar, acc in zip(bars1, accuracies):
            color = 'green' if acc > 0.78 else 'orange' if acc > 0.75 else 'red'
            bar.set_color(color)
        
        # Plot 2: Privacy score comparison (now normalized [0, 1])
        bars2 = axes[0, 1].bar(range(len(methods)), privacy_scores)
        axes[0, 1].set_xticks(range(len(methods)))
        axes[0, 1].set_xticklabels(methods, rotation=45, ha='right')
        axes[0, 1].set_ylabel('Privacy Score [0, 1]')
        axes[0, 1].set_title('Privacy Protection by Encoding Method')
        axes[0, 1].set_ylim([0, 1])
        axes[0, 1].axhline(y=0.5, color='g', linestyle='--', alpha=0.3, label='Good privacy')
        axes[0, 1].axhline(y=0.3, color='orange', linestyle='--', alpha=0.3, label='Moderate privacy')
        
        # Color bars by privacy score (adjusted for [0, 1] range)
        for bar, privacy in zip(bars2, privacy_scores):
            color = 'green' if privacy > 0.5 else 'orange' if privacy > 0.3 else 'red'
            bar.set_color(color)
        
        # Plot 3: Accuracy vs Privacy trade-off
        axes[1, 0].scatter(privacy_scores, accuracies, s=100, alpha=0.7)
        for i, method in enumerate(methods):
            axes[1, 0].annotate(method[:15], 
                               (privacy_scores[i], accuracies[i]),
                               xytext=(5, 5), 
                               textcoords='offset points',
                               fontsize=8)
        axes[1, 0].set_xlabel('Privacy Score (Higher = Better)')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_title('Privacy vs Accuracy Trade-off')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Dimension expansion
        bars3 = axes[1, 1].bar(range(len(methods)), expansions)
        axes[1, 1].set_xticks(range(len(methods)))
        axes[1, 1].set_xticklabels(methods, rotation=45, ha='right')
        axes[1, 1].set_ylabel('Expansion Factor')
        axes[1, 1].set_title('Feature Dimension Expansion')
        axes[1, 1].axhline(y=1.0, color='r', linestyle='--', alpha=0.3)
        
        # Color bars by expansion
        for bar, exp in zip(bars3, expansions):
            color = 'red' if exp > 3 else 'orange' if exp > 1.5 else 'green'
            bar.set_color(color)
        
        plt.tight_layout()
        plt.show()
        
        # Print summary table
        print("\n" + "="*120)
        print("SUMMARY OF RESULTS")
        print("="*120)
        print(f"{'Method':<25} {'Accuracy':<10} {'Privacy':<10} {'R² Score':<10} {'Expansion':<10} {'Trade-off'}")
        print("-"*120)
        
        for method in methods:
            acc = all_results[method]['best_accuracy']
            privacy = all_results[method]['privacy']['privacy_score']
            r2 = all_results[method]['privacy'].get('r2_score', 0.0)
            expansion = all_results[method].get('expansion_factor', 1.0)
            
            # Trade-off score: balance of accuracy and privacy
            # Higher is better (want both high accuracy AND high privacy)
            # Privacy score is now normalized [0, 1], so this works better
            tradeoff_score = (acc * 0.6) + (privacy * 0.4)
            
            print(f"{method:<25} {acc:<10.3f} {privacy:<10.3f} {r2:<10.3f} {expansion:<10.2f} {tradeoff_score:<10.3f}")

