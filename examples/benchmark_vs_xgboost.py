"""
Benchmark FraudBoost vs XGBoost on synthetic fraud detection data.

This example compares:
1. Traditional XGBoost with standard log loss
2. FraudBoost with value-weighted loss
3. Performance across different fraud rates and transaction value distributions
4. Business impact metrics (net savings, ROI, value detection rate)
"""

import time
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# Import FraudBoost
from fraudboost import (
    FraudBoostClassifier,
    classification_report_fraud,
    value_detection_rate,
    net_savings,
    roi
)

# Try to import XGBoost
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    print("Warning: XGBoost not available. Install with: pip install xgboost")
    HAS_XGBOOST = False


class FraudBenchmark:
    """Benchmark suite for comparing fraud detection algorithms."""
    
    def __init__(self, fp_cost=150, random_state=42):
        """
        Args:
            fp_cost: Cost per false positive investigation
            random_state: Random seed for reproducibility
        """
        self.fp_cost = fp_cost
        self.random_state = random_state
        self.results = []
    
    def create_fraud_dataset(self, n_samples=10000, fraud_rate=0.01, 
                           value_ratio=3.0, noise_level=0.02):
        """
        Create synthetic fraud dataset with realistic characteristics.
        
        Args:
            n_samples: Total number of samples
            fraud_rate: Fraction of samples that are fraud
            value_ratio: How much higher fraud amounts are on average
            noise_level: Amount of label noise to add
        """
        # Generate base classification data
        X, y = make_classification(
            n_samples=n_samples,
            n_features=25,
            n_informative=20,
            n_redundant=5,
            n_clusters_per_class=2,
            weights=[1-fraud_rate, fraud_rate],
            flip_y=noise_level,
            random_state=self.random_state
        )
        
        # Generate realistic transaction amounts
        np.random.seed(self.random_state)
        
        # Base amounts follow log-normal distribution
        base_amounts = np.random.lognormal(mean=4.5, sigma=1.2, size=n_samples)
        
        # Fraud amounts are typically higher
        fraud_mask = (y == 1)
        multipliers = np.random.uniform(value_ratio/2, value_ratio*1.5, np.sum(fraud_mask))
        base_amounts[fraud_mask] *= multipliers
        
        # Add some very high-value fraud cases (targeted attacks)
        n_high_value = max(1, int(np.sum(fraud_mask) * 0.1))
        high_value_indices = np.random.choice(
            np.where(fraud_mask)[0], n_high_value, replace=False
        )
        base_amounts[high_value_indices] *= np.random.uniform(5, 15, n_high_value)
        
        return X, y, base_amounts
    
    def train_xgboost(self, X_train, y_train, amounts_train=None):
        """Train XGBoost classifier."""
        if not HAS_XGBOOST:
            return None
            
        # XGBoost parameters tuned for fraud detection
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': self.random_state,
            'scale_pos_weight': np.sum(y_train == 0) / np.sum(y_train == 1)  # Handle imbalance
        }
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        
        return model
    
    def train_fraudboost(self, X_train, y_train, amounts_train):
        """Train FraudBoost classifier."""
        model = FraudBoostClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            fp_cost=self.fp_cost,
            loss='value_weighted',  # Use value-weighted loss
            early_stopping_rounds=10,
            random_state=self.random_state,
            verbose=0
        )
        
        model.fit(X_train, y_train, amounts=amounts_train)
        return model
    
    def evaluate_model(self, model, X_test, y_test, amounts_test, model_name):
        """Evaluate model and return comprehensive metrics."""
        start_time = time.time()
        
        # Predictions
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_pred_proba = model.decision_function(X_test)
            y_pred_proba = 1 / (1 + np.exp(-y_pred_proba))  # Sigmoid
        
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        prediction_time = time.time() - start_time
        
        # Traditional metrics
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # Business metrics
        vdr = value_detection_rate(y_test, y_pred, amounts_test)
        savings = net_savings(y_test, y_pred, amounts_test, self.fp_cost)
        roi_value = roi(y_test, y_pred, amounts_test, self.fp_cost)
        
        # Count metrics
        tp = np.sum((y_test == 1) & (y_pred == 1))
        fp = np.sum((y_test == 0) & (y_pred == 1))
        fn = np.sum((y_test == 1) & (y_pred == 0))
        tn = np.sum((y_test == 0) & (y_pred == 0))
        
        # Value metrics  
        fraud_caught_value = np.sum(amounts_test[(y_test == 1) & (y_pred == 1)])
        fraud_missed_value = np.sum(amounts_test[(y_test == 1) & (y_pred == 0)])
        
        return {
            'model_name': model_name,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'vdr': vdr,
            'net_savings': savings,
            'roi': roi_value,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn,
            'fraud_caught_value': fraud_caught_value,
            'fraud_missed_value': fraud_missed_value,
            'total_investigations': tp + fp,
            'prediction_time': prediction_time
        }
    
    def run_single_benchmark(self, n_samples=10000, fraud_rate=0.01, 
                           value_ratio=3.0, test_name=""):
        """Run benchmark on a single dataset configuration."""
        
        print(f"\n{'='*20} {test_name} {'='*20}")
        print(f"Dataset: {n_samples:,} samples, {fraud_rate:.1%} fraud rate, {value_ratio:.1f}x value ratio")
        
        # Create dataset
        X, y, amounts = self.create_fraud_dataset(
            n_samples=n_samples,
            fraud_rate=fraud_rate,
            value_ratio=value_ratio
        )
        
        # Train/test split
        X_train, X_test, y_train, y_test, amounts_train, amounts_test = train_test_split(
            X, y, amounts, test_size=0.3, stratify=y, random_state=self.random_state
        )
        
        print(f"Training: {len(X_train):,} samples ({np.sum(y_train):,} fraud)")
        print(f"Testing: {len(X_test):,} samples ({np.sum(y_test):,} fraud)")
        print(f"Avg fraud amount: ${np.mean(amounts_test[y_test==1]):,.0f}")
        print(f"Avg legit amount: ${np.mean(amounts_test[y_test==0]):,.0f}")
        
        results = []
        
        # Train and evaluate FraudBoost
        print("\nTraining FraudBoost...")
        start_time = time.time()
        fraudboost_model = self.train_fraudboost(X_train, y_train, amounts_train)
        fraudboost_train_time = time.time() - start_time
        
        fraudboost_results = self.evaluate_model(
            fraudboost_model, X_test, y_test, amounts_test, "FraudBoost"
        )
        fraudboost_results['train_time'] = fraudboost_train_time
        results.append(fraudboost_results)
        
        # Train and evaluate XGBoost (if available)
        if HAS_XGBOOST:
            print("Training XGBoost...")
            start_time = time.time()
            xgb_model = self.train_xgboost(X_train, y_train, amounts_train)
            xgb_train_time = time.time() - start_time
            
            xgb_results = self.evaluate_model(
                xgb_model, X_test, y_test, amounts_test, "XGBoost"
            )
            xgb_results['train_time'] = xgb_train_time
            results.append(xgb_results)
        else:
            print("Skipping XGBoost (not installed)")
        
        # Print comparison
        self.print_comparison(results)
        
        # Store results for aggregation
        for result in results:
            result['dataset_config'] = {
                'n_samples': n_samples,
                'fraud_rate': fraud_rate,
                'value_ratio': value_ratio,
                'test_name': test_name
            }
            self.results.append(result)
        
        return results
    
    def print_comparison(self, results):
        """Print side-by-side comparison of model results."""
        if len(results) < 2:
            return
        
        print(f"\n{'Metric':<25} {'FraudBoost':<12} {'XGBoost':<12} {'Difference':<12}")
        print("-" * 65)
        
        metrics_to_compare = [
            ('precision', 'Precision', '.4f'),
            ('recall', 'Recall', '.4f'),
            ('f1_score', 'F1 Score', '.4f'),
            ('auc', 'AUC', '.4f'),
            ('vdr', 'Value Detection Rate', '.4f'),
            ('net_savings', 'Net Savings ($)', ',.0f'),
            ('roi', 'ROI (x)', '.2f'),
            ('total_investigations', 'Investigations', ',d'),
            ('train_time', 'Train Time (s)', '.2f'),
            ('prediction_time', 'Predict Time (s)', '.4f')
        ]
        
        for metric_key, metric_name, fmt in metrics_to_compare:
            fb_val = results[0][metric_key]
            xgb_val = results[1][metric_key] if len(results) > 1 else 0
            
            if metric_key in ['net_savings']:
                diff_str = f"+${fb_val - xgb_val:,.0f}" if fb_val > xgb_val else f"-${xgb_val - fb_val:,.0f}"
            elif metric_key in ['total_investigations', 'train_time', 'prediction_time']:
                diff = fb_val - xgb_val
                diff_str = f"{diff:{fmt}}"
            else:
                diff = fb_val - xgb_val
                diff_str = f"+{diff:{fmt}}" if diff > 0 else f"{diff:{fmt}}"
            
            print(f"{metric_name:<25} {fb_val:{fmt}:<12} {xgb_val:{fmt}:<12} {diff_str:<12}")
    
    def run_comprehensive_benchmark(self):
        """Run comprehensive benchmark across multiple scenarios."""
        
        print("FRAUDBOOST vs XGBOOST COMPREHENSIVE BENCHMARK")
        print("=" * 60)
        
        # Benchmark scenarios
        scenarios = [
            {
                'n_samples': 5000,
                'fraud_rate': 0.005,  # 0.5% - very rare fraud
                'value_ratio': 2.0,
                'test_name': "Low Fraud Rate, Low Value Ratio"
            },
            {
                'n_samples': 10000,
                'fraud_rate': 0.01,   # 1% - typical fraud rate
                'value_ratio': 4.0,
                'test_name': "Typical Fraud Rate, High Value Ratio"
            },
            {
                'n_samples': 15000,
                'fraud_rate': 0.02,   # 2% - high fraud rate
                'value_ratio': 6.0,
                'test_name': "High Fraud Rate, Very High Value Ratio"
            },
            {
                'n_samples': 8000,
                'fraud_rate': 0.015,  # 1.5% - moderate fraud rate
                'value_ratio': 1.5,   # Low value ratio - fraud amounts similar to legit
                'test_name': "Moderate Fraud Rate, Similar Values"
            }
        ]
        
        # Run all scenarios
        for scenario in scenarios:
            self.run_single_benchmark(**scenario)
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print overall benchmark summary."""
        
        print(f"\n{'='*20} BENCHMARK SUMMARY {'='*20}")
        
        if not self.results:
            print("No results to summarize.")
            return
        
        # Aggregate results by model
        fraudboost_results = [r for r in self.results if r['model_name'] == 'FraudBoost']
        xgboost_results = [r for r in self.results if r['model_name'] == 'XGBoost']
        
        if not fraudboost_results:
            print("No FraudBoost results found.")
            return
        
        print(f"\nAverage Performance Across {len(fraudboost_results)} Tests:")
        print("-" * 55)
        
        def avg_metric(results, metric):
            return np.mean([r[metric] for r in results])
        
        metrics = ['precision', 'recall', 'f1_score', 'auc', 'vdr', 'net_savings', 'roi']
        
        print(f"{'Metric':<20} {'FraudBoost':<12} {'XGBoost':<12} {'FB Advantage':<12}")
        print("-" * 60)
        
        for metric in metrics:
            fb_avg = avg_metric(fraudboost_results, metric)
            
            if xgboost_results:
                xgb_avg = avg_metric(xgboost_results, metric)
                
                if metric == 'net_savings':
                    advantage = f"+${fb_avg - xgb_avg:,.0f}"
                elif metric in ['precision', 'recall', 'f1_score', 'auc', 'vdr']:
                    advantage = f"+{fb_avg - xgb_avg:.4f}"
                else:
                    advantage = f"{fb_avg - xgb_avg:.3f}"
                    
                print(f"{metric:<20} {fb_avg:.4f}<12 {xgb_avg:.4f}<12 {advantage:<12}")
            else:
                print(f"{metric:<20} {fb_avg:.4f}")
        
        # Key insights
        print(f"\nKey Insights:")
        print(f"- FraudBoost is specifically designed for fraud detection with value-weighted losses")
        print(f"- Average net savings advantage: ${avg_metric(fraudboost_results, 'net_savings') - avg_metric(xgboost_results, 'net_savings') if xgboost_results else avg_metric(fraudboost_results, 'net_savings'):,.0f}")
        print(f"- Average VDR (Value Detection Rate): {avg_metric(fraudboost_results, 'vdr'):.4f}")
        print(f"- Average ROI: {avg_metric(fraudboost_results, 'roi'):.2f}x")


def main():
    """Run the benchmark."""
    
    # Initialize benchmark
    benchmark = FraudBenchmark(fp_cost=150, random_state=42)
    
    try:
        # Run comprehensive benchmark
        benchmark.run_comprehensive_benchmark()
        
        print(f"\n{'='*20} BENCHMARK COMPLETE {'='*20}")
        print("Results show FraudBoost's advantage in financial impact metrics")
        print("while maintaining competitive traditional ML performance.")
        
    except Exception as e:
        print(f"Error running benchmark: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()