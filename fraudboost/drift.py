"""
Temporal drift detection for fraud detection models.

Monitors model degradation over time due to:
- Population shift (customer demographics change)
- Concept drift (fraud patterns evolve)
- Feature drift (data quality issues)

Provides automated retraining recommendations.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple, Union
from scipy import stats
from scipy.spatial.distance import jensenshannon
import warnings


class TemporalDriftDetector:
    """
    Detect temporal drift in fraud detection models.
    
    Monitors:
    - Population Stability Index (PSI) for feature distributions
    - Model performance degradation over time
    - Prediction distribution shift
    """
    
    def __init__(self,
                 psi_threshold: float = 0.2,
                 performance_threshold: float = 0.05,
                 min_samples: int = 1000,
                 time_decay: float = 0.95):
        """
        Args:
            psi_threshold: PSI threshold for triggering drift alert (0.1=low, 0.2=medium, 0.25=high)
            performance_threshold: Performance degradation threshold to trigger alert
            min_samples: Minimum samples required for drift computation
            time_decay: Exponential decay factor for temporal weighting (0.9-0.99)
        """
        self.psi_threshold = psi_threshold
        self.performance_threshold = performance_threshold
        self.min_samples = min_samples
        self.time_decay = time_decay
        
        # Reference distributions (from training)
        self.reference_distributions_ = {}
        self.reference_performance_ = {}
        
        # Monitoring history
        self.monitoring_history_ = []
        
    def fit_reference(self, X: np.ndarray, y: np.ndarray, 
                     y_pred_proba: Optional[np.ndarray] = None,
                     timestamps: Optional[np.ndarray] = None,
                     feature_names: Optional[List[str]] = None) -> 'TemporalDriftDetector':
        """
        Fit reference distributions from training data.
        
        Args:
            X: Training features
            y: Training labels  
            y_pred_proba: Training predictions (optional)
            timestamps: Training timestamps (optional)
            feature_names: Feature names (optional)
        """
        n_samples, n_features = X.shape
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(n_features)]
        
        # Compute reference feature distributions
        self.reference_distributions_ = {}
        
        for i, feature_name in enumerate(feature_names):
            feature_values = X[:, i]
            
            # Remove NaN values
            valid_values = feature_values[~np.isnan(feature_values)]
            
            if len(valid_values) > 0:
                # Create histogram for PSI calculation
                hist, bin_edges = self._create_histogram(valid_values)
                
                self.reference_distributions_[feature_name] = {
                    'histogram': hist,
                    'bin_edges': bin_edges,
                    'mean': np.mean(valid_values),
                    'std': np.std(valid_values),
                    'quantiles': np.percentile(valid_values, [10, 25, 50, 75, 90])
                }
        
        # Store reference performance metrics
        if y_pred_proba is not None:
            from .metrics import value_detection_rate, precision_score, recall_score
            
            y_pred = (y_pred_proba >= 0.5).astype(int)
            
            self.reference_performance_ = {
                'precision': precision_score(y, y_pred, zero_division=0),
                'recall': recall_score(y, y_pred, zero_division=0),
                'auc': self._compute_auc(y, y_pred_proba)
            }
        
        # Apply temporal weighting if timestamps provided
        if timestamps is not None:
            self._apply_temporal_weights(timestamps)
        
        return self
    
    def detect_drift(self, X: np.ndarray, y: Optional[np.ndarray] = None,
                    y_pred_proba: Optional[np.ndarray] = None,
                    timestamps: Optional[np.ndarray] = None,
                    feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Detect drift in new data compared to reference.
        
        Args:
            X: New feature data
            y: New labels (optional, for performance monitoring)
            y_pred_proba: New predictions (optional)
            timestamps: New timestamps (optional)
            feature_names: Feature names (optional)
        
        Returns:
            Dictionary with drift detection results
        """
        if not self.reference_distributions_:
            raise ValueError("Must call fit_reference() first")
        
        n_samples, n_features = X.shape
        
        if n_samples < self.min_samples:
            warnings.warn(f"Insufficient samples ({n_samples} < {self.min_samples})")
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(n_features)]
        
        drift_results = {
            'timestamp': pd.Timestamp.now(),
            'n_samples': n_samples,
            'overall_drift_detected': False,
            'drift_features': [],
            'psi_scores': {},
            'feature_drift_summary': {},
            'performance_drift': {},
            'recommendation': 'monitor'
        }
        
        # Feature drift detection using PSI
        overall_psi_score = 0.0
        n_valid_features = 0
        
        for i, feature_name in enumerate(feature_names):
            if feature_name not in self.reference_distributions_:
                continue
                
            feature_values = X[:, i]
            valid_values = feature_values[~np.isnan(feature_values)]
            
            if len(valid_values) == 0:
                continue
            
            # Compute PSI for this feature
            psi_score = self._compute_psi(
                valid_values, 
                self.reference_distributions_[feature_name]
            )
            
            drift_results['psi_scores'][feature_name] = psi_score
            
            if psi_score > self.psi_threshold:
                drift_results['drift_features'].append(feature_name)
            
            # Feature summary statistics
            drift_results['feature_drift_summary'][feature_name] = {
                'psi': psi_score,
                'mean_shift': np.mean(valid_values) - self.reference_distributions_[feature_name]['mean'],
                'std_ratio': np.std(valid_values) / max(self.reference_distributions_[feature_name]['std'], 1e-8)
            }
            
            overall_psi_score += psi_score
            n_valid_features += 1
        
        # Overall PSI assessment
        if n_valid_features > 0:
            avg_psi_score = overall_psi_score / n_valid_features
            drift_results['average_psi'] = avg_psi_score
            
            if avg_psi_score > self.psi_threshold:
                drift_results['overall_drift_detected'] = True
        
        # Performance drift detection
        if y is not None and y_pred_proba is not None and self.reference_performance_:
            performance_drift = self._detect_performance_drift(y, y_pred_proba)
            drift_results['performance_drift'] = performance_drift
            
            if performance_drift.get('significant_degradation', False):
                drift_results['overall_drift_detected'] = True
        
        # Prediction distribution drift
        if y_pred_proba is not None:
            pred_drift = self._detect_prediction_drift(y_pred_proba)
            drift_results['prediction_drift'] = pred_drift
        
        # Generate recommendation
        drift_results['recommendation'] = self._generate_recommendation(drift_results)
        
        # Store in monitoring history
        self.monitoring_history_.append(drift_results)
        
        return drift_results
    
    def _compute_psi(self, current_values: np.ndarray, reference_dist: Dict[str, Any]) -> float:
        """Compute Population Stability Index between current and reference distributions."""
        ref_hist = reference_dist['histogram']
        ref_bins = reference_dist['bin_edges']
        
        # Create histogram for current values using same bins
        current_hist, _ = np.histogram(current_values, bins=ref_bins)
        
        # Normalize to probabilities
        ref_prob = ref_hist / np.sum(ref_hist)
        current_prob = current_hist / np.sum(current_hist)
        
        # Avoid division by zero
        ref_prob = np.maximum(ref_prob, 1e-8)
        current_prob = np.maximum(current_prob, 1e-8)
        
        # PSI formula: Σ((current - reference) * ln(current / reference))
        psi = np.sum((current_prob - ref_prob) * np.log(current_prob / ref_prob))
        
        return psi
    
    def _create_histogram(self, values: np.ndarray, n_bins: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """Create histogram with appropriate binning strategy."""
        if len(np.unique(values)) <= 10:
            # Categorical-like feature
            unique_vals = np.unique(values)
            bin_edges = np.concatenate([unique_vals - 0.5, [unique_vals[-1] + 0.5]])
        else:
            # Continuous feature - use quantile-based binning
            bin_edges = np.percentile(values, np.linspace(0, 100, n_bins + 1))
            # Remove duplicates that can occur with many identical values
            bin_edges = np.unique(bin_edges)
            if len(bin_edges) < 3:
                bin_edges = np.linspace(np.min(values), np.max(values), 3)
        
        hist, bin_edges = np.histogram(values, bins=bin_edges)
        
        return hist, bin_edges
    
    def _detect_performance_drift(self, y: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """Detect performance degradation compared to reference."""
        from .metrics import precision_score, recall_score
        
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        current_performance = {
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'auc': self._compute_auc(y, y_pred_proba)
        }
        
        performance_drift = {
            'current_performance': current_performance,
            'reference_performance': self.reference_performance_,
            'degradation': {},
            'significant_degradation': False
        }
        
        # Compute performance degradation
        for metric in ['precision', 'recall', 'auc']:
            if metric in self.reference_performance_:
                ref_val = self.reference_performance_[metric]
                current_val = current_performance[metric]
                degradation = ref_val - current_val
                
                performance_drift['degradation'][metric] = degradation
                
                if degradation > self.performance_threshold:
                    performance_drift['significant_degradation'] = True
        
        return performance_drift
    
    def _detect_prediction_drift(self, y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """Detect drift in prediction distributions."""
        # Simple check for prediction distribution shift
        pred_mean = np.mean(y_pred_proba)
        pred_std = np.std(y_pred_proba)
        
        # Check for unusual prediction patterns
        pred_drift = {
            'mean_prediction': pred_mean,
            'prediction_std': pred_std,
            'high_confidence_rate': np.mean((y_pred_proba > 0.9) | (y_pred_proba < 0.1)),
            'prediction_entropy': -np.mean(
                y_pred_proba * np.log(np.maximum(y_pred_proba, 1e-8)) + 
                (1 - y_pred_proba) * np.log(np.maximum(1 - y_pred_proba, 1e-8))
            )
        }
        
        return pred_drift
    
    def _compute_auc(self, y_true: np.ndarray, y_scores: np.ndarray) -> float:
        """Compute AUC score (simplified implementation)."""
        # Sort by scores descending
        sort_idx = np.argsort(y_scores)[::-1]
        y_sorted = y_true[sort_idx]
        
        n_pos = np.sum(y_true)
        n_neg = len(y_true) - n_pos
        
        if n_pos == 0 or n_neg == 0:
            return 0.5
        
        # Compute AUC using trapezoidal rule
        tp = np.cumsum(y_sorted)
        fp = np.cumsum(1 - y_sorted)
        
        tpr = tp / n_pos
        fpr = fp / n_neg
        
        # Add endpoints
        tpr = np.concatenate([[0], tpr])
        fpr = np.concatenate([[0], fpr])
        
        auc = np.trapz(tpr, fpr)
        return auc
    
    def _apply_temporal_weights(self, timestamps: np.ndarray):
        """Apply exponential decay weighting based on timestamps."""
        if len(timestamps) == 0:
            return
        
        # Convert to numeric timestamps if needed
        if hasattr(timestamps[0], 'timestamp'):
            timestamps = np.array([t.timestamp() for t in timestamps])
        
        # Compute weights - more recent samples get higher weight
        max_timestamp = np.max(timestamps)
        time_diffs = max_timestamp - timestamps
        weights = self.time_decay ** (time_diffs / (24 * 3600))  # Daily decay
        
        # Apply weights to reference distributions (implementation depends on specific needs)
        # This is a placeholder for more sophisticated temporal weighting
        pass
    
    def _generate_recommendation(self, drift_results: Dict[str, Any]) -> str:
        """Generate actionable recommendation based on drift analysis."""
        n_drift_features = len(drift_results['drift_features'])
        overall_drift = drift_results['overall_drift_detected']
        avg_psi = drift_results.get('average_psi', 0)
        
        performance_degraded = drift_results.get('performance_drift', {}).get('significant_degradation', False)
        
        if performance_degraded:
            return 'retrain_immediately'
        elif overall_drift and n_drift_features >= 3:
            return 'retrain_soon'
        elif avg_psi > 0.1:
            return 'monitor_closely'
        else:
            return 'monitor'
    
    def get_drift_summary(self, last_n_periods: Optional[int] = None) -> pd.DataFrame:
        """Get summary of drift monitoring history."""
        if not self.monitoring_history_:
            return pd.DataFrame()
        
        history = self.monitoring_history_
        if last_n_periods:
            history = history[-last_n_periods:]
        
        summary_data = []
        for record in history:
            summary_data.append({
                'timestamp': record['timestamp'],
                'n_samples': record['n_samples'],
                'drift_detected': record['overall_drift_detected'],
                'n_drift_features': len(record['drift_features']),
                'avg_psi': record.get('average_psi', 0),
                'recommendation': record['recommendation']
            })
        
        return pd.DataFrame(summary_data)
    
    def plot_drift_trends(self, figsize: Tuple[int, int] = (12, 8), 
                         save_path: Optional[str] = None):
        """Plot drift monitoring trends over time."""
        if not self.monitoring_history_:
            raise ValueError("No monitoring history available")
        
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
        except ImportError:
            raise ImportError("matplotlib required for plotting")
        
        df = self.get_drift_summary()
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # PSI trends
        axes[0, 0].plot(df['timestamp'], df['avg_psi'], marker='o')
        axes[0, 0].axhline(y=self.psi_threshold, color='red', linestyle='--', 
                          label=f'Threshold ({self.psi_threshold})')
        axes[0, 0].set_title('Average PSI Over Time')
        axes[0, 0].set_ylabel('PSI Score')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Number of drifting features
        axes[0, 1].bar(df['timestamp'], df['n_drift_features'])
        axes[0, 1].set_title('Number of Drifting Features')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Sample size trends
        axes[1, 0].plot(df['timestamp'], df['n_samples'], marker='s', color='green')
        axes[1, 0].set_title('Sample Size Over Time')
        axes[1, 0].set_ylabel('N Samples')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Recommendations
        rec_counts = df['recommendation'].value_counts()
        axes[1, 1].pie(rec_counts.values, labels=rec_counts.index, autopct='%1.1f%%')
        axes[1, 1].set_title('Recommendation Distribution')
        
        # Format x-axis dates
        for ax in [axes[0, 0], axes[0, 1], axes[1, 0]]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.WeekdayLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig