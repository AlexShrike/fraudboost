"""
Pareto optimization for fraud detection threshold selection.

Sweeps decision thresholds and finds Pareto-optimal solutions across
multiple competing objectives: precision, recall, value detection rate,
net savings, and investigation costs.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
import matplotlib.pyplot as plt
from .metrics import (
    value_detection_rate, net_savings, roi, 
    precision_score, recall_score, f1_score,
    evaluate_at_thresholds
)


class ParetoOptimizer:
    """
    Find optimal decision thresholds using Pareto frontier analysis.
    
    Evaluates trade-offs between:
    - Precision vs Recall
    - Value Detection Rate vs False Positive Rate  
    - Net Savings vs Investigation Volume
    - ROI vs Coverage
    """
    
    def __init__(self, fp_cost: float = 100.0):
        """
        Args:
            fp_cost: Fixed cost per false positive investigation
        """
        self.fp_cost = fp_cost
        
        # Fitted attributes
        self.results_ = None
        self.pareto_indices_ = None
        self.recommended_thresholds_ = {}
    
    def fit(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
            amounts: np.ndarray, thresholds: Optional[np.ndarray] = None) -> 'ParetoOptimizer':
        """
        Evaluate performance across decision thresholds.
        
        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities
            amounts: Transaction amounts for value weighting
            thresholds: Thresholds to evaluate (default: 0.01 to 0.99 by 0.01)
        """
        if thresholds is None:
            thresholds = np.arange(0.01, 1.0, 0.01)
        
        # Evaluate all thresholds
        self.results_ = evaluate_at_thresholds(
            y_true, y_pred_proba, amounts, thresholds, self.fp_cost
        )
        
        # Find Pareto frontier
        self.pareto_indices_ = self._find_pareto_frontier()
        
        # Generate threshold recommendations
        self._generate_recommendations()
        
        return self
    
    def _find_pareto_frontier(self) -> np.ndarray:
        """
        Find Pareto-optimal solutions across multiple objectives.
        
        A solution is Pareto-optimal if no other solution dominates it
        across all objectives.
        """
        if self.results_ is None:
            raise ValueError("Must call fit() first")
        
        # Define objectives to maximize (higher is better)
        objectives = np.column_stack([
            self.results_['precision'],
            self.results_['recall'], 
            self.results_['vdr'],
            self.results_['net_savings'],
            self.results_['roi']
        ])
        
        # Handle NaN and infinite values
        objectives = np.nan_to_num(objectives, nan=0.0, posinf=1e6, neginf=-1e6)
        
        n_points = objectives.shape[0]
        pareto_mask = np.ones(n_points, dtype=bool)
        
        for i in range(n_points):
            if not pareto_mask[i]:
                continue
                
            # Check if point i is dominated by any other point
            for j in range(n_points):
                if i == j or not pareto_mask[j]:
                    continue
                
                # Point j dominates point i if j is better or equal in all objectives
                # and strictly better in at least one
                dominates = np.all(objectives[j] >= objectives[i])
                strictly_better = np.any(objectives[j] > objectives[i])
                
                if dominates and strictly_better:
                    pareto_mask[i] = False
                    break
        
        return np.where(pareto_mask)[0]
    
    def _generate_recommendations(self):
        """Generate threshold recommendations for different business priorities."""
        if self.results_ is None or self.pareto_indices_ is None:
            return
        
        results = self.results_
        pareto_idx = self.pareto_indices_
        
        # Strategy 1: Maximize Value Detection Rate
        vdr_scores = results['vdr'][pareto_idx]
        best_vdr_idx = pareto_idx[np.argmax(vdr_scores)]
        self.recommended_thresholds_['max_vdr'] = {
            'threshold': results['thresholds'][best_vdr_idx],
            'description': 'Maximize fraud value detection',
            'metrics': self._get_metrics_at_index(best_vdr_idx)
        }
        
        # Strategy 2: Maximize Net Savings
        savings_scores = results['net_savings'][pareto_idx]
        best_savings_idx = pareto_idx[np.argmax(savings_scores)]
        self.recommended_thresholds_['max_savings'] = {
            'threshold': results['thresholds'][best_savings_idx],
            'description': 'Maximize net financial savings',
            'metrics': self._get_metrics_at_index(best_savings_idx)
        }
        
        # Strategy 3: Minimize False Positives (high precision)
        precision_scores = results['precision'][pareto_idx]
        best_precision_idx = pareto_idx[np.argmax(precision_scores)]
        self.recommended_thresholds_['min_fps'] = {
            'threshold': results['thresholds'][best_precision_idx],
            'description': 'Minimize false positive investigations',
            'metrics': self._get_metrics_at_index(best_precision_idx)
        }
        
        # Strategy 4: Balanced F1 Score
        f1_scores = results['f1'][pareto_idx]
        best_f1_idx = pareto_idx[np.argmax(f1_scores)]
        self.recommended_thresholds_['balanced'] = {
            'threshold': results['thresholds'][best_f1_idx],
            'description': 'Balance precision and recall',
            'metrics': self._get_metrics_at_index(best_f1_idx)
        }
        
        # Strategy 5: Maximize ROI
        roi_scores = results['roi'][pareto_idx]
        finite_roi_mask = np.isfinite(roi_scores)
        if np.any(finite_roi_mask):
            finite_roi_idx = pareto_idx[finite_roi_mask]
            finite_roi_scores = roi_scores[finite_roi_mask]
            best_roi_idx = finite_roi_idx[np.argmax(finite_roi_scores)]
            self.recommended_thresholds_['max_roi'] = {
                'threshold': results['thresholds'][best_roi_idx],
                'description': 'Maximize return on investment',
                'metrics': self._get_metrics_at_index(best_roi_idx)
            }
    
    def _get_metrics_at_index(self, idx: int) -> Dict[str, float]:
        """Get all metrics at a specific threshold index."""
        results = self.results_
        return {
            'precision': results['precision'][idx],
            'recall': results['recall'][idx],
            'f1': results['f1'][idx],
            'vdr': results['vdr'][idx],
            'net_savings': results['net_savings'][idx],
            'roi': results['roi'][idx],
            'tp_count': int(results['tp_count'][idx]),
            'fp_count': int(results['fp_count'][idx]),
            'fn_count': int(results['fn_count'][idx]),
            'tn_count': int(results['tn_count'][idx])
        }
    
    def get_pareto_points(self) -> pd.DataFrame:
        """Get DataFrame of Pareto-optimal points."""
        if self.pareto_indices_ is None:
            raise ValueError("Must call fit() first")
        
        results = self.results_
        pareto_idx = self.pareto_indices_
        
        return pd.DataFrame({
            'threshold': results['thresholds'][pareto_idx],
            'precision': results['precision'][pareto_idx],
            'recall': results['recall'][pareto_idx],
            'f1': results['f1'][pareto_idx],
            'vdr': results['vdr'][pareto_idx],
            'net_savings': results['net_savings'][pareto_idx],
            'roi': results['roi'][pareto_idx],
            'tp_count': results['tp_count'][pareto_idx],
            'fp_count': results['fp_count'][pareto_idx]
        })
    
    def recommend_threshold(self, strategy: str = 'max_savings') -> Dict[str, Any]:
        """
        Get threshold recommendation for a business strategy.
        
        Args:
            strategy: Business strategy ('max_vdr', 'max_savings', 'min_fps', 
                     'balanced', 'max_roi')
        
        Returns:
            Dictionary with recommended threshold and metrics
        """
        if strategy not in self.recommended_thresholds_:
            available = list(self.recommended_thresholds_.keys())
            raise ValueError(f"Strategy '{strategy}' not available. Choose from: {available}")
        
        return self.recommended_thresholds_[strategy]
    
    def plot_pareto_frontier(self, x_metric: str = 'recall', y_metric: str = 'precision',
                           figsize: Tuple[int, int] = (10, 6), 
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot Pareto frontier for two competing objectives.
        
        Args:
            x_metric: Metric for x-axis ('precision', 'recall', 'vdr', 'net_savings', 'roi')
            y_metric: Metric for y-axis
            figsize: Figure size
            save_path: Path to save plot (optional)
        
        Returns:
            Matplotlib figure
        """
        if self.results_ is None:
            raise ValueError("Must call fit() first")
        
        results = self.results_
        
        # Validate metrics
        valid_metrics = ['precision', 'recall', 'f1', 'vdr', 'net_savings', 'roi']
        if x_metric not in valid_metrics or y_metric not in valid_metrics:
            raise ValueError(f"Metrics must be one of: {valid_metrics}")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot all points
        x_all = results[x_metric]
        y_all = results[y_metric]
        
        # Handle infinite/NaN values for plotting
        finite_mask = np.isfinite(x_all) & np.isfinite(y_all)
        x_all = x_all[finite_mask]
        y_all = y_all[finite_mask]
        thresholds_all = results['thresholds'][finite_mask]
        
        ax.scatter(x_all, y_all, alpha=0.3, s=20, color='lightblue', 
                  label='All thresholds')
        
        # Plot Pareto frontier
        if self.pareto_indices_ is not None:
            pareto_mask_filtered = np.isin(np.where(finite_mask)[0], self.pareto_indices_)
            x_pareto = x_all[pareto_mask_filtered]
            y_pareto = y_all[pareto_mask_filtered]
            thresholds_pareto = thresholds_all[pareto_mask_filtered]
            
            # Sort by x-axis for connecting line
            sort_idx = np.argsort(x_pareto)
            x_pareto = x_pareto[sort_idx]
            y_pareto = y_pareto[sort_idx]
            thresholds_pareto = thresholds_pareto[sort_idx]
            
            ax.plot(x_pareto, y_pareto, 'ro-', markersize=8, linewidth=2,
                   label='Pareto frontier')
            
            # Annotate some key points
            for i in range(0, len(x_pareto), max(1, len(x_pareto)//5)):
                ax.annotate(f'{thresholds_pareto[i]:.2f}', 
                           (x_pareto[i], y_pareto[i]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.8)
        
        # Highlight recommended strategies
        for strategy, rec in self.recommended_thresholds_.items():
            metrics = rec['metrics']
            if x_metric in metrics and y_metric in metrics:
                x_val = metrics[x_metric]
                y_val = metrics[y_metric]
                
                if np.isfinite(x_val) and np.isfinite(y_val):
                    ax.scatter(x_val, y_val, s=150, marker='*', 
                              label=f'{strategy.replace("_", " ").title()}')
        
        ax.set_xlabel(x_metric.replace('_', ' ').title())
        ax.set_ylabel(y_metric.replace('_', ' ').title())
        ax.set_title(f'Pareto Frontier: {y_metric.title()} vs {x_metric.title()}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_threshold_sweep(self, metrics: Optional[List[str]] = None,
                           figsize: Tuple[int, int] = (12, 8),
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot how metrics change across decision thresholds.
        
        Args:
            metrics: List of metrics to plot (default: all main metrics)
            figsize: Figure size
            save_path: Path to save plot
        
        Returns:
            Matplotlib figure
        """
        if self.results_ is None:
            raise ValueError("Must call fit() first")
        
        if metrics is None:
            metrics = ['precision', 'recall', 'f1', 'vdr', 'net_savings', 'roi']
        
        results = self.results_
        thresholds = results['thresholds']
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics[:6]):  # Max 6 subplots
            if i >= len(axes):
                break
                
            ax = axes[i]
            values = results[metric]
            
            # Handle infinite values for plotting
            if metric in ['roi', 'net_savings']:
                values = np.nan_to_num(values, nan=0, posinf=np.percentile(values[np.isfinite(values)], 95), 
                                     neginf=np.percentile(values[np.isfinite(values)], 5))
            
            ax.plot(thresholds, values, linewidth=2, label=metric.replace('_', ' ').title())
            
            # Highlight Pareto-optimal points
            if self.pareto_indices_ is not None:
                ax.scatter(thresholds[self.pareto_indices_], values[self.pareto_indices_],
                          color='red', s=30, alpha=0.7, zorder=5)
            
            # Mark recommended strategies
            for strategy, rec in self.recommended_thresholds_.items():
                if metric in rec['metrics']:
                    thresh_val = rec['threshold']
                    metric_val = rec['metrics'][metric]
                    if np.isfinite(metric_val):
                        ax.axvline(thresh_val, alpha=0.5, linestyle='--', 
                                  label=strategy.replace('_', ' '))
            
            ax.set_xlabel('Decision Threshold')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} vs Threshold')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # Remove unused subplots
        for i in range(len(metrics), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def generate_report(self) -> str:
        """Generate a comprehensive threshold optimization report."""
        if not self.recommended_thresholds_:
            return "No recommendations available. Call fit() first."
        
        report_lines = [
            "Fraud Detection Threshold Optimization Report",
            "=" * 50,
            "",
            f"Evaluated {len(self.results_['thresholds'])} thresholds from {self.results_['thresholds'][0]:.2f} to {self.results_['thresholds'][-1]:.2f}",
            f"Found {len(self.pareto_indices_)} Pareto-optimal solutions",
            f"Investigation cost: ${self.fp_cost:,.2f} per false positive",
            "",
            "RECOMMENDED THRESHOLDS:",
            "-" * 30
        ]
        
        for strategy, rec in self.recommended_thresholds_.items():
            threshold = rec['threshold']
            description = rec['description']
            metrics = rec['metrics']
            
            report_lines.extend([
                "",
                f"Strategy: {strategy.replace('_', ' ').upper()}",
                f"Description: {description}",
                f"Optimal threshold: {threshold:.3f}",
                "",
                "  Performance metrics:",
                f"    Precision:     {metrics['precision']:.4f}",
                f"    Recall:        {metrics['recall']:.4f}",
                f"    F1-Score:      {metrics['f1']:.4f}",
                f"    VDR:          {metrics['vdr']:.4f} ({metrics['vdr']*100:.1f}% of fraud value)",
                f"    Net Savings:   ${metrics['net_savings']:,.2f}",
                f"    ROI:          {metrics['roi']:.2f}x",
                "",
                "  Operational impact:",
                f"    True Positives:  {metrics['tp_count']:,} fraud cases detected",
                f"    False Positives: {metrics['fp_count']:,} false alarms",
                f"    False Negatives: {metrics['fn_count']:,} fraud cases missed",
                f"    Investigation workload: {metrics['tp_count'] + metrics['fp_count']:,} cases/day"
            ])
        
        return "\n".join(report_lines)