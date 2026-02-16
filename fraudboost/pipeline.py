"""
End-to-end fraud detection pipeline.

Combines spectral features, FraudBoost classifier, Pareto optimization,
and drift detection into a unified workflow.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any, Tuple, Union
import warnings
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from .core import FraudBoostClassifier
from .spectral import SpectralFeatureExtractor
from .pareto import ParetoOptimizer
from .drift import TemporalDriftDetector
from .metrics import classification_report_fraud, cost_benefit_analysis


class FraudDetectionPipeline:
    """
    Complete fraud detection pipeline with spectral features and optimization.
    
    Workflow:
    1. Extract spectral graph features from transaction relationships
    2. Train FraudBoost classifier with value-weighted loss
    3. Optimize decision threshold using Pareto analysis
    4. Monitor model performance and drift over time
    5. Generate comprehensive evaluation reports
    """
    
    def __init__(self,
                 # FraudBoost parameters
                 n_estimators: int = 100,
                 learning_rate: float = 0.1,
                 max_depth: int = 6,
                 fp_cost: float = 100.0,
                 loss: str = 'value_weighted',
                 
                 # Spectral features
                 use_spectral_features: bool = True,
                 spectral_components: int = 10,
                 
                 # Feature scaling
                 scale_features: bool = True,
                 
                 # Pipeline options
                 optimize_threshold: bool = True,
                 enable_drift_detection: bool = True,
                 
                 # Cross-validation
                 cv_folds: int = 3,
                 
                 random_state: Optional[int] = None,
                 verbose: int = 0):
        """
        Args:
            n_estimators: Number of boosting rounds
            learning_rate: FraudBoost learning rate
            max_depth: Maximum tree depth
            fp_cost: Fixed cost per false positive investigation
            loss: Loss function ('value_weighted', 'focal', 'logloss')
            use_spectral_features: Whether to extract graph features
            spectral_components: Number of spectral components
            scale_features: Whether to standardize features
            optimize_threshold: Whether to run Pareto optimization
            enable_drift_detection: Whether to setup drift monitoring
            cv_folds: Number of cross-validation folds
            random_state: Random seed
            verbose: Verbosity level
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.fp_cost = fp_cost
        self.loss = loss
        self.use_spectral_features = use_spectral_features
        self.spectral_components = spectral_components
        self.scale_features = scale_features
        self.optimize_threshold = optimize_threshold
        self.enable_drift_detection = enable_drift_detection
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.verbose = verbose
        
        # Pipeline components
        self.classifier_ = None
        self.spectral_extractor_ = None
        self.scaler_ = None
        self.pareto_optimizer_ = None
        self.drift_detector_ = None
        
        # Fitted attributes
        self.feature_names_ = []
        self.optimal_threshold_ = 0.5
        self.cv_results_ = None
        self.final_evaluation_ = None
        
        # Set random state
        if random_state is not None:
            np.random.seed(random_state)
    
    def fit(self, X: np.ndarray, y: np.ndarray, amounts: np.ndarray,
            timestamps: Optional[np.ndarray] = None,
            entity_columns: Optional[List[str]] = None,
            feature_names: Optional[List[str]] = None,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            amounts_val: Optional[np.ndarray] = None) -> 'FraudDetectionPipeline':
        """
        Fit the complete fraud detection pipeline.
        
        Args:
            X: Training features
            y: Training labels (0/1)
            amounts: Transaction amounts
            timestamps: Transaction timestamps for temporal analysis
            entity_columns: Column names for graph construction (if using DataFrame)
            feature_names: Names of input features
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            amounts_val: Validation amounts (optional)
        """
        X, y, amounts = self._validate_input(X, y, amounts)
        n_samples, n_features = X.shape
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(n_features)]
        
        self.feature_names_ = feature_names.copy()
        
        # Convert to DataFrame for spectral features if needed
        if self.use_spectral_features and entity_columns is not None:
            if not isinstance(X, pd.DataFrame):
                df = pd.DataFrame(X, columns=feature_names)
            else:
                df = X.copy()
            
            # Add labels and amounts for complete dataset
            df['_target'] = y
            df['_amount'] = amounts
            if timestamps is not None:
                df['_timestamp'] = timestamps
        
        if self.verbose >= 1:
            print(f"Training fraud detection pipeline on {n_samples} samples")
            fraud_rate = np.mean(y)
            print(f"Fraud rate: {fraud_rate:.4f} ({fraud_rate*100:.2f}%)")
            total_fraud_value = np.sum(amounts[y == 1])
            print(f"Total fraud value: ${total_fraud_value:,.2f}")
        
        # Step 1: Extract spectral features
        if self.use_spectral_features and entity_columns is not None:
            if self.verbose >= 1:
                print("Extracting spectral graph features...")
            
            self.spectral_extractor_ = SpectralFeatureExtractor(
                n_components=self.spectral_components,
                random_state=self.random_state
            )
            
            spectral_features = self.spectral_extractor_.fit_transform(
                df, entity_columns, weight_column='_amount'
            )
            
            # Combine original and spectral features
            X_combined = np.hstack([X, spectral_features])
            spectral_names = [f'spectral_{i}' for i in range(spectral_features.shape[1])]
            self.feature_names_.extend(spectral_names)
            
            if self.verbose >= 1:
                print(f"Added {spectral_features.shape[1]} spectral features")
        else:
            X_combined = X
        
        # Step 2: Feature scaling
        if self.scale_features:
            if self.verbose >= 1:
                print("Scaling features...")
            
            self.scaler_ = StandardScaler()
            X_combined = self.scaler_.fit_transform(X_combined)
            
            if X_val is not None:
                X_val_combined = X_val
                if self.use_spectral_features and entity_columns is not None:
                    # Apply same spectral transformation to validation set
                    val_df = pd.DataFrame(X_val, columns=feature_names[:X_val.shape[1]])
                    val_df['_target'] = y_val if y_val is not None else 0
                    val_df['_amount'] = amounts_val if amounts_val is not None else 1
                    
                    val_spectral = self.spectral_extractor_.transform(val_df, entity_columns)
                    X_val_combined = np.hstack([X_val, val_spectral])
                
                X_val_combined = self.scaler_.transform(X_val_combined)
        else:
            if X_val is not None:
                X_val_combined = X_val
                if self.use_spectral_features and entity_columns is not None:
                    val_df = pd.DataFrame(X_val, columns=feature_names[:X_val.shape[1]])
                    val_df['_target'] = y_val if y_val is not None else 0
                    val_df['_amount'] = amounts_val if amounts_val is not None else 1
                    val_spectral = self.spectral_extractor_.transform(val_df, entity_columns)
                    X_val_combined = np.hstack([X_val, val_spectral])
            else:
                X_val_combined = None
        
        # Step 3: Cross-validation evaluation
        if self.cv_folds > 1:
            if self.verbose >= 1:
                print(f"Running {self.cv_folds}-fold temporal cross-validation...")
            
            self.cv_results_ = self._cross_validate(
                X_combined, y, amounts, timestamps
            )
            
            if self.verbose >= 1:
                cv_vdr = np.mean(self.cv_results_['vdr_scores'])
                cv_savings = np.mean(self.cv_results_['net_savings'])
                print(f"CV Value Detection Rate: {cv_vdr:.4f}")
                print(f"CV Net Savings: ${cv_savings:,.2f}")
        
        # Step 4: Fit final model
        if self.verbose >= 1:
            print("Training final FraudBoost model...")
        
        self.classifier_ = FraudBoostClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            fp_cost=self.fp_cost,
            loss=self.loss,
            random_state=self.random_state,
            verbose=max(0, self.verbose - 1)
        )
        
        self.classifier_.fit(
            X_combined, y, amounts, 
            X_val_combined, y_val, amounts_val
        )
        
        # Step 5: Threshold optimization
        if self.optimize_threshold:
            if self.verbose >= 1:
                print("Optimizing decision threshold...")
            
            # Use validation set for threshold optimization if available
            if X_val_combined is not None:
                thresh_X, thresh_y, thresh_amounts = X_val_combined, y_val, amounts_val
            else:
                # Use training set (not ideal but better than nothing)
                thresh_X, thresh_y, thresh_amounts = X_combined, y, amounts
                warnings.warn("Using training set for threshold optimization - may overfit")
            
            thresh_proba = self.classifier_.predict_proba(thresh_X)[:, 1]
            
            self.pareto_optimizer_ = ParetoOptimizer(fp_cost=self.fp_cost)
            self.pareto_optimizer_.fit(thresh_y, thresh_proba, thresh_amounts)
            
            # Choose default strategy (maximize savings)
            optimal_config = self.pareto_optimizer_.recommend_threshold('max_savings')
            self.optimal_threshold_ = optimal_config['threshold']
            
            if self.verbose >= 1:
                print(f"Optimal threshold: {self.optimal_threshold_:.3f}")
                print(f"Expected net savings: ${optimal_config['metrics']['net_savings']:,.2f}")
        
        # Step 6: Setup drift detection
        if self.enable_drift_detection:
            if self.verbose >= 1:
                print("Setting up drift detection...")
            
            self.drift_detector_ = TemporalDriftDetector()
            
            train_proba = self.classifier_.predict_proba(X_combined)[:, 1]
            self.drift_detector_.fit_reference(
                X_combined, y, train_proba, timestamps, self.feature_names_
            )
        
        # Step 7: Final evaluation
        if X_val_combined is not None:
            if self.verbose >= 1:
                print("Generating final evaluation report...")
            
            self.final_evaluation_ = self._evaluate_model(
                X_val_combined, y_val, amounts_val
            )
        
        if self.verbose >= 1:
            print("Pipeline training completed!")
        
        return self
    
    def predict(self, X: np.ndarray, 
                entity_columns: Optional[List[str]] = None) -> np.ndarray:
        """Predict fraud labels using optimal threshold."""
        proba = self.predict_proba(X, entity_columns)
        return (proba[:, 1] >= self.optimal_threshold_).astype(int)
    
    def predict_proba(self, X: np.ndarray,
                     entity_columns: Optional[List[str]] = None) -> np.ndarray:
        """Predict fraud probabilities."""
        if self.classifier_ is None:
            raise ValueError("Pipeline must be fitted first")
        
        X_processed = self._preprocess_features(X, entity_columns)
        return self.classifier_.predict_proba(X_processed)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, amounts: np.ndarray,
                entity_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Comprehensive evaluation on test data."""
        return self._evaluate_model(X, y, amounts, entity_columns)
    
    def detect_drift(self, X: np.ndarray, y: Optional[np.ndarray] = None,
                    amounts: Optional[np.ndarray] = None,
                    timestamps: Optional[np.ndarray] = None,
                    entity_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Detect drift in new data."""
        if self.drift_detector_ is None:
            raise ValueError("Drift detection not enabled")
        
        X_processed = self._preprocess_features(X, entity_columns)
        
        y_pred_proba = None
        if amounts is not None:
            y_pred_proba = self.classifier_.predict_proba(X_processed)[:, 1]
        
        return self.drift_detector_.detect_drift(
            X_processed, y, y_pred_proba, timestamps, self.feature_names_
        )
    
    def _preprocess_features(self, X: np.ndarray, 
                           entity_columns: Optional[List[str]] = None) -> np.ndarray:
        """Apply same preprocessing as during training."""
        X_processed = X.copy()
        
        # Extract spectral features if enabled
        if self.use_spectral_features and self.spectral_extractor_ is not None:
            if entity_columns is not None:
                if not isinstance(X, pd.DataFrame):
                    n_orig_features = len(self.feature_names_) - self.spectral_components
                    orig_names = self.feature_names_[:n_orig_features]
                    df = pd.DataFrame(X, columns=orig_names[:X.shape[1]])
                else:
                    df = X.copy()
                
                # Add dummy columns for spectral extractor
                df['_amount'] = 1.0  # Default amount
                
                spectral_features = self.spectral_extractor_.transform(df, entity_columns)
                X_processed = np.hstack([X_processed, spectral_features])
        
        # Apply scaling if enabled
        if self.scaler_ is not None:
            X_processed = self.scaler_.transform(X_processed)
        
        return X_processed
    
    def _cross_validate(self, X: np.ndarray, y: np.ndarray, amounts: np.ndarray,
                       timestamps: Optional[np.ndarray] = None) -> Dict[str, List[float]]:
        """Perform temporal cross-validation."""
        from .metrics import value_detection_rate, net_savings, precision_score, recall_score
        
        # Use temporal splits if timestamps available
        if timestamps is not None:
            # Sort by timestamp
            sort_idx = np.argsort(timestamps)
            X_sorted = X[sort_idx]
            y_sorted = y[sort_idx]
            amounts_sorted = amounts[sort_idx]
            
            cv_splitter = TimeSeriesSplit(n_splits=self.cv_folds)
        else:
            # Fallback to regular splits
            from sklearn.model_selection import StratifiedKFold
            cv_splitter = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, 
                                        random_state=self.random_state)
            X_sorted, y_sorted, amounts_sorted = X, y, amounts
        
        cv_results = {
            'precision_scores': [],
            'recall_scores': [], 
            'vdr_scores': [],
            'net_savings': [],
            'roi_scores': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(cv_splitter.split(X_sorted, y_sorted)):
            if self.verbose >= 2:
                print(f"  Fold {fold + 1}/{self.cv_folds}")
            
            X_train_fold = X_sorted[train_idx]
            y_train_fold = y_sorted[train_idx]
            amounts_train_fold = amounts_sorted[train_idx]
            
            X_val_fold = X_sorted[val_idx]
            y_val_fold = y_sorted[val_idx]
            amounts_val_fold = amounts_sorted[val_idx]
            
            # Train model on fold
            fold_classifier = FraudBoostClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                fp_cost=self.fp_cost,
                loss=self.loss,
                random_state=self.random_state,
                verbose=0
            )
            
            fold_classifier.fit(X_train_fold, y_train_fold, amounts_train_fold)
            
            # Predict on validation
            y_pred_proba = fold_classifier.predict_proba(X_val_fold)[:, 1]
            y_pred = (y_pred_proba >= 0.5).astype(int)
            
            # Compute metrics
            precision = precision_score(y_val_fold, y_pred, zero_division=0)
            recall = recall_score(y_val_fold, y_pred, zero_division=0)
            vdr = value_detection_rate(y_val_fold, y_pred, amounts_val_fold)
            savings = net_savings(y_val_fold, y_pred, amounts_val_fold, self.fp_cost)
            roi_score = savings / (np.sum(y_pred) * self.fp_cost) if np.sum(y_pred) > 0 else 0
            
            cv_results['precision_scores'].append(precision)
            cv_results['recall_scores'].append(recall)
            cv_results['vdr_scores'].append(vdr)
            cv_results['net_savings'].append(savings)
            cv_results['roi_scores'].append(roi_score)
        
        return cv_results
    
    def _evaluate_model(self, X: np.ndarray, y: np.ndarray, amounts: np.ndarray,
                       entity_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Comprehensive model evaluation."""
        if isinstance(X, pd.DataFrame) and entity_columns is None:
            # If X is DataFrame, we can process it directly
            X_processed = self._preprocess_features(X, entity_columns)
        else:
            X_processed = self._preprocess_features(X, entity_columns)
        
        # Predictions
        y_pred_proba = self.classifier_.predict_proba(X_processed)[:, 1]
        y_pred = (y_pred_proba >= self.optimal_threshold_).astype(int)
        
        # Comprehensive metrics
        fraud_report = classification_report_fraud(
            y, y_pred, amounts, self.fp_cost, output_dict=True
        )
        
        cost_benefit = cost_benefit_analysis(y, y_pred, amounts, self.fp_cost)
        
        evaluation = {
            'threshold_used': self.optimal_threshold_,
            'classification_metrics': fraud_report,
            'cost_benefit_analysis': cost_benefit,
            'model_info': {
                'n_estimators_used': len(self.classifier_.estimators_),
                'best_iteration': self.classifier_.best_iteration_,
                'feature_importance': self.classifier_.feature_importances_
            }
        }
        
        return evaluation
    
    def _validate_input(self, X: np.ndarray, y: np.ndarray, 
                       amounts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Validate input arrays."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int32)
        amounts = np.asarray(amounts, dtype=np.float64)
        
        if X.ndim != 2:
            raise ValueError("X must be 2-dimensional")
        if y.ndim != 1:
            raise ValueError("y must be 1-dimensional")
        if amounts.ndim != 1:
            raise ValueError("amounts must be 1-dimensional")
        if len(X) != len(y) or len(X) != len(amounts):
            raise ValueError("X, y, and amounts must have same number of samples")
        if not np.all(np.isin(y, [0, 1])):
            raise ValueError("y must contain only 0 and 1 values")
        if not np.all(amounts >= 0):
            raise ValueError("amounts must be non-negative")
        
        return X, y, amounts
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get summary of pipeline configuration and performance."""
        summary = {
            'configuration': {
                'n_estimators': self.n_estimators,
                'learning_rate': self.learning_rate,
                'max_depth': self.max_depth,
                'fp_cost': self.fp_cost,
                'loss_function': self.loss,
                'use_spectral_features': self.use_spectral_features,
                'scale_features': self.scale_features,
                'optimal_threshold': self.optimal_threshold_
            },
            'model_performance': {},
            'cv_performance': {},
            'feature_info': {
                'n_features': len(self.feature_names_),
                'feature_names': self.feature_names_
            }
        }
        
        if self.cv_results_:
            summary['cv_performance'] = {
                'mean_vdr': np.mean(self.cv_results_['vdr_scores']),
                'std_vdr': np.std(self.cv_results_['vdr_scores']),
                'mean_net_savings': np.mean(self.cv_results_['net_savings']),
                'std_net_savings': np.std(self.cv_results_['net_savings'])
            }
        
        if self.final_evaluation_:
            summary['model_performance'] = self.final_evaluation_['classification_metrics']
        
        return summary