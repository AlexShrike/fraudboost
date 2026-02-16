"""
FraudBoostClassifier: Gradient boosting optimized for fraud detection.

The core innovation is using value-weighted asymmetric loss functions where
false negatives are weighted by transaction amount and false positives have
fixed investigation costs.
"""

import numpy as np
from typing import Optional, Union, Tuple, List, Dict, Any
import warnings

from .losses import BaseLoss, ValueWeightedLogLoss, LogLoss, FocalLoss
from .tree import ValueWeightedDecisionTree


class FraudBoostClassifier:
    """
    Gradient boosting classifier optimized for fraud detection.
    
    Key features:
    - Value-weighted loss functions (FN cost ∝ amount, FP cost = fixed)
    - Tree splits maximize net savings rather than information gain
    - Built-in early stopping and regularization
    - Feature importance tracking
    """
    
    def __init__(self,
                 n_estimators: int = 100,
                 learning_rate: float = 0.1,
                 max_depth: int = 6,
                 min_samples_split: int = 20,
                 min_samples_leaf: int = 10,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.8,
                 reg_lambda: float = 1.0,
                 reg_alpha: float = 0.0,
                 fp_cost: float = 100.0,
                 loss: Union[str, BaseLoss] = 'value_weighted',
                 early_stopping_rounds: Optional[int] = 10,
                 validation_fraction: float = 0.1,
                 random_state: Optional[int] = None,
                 verbose: int = 0):
        """
        Args:
            n_estimators: Number of boosting rounds
            learning_rate: Shrinkage parameter (0.01 to 0.3)
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples to split a node
            min_samples_leaf: Minimum samples in a leaf
            subsample: Fraction of samples used per tree (0.5 to 1.0)
            colsample_bytree: Fraction of features used per tree (0.5 to 1.0)
            reg_lambda: L2 regularization strength
            reg_alpha: L1 regularization strength (not implemented yet)
            fp_cost: Fixed cost of false positive investigations
            loss: Loss function ('value_weighted', 'focal', 'logloss', or BaseLoss instance)
            early_stopping_rounds: Stop if no improvement for N rounds
            validation_fraction: Fraction of training data for early stopping
            random_state: Random seed for reproducibility
            verbose: Verbosity level (0=silent, 1=progress, 2=detailed)
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.fp_cost = fp_cost
        self.loss = loss
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_fraction = validation_fraction
        self.random_state = random_state
        self.verbose = verbose
        
        # Fitted attributes
        self.estimators_: List[ValueWeightedDecisionTree] = []
        self.feature_importances_: Optional[np.ndarray] = None
        self.train_losses_: List[float] = []
        self.val_losses_: List[float] = []
        self.best_iteration_: Optional[int] = None
        self.base_score_: float = 0.0
        self.n_features_: int = 0
        
        # Random state
        if random_state is not None:
            np.random.seed(random_state)
        
        # Setup loss function
        self.loss_fn_ = self._setup_loss_function(loss)
    
    def _setup_loss_function(self, loss: Union[str, BaseLoss]) -> BaseLoss:
        """Setup the loss function."""
        if isinstance(loss, BaseLoss):
            return loss
        elif loss == 'value_weighted':
            return ValueWeightedLogLoss(fp_cost=self.fp_cost)
        elif loss == 'focal':
            return FocalLoss()
        elif loss == 'logloss':
            return LogLoss()
        else:
            raise ValueError(f"Unknown loss function: {loss}")
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            amounts: Optional[np.ndarray] = None,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            amounts_val: Optional[np.ndarray] = None) -> 'FraudBoostClassifier':
        """
        Fit the FraudBoost classifier.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,) - binary 0/1
            amounts: Transaction amounts for value weighting (n_samples,)
            X_val: Validation features for early stopping
            y_val: Validation labels for early stopping
            amounts_val: Validation amounts for early stopping
        """
        X, y = self._validate_input(X, y)
        n_samples, n_features = X.shape
        self.n_features_ = n_features
        
        if amounts is None:
            amounts = np.ones(n_samples)
        
        # Split training data for early stopping if no validation set provided
        if X_val is None and self.early_stopping_rounds is not None:
            X, X_val, y, y_val, amounts, amounts_val = self._train_val_split(
                X, y, amounts, self.validation_fraction
            )
        
        # Initialize base score (log-odds of positive class)
        pos_rate = np.mean(y)
        self.base_score_ = np.log(pos_rate / (1 - pos_rate)) if pos_rate > 0 else 0.0
        
        # Initialize predictions with base score
        train_pred = np.full(len(y), self.base_score_)
        
        if X_val is not None:
            val_pred = np.full(len(y_val), self.base_score_)
        
        # Feature importance accumulator
        feature_importance_sum = np.zeros(n_features)
        
        # Boosting iterations
        best_val_loss = np.inf
        rounds_without_improvement = 0
        
        for iteration in range(self.n_estimators):
            if self.verbose >= 1:
                print(f"Iteration {iteration + 1}/{self.n_estimators}")
            
            # Convert predictions to probabilities for loss computation
            train_proba = self._logits_to_proba(train_pred)
            
            # Compute gradients and hessians
            gradients = self.loss_fn_.gradient(y, train_proba, amounts)
            hessians = self.loss_fn_.hessian(y, train_proba, amounts)
            
            # Sample and feature subsampling
            train_indices = self._subsample_rows(len(y))
            feature_indices = self._subsample_features(n_features)
            
            X_subset = X[train_indices][:, feature_indices]
            grad_subset = gradients[train_indices]
            hess_subset = hessians[train_indices]
            amounts_subset = amounts[train_indices]
            
            # Fit tree on subset
            tree = ValueWeightedDecisionTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                min_samples_split=self.min_samples_split,
                fp_cost=self.fp_cost,
                reg_lambda=self.reg_lambda
            )
            
            tree.fit(X_subset, grad_subset, hess_subset, amounts_subset)
            
            # Store tree with feature mapping
            tree.feature_indices_ = feature_indices
            self.estimators_.append(tree)
            
            # Update feature importance
            tree_importance = np.zeros(n_features)
            tree_importance[feature_indices] = tree.feature_importances_
            feature_importance_sum += tree_importance
            
            # Update predictions
            train_pred += self.learning_rate * self._predict_tree(X, tree)
            
            # Compute training loss
            train_proba = self._logits_to_proba(train_pred)
            train_loss = self.loss_fn_(y, train_proba, amounts)
            self.train_losses_.append(train_loss)
            
            # Validation loss and early stopping
            if X_val is not None:
                val_pred += self.learning_rate * self._predict_tree(X_val, tree)
                val_proba = self._logits_to_proba(val_pred)
                val_loss = self.loss_fn_(y_val, val_proba, amounts_val)
                self.val_losses_.append(val_loss)
                
                if self.verbose >= 2:
                    print(f"  Train loss: {train_loss:.6f}, Val loss: {val_loss:.6f}")
                
                # Early stopping check
                if self.early_stopping_rounds is not None:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        self.best_iteration_ = iteration
                        rounds_without_improvement = 0
                    else:
                        rounds_without_improvement += 1
                        
                    if rounds_without_improvement >= self.early_stopping_rounds:
                        if self.verbose >= 1:
                            print(f"Early stopping at iteration {iteration + 1}")
                        break
            else:
                if self.verbose >= 2:
                    print(f"  Train loss: {train_loss:.6f}")
        
        # Normalize feature importances
        if feature_importance_sum.sum() > 0:
            self.feature_importances_ = feature_importance_sum / feature_importance_sum.sum()
        else:
            self.feature_importances_ = np.zeros(n_features)
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        X = self._validate_input_predict(X)
        
        # Get predictions from all trees
        predictions = np.full(len(X), self.base_score_)
        
        n_trees = len(self.estimators_)
        if self.best_iteration_ is not None:
            n_trees = min(n_trees, self.best_iteration_ + 1)
        
        for i in range(n_trees):
            predictions += self.learning_rate * self._predict_tree(X, self.estimators_[i])
        
        # Convert logits to probabilities
        probabilities = self._logits_to_proba(predictions)
        
        # Return both classes probabilities
        return np.column_stack([1 - probabilities, probabilities])
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary classes."""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= threshold).astype(int)
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return the decision function (raw predictions before sigmoid)."""
        X = self._validate_input_predict(X)
        
        predictions = np.full(len(X), self.base_score_)
        
        n_trees = len(self.estimators_)
        if self.best_iteration_ is not None:
            n_trees = min(n_trees, self.best_iteration_ + 1)
        
        for i in range(n_trees):
            predictions += self.learning_rate * self._predict_tree(X, self.estimators_[i])
        
        return predictions
    
    def _predict_tree(self, X: np.ndarray, tree: ValueWeightedDecisionTree) -> np.ndarray:
        """Predict using a single tree with feature mapping."""
        if hasattr(tree, 'feature_indices_'):
            X_subset = X[:, tree.feature_indices_]
        else:
            X_subset = X
        
        return tree.predict(X_subset)
    
    def _logits_to_proba(self, logits: np.ndarray) -> np.ndarray:
        """Convert logits to probabilities using sigmoid."""
        # Clip to prevent overflow
        logits = np.clip(logits, -250, 250)
        return 1.0 / (1.0 + np.exp(-logits))
    
    def _subsample_rows(self, n_samples: int) -> np.ndarray:
        """Subsample rows for bagging."""
        n_subset = int(self.subsample * n_samples)
        return np.random.choice(n_samples, size=n_subset, replace=False)
    
    def _subsample_features(self, n_features: int) -> np.ndarray:
        """Subsample features for each tree."""
        n_subset = int(self.colsample_bytree * n_features)
        return np.random.choice(n_features, size=n_subset, replace=False)
    
    def _train_val_split(self, X: np.ndarray, y: np.ndarray, amounts: np.ndarray, 
                        val_fraction: float) -> Tuple[np.ndarray, ...]:
        """Split training data into train/validation sets."""
        n_samples = len(X)
        n_val = int(val_fraction * n_samples)
        
        # Stratified split to preserve fraud rate
        fraud_indices = np.where(y == 1)[0]
        legit_indices = np.where(y == 0)[0]
        
        n_fraud_val = max(1, int(len(fraud_indices) * val_fraction))
        n_legit_val = n_val - n_fraud_val
        
        val_fraud_indices = np.random.choice(fraud_indices, n_fraud_val, replace=False)
        val_legit_indices = np.random.choice(legit_indices, n_legit_val, replace=False)
        
        val_indices = np.concatenate([val_fraud_indices, val_legit_indices])
        train_indices = np.setdiff1d(np.arange(n_samples), val_indices)
        
        return (X[train_indices], X[val_indices],
                y[train_indices], y[val_indices],
                amounts[train_indices], amounts[val_indices])
    
    def _validate_input(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Validate input arrays."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int32)
        
        if X.ndim != 2:
            raise ValueError("X must be 2-dimensional")
        if y.ndim != 1:
            raise ValueError("y must be 1-dimensional")
        if len(X) != len(y):
            raise ValueError("X and y must have same number of samples")
        if not np.all(np.isin(y, [0, 1])):
            raise ValueError("y must contain only 0 and 1 values")
        
        return X, y
    
    def _validate_input_predict(self, X: np.ndarray) -> np.ndarray:
        """Validate input for prediction."""
        X = np.asarray(X, dtype=np.float64)
        
        if X.ndim != 2:
            raise ValueError("X must be 2-dimensional")
        if X.shape[1] != self.n_features_:
            raise ValueError(f"X has {X.shape[1]} features, expected {self.n_features_}")
        
        return X
    
    def get_feature_importance(self, importance_type: str = 'gain') -> np.ndarray:
        """
        Get feature importances.
        
        Args:
            importance_type: Type of importance ('gain', 'split_count', 'permutation')
                            Currently only 'gain' is implemented
        """
        if importance_type == 'gain':
            return self.feature_importances_.copy()
        else:
            raise NotImplementedError(f"Importance type '{importance_type}' not implemented")
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'reg_lambda': self.reg_lambda,
            'reg_alpha': self.reg_alpha,
            'fp_cost': self.fp_cost,
            'loss': self.loss,
            'early_stopping_rounds': self.early_stopping_rounds,
            'validation_fraction': self.validation_fraction,
            'random_state': self.random_state,
            'verbose': self.verbose
        }