"""
Loss functions for gradient boosting optimized for fraud detection.

All gradients and hessians are computed w.r.t. raw logits (f),
where p = sigmoid(f). This is critical for gradient boosting which
operates in logit space.
"""

import numpy as np
from typing import Optional, Tuple
from abc import ABC, abstractmethod


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x))
    )


class BaseLoss(ABC):
    """Base class for loss functions."""
    
    @abstractmethod
    def __call__(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                 sample_weight: Optional[np.ndarray] = None) -> float:
        """Compute the loss value. y_pred_proba is probability [0,1]."""
        pass
    
    @abstractmethod
    def gradient(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                 sample_weight: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute gradient w.r.t. logits (not probabilities)."""
        pass
    
    @abstractmethod
    def hessian(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                sample_weight: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute hessian w.r.t. logits (not probabilities)."""
        pass


class ValueWeightedLogLoss(BaseLoss):
    """
    Value-weighted asymmetric log loss for fraud detection.
    
    Loss: L(y, p) = -[y * w_fn * log(p) + (1-y) * w_fp * log(1-p)]
    
    Gradients are computed w.r.t. logit f (where p = sigmoid(f)):
        dL/df = w_fp * p - w_fn * y  (for each sample, using appropriate weight)
        d²L/df² = p*(1-p) * (y*w_fn + (1-y)*w_fp)
    """
    
    def __init__(self, fp_cost: float = 100.0, eps: float = 1e-15):
        self.fp_cost = fp_cost
        self.eps = eps
    
    def _compute_weights(self, y_true: np.ndarray, amounts: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Compute per-sample FN and FP weights."""
        n = len(y_true)
        if amounts is None:
            return np.ones(n), np.ones(n)
        
        median_amount = np.median(amounts[amounts > 0]) if np.any(amounts > 0) else 1.0
        w_fn = amounts / median_amount
        w_fp = np.full(n, self.fp_cost / median_amount)
        return w_fn, w_fp
    
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, 
                 amounts: Optional[np.ndarray] = None) -> float:
        y_pred = np.clip(y_pred, self.eps, 1 - self.eps)
        w_fn, w_fp = self._compute_weights(y_true, amounts)
        loss = -(y_true * w_fn * np.log(y_pred) + (1 - y_true) * w_fp * np.log(1 - y_pred))
        return np.mean(loss)
    
    def gradient(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                 amounts: Optional[np.ndarray] = None) -> np.ndarray:
        """Gradient w.r.t. logits: dL/df = (1-y)*w_fp*p - y*w_fn*(1-p)"""
        p = np.clip(y_pred_proba, self.eps, 1 - self.eps)
        w_fn, w_fp = self._compute_weights(y_true, amounts)
        # Chain rule: dL/df = dL/dp * dp/df where dp/df = p*(1-p)
        # dL/dp = -y*w_fn/p + (1-y)*w_fp/(1-p)
        # dL/df = [-y*w_fn/p + (1-y)*w_fp/(1-p)] * p*(1-p)
        #       = -y*w_fn*(1-p) + (1-y)*w_fp*p
        grad = -y_true * w_fn * (1 - p) + (1 - y_true) * w_fp * p
        return grad
    
    def hessian(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                amounts: Optional[np.ndarray] = None) -> np.ndarray:
        """Hessian w.r.t. logits."""
        p = np.clip(y_pred_proba, self.eps, 1 - self.eps)
        w_fn, w_fp = self._compute_weights(y_true, amounts)
        # d²L/df² = p*(1-p) * (y*w_fn + (1-y)*w_fp)
        hess = p * (1 - p) * (y_true * w_fn + (1 - y_true) * w_fp)
        return np.maximum(hess, 1e-8)  # Ensure positive hessian


class FocalLoss(BaseLoss):
    """
    Focal loss for fraud detection — focuses on hard examples.
    Gradients w.r.t. logits.
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, eps: float = 1e-15):
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
    
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, 
                 amounts: Optional[np.ndarray] = None) -> float:
        y_pred = np.clip(y_pred, self.eps, 1 - self.eps)
        p_t = np.where(y_true == 1, y_pred, 1 - y_pred)
        alpha_t = np.where(y_true == 1, self.alpha, 1 - self.alpha)
        loss = -alpha_t * (1 - p_t) ** self.gamma * np.log(p_t)
        return np.mean(loss)
    
    def gradient(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                 amounts: Optional[np.ndarray] = None) -> np.ndarray:
        """Approximate gradient w.r.t. logits using weighted log-loss gradient."""
        p = np.clip(y_pred_proba, self.eps, 1 - self.eps)
        p_t = np.where(y_true == 1, p, 1 - p)
        alpha_t = np.where(y_true == 1, self.alpha, 1 - self.alpha)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        # Standard log-loss gradient scaled by focal weight
        grad = focal_weight * (p - y_true)
        return grad
    
    def hessian(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                amounts: Optional[np.ndarray] = None) -> np.ndarray:
        """Approximate hessian w.r.t. logits."""
        p = np.clip(y_pred_proba, self.eps, 1 - self.eps)
        p_t = np.where(y_true == 1, p, 1 - p)
        alpha_t = np.where(y_true == 1, self.alpha, 1 - self.alpha)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        hess = focal_weight * p * (1 - p)
        return np.maximum(hess, 1e-8)


class LogLoss(BaseLoss):
    """Standard log loss (baseline). Gradients w.r.t. logits."""
    
    def __init__(self, eps: float = 1e-15):
        self.eps = eps
    
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, 
                 sample_weight: Optional[np.ndarray] = None) -> float:
        y_pred = np.clip(y_pred, self.eps, 1 - self.eps)
        loss = -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)
        if sample_weight is not None:
            loss = loss * sample_weight
        return np.mean(loss)
    
    def gradient(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                 sample_weight: Optional[np.ndarray] = None) -> np.ndarray:
        """dL/df = p - y (standard GBM gradient)."""
        p = np.clip(y_pred_proba, self.eps, 1 - self.eps)
        grad = p - y_true
        if sample_weight is not None:
            grad = grad * sample_weight
        return grad
    
    def hessian(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                sample_weight: Optional[np.ndarray] = None) -> np.ndarray:
        """d²L/df² = p * (1-p)."""
        p = np.clip(y_pred_proba, self.eps, 1 - self.eps)
        hess = p * (1 - p)
        if sample_weight is not None:
            hess = hess * sample_weight
        return np.maximum(hess, 1e-8)
