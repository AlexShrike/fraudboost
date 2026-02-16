"""
Loss functions for gradient boosting optimized for fraud detection.

All loss functions provide gradients and hessians for boosting iterations,
with support for value-weighted asymmetric costs.
"""

import numpy as np
from typing import Optional, Tuple
from abc import ABC, abstractmethod


class BaseLoss(ABC):
    """Base class for loss functions."""
    
    @abstractmethod
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, 
                 sample_weight: Optional[np.ndarray] = None) -> float:
        """Compute the loss value."""
        pass
    
    @abstractmethod
    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray, 
                 sample_weight: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute the gradient (first derivative)."""
        pass
    
    @abstractmethod
    def hessian(self, y_true: np.ndarray, y_pred: np.ndarray, 
                sample_weight: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute the hessian (second derivative)."""
        pass


class ValueWeightedLogLoss(BaseLoss):
    """
    Value-weighted asymmetric log loss for fraud detection.
    
    Asymmetric weighting:
    - False negatives weighted by transaction amount (missing fraud is expensive)
    - False positives have fixed cost (investigation cost)
    
    Loss: L(y, p) = -[y * w_fn(amount) * log(p) + (1-y) * w_fp * log(1-p)]
    where:
        w_fn = amount / median_amount (higher for larger transactions)
        w_fp = fp_cost / median_amount (fixed investigation cost)
    """
    
    def __init__(self, fp_cost: float = 100.0, eps: float = 1e-15):
        """
        Args:
            fp_cost: Fixed cost of false positive (investigation cost)
            eps: Small constant to prevent log(0)
        """
        self.fp_cost = fp_cost
        self.eps = eps
    
    def _compute_weights(self, y_true: np.ndarray, amounts: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Compute per-sample weights for FN and FP costs."""
        n_samples = len(y_true)
        
        if amounts is None:
            # No amount weighting - uniform costs
            w_fn = np.ones(n_samples)
            w_fp = np.ones(n_samples)
        else:
            # Value-weighted: FN cost proportional to amount, FP cost fixed
            median_amount = np.median(amounts[amounts > 0])
            w_fn = amounts / median_amount  # Higher weight for larger transactions
            w_fp = np.full(n_samples, self.fp_cost / median_amount)
        
        return w_fn, w_fp
    
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, 
                 amounts: Optional[np.ndarray] = None) -> float:
        """Compute value-weighted log loss."""
        y_pred = np.clip(y_pred, self.eps, 1 - self.eps)
        w_fn, w_fp = self._compute_weights(y_true, amounts)
        
        # Asymmetric log loss
        loss_pos = y_true * w_fn * np.log(y_pred)
        loss_neg = (1 - y_true) * w_fp * np.log(1 - y_pred)
        
        return -np.mean(loss_pos + loss_neg)
    
    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray, 
                 amounts: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute gradient of value-weighted log loss."""
        y_pred = np.clip(y_pred, self.eps, 1 - self.eps)
        w_fn, w_fp = self._compute_weights(y_true, amounts)
        
        # d/dp[-y*w_fn*log(p) - (1-y)*w_fp*log(1-p)]
        # = -y*w_fn/p + (1-y)*w_fp/(1-p)
        grad = -y_true * w_fn / y_pred + (1 - y_true) * w_fp / (1 - y_pred)
        
        return grad
    
    def hessian(self, y_true: np.ndarray, y_pred: np.ndarray, 
                amounts: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute hessian of value-weighted log loss."""
        y_pred = np.clip(y_pred, self.eps, 1 - self.eps)
        w_fn, w_fp = self._compute_weights(y_true, amounts)
        
        # Second derivative
        hess = y_true * w_fn / (y_pred ** 2) + (1 - y_true) * w_fp / ((1 - y_pred) ** 2)
        
        return hess


class FocalLoss(BaseLoss):
    """
    Focal loss adapted for fraud detection with value weighting.
    
    Down-weights easy examples to focus on hard cases.
    From "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    
    Loss: FL(y, p) = -α * (1-p*)^γ * log(p*)
    where p* = p if y=1, else 1-p
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, eps: float = 1e-15):
        """
        Args:
            alpha: Weighting factor for rare class (fraud)
            gamma: Focusing parameter (higher = more focus on hard examples)
            eps: Small constant to prevent log(0)
        """
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
    
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, 
                 amounts: Optional[np.ndarray] = None) -> float:
        """Compute focal loss."""
        y_pred = np.clip(y_pred, self.eps, 1 - self.eps)
        
        # p* = p if y=1, else 1-p
        p_t = np.where(y_true == 1, y_pred, 1 - y_pred)
        
        # α factor: higher weight for positive class (fraud)
        alpha_t = np.where(y_true == 1, self.alpha, 1 - self.alpha)
        
        # Focal term: (1-p*)^γ
        focal_term = (1 - p_t) ** self.gamma
        
        # Focal loss: -α * (1-p*)^γ * log(p*)
        loss = -alpha_t * focal_term * np.log(p_t)
        
        # Value weighting if amounts provided
        if amounts is not None:
            median_amount = np.median(amounts[amounts > 0])
            value_weights = amounts / median_amount
            loss = loss * value_weights
        
        return np.mean(loss)
    
    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray, 
                 amounts: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute gradient of focal loss."""
        y_pred = np.clip(y_pred, self.eps, 1 - self.eps)
        
        p_t = np.where(y_true == 1, y_pred, 1 - y_pred)
        alpha_t = np.where(y_true == 1, self.alpha, 1 - self.alpha)
        focal_term = (1 - p_t) ** self.gamma
        
        # Complex gradient computation for focal loss
        # d/dp[FL] = α * (1-p*)^(γ-1) * [γ*p**log(p*) - (1-p*)]
        grad_factor = alpha_t * (1 - p_t) ** (self.gamma - 1)
        grad_term = self.gamma * p_t * np.log(p_t) - (1 - p_t)
        
        # Adjust sign based on true class
        grad = np.where(y_true == 1, 
                       -grad_factor * grad_term / y_pred,
                       grad_factor * grad_term / (1 - y_pred))
        
        # Value weighting
        if amounts is not None:
            median_amount = np.median(amounts[amounts > 0])
            value_weights = amounts / median_amount
            grad = grad * value_weights
        
        return grad
    
    def hessian(self, y_true: np.ndarray, y_pred: np.ndarray, 
                amounts: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute hessian of focal loss (approximation)."""
        # Simplified approximation for hessian
        y_pred = np.clip(y_pred, self.eps, 1 - self.eps)
        
        # Use standard log-loss hessian scaled by focal weighting
        base_hess = y_pred * (1 - y_pred)
        
        p_t = np.where(y_true == 1, y_pred, 1 - y_pred)
        alpha_t = np.where(y_true == 1, self.alpha, 1 - self.alpha)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        
        hess = base_hess * focal_weight
        
        # Value weighting
        if amounts is not None:
            median_amount = np.median(amounts[amounts > 0])
            value_weights = amounts / median_amount
            hess = hess * value_weights
        
        return hess


class LogLoss(BaseLoss):
    """Standard log loss (baseline)."""
    
    def __init__(self, eps: float = 1e-15):
        self.eps = eps
    
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, 
                 sample_weight: Optional[np.ndarray] = None) -> float:
        """Compute log loss."""
        y_pred = np.clip(y_pred, self.eps, 1 - self.eps)
        loss = -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)
        
        if sample_weight is not None:
            loss = loss * sample_weight
        
        return np.mean(loss)
    
    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray, 
                 sample_weight: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute gradient of log loss."""
        y_pred = np.clip(y_pred, self.eps, 1 - self.eps)
        grad = -y_true / y_pred + (1 - y_true) / (1 - y_pred)
        
        if sample_weight is not None:
            grad = grad * sample_weight
        
        return grad
    
    def hessian(self, y_true: np.ndarray, y_pred: np.ndarray, 
                sample_weight: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute hessian of log loss."""
        y_pred = np.clip(y_pred, self.eps, 1 - self.eps)
        hess = y_true / (y_pred ** 2) + (1 - y_true) / ((1 - y_pred) ** 2)
        
        if sample_weight is not None:
            hess = hess * sample_weight
        
        return hess