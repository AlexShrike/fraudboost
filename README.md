# FraudBoost

FraudBoost is a gradient boosting algorithm specifically designed for fraud detection that optimizes business value rather than statistical metrics. Unlike traditional models that treat all misclassifications equally, FraudBoost uses value-weighted loss functions where false negatives are weighted by transaction amount and false positives by investigation cost, resulting in models that maximize net financial savings.

## Key Innovation

FraudBoost's core innovation is **value-weighted asymmetric loss** that makes the default threshold (0.5) optimal for business impact. Traditional gradient boosting optimizes log-loss where all errors are equal, requiring manual threshold tuning. FraudBoost weights gradients by financial impact during training, so the learned decision boundary naturally optimizes net savings at threshold 0.5.

The loss function incorporates:
- **False negatives** weighted by transaction amount (missing a $10K fraud costs more than missing a $10 fraud)  
- **False positives** weighted by investigation cost (e.g., $100 per case reviewed)
- **Gradient flow** shaped by business value, not statistical purity

## Quick Start

```bash
pip install fraudboost
```

```python
import numpy as np
from fraudboost import FraudBoostClassifier

# Load your fraud data
# X = features, y = labels (0/1), amounts = transaction amounts

# Train with value weighting - Rust backend automatically used if available
model = FraudBoostClassifier(
    n_estimators=100,
    max_depth=4, 
    learning_rate=0.1,
    fp_cost=100,  # Investigation cost per false positive
    backend='auto'  # 'rust' for speed, 'python' for compatibility
)

# Fit model with transaction amounts for value weighting
model.fit(X_train, y_train, amounts=amounts_train)

# Predict - default threshold (0.5) is now optimal!
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

# Calculate business metrics
from fraudboost.metrics import net_savings, value_detection_rate
savings = net_savings(y_test, predictions, amounts_test, fp_cost=100)
vdr = value_detection_rate(y_test, predictions, amounts_test)
print(f"Net Savings: ${savings:,.0f}")
print(f"Value Detection Rate: {vdr:.1%}")
```

## Benchmark Results

### Setup

**Dataset:** Kartik2112 credit card fraud (1.85M transactions)
- **Training:** 320K transactions (Q1 2019 removed due to fraud rate outlier)
- **Validation:** 80K transactions (for hyperparameter tuning)  
- **Test:** 555K transactions (out-of-time, never seen during training)
- **Fraud rates:** ~0.52% train, ~0.39% test (natural distribution, no oversampling)
- **Features:** 13 engineered features with zero target leakage
- **Hyperparameters:** 100 trees, depth 4, learning rate 0.1 (identical for both models)

### FraudBoost vs XGBoost

**Default Threshold (0.5) on TEST:**

| Model | AUC | Recall | Precision | Net Savings | False Positives |
|-------|-----|--------|-----------|-------------|-----------------|
| XGBoost | 99.45% | 95.5% | 16.1% | $61K | 10,658 |
| **FraudBoost** | 98.07% | 71.0% | **72.3%** | **$1,018K** | **585** |

**Optimal Threshold on TEST:**

| Model | Threshold | Net Savings | VDR | False Positives |
|-------|----------|-------------|-----|-----------------|
| XGBoost | 0.95 | $978K | 92.8% | 734 |
| **FraudBoost** | 0.59 | **$1,024K** | **94.2%** | **436** |

**Key Insight:** FraudBoost achieves near-optimal performance at default threshold (0.5), eliminating the need for threshold tuning. XGBoost requires careful threshold optimization to achieve competitive results.

### FP Cost Sensitivity Analysis

FraudBoost wins at **every** false positive cost from $10 to $1000:

| FP Cost | XGBoost Net Savings | FraudBoost Net Savings | Winner |
|---------|--------------------|-----------------------|---------|
| $10 | $1,083K | **$1,086K** | FraudBoost |
| $50 | $1,017K | **$1,049K** | FraudBoost |  
| $100 | $978K | **$1,024K** | FraudBoost |
| $200 | $914K | **$986K** | FraudBoost |
| $500 | $749K | **$907K** | FraudBoost |
| $1000 | $555K | **$820K** | FraudBoost |

### Model Calibration

FraudBoost produces significantly better calibrated probabilities:
- **FraudBoost ECE:** 0.0015 
- **XGBoost ECE:** 0.0466
- **Improvement:** 30x better calibration

### 5-Fold Cross-Validation

| Model | Mean Net Savings | Std Dev | All Folds Win |
|-------|-----------------|---------|---------------|
| **FraudBoost** | **$236K** | **± $2.7K** | **5/5** |
| XGBoost | $227K | ± $2.2K | 0/5 |

### Cascade & Stacking Results

**Two-Stage Cascade** (XGBoost Stage 1 → FraudBoost Stage 2):
- **Net Savings:** $1,009K (marginal improvement over FraudBoost alone)
- **Speed Benefit:** 25x faster inference (XGBoost filters easy cases)
- **Use Case:** Production systems prioritizing inference speed

**Ensemble Stacking** (meta-learner combining both models):
- **Net Savings:** $1,003K  
- **Recall:** 75.5% (better than either model alone)
- **Trade-off:** Added complexity for modest gains

**Conclusion:** FraudBoost alone delivers the primary value. Cascade provides speed benefits for high-throughput production environments.

### Honest Limitations  

**Where XGBoost is Better:**
- **Higher AUC:** 99.45% vs 98.07% (better at ranking all transactions)
- **Better Recall:** 95.5% vs 71.0% at default threshold
- **Precision@FixedRecall:** P@80%R of 55.8% vs 22.2%
- **Speed:** 8x faster training (0.3s vs 2.4s on 320K rows)

**When to Choose XGBoost:**
- Goal is "catch every fraud regardless of cost"
- Need maximum recall with acceptable precision
- Extremely high-throughput training requirements

**When to Choose FraudBoost:**
- Optimizing business ROI and net savings
- Want well-calibrated probabilities out of the box
- Default threshold must work well (no time for optimization)
- False positive costs are significant

**Plugin Compatibility:** You can use FraudBoost's value-weighted loss as a custom objective in XGBoost for best-of-both-worlds performance.

## Architecture

FraudBoost modifies gradient boosting at three levels:

### 1. Loss Function (What the model optimizes)
```
Standard: L = -[y * log(p) + (1-y) * log(1-p)]
FraudBoost: L = -[y * w_fn(amount) * log(p) + (1-y) * w_fp * log(1-p)]

where:
  w_fn = transaction_amount / median_amount
  w_fp = fp_cost / median_amount  
```

### 2. Split Criterion (How trees are built)
Tree splits maximize value-weighted gain instead of statistical information gain. High-value transactions generate stronger gradient signals, naturally protecting large frauds.

### 3. Threshold Selection (How predictions become decisions)  
Built-in Pareto optimization finds the optimal operating point by sweeping thresholds and computing:
```
Net Savings(t) = sum(amounts[TP]) - count(FP) * fp_cost
```

## API Reference

### FraudBoostClassifier

```python
FraudBoostClassifier(
    n_estimators=100,     # Number of boosting rounds
    max_depth=4,          # Maximum tree depth  
    learning_rate=0.1,    # Learning rate (eta)
    fp_cost=100,          # Cost per false positive investigation
    loss='value_weighted', # Loss function type
    backend='auto',       # 'rust', 'python', or 'auto'
    random_state=None     # Random seed
)
```

**Key Methods:**
- `fit(X, y, amounts=None)` - Train model with optional transaction amounts
- `predict(X)` - Binary predictions using threshold 0.5  
- `predict_proba(X)` - Probability estimates
- `feature_importances_` - Gain-based feature importance

### FraudCascade

```python
from fraudboost import FraudCascade

cascade = FraudCascade(
    stage1_model=xgb_model,    # Fast first stage (XGBoost)  
    stage2_model=fb_model,     # Precise second stage (FraudBoost)
    stage1_threshold=0.1       # Stage 1 recall threshold
)
```

### FraudStacking  

```python
from fraudboost import FraudStacking

ensemble = FraudStacking(
    base_models=[xgb_model, fb_model],
    meta_model=LogisticRegression(),
    cv_folds=5
)
```

## Installation

### Basic Installation
```bash
pip install fraudboost
```

### Rust Backend (Recommended for Speed)
```bash
# Install Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Build Rust backend  
cd fraudboost/
maturin develop --release -m rust_core/Cargo.toml
```

The Python version works out-of-the-box. The Rust backend provides 10-20x speedup with identical results.

## License

MIT License - see [LICENSE](LICENSE) for details.

---

**FraudBoost: Gradient boosting that speaks money, not just metrics.**