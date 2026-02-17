# FraudBoost

**A gradient boosting framework purpose-built for fraud detection in fintech**

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Issues](https://img.shields.io/github/issues/AlexShrike/fraudboost.svg)](https://github.com/AlexShrike/fraudboost/issues)

## What is FraudBoost?

FraudBoost is a novel gradient boosting algorithm specifically designed for fraud detection in financial technology. Unlike traditional gradient boosting methods that treat all errors equally, FraudBoost uses **value-weighted asymmetric loss functions** where:

- **False negatives** are weighted by transaction amount (missing a $10K fraud costs more than missing a $10 fraud)
- **False positives** have a fixed investigation cost (e.g., $100 per case reviewed)
- **Tree splits** are evaluated by net financial savings, not information gain

This results in models that optimize for **business impact** rather than just statistical accuracy.

## Key Features

- **Value-Weighted Loss Functions**: Cost false negatives by transaction amount, false positives by investigation cost
- **Financial Impact Metrics**: Value Detection Rate (VDR), Net Savings, ROI, Cost-Benefit Analysis  
- **Spectral Graph Features**: Extract network patterns from transaction relationships (customer-merchant, customer-device)
- **Pareto-Optimal Thresholds**: Find optimal operating points across competing business objectives
- **Temporal Drift Detection**: Monitor model degradation and trigger retraining alerts
- **End-to-End Pipeline**: Complete workflow from feature extraction to production monitoring

## How FraudBoost Differs from XGBoost

### The Core Problem

Standard gradient boosting (XGBoost, LightGBM) was built for **general-purpose classification**. When applied to fraud detection, it has fundamental blind spots:

1. **All errors are equal.** Missing a $50,000 wire fraud costs the same as missing a $5 card charge. In reality, one costs 10,000x more.
2. **Thresholds are manual.** You train the model, get probabilities, then pick a threshold based on gut feeling or a grid search optimizing the wrong metric.
3. **No business awareness.** The model minimizes log-loss, not dollars lost. These are very different objectives.

### What FraudBoost Changes

FraudBoost modifies gradient boosting at three levels:

#### Level 1: The Loss Function (What the model optimizes)

**Standard XGBoost log-loss:**
```
L = -[y * log(p) + (1-y) * log(1-p)]
```
Every misclassification is weighted equally. A missed $50K fraud and a missed $5 fraud contribute the same gradient signal.

**FraudBoost value-weighted loss:**
```
L = -[y * w_fn(amount) * log(p) + (1-y) * w_fp * log(1-p)]

where:
  w_fn = transaction_amount / median_amount   (proportional to dollars at risk)
  w_fp = fp_cost / median_amount              (fixed investigation cost, e.g. $100)
```

This changes the **gradients that flow into tree building**. Trees are literally shaped differently -- they prioritize getting high-value transactions right.

**Gradients (w.r.t. logits, after chain rule):**
```
dL/df = -y * w_fn * (1-p) + (1-y) * w_fp * p
d2L/df2 = p * (1-p) * (y * w_fn + (1-y) * w_fp)
```

The chain rule application is critical. Gradient boosting operates in logit space (raw scores), not probability space. Computing gradients w.r.t. probabilities instead of logits is a common implementation bug that causes the model to fail silently (predictions collapse to the base rate).

#### Level 2: The Split Criterion (How trees are built)

**XGBoost split gain:**
```
Gain = G_L^2/(H_L + lambda) + G_R^2/(H_R + lambda) - (G_L+G_R)^2/(H_L+H_R + lambda)
```
Where G = sum of gradients, H = sum of hessians. This maximizes statistical purity.

**FraudBoost split gain:**

Same formula, but G and H incorporate value weights. A split that correctly isolates one $10K fraud generates MORE gain than a split that isolates ten $10 frauds. The tree structure adapts to protect high-value transactions.

#### Level 3: Threshold Selection (How predictions become decisions)

**XGBoost:** You pick threshold = 0.5 (default) or do a manual grid search.

**FraudBoost Pareto Optimization:**
1. Sweep thresholds from 0.01 to 0.99
2. At each threshold, compute: precision, recall, VDR, net savings, FP count, ROI
3. Find the Pareto frontier (non-dominated solutions)
4. Recommend the optimal operating point based on business priority

```
Net Savings(t) = sum(amounts[TP at threshold t]) - count(FP at threshold t) * fp_cost
```

The optimal threshold is usually NOT 0.5. On real fraud data, it's often 0.85-0.96 because the cost asymmetry is extreme.

### Side-by-Side Comparison

| Aspect | XGBoost | FraudBoost |
|--------|---------|------------|
| **Training objective** | Minimize log-loss (all errors equal) | Minimize value-weighted loss ($$$ aware) |
| **Tree split criterion** | Statistical information gain | Net financial savings gain |
| **FN treatment** | Same cost as FP | Weighted by transaction amount |
| **FP treatment** | Same cost as FN | Fixed investigation cost ($100 default) |
| **Primary metrics** | AUC, accuracy, F1 | VDR, net savings, ROI |
| **Threshold selection** | Manual (0.5 default) | Pareto-optimal (automated) |
| **Graph features** | Not built in | Spectral extraction integrated |
| **Drift detection** | External tools needed | PSI monitoring built in |
| **Speed** | C++ optimized, very fast | Pure numpy (research prototype) |

### "Can't I Just Use XGBoost with Custom Objectives?"

Yes -- and you should if you need production speed. XGBoost supports custom objective functions. You could implement FraudBoost's value-weighted loss as an XGBoost custom objective:

```python
def fraudboost_objective(y_pred, dtrain):
    y_true = dtrain.get_label()
    amounts = dtrain.get_weight()  # Hack: pass amounts as weights
    p = 1 / (1 + np.exp(-y_pred))
    
    w_fn = amounts / np.median(amounts)
    w_fp = fp_cost / np.median(amounts)
    
    grad = -y_true * w_fn * (1 - p) + (1 - y_true) * w_fp * p
    hess = p * (1 - p) * (y_true * w_fn + (1 - y_true) * w_fp)
    return grad, hess
```

But you'd still need to build the Pareto optimizer, the spectral feature extractor, the drift detector, and the business metrics layer yourself. FraudBoost bundles all of this into a single framework.

## Benchmark Results

### Kartik2112 Credit Card Fraud Dataset (1.85M transactions)

**Setup:**
- Training: 207K transactions (7K fraud + 200K legit, subsampled for 16GB machine)
- Validation: 20% of training set (for early stopping and hyperparameter selection)
- Test: 555K transactions (completely held out, never seen during training)
- Three-way split: train/val/test to prevent any overfitting
- Features: 13 engineered features, zero target leakage
- FP cost: $100 per investigation

**Performance at Default Threshold (0.5):**

| Model | AUC-ROC | Recall | F1 | VDR | Net Savings | FPs | Train Time |
|-------|---------|--------|-----|-----|-------------|-----|------------|
| **XGBoost** | **99.68%** | 96.1% | 0.294 | **99.61%** | $148K | 9,805 | 0.3s |
| **LightGBM** | 99.66% | 95.4% | 0.297 | 98.99% | $164K | 9,577 | 0.5s |
| **Random Forest** | 99.40% | 93.2% | **0.338** | 99.04% | **$355K** | 7,670 | 4.0s |
| **Logistic Regression** | 87.53% | 73.9% | 0.118 | 98.12% | -$1.2M | 23,219 | 0.1s |

**Performance at Optimal Threshold (maximizing net savings):**

| Model | Optimal Threshold | Net Savings | VDR | FPs (estimated) |
|-------|------------------|-------------|-----|-----------------|
| **LightGBM** | 0.960 | **$989K** | 92.6% | ~400 |
| **Random Forest** | 0.870 | $987K | 93.1% | ~400 |
| **XGBoost** | 0.960 | $981K | 91.7% | ~400 |
| **Logistic Regression** | 0.990 | $661K | 79.5% | ~900 |

**Key Insights:**

1. **Default threshold destroys value.** At threshold=0.5, XGBoost generates $148K in savings. At optimal threshold (0.96), it generates $981K -- a **6.6x improvement** just by picking the right threshold. This is exactly the problem FraudBoost's Pareto optimizer solves automatically.

2. **VDR vs TDR disconnect.** At default threshold, VDR is 99.6% but net savings is only $148K because of ~10K false positives costing $1M in investigations. VDR alone is misleading without considering FP costs.

3. **Logistic regression fails catastrophically.** -$1.2M net savings at default threshold. Linear models can't capture the non-linear fraud patterns in this data.

4. **All tree models converge at optimal threshold.** XGBoost, LightGBM, and Random Forest all achieve ~$980K-$989K at their optimal thresholds, suggesting diminishing returns from model complexity on this dataset.

### FraudBoost on Synthetic Data (5K transactions, 1% fraud)

| Metric | Value |
|--------|-------|
| VDR | 95.9% |
| Recall | 92.0% (46/50 frauds) |
| Precision | 46.0% |
| Net Savings | $96K |
| Probability Range | 0.0004 - 0.96 |

FraudBoost correctly learns to separate fraud from legitimate transactions with clean probability calibration.

### Current Limitations

**Speed:** FraudBoost's pure-numpy implementation is significantly slower than XGBoost/LightGBM's C++ engines. On 166K training rows, FraudBoost with 10 trees takes minutes vs XGBoost with 200 trees in 0.3 seconds. This is a research prototype -- production use should either:
- Use the FraudBoost loss function as an XGBoost custom objective (get C++ speed)
- Wait for the planned Rust core rewrite

**Scale:** Tested on up to 207K rows. For million-row datasets, use subsampling or the XGBoost custom objective approach.

### Feature Engineering (No Target Leakage)

All benchmarks use these features with zero target leakage:
- Transaction amount (raw and log-transformed)
- Category (label encoded)
- Gender (binary)
- City population
- Customer and merchant coordinates (lat/long)
- Transaction hour and day of week
- Customer age
- Haversine distance between customer and merchant

**Never used:** Category fraud rate, merchant fraud rate, or any feature derived from labels.

## Architecture Overview

```
Raw Transaction Data
        |
        v
+------------------+    +--------------------+    +------------------+
|    Features      |    |   Relationships    |    |    Amounts       |
|  - Amount        |    |  - Customer-Merch  |    |  - Value weights |
|  - Time/Location |    |  - Customer-Device |    |  - FP cost       |
|  - Demographics  |    |  - IP clusters     |    |                  |
+------------------+    +--------------------+    +------------------+
        |                        |                        |
        v                        v                        v
+------------------+    +--------------------+    +------------------+
| Feature          |    | Spectral Graph     |    | Value Weight     |
| Engineering      |    | Feature Extraction |    | Computation      |
|                  |    | - Laplacian eigen  |    | w_fn = amt/med   |
|                  |    | - 5 freq bands     |    | w_fp = cost/med  |
|                  |    | - Centrality       |    |                  |
+------------------+    +--------------------+    +------------------+
        |                        |                        |
        +----------+-------------+                        |
                   v                                      v
         +------------------+                   +------------------+
         | Combined Feature |                   | Value-Weighted   |
         | Matrix           |                   | Asymmetric Loss  |
         +------------------+                   +------------------+
                   |                                      |
                   v                                      v
         +-----------------------------------------------+
         |          FraudBoost Classifier                 |
         |  - Gradient boosting in logit space            |
         |  - Value-weighted tree splits                  |
         |  - Early stopping on validation set            |
         |  - L2 regularization + subsampling             |
         +-----------------------------------------------+
                   |
                   v
         +------------------+
         | Raw Predictions  |
         | (probabilities)  |
         +------------------+
                   |
                   v
         +------------------+       +------------------+
         | Pareto Threshold |------>| Business Impact  |
         | Optimizer        |       | Report           |
         | - Sweep 0.01-0.99|       | - VDR            |
         | - Net savings    |       | - Net Savings    |
         | - Pareto frontier|       | - ROI            |
         +------------------+       | - Cost-Benefit   |
                                    +------------------+
                   |
                   v
         +------------------+
         | Drift Detector   |
         | - PSI monitoring |
         | - Feature drift  |
         | - Retrain alerts |
         +------------------+
```

## Core Components

### 1. Value-Weighted Loss Functions (`losses.py`)

Three loss functions, all computing gradients w.r.t. logits (not probabilities):

| Loss | Use Case | Key Property |
|------|----------|-------------|
| `ValueWeightedLogLoss` | Default for fraud detection | FN cost proportional to amount |
| `FocalLoss` | When easy examples dominate | Down-weights well-classified samples |
| `LogLoss` | Baseline comparison | Standard symmetric loss |

### 2. Value-Weighted Decision Trees (`tree.py`)

Binary decision trees where splits maximize value-weighted gradient gain. Supports:
- Configurable max depth, min samples per leaf
- Missing value handling (learns default direction)
- L2 regularization on leaf values
- Feature importance tracking (gain-based)

### 3. Spectral Graph Features (`spectral.py`)

Extract network patterns from transaction relationships:
- Bipartite graph construction (customer-merchant-device)
- Laplacian eigendecomposition (sparse, max 200 eigenvalues)
- 5 frequency band energy features
- Degree centrality, local heterophily, PageRank
- Key insight: fraud graphs are heterophilic (fraud energy shifts to high frequencies)

### 4. Pareto Threshold Optimizer (`pareto.py`)

Automated threshold selection that finds non-dominated operating points across:
- Precision vs Recall
- VDR vs FP count  
- Net Savings vs investigation volume
- Supports priorities: `max_vdr`, `max_savings`, `min_fps`, `balanced`

### 5. Temporal Drift Detector (`drift.py`)

Production monitoring:
- Population Stability Index (PSI) between reference and new data
- Per-feature drift detection
- Exponential decay weighting for temporal samples
- Automated retraining recommendations

### 6. Business Metrics (`metrics.py`)

Fraud-specific evaluation beyond AUC/F1:
- **VDR**: What % of fraud dollars did we catch?
- **Net Savings**: Fraud prevented minus investigation costs
- **ROI**: Return on investment for the fraud detection system
- **Classification Report**: Enhanced report with all business metrics

## Installation

```bash
# From source
git clone https://github.com/AlexShrike/fraudboost.git
cd fraudboost
pip install -e .

# Dependencies: numpy, scipy, matplotlib (optional for plots)
```

## Quick Start

```python
import numpy as np
from fraudboost import FraudBoostClassifier
from fraudboost.metrics import value_detection_rate, net_savings
from fraudboost.pareto import ParetoOptimizer

# Load your fraud data
# X = features, y = labels (0/1), amounts = transaction amounts

# Train with value weighting
model = FraudBoostClassifier(
    n_estimators=50,
    max_depth=4,
    learning_rate=0.1,
    fp_cost=100,
    loss='value_weighted'
)
model.fit(X_train, y_train, amounts=amounts_train)

# Get probabilities
probs = model.predict_proba(X_test)[:, 1]

# Find optimal threshold
optimizer = ParetoOptimizer(fp_cost=100)
optimizer.fit(y_test, probs, amounts_test)
rec = optimizer.recommend_threshold('max_savings')
print(f"Optimal threshold: {rec['threshold']:.3f}")
print(f"Expected savings: ${rec['metrics']['net_savings']:,.0f}")

# Evaluate
preds = (probs >= rec['threshold']).astype(int)
print(f"VDR: {value_detection_rate(y_test, preds, amounts_test):.1%}")
print(f"Net Savings: ${net_savings(y_test, preds, amounts_test):,.0f}")
```

## Advanced Usage

### Custom Loss Functions

```python
from fraudboost.losses import BaseLoss

class RegulatoryCostLoss(BaseLoss):
    """Loss function incorporating regulatory fines for missed fraud."""
    
    def __call__(self, y_true, y_pred, amounts=None):
        # Your loss computation
        pass
    
    def gradient(self, y_true, y_pred_proba, amounts=None):
        # Must return gradient w.r.t. LOGITS (not probabilities!)
        p = np.clip(y_pred_proba, 1e-15, 1-1e-15)
        # ... your gradient with chain rule applied
        pass
    
    def hessian(self, y_true, y_pred_proba, amounts=None):
        # Must return hessian w.r.t. LOGITS
        pass

model = FraudBoostClassifier(loss=RegulatoryCostLoss())
```

### Using FraudBoost Loss with XGBoost (Production Speed)

For production deployments where speed matters:

```python
import xgboost as xgb
import numpy as np

def fraudboost_objective(y_pred, dtrain):
    """FraudBoost value-weighted loss as XGBoost custom objective."""
    y_true = dtrain.get_label()
    amounts = dtrain.get_weight()
    p = 1 / (1 + np.exp(-y_pred))
    
    median_amt = np.median(amounts[amounts > 0])
    w_fn = amounts / median_amt
    w_fp = np.full_like(amounts, 100.0 / median_amt)  # $100 FP cost
    
    grad = -y_true * w_fn * (1 - p) + (1 - y_true) * w_fp * p
    hess = p * (1 - p) * (y_true * w_fn + (1 - y_true) * w_fp)
    hess = np.maximum(hess, 1e-8)
    return grad, hess

dtrain = xgb.DMatrix(X_train, label=y_train, weight=amounts_train)
params = {'max_depth': 4, 'learning_rate': 0.1, 'verbosity': 0}
model = xgb.train(params, dtrain, num_boost_round=200, obj=fraudboost_objective)
```

This gives you XGBoost's C++ speed with FraudBoost's value-weighted optimization.

### Temporal Cross-Validation

```python
from sklearn.model_selection import TimeSeriesSplit

# Sort by timestamp -- never train on future data
df_sorted = df.sort_values('timestamp')
tscv = TimeSeriesSplit(n_splits=5)

for train_idx, val_idx in tscv.split(X):
    model.fit(X[train_idx], y[train_idx], amounts=amounts[train_idx])
    preds = model.predict(X[val_idx])
    vdr = value_detection_rate(y[val_idx], preds, amounts[val_idx])
    print(f"Fold VDR: {vdr:.2%}")
```

### End-to-End Pipeline with Graph Features

```python
from fraudboost.pipeline import FraudDetectionPipeline

pipeline = FraudDetectionPipeline(
    use_spectral_features=True,
    optimize_threshold=True,
    enable_drift_detection=True,
    fp_cost=100
)

pipeline.fit(X, y, amounts,
    entity_columns=['customer_id', 'merchant_id', 'device_id'])

evaluation = pipeline.evaluate(X_test, y_test, amounts_test)
```

## Testing

```bash
python -m pytest tests/ -v
```

## Roadmap

- [ ] Rust/C core for production-grade speed
- [ ] XGBoost/LightGBM backend option (use their trees with our loss)
- [ ] Streaming/online learning for real-time fraud
- [ ] SHAP integration for explainability
- [ ] Pre-trained models for common fraud patterns

## Citation

```bibtex
@software{fraudboost2026,
  author = {Gradient Mind},
  title = {FraudBoost: Value-Weighted Gradient Boosting for Fraud Detection},
  url = {https://github.com/AlexShrike/fraudboost},
  year = {2026}
}
```

## License

MIT License. See [LICENSE](LICENSE).

---

**Built by [Gradient Mind](https://gradient-mind.com) for fraud fighters.**
