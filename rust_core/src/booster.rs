use crate::tree::{TreeBuilder, TreeNode};
use ndarray::{Array1, ArrayView1, ArrayView2};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

#[derive(Debug, Clone)]
pub struct ValueWeightedLogLoss {
    fp_cost: f64,
    eps: f64,
}

impl ValueWeightedLogLoss {
    pub fn new(fp_cost: f64) -> Self {
        ValueWeightedLogLoss {
            fp_cost,
            eps: 1e-15,
        }
    }

    pub fn compute_weights(&self, y_true: &ArrayView1<f64>, amounts: Option<&ArrayView1<f64>>) -> (Array1<f64>, Array1<f64>) {
        let n = y_true.len();
        
        // Compute median amount for normalization
        let mut positive_amounts: Vec<f64> = if let Some(amounts) = amounts {
            amounts.iter()
                .zip(y_true.iter())
                .filter(|(_, &y)| y > 0.0)
                .map(|(&amt, _)| amt)
                .collect()
        } else {
            y_true.iter()
                .filter(|&&y| y > 0.0)
                .map(|_| 1.0)
                .collect()
        };
        
        positive_amounts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_amount = if positive_amounts.is_empty() {
            1.0
        } else {
            let mid = positive_amounts.len() / 2;
            if positive_amounts.len() % 2 == 0 {
                (positive_amounts[mid - 1] + positive_amounts[mid]) / 2.0
            } else {
                positive_amounts[mid]
            }
        };
        
        let w_fn = if let Some(amounts) = amounts {
            amounts.mapv(|amt| amt / median_amount)
        } else {
            Array1::ones(n)
        };
        let w_fp = Array1::from_elem(n, self.fp_cost / median_amount);
        
        (w_fn, w_fp)
    }

    pub fn sigmoid(&self, x: &ArrayView1<f64>) -> Array1<f64> {
        x.mapv(|xi| {
            if xi >= 0.0 {
                1.0 / (1.0 + (-xi).exp())
            } else {
                xi.exp() / (1.0 + xi.exp())
            }
        })
    }

    pub fn gradients(&self, y_true: &ArrayView1<f64>, logits: &ArrayView1<f64>, amounts: Option<&ArrayView1<f64>>) -> Array1<f64> {
        let probs = self.sigmoid(logits);
        let probs_clipped = probs.mapv(|p| p.clamp(self.eps, 1.0 - self.eps));
        let (w_fn, w_fp) = self.compute_weights(y_true, amounts);
        
        // grad = -y * w_fn * (1-p) + (1-y) * w_fp * p
        let mut grad = Array1::zeros(y_true.len());
        for i in 0..y_true.len() {
            let y = y_true[i];
            let p = probs_clipped[i];
            grad[i] = -y * w_fn[i] * (1.0 - p) + (1.0 - y) * w_fp[i] * p;
        }
        grad
    }

    pub fn hessians(&self, y_true: &ArrayView1<f64>, logits: &ArrayView1<f64>, amounts: Option<&ArrayView1<f64>>) -> Array1<f64> {
        let probs = self.sigmoid(logits);
        let probs_clipped = probs.mapv(|p| p.clamp(self.eps, 1.0 - self.eps));
        let (w_fn, w_fp) = self.compute_weights(y_true, amounts);
        
        // hess = p * (1-p) * (y * w_fn + (1-y) * w_fp)
        let mut hess = Array1::zeros(y_true.len());
        for i in 0..y_true.len() {
            let y = y_true[i];
            let p = probs_clipped[i];
            hess[i] = p * (1.0 - p) * (y * w_fn[i] + (1.0 - y) * w_fp[i]);
            hess[i] = hess[i].max(1e-8); // Ensure positive hessian
        }
        hess
    }
}

pub struct GradientBooster {
    pub n_estimators: usize,
    pub learning_rate: f64,
    pub max_depth: usize,
    pub min_samples_leaf: usize,
    pub min_samples_split: usize,
    pub subsample: f64,
    pub colsample_bytree: f64,
    pub reg_lambda: f64,
    pub fp_cost: f64,
    pub base_score: f64,
    pub random_state: Option<u64>,
    
    // Fitted attributes
    pub estimators: Vec<TreeNode>,
    pub feature_importances: Option<Array1<f64>>,
}

impl GradientBooster {
    pub fn new(
        n_estimators: usize,
        learning_rate: f64,
        max_depth: usize,
        min_samples_leaf: usize,
        min_samples_split: usize,
        subsample: f64,
        colsample_bytree: f64,
        reg_lambda: f64,
        fp_cost: f64,
        random_state: Option<u64>,
    ) -> Self {
        GradientBooster {
            n_estimators,
            learning_rate,
            max_depth,
            min_samples_leaf,
            min_samples_split,
            subsample,
            colsample_bytree,
            reg_lambda,
            fp_cost,
            base_score: 0.0,
            random_state,
            estimators: Vec::new(),
            feature_importances: None,
        }
    }

    pub fn fit(&mut self, x: &ArrayView2<f64>, y: &ArrayView1<f64>, amounts: Option<&ArrayView1<f64>>) {
        let (n_samples, n_features) = x.dim();
        
        // Initialize base score (log-odds)
        let pos_rate = y.sum() / n_samples as f64;
        self.base_score = if pos_rate > 0.0 && pos_rate < 1.0 {
            (pos_rate / (1.0 - pos_rate)).ln()
        } else {
            0.0
        };

        // Initialize predictions with base score
        let mut predictions = Array1::from_elem(n_samples, self.base_score);

        // Initialize random number generator
        let mut rng = if let Some(seed) = self.random_state {
            ChaCha8Rng::seed_from_u64(seed)
        } else {
            ChaCha8Rng::from_entropy()
        };

        // Initialize loss function
        let loss_fn = ValueWeightedLogLoss::new(self.fp_cost);

        // Feature importance accumulator
        let mut feature_importance_sum = Array1::zeros(n_features);

        // Build trees
        for _iteration in 0..self.n_estimators {
            // Compute gradients and hessians
            let gradients = loss_fn.gradients(y, &predictions.view(), amounts);
            let hessians = loss_fn.hessians(y, &predictions.view(), amounts);

            // Sample subsets
            let sample_indices = self.subsample_rows(n_samples, &mut rng);
            let feature_indices = self.subsample_features(n_features, &mut rng);

            // Build tree
            let tree_builder = TreeBuilder::new(
                self.max_depth,
                self.min_samples_leaf,
                self.min_samples_split,
                self.reg_lambda,
            );

            let tree = tree_builder.build_tree(
                x,
                &gradients.view(),
                &hessians.view(),
                &sample_indices,
                &feature_indices,
                0,
            );

            // Update predictions
            for i in 0..n_samples {
                let tree_pred = tree.predict(&x.row(i));
                predictions[i] += self.learning_rate * tree_pred;
            }

            // Update feature importances (simplified - should use actual gains)
            let tree_importances = crate::tree::compute_feature_importances(&tree, n_features);
            feature_importance_sum = &feature_importance_sum + &tree_importances;

            self.estimators.push(tree);
        }

        // Normalize feature importances
        let total: f64 = feature_importance_sum.sum();
        if total > 0.0 {
            self.feature_importances = Some(feature_importance_sum / total);
        } else {
            self.feature_importances = Some(Array1::zeros(n_features));
        }
    }

    pub fn predict_raw(&self, x: &ArrayView2<f64>) -> Array1<f64> {
        let n_samples = x.nrows();
        let mut predictions = Array1::from_elem(n_samples, self.base_score);

        for tree in &self.estimators {
            for i in 0..n_samples {
                let tree_pred = tree.predict(&x.row(i));
                predictions[i] += self.learning_rate * tree_pred;
            }
        }

        predictions
    }

    pub fn predict_proba(&self, x: &ArrayView2<f64>) -> Array1<f64> {
        let logits = self.predict_raw(x);
        let loss_fn = ValueWeightedLogLoss::new(self.fp_cost);
        loss_fn.sigmoid(&logits.view())
    }

    fn subsample_rows(&self, n_samples: usize, rng: &mut ChaCha8Rng) -> Vec<usize> {
        if self.subsample >= 1.0 {
            (0..n_samples).collect()
        } else {
            let n_subsample = (n_samples as f64 * self.subsample) as usize;
            let mut indices: Vec<usize> = (0..n_samples).collect();
            indices.shuffle(rng);
            indices.truncate(n_subsample);
            indices.sort_unstable(); // Keep sorted for better cache locality
            indices
        }
    }

    fn subsample_features(&self, n_features: usize, rng: &mut ChaCha8Rng) -> Vec<usize> {
        if self.colsample_bytree >= 1.0 {
            (0..n_features).collect()
        } else {
            let n_subsample = (n_features as f64 * self.colsample_bytree) as usize;
            let mut indices: Vec<usize> = (0..n_features).collect();
            indices.shuffle(rng);
            indices.truncate(n_subsample);
            indices.sort_unstable();
            indices
        }
    }
}