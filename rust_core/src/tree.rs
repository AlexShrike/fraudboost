use ndarray::{Array1, ArrayView1, ArrayView2};
use rayon::prelude::*;

#[derive(Debug, Clone)]
pub struct SplitInfo {
    pub feature: usize,
    pub threshold: f64,
    pub gain: f64,
    pub n_samples_left: usize,
    pub n_samples_right: usize,
}

#[derive(Debug, Clone)]
pub struct TreeNode {
    pub is_leaf: bool,
    pub value: f64,
    pub feature: Option<usize>,
    pub threshold: Option<f64>,
    pub left: Option<Box<TreeNode>>,
    pub right: Option<Box<TreeNode>>,
    pub n_samples: usize,
}

impl TreeNode {
    pub fn new_leaf(value: f64, n_samples: usize) -> Self {
        TreeNode {
            is_leaf: true,
            value,
            feature: None,
            threshold: None,
            left: None,
            right: None,
            n_samples,
        }
    }

    pub fn new_internal(
        feature: usize,
        threshold: f64,
        left: TreeNode,
        right: TreeNode,
        n_samples: usize,
    ) -> Self {
        TreeNode {
            is_leaf: false,
            value: 0.0, // Not used for internal nodes
            feature: Some(feature),
            threshold: Some(threshold),
            left: Some(Box::new(left)),
            right: Some(Box::new(right)),
            n_samples,
        }
    }

    pub fn predict(&self, x: &ArrayView1<f64>) -> f64 {
        if self.is_leaf {
            return self.value;
        }

        let feature_idx = self.feature.unwrap();
        let threshold = self.threshold.unwrap();
        let feature_value = x[feature_idx];

        // Handle missing values (NaN) by going to the larger child
        if feature_value.is_nan() {
            let left_samples = self.left.as_ref().unwrap().n_samples;
            let right_samples = self.right.as_ref().unwrap().n_samples;
            if left_samples >= right_samples {
                return self.left.as_ref().unwrap().predict(x);
            } else {
                return self.right.as_ref().unwrap().predict(x);
            }
        }

        if feature_value <= threshold {
            self.left.as_ref().unwrap().predict(x)
        } else {
            self.right.as_ref().unwrap().predict(x)
        }
    }
}

pub struct TreeBuilder {
    max_depth: usize,
    min_samples_leaf: usize,
    min_samples_split: usize,
    reg_lambda: f64,
    min_gain: f64,
}

impl TreeBuilder {
    pub fn new(
        max_depth: usize,
        min_samples_leaf: usize,
        min_samples_split: usize,
        reg_lambda: f64,
    ) -> Self {
        TreeBuilder {
            max_depth,
            min_samples_leaf,
            min_samples_split,
            reg_lambda,
            min_gain: 1e-7,
        }
    }

    pub fn build_tree(
        &self,
        x: &ArrayView2<f64>,
        gradients: &ArrayView1<f64>,
        hessians: &ArrayView1<f64>,
        sample_indices: &[usize],
        feature_indices: &[usize],
        depth: usize,
    ) -> TreeNode {
        let n_samples = sample_indices.len();

        // Compute leaf value using Newton-Raphson step
        let grad_sum: f64 = sample_indices.iter().map(|&i| gradients[i]).sum();
        let hess_sum: f64 = sample_indices.iter().map(|&i| hessians[i]).sum();
        let leaf_value = -grad_sum / (hess_sum + self.reg_lambda);

        // Check stopping conditions
        if depth >= self.max_depth
            || n_samples < self.min_samples_split
            || n_samples <= self.min_samples_leaf
        {
            return TreeNode::new_leaf(leaf_value, n_samples);
        }

        // Find the best split
        let best_split = self.find_best_split(
            x,
            gradients,
            hessians,
            sample_indices,
            feature_indices,
            grad_sum,
            hess_sum,
        );

        if let Some(split) = best_split {
            if split.gain >= self.min_gain
                && split.n_samples_left >= self.min_samples_leaf
                && split.n_samples_right >= self.min_samples_leaf
            {
                // Create the split
                let (left_indices, right_indices) =
                    self.split_samples(x, sample_indices, split.feature, split.threshold);

                let left_child = self.build_tree(
                    x,
                    gradients,
                    hessians,
                    &left_indices,
                    feature_indices,
                    depth + 1,
                );

                let right_child = self.build_tree(
                    x,
                    gradients,
                    hessians,
                    &right_indices,
                    feature_indices,
                    depth + 1,
                );

                return TreeNode::new_internal(
                    split.feature,
                    split.threshold,
                    left_child,
                    right_child,
                    n_samples,
                );
            }
        }

        // No valid split found, return leaf
        TreeNode::new_leaf(leaf_value, n_samples)
    }

    fn find_best_split(
        &self,
        x: &ArrayView2<f64>,
        gradients: &ArrayView1<f64>,
        hessians: &ArrayView1<f64>,
        sample_indices: &[usize],
        feature_indices: &[usize],
        total_grad: f64,
        total_hess: f64,
    ) -> Option<SplitInfo> {
        // Use rayon for parallel feature processing
        let splits: Vec<Option<SplitInfo>> = feature_indices
            .par_iter()
            .map(|&feature_idx| {
                self.find_best_split_for_feature(
                    x,
                    gradients,
                    hessians,
                    sample_indices,
                    feature_idx,
                    total_grad,
                    total_hess,
                )
            })
            .collect();

        // Find the best split across all features
        splits
            .into_iter()
            .filter_map(|s| s)
            .max_by(|a, b| a.gain.partial_cmp(&b.gain).unwrap_or(std::cmp::Ordering::Equal))
    }

    fn find_best_split_for_feature(
        &self,
        x: &ArrayView2<f64>,
        gradients: &ArrayView1<f64>,
        hessians: &ArrayView1<f64>,
        sample_indices: &[usize],
        feature_idx: usize,
        total_grad: f64,
        total_hess: f64,
    ) -> Option<SplitInfo> {
        // Collect feature values with their corresponding gradient/hessian
        let mut samples: Vec<(f64, f64, f64, usize)> = sample_indices
            .iter()
            .map(|&i| (x[[i, feature_idx]], gradients[i], hessians[i], i))
            .filter(|(val, _, _, _)| !val.is_nan()) // Filter out NaN values
            .collect();

        if samples.len() < 2 * self.min_samples_leaf {
            return None;
        }

        // Sort by feature value
        samples.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut best_split: Option<SplitInfo> = None;
        let mut best_gain = f64::NEG_INFINITY;

        let mut left_grad = 0.0;
        let mut left_hess = 0.0;

        // Try all possible thresholds (between unique values)
        for i in 0..samples.len() - 1 {
            left_grad += samples[i].1;
            left_hess += samples[i].2;

            let right_grad = total_grad - left_grad;
            let right_hess = total_hess - left_hess;

            // Only consider splits that satisfy min_samples_leaf
            let left_count = i + 1;
            let right_count = samples.len() - left_count;

            if left_count < self.min_samples_leaf || right_count < self.min_samples_leaf {
                continue;
            }

            // Use threshold between consecutive different values
            if samples[i].0 >= samples[i + 1].0 - f64::EPSILON {
                continue; // Skip if values are the same
            }

            let threshold = (samples[i].0 + samples[i + 1].0) / 2.0;

            // Compute gain using gradient boosting criterion
            let gain = self.compute_split_gain(left_grad, left_hess, right_grad, right_hess, total_grad, total_hess);

            if gain > best_gain {
                best_gain = gain;
                best_split = Some(SplitInfo {
                    feature: feature_idx,
                    threshold,
                    gain,
                    n_samples_left: left_count,
                    n_samples_right: right_count,
                });
            }
        }

        best_split
    }

    fn compute_split_gain(
        &self,
        left_grad: f64,
        left_hess: f64,
        right_grad: f64,
        right_hess: f64,
        total_grad: f64,
        total_hess: f64,
    ) -> f64 {
        // XGBoost/LightGBM gain formula:
        // Gain = 0.5 * [L²/(H+λ) + R²/(H+λ) - (L+R)²/(H+λ)]
        let left_score = (left_grad * left_grad) / (left_hess + self.reg_lambda);
        let right_score = (right_grad * right_grad) / (right_hess + self.reg_lambda);
        let parent_score = (total_grad * total_grad) / (total_hess + self.reg_lambda);

        0.5 * (left_score + right_score - parent_score)
    }

    fn split_samples(
        &self,
        x: &ArrayView2<f64>,
        sample_indices: &[usize],
        feature_idx: usize,
        threshold: f64,
    ) -> (Vec<usize>, Vec<usize>) {
        let mut left_indices = Vec::new();
        let mut right_indices = Vec::new();

        for &sample_idx in sample_indices {
            let feature_value = x[[sample_idx, feature_idx]];
            
            if feature_value.is_nan() {
                // For missing values, use the same logic as prediction (go to larger child)
                // This is a heuristic - we'll assign based on which side currently has more samples
                if left_indices.len() >= right_indices.len() {
                    left_indices.push(sample_idx);
                } else {
                    right_indices.push(sample_idx);
                }
            } else if feature_value <= threshold {
                left_indices.push(sample_idx);
            } else {
                right_indices.push(sample_idx);
            }
        }

        (left_indices, right_indices)
    }
}

/// Compute feature importances by summing gains
pub fn compute_feature_importances(tree: &TreeNode, n_features: usize) -> Array1<f64> {
    let mut importances = Array1::zeros(n_features);
    collect_importances(tree, &mut importances);
    
    // Normalize
    let total: f64 = importances.sum();
    if total > 0.0 {
        importances /= total;
    }
    
    importances
}

fn collect_importances(node: &TreeNode, importances: &mut Array1<f64>) {
    if !node.is_leaf {
        if let Some(feature) = node.feature {
            // In a real implementation, you'd store the gain in the node
            // For now, we'll use a simple heuristic based on samples
            importances[feature] += node.n_samples as f64;
            
            if let Some(ref left) = node.left {
                collect_importances(left, importances);
            }
            if let Some(ref right) = node.right {
                collect_importances(right, importances);
            }
        }
    }
}