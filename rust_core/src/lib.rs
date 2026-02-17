pub mod tree;
pub mod booster;

use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use ndarray::{Array1, Array2};
use crate::booster::GradientBooster;

#[pyclass(name = "RustBooster")]
struct PyGradientBooster {
    inner: GradientBooster,
}

#[pymethods]
impl PyGradientBooster {
    #[new]
    #[pyo3(signature = (
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        min_samples_leaf=10,
        min_samples_split=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        fp_cost=100.0,
        random_state=None
    ))]
    fn new(
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
        let booster = GradientBooster::new(
            n_estimators,
            learning_rate,
            max_depth,
            min_samples_leaf,
            min_samples_split,
            subsample,
            colsample_bytree,
            reg_lambda,
            fp_cost,
            random_state,
        );
        
        PyGradientBooster { inner: booster }
    }

    #[pyo3(signature = (x, y, amounts=None))]
    fn fit(
        &mut self,
        x: PyReadonlyArray2<f64>,
        y: PyReadonlyArray1<f64>,
        amounts: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<()> {
        let x_array = x.as_array();
        let y_array = y.as_array();
        let amounts_array = amounts.as_ref().map(|a| a.as_array());
        
        self.inner.fit(&x_array, &y_array, amounts_array.as_ref());
        Ok(())
    }

    fn predict_raw<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<f64>) -> Bound<'py, PyArray1<f64>> {
        let x_array = x.as_array();
        let predictions = self.inner.predict_raw(&x_array);
        PyArray1::from_vec(py, predictions.to_vec())
    }

    fn predict_proba<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<f64>) -> Bound<'py, PyArray1<f64>> {
        let x_array = x.as_array();
        let probabilities = self.inner.predict_proba(&x_array);
        PyArray1::from_vec(py, probabilities.to_vec())
    }

    #[getter]
    fn feature_importances<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner.feature_importances.as_ref().map(|importances| {
            PyArray1::from_vec(py, importances.to_vec())
        })
    }

    #[getter]
    fn base_score(&self) -> f64 {
        self.inner.base_score
    }

    #[getter]
    fn n_estimators_built(&self) -> usize {
        self.inner.estimators.len()
    }
}

/// Fast tree building function for advanced users
#[pyfunction]
#[pyo3(signature = (
    x,
    gradients,
    hessians,
    max_depth=6,
    min_samples_leaf=10,
    min_samples_split=20,
    reg_lambda=1.0
))]
fn build_tree(
    x: PyReadonlyArray2<f64>,
    gradients: PyReadonlyArray1<f64>,
    hessians: PyReadonlyArray1<f64>,
    max_depth: usize,
    min_samples_leaf: usize,
    min_samples_split: usize,
    reg_lambda: f64,
) -> PyResult<f64> {
    let x_array = x.as_array();
    let grad_array = gradients.as_array();
    let hess_array = hessians.as_array();
    
    let n_samples = x_array.nrows();
    let n_features = x_array.ncols();
    
    let tree_builder = crate::tree::TreeBuilder::new(
        max_depth,
        min_samples_leaf,
        min_samples_split,
        reg_lambda,
    );
    
    let sample_indices: Vec<usize> = (0..n_samples).collect();
    let feature_indices: Vec<usize> = (0..n_features).collect();
    
    let tree = tree_builder.build_tree(
        &x_array,
        &grad_array,
        &hess_array,
        &sample_indices,
        &feature_indices,
        0,
    );
    
    // Return the root node value as a simple test
    Ok(tree.value)
}

/// Compute gradients for value-weighted log loss
#[pyfunction]
#[pyo3(signature = (y_true, logits, amounts=None, fp_cost=100.0))]
fn compute_gradients<'py>(
    py: Python<'py>,
    y_true: PyReadonlyArray1<f64>,
    logits: PyReadonlyArray1<f64>,
    amounts: Option<PyReadonlyArray1<f64>>,
    fp_cost: f64,
) -> Bound<'py, PyArray1<f64>> {
    let y_array = y_true.as_array();
    let logits_array = logits.as_array();
    let amounts_array = amounts.as_ref().map(|a| a.as_array());
    
    let loss_fn = crate::booster::ValueWeightedLogLoss::new(fp_cost);
    let gradients = loss_fn.gradients(&y_array, &logits_array, amounts_array.as_ref());
    
    PyArray1::from_vec(py, gradients.to_vec())
}

/// Compute hessians for value-weighted log loss
#[pyfunction]
#[pyo3(signature = (y_true, logits, amounts=None, fp_cost=100.0))]
fn compute_hessians<'py>(
    py: Python<'py>,
    y_true: PyReadonlyArray1<f64>,
    logits: PyReadonlyArray1<f64>,
    amounts: Option<PyReadonlyArray1<f64>>,
    fp_cost: f64,
) -> Bound<'py, PyArray1<f64>> {
    let y_array = y_true.as_array();
    let logits_array = logits.as_array();
    let amounts_array = amounts.as_ref().map(|a| a.as_array());
    
    let loss_fn = crate::booster::ValueWeightedLogLoss::new(fp_cost);
    let hessians = loss_fn.hessians(&y_array, &logits_array, amounts_array.as_ref());
    
    PyArray1::from_vec(py, hessians.to_vec())
}

/// Create a simple benchmark dataset
#[pyfunction]
fn create_fraud_dataset<'py>(
    py: Python<'py>,
    n_samples: usize,
    n_features: usize,
    fraud_rate: f64,
    random_state: Option<u64>,
) -> (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) {
    use rand::prelude::*;
    use rand_chacha::ChaCha8Rng;
    use rand::distributions::Bernoulli;

    let mut rng = if let Some(seed) = random_state {
        ChaCha8Rng::seed_from_u64(seed)
    } else {
        ChaCha8Rng::from_entropy()
    };

    // Generate features
    let mut x_data = Vec::with_capacity(n_samples * n_features);
    for _ in 0..n_samples {
        for _ in 0..n_features {
            x_data.push(rng.gen::<f64>() * 2.0 - 1.0); // [-1, 1]
        }
    }
    
    let mut x = Array2::from_shape_vec((n_samples, n_features), x_data).unwrap();

    // Generate labels with some signal
    let fraud_dist = Bernoulli::new(fraud_rate).unwrap();
    let mut y_data = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let is_fraud = if i < n_features {
            // First few samples are definitely fraud with strong signal
            x[[i, 0]] += 2.0; // Strong positive signal in first feature
            true
        } else {
            fraud_dist.sample(&mut rng)
        };
        y_data.push(if is_fraud { 1.0 } else { 0.0 });
    }

    // Generate amounts (higher for fraud)
    let mut amounts_data = Vec::with_capacity(n_samples);
    for &label in &y_data {
        let base_amount = if label > 0.5 { 500.0 } else { 100.0 };
        let amount = base_amount * (0.5 + rng.gen::<f64>() * 1.5); // [0.5x, 2x]
        amounts_data.push(amount);
    }

    // Convert to numpy arrays
    let py_x = PyArray2::from_vec2(py, &x.outer_iter()
        .map(|row| row.to_vec())
        .collect::<Vec<_>>()).unwrap();
    
    let py_y = PyArray1::from_vec(py, y_data);
    let py_amounts = PyArray1::from_vec(py, amounts_data);

    (py_x, py_y, py_amounts)
}

#[pymodule]
fn fraudboost_rust_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGradientBooster>()?;
    m.add_function(wrap_pyfunction!(build_tree, m)?)?;
    m.add_function(wrap_pyfunction!(compute_gradients, m)?)?;
    m.add_function(wrap_pyfunction!(compute_hessians, m)?)?;
    m.add_function(wrap_pyfunction!(create_fraud_dataset, m)?)?;
    Ok(())
}