#![recursion_limit = "8192"]

use burn::backend::{Autodiff, NdArray};
use burn::module::AutodiffModule;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use numpy::{PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::sync::Mutex;

type B = NdArray;
type AB = Autodiff<NdArray>;

// ─── Helpers ────────────────────────────────────────────────────────────────

fn np2burn(arr: &PyReadonlyArray2<'_, f32>, device: &<B as Backend>::Device) -> Tensor<B, 2> {
    let view = arr.as_array();
    let (n, d) = (view.nrows(), view.ncols());
    let data: Vec<f32> = arr.as_slice().unwrap().to_vec();
    Tensor::from_floats(TensorData::new(data, [n, d]), device)
}

fn np2burn_ad(arr: &PyReadonlyArray2<'_, f32>, device: &<AB as Backend>::Device) -> Tensor<AB, 2> {
    let view = arr.as_array();
    let (n, d) = (view.nrows(), view.ncols());
    let data: Vec<f32> = arr.as_slice().unwrap().to_vec();
    Tensor::from_floats(TensorData::new(data, [n, d]), device)
}

fn burn2np_2d<'py>(py: Python<'py>, t: Tensor<B, 2>) -> Bound<'py, PyArray2<f32>> {
    let dims = t.dims();
    let d = dims[1];
    let data: Vec<f32> = t.to_data().to_vec().unwrap();
    let rows: Vec<Vec<f32>> = data.chunks(d).map(|c| c.to_vec()).collect();
    PyArray2::from_vec2(py, &rows).unwrap()
}

fn burn2np_1d<'py>(py: Python<'py>, t: Tensor<B, 1>) -> Bound<'py, PyArray1<f32>> {
    let data: Vec<f32> = t.to_data().to_vec().unwrap();
    PyArray1::from_vec(py, data)
}

// ─── Training history ───────────────────────────────────────────────────────

#[pyclass]
#[derive(Clone)]
struct TrainHistory {
    #[pyo3(get)]
    steps: Vec<usize>,
    #[pyo3(get)]
    train_loss: Vec<f32>,
    #[pyo3(get)]
    val_loss: Vec<f32>,
}

// ─── Batching helper ────────────────────────────────────────────────────────

fn sample_batch_flat(
    data: &[f32],
    d: usize,
    batch_size: usize,
    rng: &mut StdRng,
) -> Vec<f32> {
    let n = data.len() / d;
    let indices: Vec<usize> = (0..n)
        .collect::<Vec<_>>()
        .choose_multiple(rng, batch_size)
        .cloned()
        .collect();
    indices
        .iter()
        .flat_map(|&i| data[i * d..(i + 1) * d].iter().copied())
        .collect()
}

// ─── MAF ────────────────────────────────────────────────────────────────────

#[pyclass]
struct MAF {
    model: Mutex<flowrs::Maf<B>>,
    config: flowrs::MafConfig,
    d_input: usize,
    d_context: Option<usize>,
    device: <B as Backend>::Device,
}

#[pymethods]
impl MAF {
    #[new]
    #[pyo3(signature = (d_input, num_flows, hidden_sizes, d_context=None, seed=42))]
    fn new(
        d_input: usize,
        num_flows: usize,
        hidden_sizes: Vec<usize>,
        d_context: Option<usize>,
        seed: u64,
    ) -> Self {
        let device = Default::default();
        let config = flowrs::MafConfig::new(d_input, num_flows, hidden_sizes)
            .with_seed(seed)
            .with_d_context(d_context);
        let model = config.init::<B>(&device);
        MAF {
            model: Mutex::new(model),
            config,
            d_input,
            d_context,
            device,
        }
    }

    /// log p(x) or log p(x|context). Returns 1D array [N].
    #[pyo3(signature = (x, context=None))]
    fn log_prob<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f32>,
        context: Option<PyReadonlyArray2<'py, f32>>,
    ) -> Bound<'py, PyArray1<f32>> {
        let m = self.model.lock().unwrap();
        let xt = np2burn(&x, &self.device);
        let lp = match context {
            Some(ref ctx) => m.log_prob_conditional(xt, np2burn(ctx, &self.device)),
            None => m.log_prob(xt),
        };
        burn2np_1d(py, lp)
    }

    /// Forward: x -> (z, log_det).
    #[pyo3(signature = (x, context=None))]
    fn forward<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f32>,
        context: Option<PyReadonlyArray2<'py, f32>>,
    ) -> (Bound<'py, PyArray2<f32>>, Bound<'py, PyArray1<f32>>) {
        let m = self.model.lock().unwrap();
        let xt = np2burn(&x, &self.device);
        let (z, ld) = match context {
            Some(ref ctx) => m.forward_conditional(xt, Some(np2burn(ctx, &self.device))),
            None => m.forward(xt),
        };
        (burn2np_2d(py, z), burn2np_1d(py, ld))
    }

    /// Inverse: z -> x.
    #[pyo3(signature = (z, context=None))]
    fn inverse<'py>(
        &self,
        py: Python<'py>,
        z: PyReadonlyArray2<'py, f32>,
        context: Option<PyReadonlyArray2<'py, f32>>,
    ) -> Bound<'py, PyArray2<f32>> {
        let m = self.model.lock().unwrap();
        let zt = np2burn(&z, &self.device);
        let x = match context {
            Some(ref ctx) => m.inverse_conditional(zt, Some(np2burn(ctx, &self.device))),
            None => m.inverse(zt),
        };
        burn2np_2d(py, x)
    }

    /// Sample from the flow. context shape [1, d_context] is broadcast.
    #[pyo3(signature = (num_samples, context=None))]
    fn sample<'py>(
        &self,
        py: Python<'py>,
        num_samples: usize,
        context: Option<PyReadonlyArray2<'py, f32>>,
    ) -> Bound<'py, PyArray2<f32>> {
        let m = self.model.lock().unwrap();
        let z = Tensor::<B, 2>::random(
            [num_samples, self.d_input],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &self.device,
        );
        let ctx = context.map(|c| {
            let ct = np2burn(&c, &self.device);
            let dims = ct.dims();
            if dims[0] == 1 {
                ct.repeat_dim(0, num_samples)
            } else {
                ct
            }
        });
        let x = m.inverse_conditional(z, ctx);
        burn2np_2d(py, x)
    }

    /// Train the flow. Returns TrainHistory.
    #[pyo3(signature = (
        x, y=None, x_val=None, y_val=None,
        num_steps=20000, batch_size=64, lr=3e-4,
        noise_std=0.0, weight_decay=0.0,
        patience=5000, eval_every=500, verbose=true, seed=42
    ))]
    #[allow(clippy::too_many_arguments)]
    fn fit(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f32>,
        y: Option<PyReadonlyArray2<'_, f32>>,
        x_val: Option<PyReadonlyArray2<'_, f32>>,
        y_val: Option<PyReadonlyArray2<'_, f32>>,
        num_steps: usize,
        batch_size: usize,
        lr: f64,
        noise_std: f64,
        weight_decay: f64,
        patience: usize,
        eval_every: usize,
        verbose: bool,
        seed: u64,
    ) -> PyResult<TrainHistory> {
        let x_view = x.as_array();
        let (n, d_x) = (x_view.nrows(), x_view.ncols());
        let x_flat: Vec<f32> = x.as_slice().unwrap().to_vec();

        let y_flat: Option<Vec<f32>> = y.as_ref().map(|a| a.as_slice().unwrap().to_vec());
        let d_y = y.as_ref().map(|a| a.as_array().ncols()).unwrap_or(0);

        let xv_flat: Option<Vec<f32>> = x_val.as_ref().map(|a| a.as_slice().unwrap().to_vec());
        let yv_flat: Option<Vec<f32>> = y_val.as_ref().map(|a| a.as_slice().unwrap().to_vec());
        let n_val = x_val.as_ref().map(|a| a.as_array().nrows()).unwrap_or(0);

        // Initialize autodiff model from current weights
        let mut ad_model = self.config.init::<AB>(&self.device);
        // TODO: ideally we'd copy weights from self.model, but for now re-init is fine
        // since fit() is typically called once right after construction

        let mut rng = StdRng::seed_from_u64(seed);
        let mut optim = if weight_decay > 0.0 {
            AdamConfig::new()
                .with_weight_decay(Some(burn::optim::decay::WeightDecayConfig::new(weight_decay as f32)))
                .init()
        } else {
            AdamConfig::new().init()
        };

        let mut history = TrainHistory {
            steps: vec![],
            train_loss: vec![],
            val_loss: vec![],
        };
        let mut best_val = f32::MAX;
        let mut steps_no_improve = 0usize;

        for step in 0..=num_steps {
            let xb = sample_batch_flat(&x_flat, d_x, batch_size, &mut rng);
            let x_tensor = Tensor::<AB, 2>::from_floats(
                TensorData::new(xb, [batch_size, d_x]),
                &self.device,
            );
            let x_tensor = if noise_std > 0.0 {
                x_tensor
                    + Tensor::<AB, 2>::random(
                        [batch_size, d_x],
                        burn::tensor::Distribution::Normal(0.0, noise_std),
                        &self.device,
                    )
            } else {
                x_tensor
            };

            let loss = if let Some(ref yf) = y_flat {
                let yb = sample_batch_flat(yf, d_y, batch_size, &mut rng);
                let y_tensor = Tensor::<AB, 2>::from_floats(
                    TensorData::new(yb, [batch_size, d_y]),
                    &self.device,
                );
                -ad_model.log_prob_conditional(x_tensor, y_tensor).mean()
            } else {
                -ad_model.log_prob(x_tensor).mean()
            };

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &ad_model);
            let progress = step as f64 / num_steps as f64;
            let eff_lr = lr * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
            ad_model = optim.step(eff_lr, ad_model, grads);

            if step % eval_every == 0 {
                let mv = ad_model.valid();

                // Train NLL
                let txb = sample_batch_flat(&x_flat, d_x, batch_size.min(n), &mut rng);
                let tx = Tensor::<B, 2>::from_floats(
                    TensorData::new(txb, [batch_size.min(n), d_x]),
                    &self.device,
                );
                let train_nll: f32 = if let Some(ref yf) = y_flat {
                    let tyb = sample_batch_flat(yf, d_y, batch_size.min(n), &mut rng);
                    let ty = Tensor::<B, 2>::from_floats(
                        TensorData::new(tyb, [batch_size.min(n), d_y]),
                        &self.device,
                    );
                    (-mv.log_prob_conditional(tx, ty).mean()).into_scalar().elem()
                } else {
                    (-mv.log_prob(tx).mean()).into_scalar().elem()
                };

                // Val NLL
                let val_nll: f32 = if let Some(ref xvf) = xv_flat {
                    let xv = Tensor::<B, 2>::from_floats(
                        TensorData::new(xvf.clone(), [n_val, d_x]),
                        &self.device,
                    );
                    if let Some(ref yvf) = yv_flat {
                        let yv = Tensor::<B, 2>::from_floats(
                            TensorData::new(yvf.clone(), [n_val, d_y]),
                            &self.device,
                        );
                        (-mv.log_prob_conditional(xv, yv).mean()).into_scalar().elem()
                    } else {
                        (-mv.log_prob(xv).mean()).into_scalar().elem()
                    }
                } else {
                    train_nll
                };

                history.steps.push(step);
                history.train_loss.push(train_nll);
                history.val_loss.push(val_nll);

                let improved = if val_nll < best_val {
                    best_val = val_nll;
                    steps_no_improve = 0;
                    " *"
                } else {
                    steps_no_improve += eval_every;
                    ""
                };

                if verbose {
                    println!(
                        "Step {step:>5}/{num_steps}  train: {train_nll:>7.3}  val: {val_nll:>7.3}  best: {best_val:>7.3}  lr: {eff_lr:.2e}{improved}"
                    );
                }

                if patience > 0 && steps_no_improve >= patience {
                    if verbose {
                        println!("Early stopping at step {step}");
                    }
                    break;
                }
            }

            if step % 100 == 0 {
                py.check_signals()?;
            }
        }

        // Store trained inference model
        *self.model.lock().unwrap() = ad_model.valid();

        Ok(history)
    }

    fn __repr__(&self) -> String {
        format!(
            "MAF(d_input={}, d_context={:?})",
            self.d_input, self.d_context
        )
    }
}

// ─── NSF ────────────────────────────────────────────────────────────────────

#[pyclass]
struct NSF {
    model: Mutex<flowrs::Nsf<B>>,
    d_input: usize,
    device: <B as Backend>::Device,
    seed: u64,
    num_layers: usize,
    hidden_sizes: Vec<usize>,
    num_bins: usize,
    tail_bound: f32,
}

#[pymethods]
impl NSF {
    #[new]
    #[pyo3(signature = (d_input, num_layers, hidden_sizes, num_bins=8, tail_bound=3.0, seed=42))]
    fn new(
        d_input: usize,
        num_layers: usize,
        hidden_sizes: Vec<usize>,
        num_bins: usize,
        tail_bound: f32,
        seed: u64,
    ) -> Self {
        let device = Default::default();
        let config = flowrs::NsfConfig::new(d_input, num_layers, hidden_sizes.clone())
            .with_num_bins(num_bins)
            .with_tail_bound(tail_bound);
        let model = config.init::<B>(&device);
        NSF {
            model: Mutex::new(model),
            d_input,
            device,
            seed,
            num_layers,
            hidden_sizes,
            num_bins,
            tail_bound,
        }
    }

    #[pyo3(signature = (x,))]
    fn log_prob<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f32>,
    ) -> Bound<'py, PyArray1<f32>> {
        let m = self.model.lock().unwrap();
        let xt = np2burn(&x, &self.device);
        burn2np_1d(py, m.log_prob(xt))
    }

    #[pyo3(signature = (x,))]
    fn forward<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f32>,
    ) -> (Bound<'py, PyArray2<f32>>, Bound<'py, PyArray1<f32>>) {
        let m = self.model.lock().unwrap();
        let xt = np2burn(&x, &self.device);
        let (z, ld) = m.forward(xt);
        (burn2np_2d(py, z), burn2np_1d(py, ld))
    }

    #[pyo3(signature = (z,))]
    fn inverse<'py>(
        &self,
        py: Python<'py>,
        z: PyReadonlyArray2<'py, f32>,
    ) -> Bound<'py, PyArray2<f32>> {
        let m = self.model.lock().unwrap();
        let zt = np2burn(&z, &self.device);
        burn2np_2d(py, m.inverse(zt))
    }

    #[pyo3(signature = (num_samples,))]
    fn sample<'py>(&self, py: Python<'py>, num_samples: usize) -> Bound<'py, PyArray2<f32>> {
        let m = self.model.lock().unwrap();
        let z = Tensor::<B, 2>::random(
            [num_samples, self.d_input],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &self.device,
        );
        burn2np_2d(py, m.inverse(z))
    }

    /// Train the NSF. Returns TrainHistory.
    #[pyo3(signature = (
        x, x_val=None,
        num_steps=20000, batch_size=512, lr=5e-4,
        noise_std=0.0, weight_decay=0.0,
        patience=5000, eval_every=200, verbose=true, seed=42
    ))]
    #[allow(clippy::too_many_arguments)]
    fn fit(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f32>,
        x_val: Option<PyReadonlyArray2<'_, f32>>,
        num_steps: usize,
        batch_size: usize,
        lr: f64,
        noise_std: f64,
        weight_decay: f64,
        patience: usize,
        eval_every: usize,
        verbose: bool,
        seed: u64,
    ) -> PyResult<TrainHistory> {
        let x_view = x.as_array();
        let (n, d) = (x_view.nrows(), x_view.ncols());
        let x_flat: Vec<f32> = x.as_slice().unwrap().to_vec();

        let xv_flat: Option<Vec<f32>> = x_val.as_ref().map(|a| a.as_slice().unwrap().to_vec());
        let n_val = x_val.as_ref().map(|a| a.as_array().nrows()).unwrap_or(0);

        let config = flowrs::NsfConfig::new(d, self.num_layers, self.hidden_sizes.clone())
            .with_num_bins(self.num_bins)
            .with_tail_bound(self.tail_bound);
        let mut ad_model = config.init::<AB>(&self.device);

        let mut rng = StdRng::seed_from_u64(seed);
        let mut optim = if weight_decay > 0.0 {
            AdamConfig::new()
                .with_weight_decay(Some(burn::optim::decay::WeightDecayConfig::new(weight_decay as f32)))
                .init()
        } else {
            AdamConfig::new().init()
        };

        let mut history = TrainHistory {
            steps: vec![],
            train_loss: vec![],
            val_loss: vec![],
        };
        let mut best_val = f32::MAX;
        let mut steps_no_improve = 0usize;

        for step in 0..=num_steps {
            let xb = sample_batch_flat(&x_flat, d, batch_size, &mut rng);
            let x_tensor = Tensor::<AB, 2>::from_floats(
                TensorData::new(xb, [batch_size, d]),
                &self.device,
            );
            let x_tensor = if noise_std > 0.0 {
                x_tensor
                    + Tensor::<AB, 2>::random(
                        [batch_size, d],
                        burn::tensor::Distribution::Normal(0.0, noise_std),
                        &self.device,
                    )
            } else {
                x_tensor
            };

            let loss = -ad_model.log_prob(x_tensor).mean();
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &ad_model);
            let progress = step as f64 / num_steps as f64;
            let eff_lr = lr * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
            ad_model = optim.step(eff_lr, ad_model, grads);

            if step % eval_every == 0 {
                let mv = ad_model.valid();
                let txb = sample_batch_flat(&x_flat, d, batch_size.min(n), &mut rng);
                let tx = Tensor::<B, 2>::from_floats(
                    TensorData::new(txb, [batch_size.min(n), d]),
                    &self.device,
                );
                let train_nll: f32 = (-mv.log_prob(tx).mean()).into_scalar().elem();

                let val_nll: f32 = if let Some(ref xvf) = xv_flat {
                    let xv = Tensor::<B, 2>::from_floats(
                        TensorData::new(xvf.clone(), [n_val, d]),
                        &self.device,
                    );
                    (-mv.log_prob(xv).mean()).into_scalar().elem()
                } else {
                    train_nll
                };

                history.steps.push(step);
                history.train_loss.push(train_nll);
                history.val_loss.push(val_nll);

                let improved = if val_nll < best_val {
                    best_val = val_nll;
                    steps_no_improve = 0;
                    " *"
                } else {
                    steps_no_improve += eval_every;
                    ""
                };

                if verbose {
                    println!(
                        "Step {step:>5}/{num_steps}  train: {train_nll:>7.3}  val: {val_nll:>7.3}  best: {best_val:>7.3}  lr: {eff_lr:.2e}{improved}"
                    );
                }

                if patience > 0 && steps_no_improve >= patience {
                    if verbose {
                        println!("Early stopping at step {step}");
                    }
                    break;
                }
            }

            if step % 100 == 0 {
                py.check_signals()?;
            }
        }

        *self.model.lock().unwrap() = ad_model.valid();
        Ok(history)
    }

    fn __repr__(&self) -> String {
        format!("NSF(d_input={})", self.d_input)
    }
}

// ─── RealNVP ────────────────────────────────────────────────────────────────

#[pyclass]
struct RealNVP {
    model: Mutex<flowrs::RealNvp<B>>,
    d_input: usize,
    device: <B as Backend>::Device,
}

#[pymethods]
impl RealNVP {
    #[new]
    #[pyo3(signature = (d_input, num_layers, hidden_sizes))]
    fn new(d_input: usize, num_layers: usize, hidden_sizes: Vec<usize>) -> Self {
        let device = Default::default();
        let config = flowrs::RealNvpConfig::new(d_input, num_layers, hidden_sizes);
        let model = config.init::<B>(&device);
        RealNVP {
            model: Mutex::new(model),
            d_input,
            device,
        }
    }

    #[pyo3(signature = (x,))]
    fn log_prob<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f32>,
    ) -> Bound<'py, PyArray1<f32>> {
        let m = self.model.lock().unwrap();
        let xt = np2burn(&x, &self.device);
        burn2np_1d(py, m.log_prob(xt))
    }

    #[pyo3(signature = (num_samples,))]
    fn sample<'py>(&self, py: Python<'py>, num_samples: usize) -> Bound<'py, PyArray2<f32>> {
        let m = self.model.lock().unwrap();
        let z = Tensor::<B, 2>::random(
            [num_samples, self.d_input],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &self.device,
        );
        burn2np_2d(py, m.inverse(z))
    }

    fn __repr__(&self) -> String {
        format!("RealNVP(d_input={})", self.d_input)
    }
}

// ─── Module ─────────────────────────────────────────────────────────────────

#[pymodule]
fn pyflowrs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<MAF>()?;
    m.add_class::<NSF>()?;
    m.add_class::<RealNVP>()?;
    m.add_class::<TrainHistory>()?;
    Ok(())
}
