use crate::masked_linear::MaskedLinear;
use burn::module::Ignored;
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Configuration for a MADE (Masked Autoencoder for Distribution Estimation) network.
#[derive(Config, Debug)]
pub struct MadeConfig {
    /// Dimensionality of the input.
    pub d_input: usize,
    /// Hidden layer sizes.
    ///
    /// In residual mode, `hidden_sizes[0]` is the width and `hidden_sizes.len()`
    /// is the number of residual blocks (matching nflows' `hidden_features`/`num_blocks`).
    pub hidden_sizes: Vec<usize>,
    /// Random seed for mask generation.
    #[config(default = 42)]
    pub seed: u64,
    /// Optional context dimensionality for conditional flows.
    pub d_context: Option<usize>,
    /// Use residual blocks (matches nflows). Default: true.
    #[config(default = true)]
    pub use_residual_blocks: bool,
}

/// Build a binary mask from degree arrays.
fn build_mask<B: Backend>(
    prev_degrees: &[usize],
    curr_degrees: &[usize],
    strict: bool,
    device: &B::Device,
) -> Tensor<B, 2> {
    let d_in = prev_degrees.len();
    let d_out = curr_degrees.len();
    let mut mask_data = vec![0.0f32; d_in * d_out];
    for r in 0..d_in {
        for c in 0..d_out {
            let cond = if strict {
                prev_degrees[r] < curr_degrees[c]
            } else {
                prev_degrees[r] <= curr_degrees[c]
            };
            if cond {
                mask_data[r * d_out + c] = 1.0;
            }
        }
    }
    Tensor::from_floats(TensorData::new(mask_data, [d_in, d_out]), device)
}

impl MadeConfig {
    /// Build a MADE network with autoregressive masks.
    ///
    /// When `use_residual_blocks` is true, the architecture matches nflows:
    /// `initial_linear → [ResBlock × num_blocks] → final_linear`, where each
    /// residual block has two masked linears with a skip connection and
    /// pre-activation (ReLU before linear).
    ///
    /// When false, uses simple feedforward layers (legacy behavior).
    #[must_use]
    pub fn init<B: Backend>(&self, device: &B::Device) -> Made<B> {
        if self.use_residual_blocks {
            self.init_residual(device)
        } else {
            self.init_feedforward(device)
        }
    }

    /// Residual block architecture matching nflows.
    fn init_residual<B: Backend>(&self, device: &B::Device) -> Made<B> {
        let d = self.d_input;
        let h = self.hidden_sizes[0];
        let num_blocks = self.hidden_sizes.len();
        let mut rng = StdRng::seed_from_u64(self.seed);

        let input_degrees: Vec<usize> = (0..d).collect();
        let hidden_degrees: Vec<usize> = (0..h).map(|_| rng.gen_range(0..d)).collect();
        let output_degrees: Vec<usize> = (0..d).chain(0..d).collect();

        let mut layers = Vec::new();

        // Initial layer: input → hidden (non-strict)
        let mask = build_mask::<B>(&input_degrees, &hidden_degrees, false, device);
        layers.push(MaskedLinear::new(d, h, mask, device));

        // Residual blocks: each has 2 masked linears (hidden → hidden)
        let mask_hh = build_mask::<B>(&hidden_degrees, &hidden_degrees, false, device);
        for _ in 0..num_blocks {
            layers.push(MaskedLinear::new(h, h, mask_hh.clone(), device));
            layers.push(MaskedLinear::new_zero_init(h, h, mask_hh.clone(), device));
        }

        // Final layer: hidden → output (strict)
        let mask = build_mask::<B>(&hidden_degrees, &output_degrees, true, device);
        layers.push(MaskedLinear::new(h, 2 * d, mask, device));

        // Context layers: one for initial, one per residual block
        let context_layers = if let Some(d_ctx) = self.d_context {
            let mut ctx = Vec::new();
            ctx.push(LinearConfig::new(d_ctx, h).init(device));
            for _ in 0..num_blocks {
                ctx.push(LinearConfig::new(d_ctx, h).init(device));
            }
            ctx
        } else {
            Vec::new()
        };

        Made {
            layers,
            context_layers,
            d_input: d,
            use_residual_blocks: Ignored(true),
            num_res_blocks: Ignored(num_blocks),
        }
    }

    /// Simple feedforward architecture (legacy).
    fn init_feedforward<B: Backend>(&self, device: &B::Device) -> Made<B> {
        let d = self.d_input;
        let mut rng = StdRng::seed_from_u64(self.seed);

        let input_degrees: Vec<usize> = (0..d).collect();
        let mut all_hidden_degrees: Vec<Vec<usize>> = Vec::new();
        for &h in &self.hidden_sizes {
            let degrees: Vec<usize> = (0..h).map(|_| rng.gen_range(0..d)).collect();
            all_hidden_degrees.push(degrees);
        }
        let output_degrees: Vec<usize> = (0..d).chain(0..d).collect();

        let mut layers = Vec::new();
        let num_hidden = self.hidden_sizes.len();

        for i in 0..=num_hidden {
            let (prev_degrees, curr_degrees, strict) = if i == 0 {
                (&input_degrees, &all_hidden_degrees[0], false)
            } else if i == num_hidden {
                (&all_hidden_degrees[num_hidden - 1], &output_degrees, true)
            } else {
                (&all_hidden_degrees[i - 1], &all_hidden_degrees[i], false)
            };

            let mask = build_mask::<B>(prev_degrees, curr_degrees, strict, device);
            layers.push(MaskedLinear::new(prev_degrees.len(), curr_degrees.len(), mask, device));
        }

        let context_layers = if let Some(d_ctx) = self.d_context {
            let mut ctx_layers = Vec::new();
            for &h in &self.hidden_sizes {
                ctx_layers.push(LinearConfig::new(d_ctx, h).init(device));
            }
            ctx_layers
        } else {
            Vec::new()
        };

        Made {
            layers,
            context_layers,
            d_input: d,
            use_residual_blocks: Ignored(false),
            num_res_blocks: Ignored(0),
        }
    }
}

/// MADE (Masked Autoencoder for Distribution Estimation).
///
/// An autoregressive network where output dimension `i` depends only on
/// input dimensions `0..i`, enforced by masked weight matrices. Returns
/// `(mu, log_sigma)` for use in autoregressive flows.
///
/// Supports two modes:
/// - **Residual** (default): `initial → [ResBlock × N] → final` with skip connections,
///   matching nflows' architecture for apples-to-apples comparison.
/// - **Feedforward**: plain stacked masked linears with ReLU (legacy).
#[derive(Module, Debug)]
pub struct Made<B: Backend> {
    pub(crate) layers: Vec<MaskedLinear<B>>,
    pub(crate) context_layers: Vec<Linear<B>>,
    pub(crate) d_input: usize,
    pub(crate) use_residual_blocks: Ignored<bool>,
    pub(crate) num_res_blocks: Ignored<usize>,
}

impl<B: Backend> Made<B> {
    /// Forward pass: returns `(mu, log_sigma)` each with shape `[batch, d]`.
    #[must_use]
    pub fn forward(&self, x: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        self.forward_conditional(x, None)
    }

    /// Conditional forward pass: returns `(mu, log_sigma)` each with shape `[batch, d]`.
    ///
    /// When `context` is provided, it is projected and added at each layer/block.
    #[must_use]
    pub fn forward_conditional(
        &self,
        x: Tensor<B, 2>,
        context: Option<Tensor<B, 2>>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        if *self.use_residual_blocks {
            self.forward_residual(x, context)
        } else {
            self.forward_feedforward(x, context)
        }
    }

    /// Residual forward: initial → context → [ResBlock × N] → final
    fn forward_residual(
        &self,
        x: Tensor<B, 2>,
        context: Option<Tensor<B, 2>>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let num_blocks = *self.num_res_blocks;

        // Initial layer
        let mut h = self.layers[0].forward(x);
        if let Some(ref ctx) = context {
            if !self.context_layers.is_empty() {
                h = h + self.context_layers[0].forward(ctx.clone());
            }
        }

        // Residual blocks
        for b in 0..num_blocks {
            let residual = h.clone();
            // Pre-activation
            h = activation::relu(h);
            // First masked linear
            h = self.layers[1 + b * 2].forward(h);
            // Context injection
            if let Some(ref ctx) = context {
                let ctx_idx = 1 + b;
                if ctx_idx < self.context_layers.len() {
                    h = h + self.context_layers[ctx_idx].forward(ctx.clone());
                }
            }
            // Second pre-activation
            h = activation::relu(h);
            // Second masked linear (zero-initialized)
            h = self.layers[2 + b * 2].forward(h);
            // Skip connection
            h = residual + h;
        }

        // Final layer
        h = self.layers[1 + num_blocks * 2].forward(h);

        let d = self.d_input;
        let mu = h.clone().narrow(1, 0, d);
        let log_sigma = h.narrow(1, d, d);
        (mu, log_sigma)
    }

    /// Feedforward forward (legacy).
    fn forward_feedforward(
        &self,
        x: Tensor<B, 2>,
        context: Option<Tensor<B, 2>>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let num_layers = self.layers.len();
        let mut h = x;
        let mut ctx_idx = 0;

        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(h);
            if i < num_layers - 1 {
                if let Some(ref ctx) = context {
                    if ctx_idx < self.context_layers.len() {
                        h = h + self.context_layers[ctx_idx].forward(ctx.clone());
                        ctx_idx += 1;
                    }
                }
                h = activation::relu(h);
            }
        }

        let d = self.d_input;
        let mu = h.clone().narrow(1, 0, d);
        let log_sigma = h.narrow(1, d, d);
        (mu, log_sigma)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray;

    #[test]
    fn output_shapes() {
        let device = Default::default();
        let model = MadeConfig::new(4, vec![16, 16]).init::<B>(&device);
        let x = Tensor::<B, 2>::random(
            [8, 4],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let (mu, log_sigma) = model.forward(x);
        assert_eq!(mu.dims(), [8, 4]);
        assert_eq!(log_sigma.dims(), [8, 4]);
    }

    #[test]
    fn autoregressive_property() {
        let device = Default::default();
        let d = 4usize;
        let model = MadeConfig::new(d, vec![16, 16]).init::<B>(&device);
        check_autoregressive(&model, d, &device);
    }

    #[test]
    fn autoregressive_property_feedforward() {
        let device = Default::default();
        let d = 4usize;
        let model = MadeConfig::new(d, vec![16, 16])
            .with_use_residual_blocks(false)
            .init::<B>(&device);
        check_autoregressive(&model, d, &device);
    }

    #[test]
    fn output_shapes_feedforward() {
        let device = Default::default();
        let model = MadeConfig::new(4, vec![16, 16])
            .with_use_residual_blocks(false)
            .init::<B>(&device);
        let x = Tensor::<B, 2>::random(
            [8, 4],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let (mu, log_sigma) = model.forward(x);
        assert_eq!(mu.dims(), [8, 4]);
        assert_eq!(log_sigma.dims(), [8, 4]);
    }

    fn check_autoregressive(model: &Made<B>, d: usize, device: &<B as Backend>::Device) {
        let x1 = Tensor::<B, 2>::random(
            [2, d],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            device,
        );
        let mut x2_data: Vec<f32> = x1.to_data().to_vec().unwrap();
        for b in 0..2 {
            x2_data[b * d + (d - 1)] += 1.0;
        }
        let x2 = Tensor::<B, 2>::from_floats(TensorData::new(x2_data, [2, d]), device);
        let (mu1, ls1) = model.forward(x1);
        let (mu2, ls2) = model.forward(x2);

        let mu_diff: Vec<f32> = (mu1 - mu2).to_data().to_vec().unwrap();
        let ls_diff: Vec<f32> = (ls1 - ls2).to_data().to_vec().unwrap();
        let max_mu_diff = mu_diff.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let max_ls_diff = ls_diff.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        assert!(
            max_mu_diff < 1e-6,
            "mu changed when modifying x_{{d-1}}: max diff = {max_mu_diff}"
        );
        assert!(
            max_ls_diff < 1e-6,
            "log_sigma changed when modifying x_{{d-1}}: max diff = {max_ls_diff}"
        );
    }
}
