use burn::prelude::*;
use burn::module::Ignored;
use crate::mlp::{Mlp, MlpConfig};
use crate::spline;

// ──────────────────────────── Affine Coupling ────────────────────────────

#[derive(Config, Debug)]
pub struct AffineCouplingConfig {
    pub d_input: usize,
    pub hidden_sizes: Vec<usize>,
    #[config(default = true)]
    pub mask_even: bool,
}

impl AffineCouplingConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> AffineCoupling<B> {
        let d_identity = self.d_input / 2;
        let d_transform = self.d_input - d_identity;

        let conditioner = MlpConfig::new(d_identity, d_transform * 2, self.hidden_sizes.clone())
            .init(device);

        AffineCoupling {
            conditioner,
            d_input: self.d_input,
            d_identity,
            d_transform,
            mask_even: Ignored(self.mask_even),
        }
    }
}

#[derive(Module, Debug)]
pub struct AffineCoupling<B: Backend> {
    pub(crate) conditioner: Mlp<B>,
    pub(crate) d_input: usize,
    pub(crate) d_identity: usize,
    pub(crate) d_transform: usize,
    pub(crate) mask_even: Ignored<bool>,
}

impl<B: Backend> AffineCoupling<B> {
    /// Split input into identity and transform parts based on mask_even.
    fn split(&self, x: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        if *self.mask_even {
            // first half = identity, second half = transform
            let id = x.clone().narrow(1, 0, self.d_identity);
            let tr = x.narrow(1, self.d_identity, self.d_transform);
            (id, tr)
        } else {
            // second half = identity, first half = transform
            let tr = x.clone().narrow(1, 0, self.d_transform);
            let id = x.narrow(1, self.d_transform, self.d_identity);
            (id, tr)
        }
    }

    /// Merge identity and transform parts back together.
    fn merge(&self, id: Tensor<B, 2>, tr: Tensor<B, 2>) -> Tensor<B, 2> {
        if *self.mask_even {
            Tensor::cat(vec![id, tr], 1)
        } else {
            Tensor::cat(vec![tr, id], 1)
        }
    }

    /// Forward: y_tr = x_tr * exp(log_scale) + shift, y_id = x_id
    /// Returns (y, log_det \[batch\])
    pub fn forward(&self, x: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 1>) {
        let (x_id, x_tr) = self.split(x);

        let params = self.conditioner.forward(x_id.clone()); // [B, 2*d_transform]
        let log_scale = params.clone().narrow(1, 0, self.d_transform).clamp(-5.0, 5.0);
        let shift = params.narrow(1, self.d_transform, self.d_transform);

        let y_tr = x_tr * log_scale.clone().exp() + shift;
        let log_det: Tensor<B, 1> = log_scale.sum_dim(1).squeeze();

        (self.merge(x_id, y_tr), log_det)
    }

    /// Inverse: x_tr = (y_tr - shift) * exp(-log_scale)
    pub fn inverse(&self, y: Tensor<B, 2>) -> Tensor<B, 2> {
        let (y_id, y_tr) = self.split(y);

        let params = self.conditioner.forward(y_id.clone());
        let log_scale = params.clone().narrow(1, 0, self.d_transform).clamp(-5.0, 5.0);
        let shift = params.narrow(1, self.d_transform, self.d_transform);

        let x_tr = (y_tr - shift) * (-log_scale).exp();

        self.merge(y_id, x_tr)
    }
}

// ──────────────────────────── Spline Coupling ────────────────────────────

#[derive(Config, Debug)]
pub struct SplineCouplingConfig {
    pub d_input: usize,
    pub hidden_sizes: Vec<usize>,
    #[config(default = 8)]
    pub num_bins: usize,
    #[config(default = 3.0)]
    pub tail_bound: f32,
    #[config(default = true)]
    pub mask_even: bool,
}

impl SplineCouplingConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> SplineCoupling<B> {
        let d_identity = self.d_input / 2;
        let d_transform = self.d_input - d_identity;

        // Conditioner outputs: K widths + K heights + (K-1) derivatives per transformed dim
        let params_per_dim = 3 * self.num_bins - 1;
        let conditioner = MlpConfig::new(d_identity, d_transform * params_per_dim, self.hidden_sizes.clone())
            .init(device);

        SplineCoupling {
            conditioner,
            d_input: self.d_input,
            d_identity,
            d_transform,
            num_bins: self.num_bins,
            tail_bound: Ignored(self.tail_bound),
            mask_even: Ignored(self.mask_even),
        }
    }
}

#[derive(Module, Debug)]
pub struct SplineCoupling<B: Backend> {
    pub(crate) conditioner: Mlp<B>,
    pub(crate) d_input: usize,
    pub(crate) d_identity: usize,
    pub(crate) d_transform: usize,
    pub(crate) num_bins: usize,
    pub(crate) tail_bound: Ignored<f32>,
    pub(crate) mask_even: Ignored<bool>,
}

impl<B: Backend> SplineCoupling<B> {
    fn split(&self, x: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        if *self.mask_even {
            let id = x.clone().narrow(1, 0, self.d_identity);
            let tr = x.narrow(1, self.d_identity, self.d_transform);
            (id, tr)
        } else {
            let tr = x.clone().narrow(1, 0, self.d_transform);
            let id = x.narrow(1, self.d_transform, self.d_identity);
            (id, tr)
        }
    }

    fn merge(&self, id: Tensor<B, 2>, tr: Tensor<B, 2>) -> Tensor<B, 2> {
        if *self.mask_even {
            Tensor::cat(vec![id, tr], 1)
        } else {
            Tensor::cat(vec![tr, id], 1)
        }
    }

    /// Parse conditioner output into spline parameters.
    /// Returns (widths [B, D_tr, K], heights [B, D_tr, K], derivs [B, D_tr, K-1])
    fn parse_params(&self, raw: Tensor<B, 2>) -> (Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 3>) {
        let batch = raw.dims()[0];
        let k = self.num_bins;
        let d_tr = self.d_transform;

        // raw is [B, d_tr * (3K - 1)], reshape to [B, d_tr, 3K-1]
        let raw_3d = raw.reshape([batch, d_tr, 3 * k - 1]);

        let widths = raw_3d.clone().narrow(2, 0, k);
        let heights = raw_3d.clone().narrow(2, k, k);
        let derivs = raw_3d.narrow(2, 2 * k, k - 1);

        (widths, heights, derivs)
    }

    /// Forward pass through spline coupling.
    pub fn forward(&self, x: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 1>) {
        let (x_id, x_tr) = self.split(x);

        let raw = self.conditioner.forward(x_id.clone());
        let (widths, heights, derivs) = self.parse_params(raw);

        let (y_tr, logdet_2d) = spline::rqs_forward(x_tr, widths, heights, derivs, *self.tail_bound);
        let log_det: Tensor<B, 1> = logdet_2d.sum_dim(1).squeeze(); // sum over D_tr

        (self.merge(x_id, y_tr), log_det)
    }

    /// Inverse pass through spline coupling.
    pub fn inverse(&self, y: Tensor<B, 2>) -> Tensor<B, 2> {
        let (y_id, y_tr) = self.split(y);

        let raw = self.conditioner.forward(y_id.clone());
        let (widths, heights, derivs) = self.parse_params(raw);

        let (x_tr, _) = spline::rqs_inverse(y_tr, widths, heights, derivs, *self.tail_bound);

        self.merge(y_id, x_tr)
    }
}
