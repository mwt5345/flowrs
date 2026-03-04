use crate::actnorm::{ActNorm, ActNormConfig};
use crate::coupling::{AffineCoupling, AffineCouplingConfig};
use crate::flow::{Flow, standard_normal_log_prob};
use burn::prelude::*;

/// Configuration for a RealNVP flow.
///
/// Alternates activation normalization and affine coupling layers with
/// alternating masks, producing a simple but effective normalizing flow.
#[derive(Config, Debug)]
pub struct RealNvpConfig {
    /// Dimensionality of the input data.
    pub d_input: usize,
    /// Number of coupling layers.
    pub num_layers: usize,
    /// Hidden layer sizes for the conditioner MLPs.
    pub hidden_sizes: Vec<usize>,
}

impl RealNvpConfig {
    /// Build a RealNVP flow.
    #[must_use]
    pub fn init<B: Backend>(&self, device: &B::Device) -> RealNvp<B> {
        let mut actnorms = Vec::with_capacity(self.num_layers);
        let mut couplings = Vec::with_capacity(self.num_layers);

        for i in 0..self.num_layers {
            actnorms.push(ActNormConfig::new(self.d_input).init(device));
            couplings.push(
                AffineCouplingConfig::new(self.d_input, self.hidden_sizes.clone())
                    .with_mask_even(i % 2 == 0)
                    .init(device),
            );
        }

        RealNvp {
            actnorms,
            couplings,
            d_input: self.d_input,
        }
    }
}

/// RealNVP (Real-valued Non-Volume Preserving) normalizing flow.
///
/// A classic normalizing flow architecture that alternates activation
/// normalization and affine coupling layers. Simpler and faster than NSF,
/// making it a good baseline for density estimation.
#[derive(Module, Debug)]
pub struct RealNvp<B: Backend> {
    pub(crate) actnorms: Vec<ActNorm<B>>,
    pub(crate) couplings: Vec<AffineCoupling<B>>,
    pub(crate) d_input: usize,
}

impl<B: Backend> RealNvp<B> {
    /// Forward: x -> z, returns `(z, total_log_det)`.
    #[must_use]
    pub fn forward(&self, x: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 1>) {
        let batch = x.dims()[0];
        let device = x.device();
        let mut z = x;
        let mut total_log_det = Tensor::<B, 1>::zeros([batch], &device);

        for i in 0..self.actnorms.len() {
            let (out, ld) = self.actnorms[i].forward(z);
            z = out;
            total_log_det = total_log_det + ld;

            let (out, ld) = self.couplings[i].forward(z);
            z = out;
            total_log_det = total_log_det + ld;
        }

        (z, total_log_det)
    }

    /// Inverse: z -> x.
    #[must_use]
    pub fn inverse(&self, z: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut x = z;
        for i in (0..self.actnorms.len()).rev() {
            x = self.couplings[i].inverse(x);
            x = self.actnorms[i].inverse(x);
        }
        x
    }

    /// Compute `log p(x) = log N(z; 0, I) + log_det`.
    #[must_use]
    pub fn log_prob(&self, x: Tensor<B, 2>) -> Tensor<B, 1> {
        let (z, log_det) = self.forward(x);
        standard_normal_log_prob(z, log_det)
    }
}

impl<B: Backend> Flow<B> for RealNvp<B> {
    fn forward(&self, x: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 1>) {
        self.forward(x)
    }
    fn inverse(&self, z: Tensor<B, 2>) -> Tensor<B, 2> {
        self.inverse(z)
    }
    fn log_prob(&self, x: Tensor<B, 2>) -> Tensor<B, 1> {
        self.log_prob(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray;

    #[test]
    fn forward_inverse_roundtrip() {
        let device = Default::default();
        let model = RealNvpConfig::new(4, 2, vec![16, 16]).init::<B>(&device);
        let x = Tensor::<B, 2>::random(
            [2, 4],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let (z, _) = model.forward(x.clone());
        let x_rec = model.inverse(z);
        let diff: Vec<f32> = (x - x_rec).to_data().to_vec().unwrap();
        let max_diff = diff.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        assert!(max_diff < 1e-5, "max diff: {max_diff}");
    }

    #[test]
    fn log_prob_shape() {
        let device = Default::default();
        let model = RealNvpConfig::new(4, 2, vec![16, 16]).init::<B>(&device);
        let x = Tensor::<B, 2>::random(
            [8, 4],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let lp = model.log_prob(x);
        assert_eq!(lp.dims(), [8]);
    }

    #[test]
    fn batch_1_no_panic() {
        let device = Default::default();
        let model = RealNvpConfig::new(4, 2, vec![16, 16]).init::<B>(&device);
        let x = Tensor::<B, 2>::random(
            [1, 4],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let (z, log_det) = model.forward(x.clone());
        assert_eq!(z.dims(), [1, 4]);
        assert_eq!(log_det.dims(), [1]);
        let lp = model.log_prob(x);
        assert_eq!(lp.dims(), [1]);
    }
}
