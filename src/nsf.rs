use crate::actnorm::{ActNorm, ActNormConfig};
use crate::coupling::{SplineCoupling, SplineCouplingConfig};
use crate::flow::standard_normal_log_prob;
use crate::lu_linear::{LULinear, LULinearConfig};
use burn::prelude::*;

#[derive(Config, Debug)]
pub struct NsfConfig {
    pub d_input: usize,
    pub num_layers: usize,
    pub hidden_sizes: Vec<usize>,
    #[config(default = 8)]
    pub num_bins: usize,
    #[config(default = 3.0)]
    pub tail_bound: f32,
}

impl NsfConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Nsf<B> {
        let mut actnorms = Vec::with_capacity(self.num_layers);
        let mut lu_linears = Vec::with_capacity(self.num_layers);
        let mut couplings = Vec::with_capacity(self.num_layers);

        for i in 0..self.num_layers {
            actnorms.push(ActNormConfig::new(self.d_input).init(device));
            lu_linears.push(LULinearConfig::new(self.d_input).init(device));
            couplings.push(
                SplineCouplingConfig::new(self.d_input, self.hidden_sizes.clone())
                    .with_num_bins(self.num_bins)
                    .with_tail_bound(self.tail_bound)
                    .with_mask_even(i % 2 == 0)
                    .init(device),
            );
        }

        Nsf {
            actnorms,
            lu_linears,
            couplings,
            d_input: self.d_input,
        }
    }
}

#[derive(Module, Debug)]
pub struct Nsf<B: Backend> {
    pub(crate) actnorms: Vec<ActNorm<B>>,
    pub(crate) lu_linears: Vec<LULinear<B>>,
    pub(crate) couplings: Vec<SplineCoupling<B>>,
    pub(crate) d_input: usize,
}

impl<B: Backend> Nsf<B> {
    /// Forward: x -> z, returns (z, total_log_det)
    pub fn forward(&self, x: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 1>) {
        let batch = x.dims()[0];
        let device = x.device();
        let mut z = x;
        let mut total_log_det = Tensor::<B, 1>::zeros([batch], &device);

        for i in 0..self.actnorms.len() {
            let (out, ld) = self.actnorms[i].forward(z);
            z = out;
            total_log_det = total_log_det + ld;

            let (out, ld) = self.lu_linears[i].forward(z);
            z = out;
            total_log_det = total_log_det + ld;

            let (out, ld) = self.couplings[i].forward(z);
            z = out;
            total_log_det = total_log_det + ld;
        }

        (z, total_log_det)
    }

    /// Inverse: z -> x
    pub fn inverse(&self, z: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut x = z;
        for i in (0..self.actnorms.len()).rev() {
            x = self.couplings[i].inverse(x);
            x = self.lu_linears[i].inverse(x);
            x = self.actnorms[i].inverse(x);
        }
        x
    }

    /// Compute log p(x) = log N(z; 0, I) + log_det
    pub fn log_prob(&self, x: Tensor<B, 2>) -> Tensor<B, 1> {
        let (z, log_det) = self.forward(x);
        standard_normal_log_prob(z, log_det)
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
        let model = NsfConfig::new(4, 2, vec![16, 16]).init::<B>(&device);
        let x = Tensor::<B, 2>::random(
            [2, 4],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let (z, _) = model.forward(x.clone());
        let x_rec = model.inverse(z);
        let diff: Vec<f32> = (x - x_rec).to_data().to_vec().unwrap();
        let max_diff = diff.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        assert!(max_diff < 1e-3, "max diff: {max_diff}");
    }

    #[test]
    fn log_prob_shape() {
        let device = Default::default();
        let model = NsfConfig::new(4, 2, vec![16, 16]).init::<B>(&device);
        let x = Tensor::<B, 2>::random(
            [8, 4],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let lp = model.log_prob(x);
        assert_eq!(lp.dims(), [8]);
    }
}
