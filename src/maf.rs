use crate::made::{Made, MadeConfig};
use burn::module::Ignored;
use burn::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[derive(Config, Debug)]
pub struct MafConfig {
    pub d_input: usize,
    pub num_flows: usize,
    pub hidden_sizes: Vec<usize>,
    #[config(default = 42)]
    pub seed: u64,
}

impl MafConfig {
    /// Build a MAF normalizing flow (stack of MADEs with alternating permutations).
    pub fn init<B: Backend>(&self, device: &B::Device) -> Maf<B> {
        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut flows = Vec::with_capacity(self.num_flows);
        let mut permutations = Vec::with_capacity(self.num_flows);

        for i in 0..self.num_flows {
            let flow_seed = rng.r#gen::<u64>();
            let made_config =
                MadeConfig::new(self.d_input, self.hidden_sizes.clone()).with_seed(flow_seed);
            flows.push(made_config.init(device));

            if i % 2 == 0 {
                permutations.push(None); // identity
            } else {
                let rev: Vec<usize> = (0..self.d_input).rev().collect();
                permutations.push(Some(rev));
            }
        }

        Maf {
            flows,
            permutations: Ignored(permutations),
            d_input: self.d_input,
        }
    }
}

#[derive(Module, Debug)]
pub struct Maf<B: Backend> {
    pub(crate) flows: Vec<Made<B>>,
    /// Permutation indices per flow step: None = identity, Some = reverse
    pub(crate) permutations: Ignored<Vec<Option<Vec<usize>>>>,
    pub(crate) d_input: usize,
}

impl<B: Backend> Maf<B> {
    /// Apply permutation to tensor columns
    fn permute_cols(&self, x: Tensor<B, 2>, perm: &Option<Vec<usize>>) -> Tensor<B, 2> {
        match perm {
            None => x,
            Some(indices) => {
                let cols: Vec<Tensor<B, 2>> =
                    indices.iter().map(|&i| x.clone().narrow(1, i, 1)).collect();
                Tensor::cat(cols, 1)
            }
        }
    }

    /// Inverse permutation
    fn inv_permute_cols(&self, x: Tensor<B, 2>, perm: &Option<Vec<usize>>) -> Tensor<B, 2> {
        match perm {
            None => x,
            Some(indices) => {
                let mut inv = vec![0usize; indices.len()];
                for (i, &j) in indices.iter().enumerate() {
                    inv[j] = i;
                }
                let cols: Vec<Tensor<B, 2>> =
                    inv.iter().map(|&i| x.clone().narrow(1, i, 1)).collect();
                Tensor::cat(cols, 1)
            }
        }
    }

    /// Forward: x -> z (fast, parallel)
    /// Returns (z, log_det_jacobian \[batch\])
    pub fn forward(&self, x: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 1>) {
        let batch = x.dims()[0];
        let device = x.device();
        let mut z = x;
        let mut total_log_det = Tensor::<B, 1>::zeros([batch], &device);

        for (i, flow) in self.flows.iter().enumerate() {
            // Apply permutation
            z = self.permute_cols(z, &self.permutations.0[i]);

            let (mu, log_sigma) = flow.forward(z.clone());

            // Clamp log_sigma for numerical stability
            let log_sigma = log_sigma.clamp(-5.0, 5.0);

            // z_i = (x_i - mu_i) * exp(-log_sigma_i)
            z = (z - mu) * (-log_sigma.clone()).exp();

            // log_det = -sum(log_sigma) per sample -> [batch, 1] -> squeeze to [batch]
            let log_det: Tensor<B, 1> = log_sigma.sum_dim(1).squeeze();
            total_log_det = total_log_det - log_det;
        }

        (z, total_log_det)
    }

    /// Inverse: z -> x (sequential, for sampling)
    pub fn inverse(&self, z: Tensor<B, 2>) -> Tensor<B, 2> {
        let d = self.d_input;
        let batch = z.dims()[0];
        let device = z.device();
        let mut x = z;

        // Process flows in reverse order
        for i in (0..self.flows.len()).rev() {
            let flow = &self.flows[i];

            // Dimension-by-dimension sequential reconstruction
            let mut out = Tensor::<B, 2>::zeros([batch, d], &device);

            for dim in 0..d {
                let (mu, log_sigma) = flow.forward(out.clone());
                let log_sigma = log_sigma.clamp(-5.0, 5.0);

                // x_dim = z_dim * exp(log_sigma_dim) + mu_dim
                let mu_d = mu.narrow(1, dim, 1);
                let ls_d = log_sigma.narrow(1, dim, 1);
                let x_d = x.clone().narrow(1, dim, 1);
                let out_d = x_d * ls_d.exp() + mu_d;

                // Slice-assign: build new tensor with this column updated
                if dim == 0 {
                    if d > 1 {
                        out = Tensor::cat(vec![out_d, out.narrow(1, 1, d - 1)], 1);
                    } else {
                        out = out_d;
                    }
                } else if dim == d - 1 {
                    out = Tensor::cat(vec![out.narrow(1, 0, dim), out_d], 1);
                } else {
                    out = Tensor::cat(
                        vec![
                            out.clone().narrow(1, 0, dim),
                            out_d,
                            out.narrow(1, dim + 1, d - dim - 1),
                        ],
                        1,
                    );
                }
            }

            x = out;

            // Undo permutation
            x = self.inv_permute_cols(x, &self.permutations.0[i]);
        }

        x
    }

    /// Compute log p(x) = log N(z; 0, I) + log_det_jacobian
    pub fn log_prob(&self, x: Tensor<B, 2>) -> Tensor<B, 1> {
        let (z, log_det) = self.forward(x);
        crate::flow::standard_normal_log_prob(z, log_det)
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
        let model = MafConfig::new(4, 2, vec![16, 16]).init::<B>(&device);
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
        let model = MafConfig::new(4, 2, vec![16, 16]).init::<B>(&device);
        let x = Tensor::<B, 2>::random(
            [8, 4],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let lp = model.log_prob(x);
        assert_eq!(lp.dims(), [8]);
    }
}
