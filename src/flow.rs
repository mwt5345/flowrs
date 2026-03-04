use burn::prelude::*;

/// Compute the log-probability of samples under a standard normal base distribution.
///
/// Uses the change-of-variables formula:
/// `log p(x) = log N(z; 0, I) + log|det(dz/dx)|`
///
/// where `z` has shape `[batch, d]` and `log_det` has shape `[batch]`.
#[must_use]
pub fn standard_normal_log_prob<B: Backend>(
    z: Tensor<B, 2>,
    log_det: Tensor<B, 1>,
) -> Tensor<B, 1> {
    let batch = z.dims()[0];
    let d = z.dims()[1] as f32;
    let log_base = -0.5 * d * (2.0 * std::f32::consts::PI).ln();
    let z_sq_sum: Tensor<B, 1> = z.powf_scalar(2.0).sum_dim(1).reshape([batch]);
    let log_normal = z_sq_sum.mul_scalar(-0.5).add_scalar(log_base);
    log_normal + log_det
}

/// Trait abstracting a normalizing flow model.
///
/// All flow models (`Maf`, `Nsf`, `RealNvp`) implement this trait, enabling
/// generic code over different flow architectures.
pub trait Flow<B: Backend> {
    /// Transform data `x` to latent space `z`, returning `(z, log_det_jacobian)`.
    #[must_use]
    fn forward(&self, x: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 1>);

    /// Transform latent samples `z` back to data space.
    #[must_use]
    fn inverse(&self, z: Tensor<B, 2>) -> Tensor<B, 2>;

    /// Compute `log p(x)` under the flow model.
    #[must_use]
    fn log_prob(&self, x: Tensor<B, 2>) -> Tensor<B, 1>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray;

    #[test]
    fn log_prob_shape() {
        let device = Default::default();
        let z = Tensor::<B, 2>::random(
            [8, 4],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let log_det = Tensor::<B, 1>::zeros([8], &device);
        let result = standard_normal_log_prob(z, log_det);
        assert_eq!(result.dims(), [8]);
    }

    #[test]
    fn log_prob_known_value() {
        let device = Default::default();
        let d = 4usize;
        let z = Tensor::<B, 2>::zeros([2, d], &device);
        let log_det = Tensor::<B, 1>::zeros([2], &device);
        let result = standard_normal_log_prob(z, log_det);
        let expected = -0.5 * (d as f32) * (2.0 * std::f32::consts::PI).ln();
        let val: Vec<f32> = result.to_data().to_vec().unwrap();
        assert!(
            (val[0] - expected).abs() < 1e-4,
            "expected {expected}, got {}",
            val[0]
        );
        assert!(
            (val[1] - expected).abs() < 1e-4,
            "expected {expected}, got {}",
            val[1]
        );
    }

    #[test]
    fn log_prob_batch_1() {
        let device = Default::default();
        let d = 4usize;
        let z = Tensor::<B, 2>::zeros([1, d], &device);
        let log_det = Tensor::<B, 1>::zeros([1], &device);
        let result = standard_normal_log_prob(z, log_det);
        assert_eq!(result.dims(), [1]);
        let expected = -0.5 * (d as f32) * (2.0 * std::f32::consts::PI).ln();
        let val: Vec<f32> = result.to_data().to_vec().unwrap();
        assert!(
            (val[0] - expected).abs() < 1e-4,
            "expected {expected}, got {}",
            val[0]
        );
    }

    #[test]
    fn flow_trait_dispatch() {
        use crate::{MafConfig, NsfConfig, RealNvpConfig};

        let device = Default::default();

        fn check_flow<B: Backend>(flow: &impl Flow<B>, device: &B::Device) {
            let x = Tensor::<B, 2>::random(
                [4, 2],
                burn::tensor::Distribution::Normal(0.0, 1.0),
                device,
            );
            let (z, ld) = flow.forward(x.clone());
            assert_eq!(z.dims(), [4, 2]);
            assert_eq!(ld.dims(), [4]);
            let lp = flow.log_prob(x);
            assert_eq!(lp.dims(), [4]);
            let z = Tensor::<B, 2>::random(
                [4, 2],
                burn::tensor::Distribution::Normal(0.0, 1.0),
                device,
            );
            let x_rec = flow.inverse(z);
            assert_eq!(x_rec.dims(), [4, 2]);
        }

        let maf = MafConfig::new(2, 2, vec![16, 16]).init::<B>(&device);
        check_flow(&maf, &device);

        let nsf = NsfConfig::new(2, 2, vec![16, 16]).init::<B>(&device);
        check_flow(&nsf, &device);

        let nvp = RealNvpConfig::new(2, 2, vec![16, 16]).init::<B>(&device);
        check_flow(&nvp, &device);
    }
}
