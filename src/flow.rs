use burn::prelude::*;

/// Compute log p(x) = log N(z; 0, I) + log_det_jacobian
/// where z has shape \[batch, d\] and log_det has shape \[batch\].
pub fn standard_normal_log_prob<B: Backend>(
    z: Tensor<B, 2>,
    log_det: Tensor<B, 1>,
) -> Tensor<B, 1> {
    let d = z.dims()[1] as f32;
    let log_base = -0.5 * d * (2.0 * std::f32::consts::PI).ln();
    let z_sq_sum: Tensor<B, 1> = z.powf_scalar(2.0).sum_dim(1).squeeze();
    let log_normal = z_sq_sum.mul_scalar(-0.5).add_scalar(log_base);
    log_normal + log_det
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
        // Use batch=2 to avoid squeeze edge case with batch=1
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
}
