use burn::prelude::*;

/// Compute log p(x) = log N(z; 0, I) + log_det_jacobian
/// where z has shape \[batch, d\] and log_det has shape \[batch\].
pub fn standard_normal_log_prob<B: Backend>(z: Tensor<B, 2>, log_det: Tensor<B, 1>) -> Tensor<B, 1> {
    let d = z.dims()[1] as f32;
    let log_base = -0.5 * d * (2.0 * std::f32::consts::PI).ln();
    let z_sq_sum: Tensor<B, 1> = z.powf_scalar(2.0).sum_dim(1).squeeze();
    let log_normal = z_sq_sum.mul_scalar(-0.5).add_scalar(log_base);
    log_normal + log_det
}
