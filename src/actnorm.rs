use burn::module::Param;
use burn::prelude::*;

/// Configuration for an activation normalization layer.
#[derive(Config, Debug)]
pub struct ActNormConfig {
    /// Number of features (channels) to normalize.
    pub features: usize,
}

impl ActNormConfig {
    /// Build an activation normalization layer initialized to the identity transform.
    #[must_use]
    pub fn init<B: Backend>(&self, device: &B::Device) -> ActNorm<B> {
        let log_scale = Tensor::zeros([self.features], device);
        let shift = Tensor::zeros([self.features], device);
        ActNorm {
            log_scale: Param::from_tensor(log_scale),
            shift: Param::from_tensor(shift),
            features: self.features,
        }
    }
}

/// Activation normalization layer.
///
/// Learnable per-feature affine transformation: `y = x * exp(log_scale) + shift`.
/// Commonly used in normalizing flows to stabilize training.
#[derive(Module, Debug)]
pub struct ActNorm<B: Backend> {
    pub(crate) log_scale: Param<Tensor<B, 1>>,
    pub(crate) shift: Param<Tensor<B, 1>>,
    pub(crate) features: usize,
}

impl<B: Backend> ActNorm<B> {
    /// Forward: `y = x * exp(log_scale) + shift`.
    ///
    /// Returns `(y, log_det)` where `log_det` has shape `[batch]`.
    #[must_use]
    pub fn forward(&self, x: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 1>) {
        let batch = x.dims()[0];
        let device = x.device();

        let scale = self.log_scale.val().unsqueeze_dim(0); // [1, features]
        let shift = self.shift.val().unsqueeze_dim(0); // [1, features]

        let y = x * scale.clone().exp() + shift;

        let ones = Tensor::<B, 1>::ones([batch], &device);
        let log_det = ones * self.log_scale.val().sum();

        (y, log_det)
    }

    /// Inverse: `x = (y - shift) * exp(-log_scale)`.
    #[must_use]
    pub fn inverse(&self, y: Tensor<B, 2>) -> Tensor<B, 2> {
        let shift = self.shift.val().unsqueeze_dim(0);
        let neg_log_scale = self.log_scale.val().neg().unsqueeze_dim(0);
        (y - shift) * neg_log_scale.exp()
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
        let model = ActNormConfig::new(4).init::<B>(&device);
        let x = Tensor::<B, 2>::random(
            [8, 4],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let (y, _) = model.forward(x.clone());
        let x_rec = model.inverse(y);
        let diff: Vec<f32> = (x - x_rec).to_data().to_vec().unwrap();
        let max_diff = diff.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        assert!(max_diff < 1e-5, "max diff: {max_diff}");
    }

    #[test]
    fn log_det_shape() {
        let device = Default::default();
        let model = ActNormConfig::new(4).init::<B>(&device);
        let x = Tensor::<B, 2>::random(
            [8, 4],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let (_, log_det) = model.forward(x);
        assert_eq!(log_det.dims(), [8]);
    }
}
