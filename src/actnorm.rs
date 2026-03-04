use burn::prelude::*;
use burn::module::Param;

#[derive(Config, Debug)]
pub struct ActNormConfig {
    pub features: usize,
}

impl ActNormConfig {
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

#[derive(Module, Debug)]
pub struct ActNorm<B: Backend> {
    pub(crate) log_scale: Param<Tensor<B, 1>>,
    pub(crate) shift: Param<Tensor<B, 1>>,
    pub(crate) features: usize,
}

impl<B: Backend> ActNorm<B> {
    /// Forward: y = x * exp(log_scale) + shift
    /// Returns (y, log_det) where log_det has shape \[batch\].
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

    /// Inverse: x = (y - shift) * exp(-log_scale)
    pub fn inverse(&self, y: Tensor<B, 2>) -> Tensor<B, 2> {
        let shift = self.shift.val().unsqueeze_dim(0);
        let neg_log_scale = self.log_scale.val().neg().unsqueeze_dim(0);
        (y - shift) * neg_log_scale.exp()
    }
}
