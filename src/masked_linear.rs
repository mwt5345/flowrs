use burn::prelude::*;
use burn::module::Param;

#[derive(Module, Debug)]
pub(crate) struct MaskedLinear<B: Backend> {
    pub(crate) weight: Param<Tensor<B, 2>>,
    pub(crate) bias: Param<Tensor<B, 1>>,
    pub(crate) mask: Tensor<B, 2>,
}

impl<B: Backend> MaskedLinear<B> {
    pub(crate) fn new(d_in: usize, d_out: usize, mask: Tensor<B, 2>, device: &B::Device) -> Self {
        // Kaiming uniform initialization
        let k = 1.0 / (d_in as f64).sqrt();
        let weight = Tensor::random([d_in, d_out], burn::tensor::Distribution::Uniform(-k, k), device);
        let bias = Tensor::zeros([d_out], device);
        Self {
            weight: Param::from_tensor(weight),
            bias: Param::from_tensor(bias),
            mask,
        }
    }

    pub(crate) fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let masked_weight = self.weight.val().mul(self.mask.clone());
        input.matmul(masked_weight) + self.bias.val().unsqueeze_dim(0)
    }
}
