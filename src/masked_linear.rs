use burn::module::Param;
use burn::prelude::*;

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
        let weight = Tensor::random(
            [d_in, d_out],
            burn::tensor::Distribution::Uniform(-k, k),
            device,
        );
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

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray;

    #[test]
    fn masking_zeros_correct_entries() {
        let device = Default::default();
        let d_in = 3;
        let d_out = 4;
        let mut mask_data = vec![1.0f32; d_in * d_out];
        mask_data[0 * d_out + 1] = 0.0; // row 0, col 1
        mask_data[2 * d_out + 3] = 0.0; // row 2, col 3
        let mask = Tensor::<B, 2>::from_floats(TensorData::new(mask_data, [d_in, d_out]), &device);
        let layer = MaskedLinear::new(d_in, d_out, mask, &device);
        let masked_weight: Vec<f32> = layer
            .weight
            .val()
            .mul(layer.mask.clone())
            .to_data()
            .to_vec()
            .unwrap();
        assert_eq!(masked_weight[0 * d_out + 1], 0.0);
        assert_eq!(masked_weight[2 * d_out + 3], 0.0);
    }
}
