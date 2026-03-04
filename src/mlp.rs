use burn::prelude::*;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation;

#[derive(Config, Debug)]
pub(crate) struct MlpConfig {
    pub d_input: usize,
    pub d_output: usize,
    pub hidden_sizes: Vec<usize>,
}

impl MlpConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Mlp<B> {
        let mut layers = Vec::new();
        let mut d_in = self.d_input;

        for &h in &self.hidden_sizes {
            layers.push(LinearConfig::new(d_in, h).init(device));
            d_in = h;
        }
        layers.push(LinearConfig::new(d_in, self.d_output).init(device));

        Mlp { layers }
    }
}

#[derive(Module, Debug)]
pub(crate) struct Mlp<B: Backend> {
    pub(crate) layers: Vec<Linear<B>>,
}

impl<B: Backend> Mlp<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let num_layers = self.layers.len();
        let mut h = x;
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(h);
            if i < num_layers - 1 {
                h = activation::relu(h);
            }
        }
        h
    }
}
