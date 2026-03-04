use burn::prelude::*;
use burn::tensor::activation;
use crate::masked_linear::MaskedLinear;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[derive(Config, Debug)]
pub struct MadeConfig {
    pub d_input: usize,
    pub hidden_sizes: Vec<usize>,
    #[config(default = 42)]
    pub seed: u64,
}

impl MadeConfig {
    /// Build a MADE network with autoregressive masks.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Made<B> {
        let d = self.d_input;
        let mut rng = StdRng::seed_from_u64(self.seed);

        // Input degrees: 0, 1, ..., d-1
        let input_degrees: Vec<usize> = (0..d).collect();

        // Hidden degrees for each layer
        let mut all_hidden_degrees: Vec<Vec<usize>> = Vec::new();
        for &h in &self.hidden_sizes {
            let degrees: Vec<usize> = (0..h).map(|_| rng.gen_range(0..d)).collect();
            all_hidden_degrees.push(degrees);
        }

        // Output degrees: 0, 1, ..., d-1 for mu, then 0, 1, ..., d-1 for log_sigma
        let output_degrees: Vec<usize> = (0..d).chain(0..d).collect();

        // Build masks and layers
        let mut layers = Vec::new();
        let num_hidden = self.hidden_sizes.len();

        for i in 0..=num_hidden {
            let (prev_degrees, curr_degrees, strict) = if i == 0 {
                (&input_degrees, &all_hidden_degrees[0], false)
            } else if i == num_hidden {
                (&all_hidden_degrees[num_hidden - 1], &output_degrees, true)
            } else {
                (&all_hidden_degrees[i - 1], &all_hidden_degrees[i], false)
            };

            let d_in = prev_degrees.len();
            let d_out = curr_degrees.len();

            // M[i,j] = 1 if prev_degrees[i] <= curr_degrees[j] (or < for strict)
            let mut mask_data = vec![0.0f32; d_in * d_out];
            for r in 0..d_in {
                for c in 0..d_out {
                    let cond = if strict {
                        prev_degrees[r] < curr_degrees[c]
                    } else {
                        prev_degrees[r] <= curr_degrees[c]
                    };
                    if cond {
                        mask_data[r * d_out + c] = 1.0;
                    }
                }
            }

            let mask = Tensor::from_floats(TensorData::new(mask_data, [d_in, d_out]), device);
            layers.push(MaskedLinear::new(d_in, d_out, mask, device));
        }

        Made { layers, d_input: d }
    }
}

#[derive(Module, Debug)]
pub struct Made<B: Backend> {
    pub(crate) layers: Vec<MaskedLinear<B>>,
    pub(crate) d_input: usize,
}

impl<B: Backend> Made<B> {
    /// Forward pass: returns (mu, log_sigma) each [batch, d]
    pub fn forward(&self, x: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let num_layers = self.layers.len();
        let mut h = x;

        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(h);
            // Apply ReLU to all but the last layer
            if i < num_layers - 1 {
                h = activation::relu(h);
            }
        }

        // Split output into mu and log_sigma
        let d = self.d_input;
        let mu = h.clone().narrow(1, 0, d);
        let log_sigma = h.narrow(1, d, d);

        (mu, log_sigma)
    }
}
