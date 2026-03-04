use burn::module::Param;
use burn::prelude::*;

#[derive(Config, Debug)]
pub struct LULinearConfig {
    pub features: usize,
}

impl LULinearConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> LULinear<B> {
        let d = self.features;

        // Number of strictly lower/upper triangular entries: d*(d-1)/2
        let num_off_diag = d * (d - 1) / 2;

        // Identity init: off-diagonal entries = 0
        let lower_entries = Tensor::zeros([num_off_diag], device);
        let upper_entries = Tensor::zeros([num_off_diag], device);

        // log_upper_diag: softplus_inv(1.0) = ln(e^1 - 1) ≈ 0.5413
        let sp_inv_one = (1.0_f32.exp() - 1.0).ln();
        let log_upper_diag = Tensor::zeros([d], device).add_scalar(sp_inv_one);

        let bias = Tensor::zeros([d], device);

        LULinear {
            lower_entries: Param::from_tensor(lower_entries),
            upper_entries: Param::from_tensor(upper_entries),
            log_upper_diag: Param::from_tensor(log_upper_diag),
            bias: Param::from_tensor(bias),
            features: d,
        }
    }
}

#[derive(Module, Debug)]
pub struct LULinear<B: Backend> {
    pub(crate) lower_entries: Param<Tensor<B, 1>>,
    pub(crate) upper_entries: Param<Tensor<B, 1>>,
    pub(crate) log_upper_diag: Param<Tensor<B, 1>>,
    pub(crate) bias: Param<Tensor<B, 1>>,
    pub(crate) features: usize,
}

impl<B: Backend> LULinear<B> {
    /// Assemble the weight matrix W = L @ U where L is unit lower triangular
    /// and U is upper triangular with positive diagonal (via softplus).
    fn assemble_w(&self) -> Tensor<B, 2> {
        let d = self.features;
        let device = self.lower_entries.val().device();

        // Build L: unit lower triangular (1s on diagonal, learned below)
        let mut l_data = vec![0.0f32; d * d];
        for i in 0..d {
            l_data[i * d + i] = 1.0; // diagonal = 1
        }
        let mut l_mat = Tensor::<B, 2>::from_floats(TensorData::new(l_data, [d, d]), &device);

        // Fill in lower triangular entries
        let mut idx = 0;
        for i in 1..d {
            for j in 0..i {
                // We need to add the learned value at position (i, j)
                let mut mask_data = vec![0.0f32; d * d];
                mask_data[i * d + j] = 1.0;
                let mask = Tensor::<B, 2>::from_floats(TensorData::new(mask_data, [d, d]), &device);
                // Extract the scalar and multiply by mask
                let val = self.lower_entries.val().narrow(0, idx, 1);
                let val_2d: Tensor<B, 2> = val.unsqueeze_dim(0); // [1, 1]
                l_mat = l_mat + mask * val_2d;
                idx += 1;
            }
        }

        // Build U: upper triangular with softplus diagonal
        let diag = burn::tensor::activation::softplus(self.log_upper_diag.val(), 1.0);
        let mut u_mat = Tensor::<B, 2>::zeros([d, d], &device);

        // Set diagonal
        for i in 0..d {
            let mut mask_data = vec![0.0f32; d * d];
            mask_data[i * d + i] = 1.0;
            let mask = Tensor::<B, 2>::from_floats(TensorData::new(mask_data, [d, d]), &device);
            let val = diag.clone().narrow(0, i, 1).unsqueeze_dim(0);
            u_mat = u_mat + mask * val;
        }

        // Fill upper triangular entries
        idx = 0;
        for i in 0..d {
            for j in (i + 1)..d {
                let mut mask_data = vec![0.0f32; d * d];
                mask_data[i * d + j] = 1.0;
                let mask = Tensor::<B, 2>::from_floats(TensorData::new(mask_data, [d, d]), &device);
                let val = self.upper_entries.val().narrow(0, idx, 1).unsqueeze_dim(0);
                u_mat = u_mat + mask * val;
                idx += 1;
            }
        }

        l_mat.matmul(u_mat)
    }

    /// Forward: y = x @ W^T + bias
    /// Returns (y, log_det) where log_det = sum(log(softplus(log_upper_diag)))
    pub fn forward(&self, x: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 1>) {
        let batch = x.dims()[0];
        let device = x.device();

        let w = self.assemble_w(); // [d, d]
        let y = x.matmul(w.transpose()) + self.bias.val().unsqueeze_dim(0);

        // log_det = sum(log(softplus(log_upper_diag)))
        let diag = burn::tensor::activation::softplus(self.log_upper_diag.val(), 1.0);
        let log_det_scalar = diag.log().sum();
        let log_det = Tensor::<B, 1>::ones([batch], &device) * log_det_scalar;

        (y, log_det)
    }

    /// Inverse: x = (y - bias) @ W^{-T}
    /// Computes W then inverts via Gauss-Jordan (suitable for small D).
    pub fn inverse(&self, y: Tensor<B, 2>) -> Tensor<B, 2> {
        let d = self.features;
        let device = y.device();

        let w = self.assemble_w();
        let w_data = w.to_data();
        let w_vec: Vec<f32> = w_data.to_vec().unwrap();

        // Gauss-Jordan inversion in f32
        let mut aug = vec![0.0f32; d * 2 * d];
        for i in 0..d {
            for j in 0..d {
                aug[i * 2 * d + j] = w_vec[i * d + j];
            }
            aug[i * 2 * d + d + i] = 1.0;
        }

        for col in 0..d {
            // Find pivot
            let mut pivot_row = col;
            let mut pivot_val = aug[col * 2 * d + col].abs();
            for row in (col + 1)..d {
                let val = aug[row * 2 * d + col].abs();
                if val > pivot_val {
                    pivot_row = row;
                    pivot_val = val;
                }
            }
            // Swap rows
            if pivot_row != col {
                for k in 0..(2 * d) {
                    aug.swap(col * 2 * d + k, pivot_row * 2 * d + k);
                }
            }
            // Scale pivot row
            let pv = aug[col * 2 * d + col];
            for k in 0..(2 * d) {
                aug[col * 2 * d + k] /= pv;
            }
            // Eliminate
            for row in 0..d {
                if row == col {
                    continue;
                }
                let factor = aug[row * 2 * d + col];
                for k in 0..(2 * d) {
                    aug[row * 2 * d + k] -= factor * aug[col * 2 * d + k];
                }
            }
        }

        // Extract inverse
        let mut inv_data = vec![0.0f32; d * d];
        for i in 0..d {
            for j in 0..d {
                inv_data[i * d + j] = aug[i * 2 * d + d + j];
            }
        }
        let w_inv = Tensor::<B, 2>::from_floats(TensorData::new(inv_data, [d, d]), &device);

        // x = (y - bias) @ W^{-T}
        let x = y - self.bias.val().unsqueeze_dim(0);
        x.matmul(w_inv.transpose())
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
        let model = LULinearConfig::new(4).init::<B>(&device);
        let x = Tensor::<B, 2>::random(
            [8, 4],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let (y, _) = model.forward(x.clone());
        let x_rec = model.inverse(y);
        let diff: Vec<f32> = (x - x_rec).to_data().to_vec().unwrap();
        let max_diff = diff.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        assert!(max_diff < 1e-4, "max diff: {max_diff}");
    }

    #[test]
    fn log_det_shape() {
        let device = Default::default();
        let model = LULinearConfig::new(4).init::<B>(&device);
        let x = Tensor::<B, 2>::random(
            [8, 4],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let (_, log_det) = model.forward(x);
        assert_eq!(log_det.dims(), [8]);
    }

    #[test]
    fn w_is_invertible() {
        let device = Default::default();
        let model = LULinearConfig::new(4).init::<B>(&device);
        let w = model.assemble_w();
        let w_data: Vec<f32> = w.to_data().to_vec().unwrap();
        let d = 4;
        for i in 0..d {
            assert!(
                w_data[i * d + i].abs() > 1e-6,
                "diagonal element {i} is too small"
            );
        }
    }
}
