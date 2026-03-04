use burn::prelude::*;
use burn::tensor::activation;

/// Rational Quadratic Spline forward transform.
///
/// # Arguments
/// * `inputs` - [B, D] tensor of input values
/// * `unnorm_widths` - [B, D, K] unnormalized bin widths
/// * `unnorm_heights` - [B, D, K] unnormalized bin heights
/// * `unnorm_derivs` - [B, D, K-1] unnormalized interior derivatives
/// * `tail_bound` - scalar; identity outside [-tail_bound, tail_bound]
///
/// # Returns
/// (outputs [B, D], log_abs_det [B, D])
pub fn rqs_forward<B: Backend>(
    inputs: Tensor<B, 2>,
    unnorm_widths: Tensor<B, 3>,
    unnorm_heights: Tensor<B, 3>,
    unnorm_derivs: Tensor<B, 3>,
    tail_bound: f32,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    rqs_transform(inputs, unnorm_widths, unnorm_heights, unnorm_derivs, tail_bound, false)
}

/// Rational Quadratic Spline inverse transform.
pub fn rqs_inverse<B: Backend>(
    inputs: Tensor<B, 2>,
    unnorm_widths: Tensor<B, 3>,
    unnorm_heights: Tensor<B, 3>,
    unnorm_derivs: Tensor<B, 3>,
    tail_bound: f32,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    rqs_transform(inputs, unnorm_widths, unnorm_heights, unnorm_derivs, tail_bound, true)
}

fn rqs_transform<B: Backend>(
    inputs: Tensor<B, 2>,
    unnorm_widths: Tensor<B, 3>,
    unnorm_heights: Tensor<B, 3>,
    unnorm_derivs: Tensor<B, 3>,
    tail_bound: f32,
    inverse: bool,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let [batch, d, num_bins] = unnorm_widths.dims();
    let device = inputs.device();

    // Normalize widths and heights via softmax, then scale to span [-tail_bound, tail_bound]
    let widths = activation::softmax(unnorm_widths, 2).mul_scalar(2.0 * tail_bound); // [B, D, K]
    let heights = activation::softmax(unnorm_heights, 2).mul_scalar(2.0 * tail_bound); // [B, D, K]

    // Derivatives: softplus to ensure positivity, pad boundaries with 1.0
    let derivs_inner = activation::softplus(unnorm_derivs, 1.0); // [B, D, K-1]

    // Pad derivatives: prepend and append 1.0 -> [B, D, K+1]
    let ones_pad = Tensor::<B, 3>::ones([batch, d, 1], &device);
    let derivs = Tensor::cat(vec![ones_pad.clone(), derivs_inner, ones_pad], 2); // [B, D, K+1]

    // Cumulative widths and heights (bin edges)
    // cumwidths[..., 0] = -tail_bound, cumwidths[..., k] = cumwidths[..., k-1] + widths[..., k-1]
    let mut cumwidths_slices = Vec::with_capacity(num_bins + 1);
    let mut cumheights_slices = Vec::with_capacity(num_bins + 1);

    let start = Tensor::<B, 3>::ones([batch, d, 1], &device).mul_scalar(-tail_bound);
    cumwidths_slices.push(start.clone());
    cumheights_slices.push(start);

    for k in 0..num_bins {
        let prev_cw = cumwidths_slices.last().unwrap().clone();
        let w_k = widths.clone().narrow(2, k, 1);
        cumwidths_slices.push(prev_cw + w_k);

        let prev_ch = cumheights_slices.last().unwrap().clone();
        let h_k = heights.clone().narrow(2, k, 1);
        cumheights_slices.push(prev_ch + h_k);
    }
    let cumwidths = Tensor::cat(cumwidths_slices, 2); // [B, D, K+1]
    let cumheights = Tensor::cat(cumheights_slices, 2); // [B, D, K+1]

    // Determine which bin each input falls into
    // Use the appropriate cumulative values depending on forward vs inverse
    let cum_lookup = if inverse { cumheights.clone() } else { cumwidths.clone() };

    // inputs: [B, D] -> [B, D, 1] for comparison
    let inputs_3d = inputs.clone().unsqueeze_dim(2); // [B, D, 1]

    // inside_mask: points inside [-tail_bound, tail_bound]
    let inside_mask = inputs.clone().greater_equal(
        Tensor::<B, 2>::ones([batch, d], &device).mul_scalar(-tail_bound),
    ).bool_and(
        inputs.clone().lower(
            Tensor::<B, 2>::ones([batch, d], &device).mul_scalar(tail_bound),
        ),
    ); // [B, D] bool

    // Bin indices: count how many bin edges are <= input, subtract 1
    // Compare inputs [B, D, 1] >= cum_lookup [B, D, K+1] -> [B, D, K+1] bool
    let ge_mask = inputs_3d.greater_equal(cum_lookup); // [B, D, K+1]
    let ge_int = Tensor::<B, 3>::zeros([batch, d, num_bins + 1], &device)
        .mask_fill(ge_mask, 1.0); // [B, D, K+1]
    let bin_counts: Tensor<B, 2> = ge_int.sum_dim(2).reshape([batch, d]); // [B, D]

    // bin_idx = clamp(count - 1, 0, K-1)
    let bin_idx_f = (bin_counts - 1.0).clamp(0.0, (num_bins - 1) as f32); // [B, D]
    let bin_idx: Tensor<B, 3, Int> = bin_idx_f.int().unsqueeze_dim(2); // [B, D, 1]

    // Gather per-bin parameters
    let input_cumwidths: Tensor<B, 2> = cumwidths.clone().gather(2, bin_idx.clone()).reshape([batch, d]); // [B, D]
    let bin_idx_p1: Tensor<B, 3, Int> = bin_idx.clone() + 1;
    let input_cumwidths_p1: Tensor<B, 2> = cumwidths.gather(2, bin_idx_p1.clone()).reshape([batch, d]);
    let input_bin_widths = input_cumwidths_p1 - input_cumwidths.clone();

    let input_cumheights: Tensor<B, 2> = cumheights.clone().gather(2, bin_idx.clone()).reshape([batch, d]);
    let input_cumheights_p1: Tensor<B, 2> = cumheights.gather(2, bin_idx_p1.clone()).reshape([batch, d]);
    let input_bin_heights = input_cumheights_p1 - input_cumheights.clone();

    let input_delta = input_bin_heights.clone() / input_bin_widths.clone(); // s_k

    let input_derivs: Tensor<B, 2> = derivs.clone().gather(2, bin_idx.clone()).reshape([batch, d]); // d_k
    let input_derivs_p1: Tensor<B, 2> = derivs.gather(2, bin_idx_p1).reshape([batch, d]); // d_{k+1}

    // Compute the spline
    let (outputs_inside, logdet_inside) = if inverse {
        // Inverse: given y, find x
        let a = input_bin_heights.clone()
            * (input_delta.clone() - input_derivs.clone())
            + (inputs.clone() - input_cumheights.clone()) * (input_derivs_p1.clone() + input_derivs.clone() - input_delta.clone() * 2.0);

        let b = input_bin_heights.clone() * input_derivs.clone()
            - (inputs.clone() - input_cumheights.clone()) * (input_derivs_p1.clone() + input_derivs.clone() - input_delta.clone() * 2.0);

        let c = -input_delta.clone() * (inputs.clone() - input_cumheights.clone());

        let discriminant = (b.clone().powf_scalar(2.0) - a.clone() * c * 4.0).clamp(0.0, f32::MAX);
        let theta = (discriminant.sqrt() - b) / (a * 2.0);
        let theta = theta.clamp(0.0, 1.0);

        let outputs = theta.clone() * input_bin_widths.clone() + input_cumwidths.clone();

        // Compute derivative for log_det (same formula as forward, evaluated at theta)
        let one_m_theta = (-theta.clone()).add_scalar(1.0);
        let denom = input_delta.clone()
            + (input_derivs_p1.clone() + input_derivs.clone() - input_delta.clone() * 2.0)
            * theta.clone() * one_m_theta.clone();

        let numer = input_delta.clone().powf_scalar(2.0)
            * (input_derivs_p1.clone() * theta.clone().powf_scalar(2.0)
                + input_delta.clone() * theta.clone() * one_m_theta.clone() * 2.0
                + input_derivs.clone() * one_m_theta.clone().powf_scalar(2.0));

        let logdet = numer.log() - denom.powf_scalar(2.0).log();
        // Inverse log_det is negative of forward
        (outputs, -logdet)
    } else {
        // Forward: given x, find y
        let theta = (inputs.clone() - input_cumwidths.clone()) / input_bin_widths.clone();
        let theta = theta.clamp(0.0, 1.0);
        let one_m_theta = (-theta.clone()).add_scalar(1.0);

        let numer = input_bin_heights.clone()
            * (input_delta.clone() * theta.clone().powf_scalar(2.0)
                + input_derivs.clone() * theta.clone() * one_m_theta.clone());

        let denom = input_delta.clone()
            + (input_derivs_p1.clone() + input_derivs.clone() - input_delta.clone() * 2.0)
            * theta.clone() * one_m_theta.clone();

        let outputs = input_cumheights.clone() + numer / denom.clone();

        // Log absolute derivative
        let deriv_numer = input_delta.clone().powf_scalar(2.0)
            * (input_derivs_p1.clone() * theta.clone().powf_scalar(2.0)
                + input_delta.clone() * theta.clone() * one_m_theta.clone() * 2.0
                + input_derivs.clone() * one_m_theta.clone().powf_scalar(2.0));

        let logdet = deriv_numer.log() - denom.powf_scalar(2.0).log();

        (outputs, logdet)
    };

    // Apply linear tails: identity + logdet=0 outside [-tail_bound, tail_bound]
    let inside_mask_2d = inside_mask; // [B, D]

    let outputs = inputs.clone().mask_where(inside_mask_2d.clone(), outputs_inside);
    let logdet = Tensor::<B, 2>::zeros([batch, d], &device).mask_where(inside_mask_2d, logdet_inside);

    (outputs, logdet)
}
