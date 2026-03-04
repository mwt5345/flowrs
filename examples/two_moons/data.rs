use burn::prelude::*;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand_distr::{Distribution, Normal, Uniform};

#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
pub enum ToyDataset {
    TwoMoons,
    ConcentricRings,
}

pub fn generate_dataset(kind: ToyDataset, n: usize, noise: f64, seed: u64) -> Vec<[f32; 2]> {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, noise).unwrap();

    match kind {
        ToyDataset::TwoMoons => {
            let mut points = Vec::with_capacity(n);
            let half = n / 2;
            for i in 0..half {
                let angle = std::f64::consts::PI * (i as f64) / (half as f64);
                let x = angle.cos() + normal.sample(&mut rng);
                let y = angle.sin() + normal.sample(&mut rng);
                points.push([x as f32, y as f32]);
            }
            for i in 0..(n - half) {
                let angle = std::f64::consts::PI * (i as f64) / ((n - half) as f64);
                let x = 1.0 - angle.cos() + normal.sample(&mut rng);
                let y = 1.0 - angle.sin() - 0.5 + normal.sample(&mut rng);
                points.push([x as f32, y as f32]);
            }
            points
        }
        ToyDataset::ConcentricRings => {
            let mut points = Vec::with_capacity(n);
            let half = n / 2;
            let uniform = Uniform::new(0.0, 2.0 * std::f64::consts::PI);
            for _ in 0..half {
                let angle = uniform.sample(&mut rng);
                let r = 1.0 + normal.sample(&mut rng);
                points.push([(r * angle.cos()) as f32, (r * angle.sin()) as f32]);
            }
            for _ in 0..(n - half) {
                let angle = uniform.sample(&mut rng);
                let r = 2.5 + normal.sample(&mut rng);
                points.push([(r * angle.cos()) as f32, (r * angle.sin()) as f32]);
            }
            points
        }
    }
}

pub fn sample_batch<B: Backend>(
    points: &[[f32; 2]],
    batch_size: usize,
    device: &B::Device,
    rng: &mut StdRng,
) -> Tensor<B, 2> {
    let chosen: Vec<[f32; 2]> = points.choose_multiple(rng, batch_size).cloned().collect();
    let flat: Vec<f32> = chosen.iter().flat_map(|p| p.iter().copied()).collect();
    Tensor::from_floats(TensorData::new(flat, [batch_size, 2]), device)
}
