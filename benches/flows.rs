use burn::backend::NdArray;
use burn::prelude::*;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use flowrs::{MafConfig, NsfConfig, RealNvpConfig};

type B = NdArray;

const D: usize = 2;
const BATCH: usize = 512;
const HIDDEN: [usize; 2] = [128, 128];

fn bench_forward(c: &mut Criterion) {
    let device = Default::default();
    let mut group = c.benchmark_group("forward");

    let maf = MafConfig::new(D, 8, HIDDEN.to_vec()).init::<B>(&device);
    let nsf = NsfConfig::new(D, 8, HIDDEN.to_vec()).init::<B>(&device);
    let nvp = RealNvpConfig::new(D, 8, HIDDEN.to_vec()).init::<B>(&device);

    group.bench_function("MAF", |b| {
        b.iter(|| {
            let x = Tensor::<B, 2>::random(
                [BATCH, D],
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &device,
            );
            maf.forward(x)
        })
    });
    group.bench_function("NSF", |b| {
        b.iter(|| {
            let x = Tensor::<B, 2>::random(
                [BATCH, D],
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &device,
            );
            nsf.forward(x)
        })
    });
    group.bench_function("RealNVP", |b| {
        b.iter(|| {
            let x = Tensor::<B, 2>::random(
                [BATCH, D],
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &device,
            );
            nvp.forward(x)
        })
    });

    group.finish();
}

fn bench_inverse(c: &mut Criterion) {
    let device = Default::default();
    let mut group = c.benchmark_group("inverse");

    let maf = MafConfig::new(D, 8, HIDDEN.to_vec()).init::<B>(&device);
    let nsf = NsfConfig::new(D, 8, HIDDEN.to_vec()).init::<B>(&device);
    let nvp = RealNvpConfig::new(D, 8, HIDDEN.to_vec()).init::<B>(&device);

    group.bench_function("MAF", |b| {
        b.iter(|| {
            let z = Tensor::<B, 2>::random(
                [BATCH, D],
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &device,
            );
            maf.inverse(z)
        })
    });
    group.bench_function("NSF", |b| {
        b.iter(|| {
            let z = Tensor::<B, 2>::random(
                [BATCH, D],
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &device,
            );
            nsf.inverse(z)
        })
    });
    group.bench_function("RealNVP", |b| {
        b.iter(|| {
            let z = Tensor::<B, 2>::random(
                [BATCH, D],
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &device,
            );
            nvp.inverse(z)
        })
    });

    group.finish();
}

fn bench_log_prob(c: &mut Criterion) {
    let device = Default::default();
    let mut group = c.benchmark_group("log_prob");

    let maf = MafConfig::new(D, 8, HIDDEN.to_vec()).init::<B>(&device);
    let nsf = NsfConfig::new(D, 8, HIDDEN.to_vec()).init::<B>(&device);
    let nvp = RealNvpConfig::new(D, 8, HIDDEN.to_vec()).init::<B>(&device);

    group.bench_function("MAF", |b| {
        b.iter(|| {
            let x = Tensor::<B, 2>::random(
                [BATCH, D],
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &device,
            );
            maf.log_prob(x)
        })
    });
    group.bench_function("NSF", |b| {
        b.iter(|| {
            let x = Tensor::<B, 2>::random(
                [BATCH, D],
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &device,
            );
            nsf.log_prob(x)
        })
    });
    group.bench_function("RealNVP", |b| {
        b.iter(|| {
            let x = Tensor::<B, 2>::random(
                [BATCH, D],
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &device,
            );
            nvp.log_prob(x)
        })
    });

    group.finish();
}

fn bench_batch_scaling_nsf(c: &mut Criterion) {
    let device = Default::default();
    let mut group = c.benchmark_group("batch_scaling_nsf");

    let nsf = NsfConfig::new(D, 8, HIDDEN.to_vec()).init::<B>(&device);

    for &batch_size in &[32usize, 128, 512, 2048] {
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &batch_size,
            |b, &bs| {
                b.iter(|| {
                    let x = Tensor::<B, 2>::random(
                        [bs, D],
                        burn::tensor::Distribution::Normal(0.0, 1.0),
                        &device,
                    );
                    nsf.forward(x)
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_forward,
    bench_inverse,
    bench_log_prob,
    bench_batch_scaling_nsf
);
criterion_main!(benches);
