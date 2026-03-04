//! # flowrs — Normalizing Flows in Rust
//!
//! A normalizing flows library built on the [Burn](https://burn.dev) deep learning framework.
//!
//! ## Flow architectures
//!
//! | Model | Module | Description |
//! |-------|--------|-------------|
//! | MAF | [`Maf`] | Masked Autoregressive Flow — fast parallel forward, sequential inverse |
//! | NSF | [`Nsf`] | Neural Spline Flow — rational-quadratic spline coupling layers |
//! | RealNVP | [`RealNvp`] | Real-valued Non-Volume Preserving — affine coupling layers |
//!
//! All three implement the [`Flow`] trait, so you can write generic code over any
//! architecture.
//!
//! ## Quick start
//!
//! ```rust,ignore
//! use burn::backend::NdArray;
//! use flowrs::{Flow, NsfConfig};
//!
//! let device = Default::default();
//! let model = NsfConfig::new(2, 8, vec![128, 128]).init::<NdArray>(&device);
//! // forward / inverse / log_prob via the Flow trait
//! ```

#![recursion_limit = "8192"]

pub mod actnorm;
pub mod coupling;
pub mod flow;
pub mod lu_linear;
pub mod made;
pub mod maf;
mod masked_linear;
pub(crate) mod mlp;
pub mod nsf;
pub mod realnvp;
pub mod spline;

pub use actnorm::{ActNorm, ActNormConfig};
pub use coupling::{AffineCoupling, AffineCouplingConfig, SplineCoupling, SplineCouplingConfig};
pub use flow::{Flow, standard_normal_log_prob};
pub use lu_linear::{LULinear, LULinearConfig};
pub use made::{Made, MadeConfig};
pub use maf::{Maf, MafConfig};
pub use nsf::{Nsf, NsfConfig};
pub use realnvp::{RealNvp, RealNvpConfig};
