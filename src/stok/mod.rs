//! State-Time Option Kernel (STOK) structures.
//!
//! STOKs are predictive probability kernels over termination events,
//! enabling exact composition via Chapman-Kolmogorov equations.
//!
//! Reference: Ringstrom & Schrater (2025), Equations [15-17]

mod kernel;

pub use kernel::{STOKKernel, STOKData};