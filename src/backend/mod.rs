//! Backend abstraction for GPU-accelerated computation.
//!
//! This module provides device management and backend selection
//! for running STOK computations on GPU or CPU.
//!
//! # Default Backend
//!
//! The default backend is WGPU, which provides cross-platform GPU support
//! via Vulkan (Linux/Windows), Metal (macOS), or DirectX 12 (Windows).
//!
//! # Usage
//!
//! ```rust,ignore
//! use stok_core::backend::{default_device, DefaultBackend};
//! use burn::prelude::*;
//!
//! let device = default_device();
//! let tensor: Tensor<DefaultBackend, 2> = Tensor::zeros([10, 10], &device);
//! ```
//!
//! # Feature Flags
//!
//! | Feature | Description |
//! |---------|-------------|
//! | (default) | WGPU backend |
//! | `cuda` | NVIDIA CUDA backend |
//!
//! # Performance Notes
//!
//! - WGPU: Best cross-platform compatibility, good performance
//! - CUDA: Best performance on NVIDIA GPUs, requires CUDA toolkit
//! - CPU: Slowest, use only for testing

mod device;

pub use device::{
    DefaultBackend,
    DefaultDevice,
    DeviceConfig,
    DeviceManager,
    default_device,
    cpu_device,
};