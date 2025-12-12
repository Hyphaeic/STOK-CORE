//! Backend device management for STOK computations.
//!
//! Provides abstraction over Burn backends (WGPU, CPU, CUDA) for
//! GPU-accelerated tensor operations.
//!
//! # Backend Selection
//!
//! The default backend is WGPU, which provides cross-platform GPU support:
//!
//! | Platform | Graphics API |
//! |----------|--------------|
//! | Windows | DirectX 12, Vulkan |
//! | macOS | Metal |
//! | Linux | Vulkan |
//!
//! # Usage
//!
//! ```rust,ignore
//! use stok_core::backend::{default_device, DefaultBackend};
//!
//! let device = default_device();
//! // Use device for tensor allocation
//! ```
//!
//! # Feature Flags
//!
//! - `cuda`: Enable CUDA backend (requires NVIDIA GPU + CUDA toolkit)

use crate::types::StokError;
use burn::backend::wgpu::{Wgpu, WgpuDevice};

// ============================================================================
// Type Aliases
// ============================================================================

/// Default backend type (WGPU for cross-platform GPU support).
///
/// WGPU provides excellent cross-platform compatibility without
/// requiring specific GPU drivers or SDKs.
pub type DefaultBackend = Wgpu;

/// Default device type for the WGPU backend.
pub type DefaultDevice = WgpuDevice;

// ============================================================================
// Device Configuration
// ============================================================================

/// Device configuration options.
///
/// Specifies which backend and device to use for tensor operations.
///
/// # Example
///
/// ```rust,ignore
/// use stok_core::backend::DeviceConfig;
///
/// let config = DeviceConfig::Wgpu { device_index: 0 };
/// ```
#[derive(Debug, Clone)]
pub enum DeviceConfig {
    /// WGPU backend (Vulkan/Metal/DX12).
    ///
    /// Recommended for cross-platform deployment.
    Wgpu {
        /// GPU device index (0 = primary GPU)
        device_index: usize,
    },
    
    /// CPU backend (for testing/fallback).
    ///
    /// Uses software rendering, significantly slower than GPU.
    Cpu,
    
    /// CUDA backend (feature-gated).
    ///
    /// Requires `cuda` feature and NVIDIA GPU.
    #[cfg(feature = "cuda")]
    Cuda {
        /// CUDA device index
        device_index: usize,
    },
}

impl Default for DeviceConfig {
    fn default() -> Self {
        Self::Wgpu { device_index: 0 }
    }
}

// ============================================================================
// Device Manager
// ============================================================================

/// Manages device initialization and access.
///
/// Wraps the underlying Burn device with configuration tracking.
///
/// # Example
///
/// ```rust,ignore
/// use stok_core::backend::{DeviceManager, DeviceConfig};
///
/// let manager = DeviceManager::new(DeviceConfig::default())?;
/// let device = manager.device();
/// ```
#[derive(Debug)]
pub struct DeviceManager {
    device: WgpuDevice,
    config: DeviceConfig,
}

impl DeviceManager {
    /// Create a new device manager with the given configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Device configuration specifying backend and device index
    ///
    /// # Returns
    ///
    /// `Ok(DeviceManager)` on success, `Err(StokError::DeviceError)` on failure.
    ///
    /// # Errors
    ///
    /// - `StokError::DeviceError` if device initialization fails
    pub fn new(config: DeviceConfig) -> Result<Self, StokError> {
        let device = match &config {
            DeviceConfig::Wgpu { device_index } => {
                if *device_index == 0 {
                    WgpuDevice::default()
                } else {
                    WgpuDevice::DiscreteGpu(*device_index)
                }
            }
            DeviceConfig::Cpu => {
                WgpuDevice::Cpu
            }
            #[cfg(feature = "cuda")]
            DeviceConfig::Cuda { .. } => {
                return Err(StokError::DeviceError(
                    "CUDA backend requires burn-cuda crate".into(),
                ));
            }
        };

        Ok(Self { device, config })
    }

    /// Get reference to the underlying device.
    ///
    /// Use this to pass to tensor allocation functions.
    pub fn device(&self) -> &WgpuDevice {
        &self.device
    }

    /// Get the device configuration.
    pub fn config(&self) -> &DeviceConfig {
        &self.config
    }

    /// Check if running on GPU (vs CPU fallback).
    pub fn is_gpu(&self) -> bool {
        !matches!(self.config, DeviceConfig::Cpu)
    }
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Get the default GPU device.
///
/// Returns a WGPU device configured for the primary GPU.
/// This is the recommended way to get a device for most use cases.
///
/// # Example
///
/// ```rust,ignore
/// use stok_core::backend::default_device;
///
/// let device = default_device();
/// ```
pub fn default_device() -> WgpuDevice {
    WgpuDevice::default()
}

/// Get a CPU device for testing.
///
/// Returns a device that runs on CPU via software rendering.
/// Useful for testing on machines without GPU support.
///
/// # Warning
///
/// CPU execution is significantly slower than GPU. Use only for
/// small test cases or when GPU is unavailable.
pub fn cpu_device() -> WgpuDevice {
    WgpuDevice::Cpu
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_device() {
        let device = default_device();
        // Should not panic
        let _ = device;
    }

    #[test]
    fn test_cpu_device() {
        let device = cpu_device();
        // Should not panic
        let _ = device;
    }

    #[test]
    fn test_device_manager_default() {
        let manager = DeviceManager::new(DeviceConfig::default());
        assert!(manager.is_ok());
    }

    #[test]
    fn test_device_manager_cpu() {
        let manager = DeviceManager::new(DeviceConfig::Cpu).unwrap();
        assert!(!manager.is_gpu());
    }

    #[test]
    fn test_device_config_default() {
        let config = DeviceConfig::default();
        match config {
            DeviceConfig::Wgpu { device_index } => {
                assert_eq!(device_index, 0);
            }
            _ => panic!("Default should be WGPU"),
        }
    }
}