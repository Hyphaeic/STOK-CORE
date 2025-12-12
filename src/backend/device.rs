//! Backend device management for STOK computations.
//!
//! Provides abstraction over Burn backends (WGPU, CPU, CUDA).

use crate::types::StokError;
use burn::backend::wgpu::{Wgpu, WgpuDevice};

// ============================================================================
// Type Aliases
// ============================================================================

/// Default backend type (WGPU for cross-platform GPU support)
pub type DefaultBackend = Wgpu;

/// Default device type
pub type DefaultDevice = WgpuDevice;

// ============================================================================
// Device Configuration
// ============================================================================

/// Device configuration options
#[derive(Debug, Clone)]
pub enum DeviceConfig {
    /// WGPU backend (Vulkan/Metal/DX12)
    Wgpu { device_index: usize },
    /// CPU backend (for testing/fallback)
    Cpu,
    #[cfg(feature = "cuda")]
    /// CUDA backend (feature-gated)
    Cuda { device_index: usize },
}

impl Default for DeviceConfig {
    fn default() -> Self {
        Self::Wgpu { device_index: 0 }
    }
}

// ============================================================================
// Device Manager
// ============================================================================

/// Manages device initialization and access
#[derive(Debug)]
pub struct DeviceManager {
    device: WgpuDevice,
    config: DeviceConfig,
}

impl DeviceManager {
    /// Create a new device manager with the given configuration
    pub fn new(config: DeviceConfig) -> Result<Self, StokError> {
        let device = match &config {
            DeviceConfig::Wgpu { device_index } => {
                // WgpuDevice::default() selects best available
                // For specific device selection, use DiscreteGpu(index) or IntegratedGpu(index)
                if *device_index == 0 {
                    WgpuDevice::default()
                } else {
                    WgpuDevice::DiscreteGpu(*device_index)
                }
            }
            DeviceConfig::Cpu => {
                // WGPU can run on CPU via software rasterization
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

    /// Get reference to the device
    pub fn device(&self) -> &WgpuDevice {
        &self.device
    }

    /// Get the device configuration
    pub fn config(&self) -> &DeviceConfig {
        &self.config
    }

    /// Clone the device (for tensor creation)
    pub fn device_clone(&self) -> WgpuDevice {
        self.device.clone()
    }
}

impl Default for DeviceManager {
    fn default() -> Self {
        Self::new(DeviceConfig::default()).expect("Failed to create default device")
    }
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Get the default WGPU device
pub fn default_device() -> WgpuDevice {
    WgpuDevice::default()
}

/// Get a CPU device (for testing)
pub fn cpu_device() -> WgpuDevice {
    WgpuDevice::Cpu
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Tensor;

    #[test]
    fn test_default_device_creation() {
        let manager = DeviceManager::default();
        let _device = manager.device();
        // If we get here without panic, device was created successfully
    }

    #[test]
    fn test_tensor_on_device() {
        let device = default_device();
        let tensor: Tensor<Wgpu, 1> = Tensor::zeros([10], &device);
        assert_eq!(tensor.dims(), [10]);
    }

    #[test]
    fn test_cpu_device() {
        let device = cpu_device();
        let tensor: Tensor<Wgpu, 1> = Tensor::ones([5], &device);
        let data: Vec<f32> = tensor.into_data().to_vec().unwrap();
        assert_eq!(data, vec![1.0; 5]);
    }
}
