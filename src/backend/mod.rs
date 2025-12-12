//! Backend abstraction for GPU/CPU computation.

mod device;

pub use device::{
    cpu_device, default_device, DefaultBackend, DefaultDevice, DeviceConfig, DeviceManager,
};