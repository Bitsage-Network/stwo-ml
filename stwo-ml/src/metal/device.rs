//! Metal/wgpu device initialization and management.

use std::sync::OnceLock;

/// Global wgpu device + queue (initialized once, shared across all kernels).
static DEVICE: OnceLock<MetalDevice> = OnceLock::new();

pub struct MetalDevice {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter_name: String,
    pub max_buffer_size: u64,
}

impl MetalDevice {
    /// Initialize the Metal device. Panics if no GPU is available.
    pub fn init() -> Self {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::METAL,
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .expect("No Metal GPU adapter found");

        let adapter_name = adapter.get_info().name.clone();
        let max_buffer_size = adapter.limits().max_buffer_size;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("stwo-ml-metal"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits {
                    max_buffer_size,
                    max_storage_buffer_binding_size: max_buffer_size as u32,
                    max_compute_workgroup_size_x: 256,
                    max_compute_workgroup_size_y: 256,
                    max_compute_workgroup_size_z: 64,
                    max_compute_invocations_per_workgroup: 256,
                    ..Default::default()
                },
                memory_hints: Default::default(),
            },
            None,
        ))
        .expect("Failed to create Metal device");

        eprintln!(
            "[Metal] Initialized: {} (max buffer: {} MB)",
            adapter_name,
            max_buffer_size / (1024 * 1024),
        );

        Self {
            device,
            queue,
            adapter_name,
            max_buffer_size,
        }
    }

    /// Get or initialize the global Metal device.
    pub fn global() -> &'static MetalDevice {
        DEVICE.get_or_init(|| MetalDevice::init())
    }
}
