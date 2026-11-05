//! Synchronous batch-one compute; no subgroup-size or FP16 extension requirement.
use anyhow::{Context, Result, bail, ensure};
use ash::{Entry, vk};
use std::{
    ffi::{CStr, CString},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

struct Instance {
    _entry: Entry,
    raw: ash::Instance,
    abandoned: Arc<AtomicBool>,
}
impl Drop for Instance {
    fn drop(&mut self) {
        if !self.abandoned.load(Ordering::Relaxed) {
            unsafe { self.raw.destroy_instance(None) };
        }
    }
}
struct Device {
    _instance: Instance,
    raw: ash::Device,
    memory: vk::PhysicalDeviceMemoryProperties,
    abandoned: Arc<AtomicBool>,
}
impl Drop for Device {
    fn drop(&mut self) {
        if !self.abandoned.load(Ordering::Relaxed) {
            unsafe { self.raw.destroy_device(None) };
        }
    }
}

pub struct Buffer {
    device: Arc<Device>,
    raw: vk::Buffer,
    memory: vk::DeviceMemory,
    size: usize,
}
impl Buffer {
    pub fn write(&self, bytes: &[u8]) -> Result<()> {
        ensure!(
            bytes.len() <= self.size,
            "GPU buffer write exceeds allocation"
        );
        ensure!(
            !self.device.abandoned.load(Ordering::Relaxed),
            "GPU timed out; restart the app"
        );
        unsafe {
            let ptr = self.device.raw.map_memory(
                self.memory,
                0,
                self.size as u64,
                vk::MemoryMapFlags::empty(),
            )?;
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.cast(), bytes.len());
            self.device.raw.unmap_memory(self.memory);
        }
        Ok(())
    }
    fn read_f32(&self, count: usize) -> Result<Vec<f32>> {
        ensure!(
            count.checked_mul(4).is_some_and(|n| n <= self.size),
            "GPU buffer read exceeds allocation"
        );
        let mut output = vec![0.; count];
        unsafe {
            let ptr = self.device.raw.map_memory(
                self.memory,
                0,
                self.size as u64,
                vk::MemoryMapFlags::empty(),
            )?;
            std::ptr::copy_nonoverlapping(ptr.cast::<f32>(), output.as_mut_ptr(), count);
            self.device.raw.unmap_memory(self.memory);
        }
        ensure!(
            output.iter().all(|x| x.is_finite()),
            "GPU returned non-finite values"
        );
        Ok(output)
    }
}
impl Drop for Buffer {
    fn drop(&mut self) {
        if !self.device.abandoned.load(Ordering::Relaxed) {
            unsafe {
                self.device.raw.destroy_buffer(self.raw, None);
                self.device.raw.free_memory(self.memory, None);
            }
        }
    }
}

pub struct Gpu {
    device: Arc<Device>,
    queue: vk::Queue,
    pool: vk::CommandPool,
    command: vk::CommandBuffer,
    fence: vk::Fence,
    set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor: vk::DescriptorSet,
    layout: vk::PipelineLayout,
    qmv: vk::Pipeline,
    dense: vk::Pipeline,
    input: Option<Buffer>,
    output: Option<Buffer>,
    pub name: String,
    pub max_range: usize,
    max_groups: u32,
    allocated: usize,
}

impl Gpu {
    pub fn new() -> Result<Self> {
        let entry = unsafe { Entry::load().context("Vulkan loader is unavailable")? };
        let app_name = CString::new("MLXL3 Android")?;
        let app = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .api_version(vk::API_VERSION_1_1);
        let mut info = vk::InstanceCreateInfo::default().application_info(&app);
        let extensions = [ash::khr::portability_enumeration::NAME.as_ptr()];
        if cfg!(target_os = "macos") {
            info = info
                .enabled_extension_names(&extensions)
                .flags(vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR);
        }
        let abandoned = Arc::new(AtomicBool::new(false));
        let instance = Instance {
            raw: unsafe { entry.create_instance(&info, None)? },
            _entry: entry,
            abandoned: abandoned.clone(),
        };
        let mut chosen = None;
        for physical in unsafe { instance.raw.enumerate_physical_devices()? } {
            let props = unsafe { instance.raw.get_physical_device_properties(physical) };
            if props.api_version < vk::API_VERSION_1_1
                || props.limits.max_compute_work_group_invocations < 128
                || props.limits.max_compute_work_group_size[0] < 128
                || props.limits.max_push_constants_size < 16
            {
                continue;
            }
            let families = unsafe {
                instance
                    .raw
                    .get_physical_device_queue_family_properties(physical)
            };
            if let Some(family) = families
                .iter()
                .position(|f| f.queue_count > 0 && f.queue_flags.contains(vk::QueueFlags::COMPUTE))
            {
                chosen = Some((physical, family as u32, props));
                break;
            }
        }
        let (physical, family, props) =
            chosen.context("No compatible Vulkan 1.1 compute device")?;
        let priorities = [1.];
        let queues = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(family)
            .queue_priorities(&priorities)];
        let device_extensions = unsafe {
            instance
                .raw
                .enumerate_device_extension_properties(physical)?
        };
        let portability = device_extensions.iter().any(|e| unsafe { CStr::from_ptr(e.extension_name.as_ptr()) } == ash::khr::portability_subset::NAME);
        let extension_names = [ash::khr::portability_subset::NAME.as_ptr()];
        let mut device_info = vk::DeviceCreateInfo::default().queue_create_infos(&queues);
        if portability {
            device_info = device_info.enabled_extension_names(&extension_names);
        }
        let raw = unsafe { instance.raw.create_device(physical, &device_info, None)? };
        let queue = unsafe { raw.get_device_queue(family, 0) };
        let memory = unsafe { instance.raw.get_physical_device_memory_properties(physical) };
        let device = Arc::new(Device {
            _instance: instance,
            raw,
            memory,
            abandoned,
        });
        let mut gpu = Self {
            device,
            queue,
            pool: vk::CommandPool::null(),
            command: vk::CommandBuffer::null(),
            fence: vk::Fence::null(),
            set_layout: vk::DescriptorSetLayout::null(),
            descriptor_pool: vk::DescriptorPool::null(),
            descriptor: vk::DescriptorSet::null(),
            layout: vk::PipelineLayout::null(),
            qmv: vk::Pipeline::null(),
            dense: vk::Pipeline::null(),
            input: None,
            output: None,
            name: unsafe { CStr::from_ptr(props.device_name.as_ptr()) }
                .to_string_lossy()
                .into_owned(),
            max_range: props.limits.max_storage_buffer_range as usize,
            max_groups: props.limits.max_compute_work_group_count[0],
            allocated: 0,
        };
        unsafe {
            let d = &gpu.device.raw;
            gpu.pool = d.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )?;
            gpu.command = d.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(gpu.pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )?[0];
            gpu.fence = d.create_fence(&vk::FenceCreateInfo::default(), None)?;
            let bindings: Vec<_> = (0..3)
                .map(|i| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(i)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                })
                .collect();
            gpu.set_layout = d.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )?;
            let sizes = [vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 3,
            }];
            gpu.descriptor_pool = d.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(1)
                    .pool_sizes(&sizes),
                None,
            )?;
            let layouts = [gpu.set_layout];
            gpu.descriptor = d.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(gpu.descriptor_pool)
                    .set_layouts(&layouts),
            )?[0];
            let ranges = [vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .size(16)];
            gpu.layout = d.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&layouts)
                    .push_constant_ranges(&ranges),
                None,
            )?;
        }
        gpu.qmv = gpu.pipeline(include_bytes!(concat!(env!("OUT_DIR"), "/qmv.spv")))?;
        gpu.dense = gpu.pipeline(include_bytes!(concat!(env!("OUT_DIR"), "/dense.spv")))?;
        gpu.input = Some(gpu.buffer(&vec![0; 32768 * 4])?);
        gpu.output = Some(gpu.buffer(&vec![0; 32768 * 4])?);
        Ok(gpu)
    }

    fn pipeline(&self, bytes: &[u8]) -> Result<vk::Pipeline> {
        let code = ash::util::read_spv(&mut std::io::Cursor::new(bytes))?;
        unsafe {
            let module = self
                .device
                .raw
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)?;
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(c"main");
            let infos = [vk::ComputePipelineCreateInfo::default()
                .stage(stage)
                .layout(self.layout)];
            let result =
                self.device
                    .raw
                    .create_compute_pipelines(vk::PipelineCache::null(), &infos, None);
            self.device.raw.destroy_shader_module(module, None);
            match result {
                Ok(pipelines) => Ok(pipelines[0]),
                Err((pipelines, err)) => {
                    for pipeline in pipelines {
                        self.device.raw.destroy_pipeline(pipeline, None);
                    }
                    bail!("Vulkan compute pipeline: {err:?}")
                }
            }
        }
    }

    pub fn buffer(&mut self, bytes: &[u8]) -> Result<Buffer> {
        let size = bytes
            .len()
            .checked_add(3)
            .map(|n| n & !3)
            .context("buffer size overflow")?
            .max(4);
        ensure!(
            size <= self.max_range,
            "Tensor exceeds Vulkan storage-buffer limit"
        );
        ensure!(
            self.allocated
                .checked_add(size)
                .is_some_and(|n| n <= 3 * 1024 * 1024 * 1024usize),
            "Model exceeds the 3 GiB GPU budget"
        );
        let d = &self.device.raw;
        unsafe {
            let raw = d.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size as u64)
                    .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )?;
            let req = d.get_buffer_memory_requirements(raw);
            let ty = (0..self.device.memory.memory_type_count).find(|i| {
                req.memory_type_bits & (1 << i) != 0
                    && self.device.memory.memory_types[*i as usize]
                        .property_flags
                        .contains(
                            vk::MemoryPropertyFlags::HOST_VISIBLE
                                | vk::MemoryPropertyFlags::HOST_COHERENT,
                        )
            });
            let Some(ty) = ty else {
                d.destroy_buffer(raw, None);
                bail!("No host-visible coherent Vulkan memory");
            };
            let memory = match d.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(ty),
                None,
            ) {
                Ok(memory) => memory,
                Err(err) => {
                    d.destroy_buffer(raw, None);
                    return Err(err.into());
                }
            };
            let buffer = Buffer {
                device: self.device.clone(),
                raw,
                memory,
                size,
            };
            d.bind_buffer_memory(raw, memory, 0)?;
            buffer.write(bytes)?;
            self.allocated += size;
            Ok(buffer)
        }
    }

    pub fn multiply(
        &mut self,
        weights: &Buffer,
        x: &[f32],
        outputs: usize,
        bits: u32,
        codebook: u32,
    ) -> Result<Vec<f32>> {
        ensure!(
            !self.device.abandoned.load(Ordering::Relaxed),
            "GPU timed out; restart the app"
        );
        ensure!(
            !x.is_empty()
                && x.len() <= 32768
                && outputs > 0
                && outputs <= 32768
                && outputs <= self.max_groups as usize,
            "Invalid GPU projection dimensions"
        );
        ensure!(
            x.iter().all(|v| v.is_finite()),
            "Projection input is non-finite"
        );
        ensure!(
            Arc::ptr_eq(&weights.device, &self.device),
            "Weights belong to another GPU"
        );
        let expected = if bits == 0 {
            ensure!(codebook <= 1, "Invalid dense weight dtype");
            x.len().checked_mul(outputs).and_then(|n| n.checked_mul(2))
        } else {
            ensure!(
                (1..=8).contains(&bits)
                    && codebook <= 2
                    && x.len().is_multiple_of(16)
                    && outputs.is_multiple_of(16),
                "Invalid EXL3 layout"
            );
            x.len()
                .checked_mul(outputs)
                .and_then(|n| n.checked_mul(bits as usize))
                .map(|n| n / 8)
        }
        .context("Projection size overflow")?;
        ensure!(
            weights.size == (expected + 3) & !3,
            "Weight buffer does not match projection shape"
        );
        let input = self.input.as_ref().context("GPU input unavailable")?;
        let output = self.output.as_ref().context("GPU output unavailable")?;
        let params = [
            x.len() as u32,
            outputs as u32,
            if bits == 0 { codebook } else { bits },
            codebook,
        ];
        let bytes = |values: &[f32]| unsafe {
            std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
        };
        input.write(bytes(x))?;
        unsafe {
            let d = &self.device.raw;
            let infos = [
                [vk::DescriptorBufferInfo::default()
                    .buffer(input.raw)
                    .range(input.size as u64)],
                [vk::DescriptorBufferInfo::default()
                    .buffer(weights.raw)
                    .range(weights.size as u64)],
                [vk::DescriptorBufferInfo::default()
                    .buffer(output.raw)
                    .range(output.size as u64)],
            ];
            let writes: Vec<_> = infos
                .iter()
                .enumerate()
                .map(|(i, info)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(self.descriptor)
                        .dst_binding(i as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(info)
                })
                .collect();
            d.update_descriptor_sets(&writes, &[]);
            d.reset_command_buffer(self.command, vk::CommandBufferResetFlags::empty())?;
            d.reset_fences(&[self.fence])?;
            d.begin_command_buffer(
                self.command,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            d.cmd_bind_pipeline(
                self.command,
                vk::PipelineBindPoint::COMPUTE,
                if bits == 0 { self.dense } else { self.qmv },
            );
            d.cmd_bind_descriptor_sets(
                self.command,
                vk::PipelineBindPoint::COMPUTE,
                self.layout,
                0,
                &[self.descriptor],
                &[],
            );
            let push = std::slice::from_raw_parts(params.as_ptr().cast::<u8>(), 16);
            d.cmd_push_constants(
                self.command,
                self.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                push,
            );
            d.cmd_dispatch(self.command, outputs as u32, 1, 1);
            let barriers = [vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)];
            d.cmd_pipeline_barrier(
                self.command,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &barriers,
                &[],
                &[],
            );
            d.end_command_buffer(self.command)?;
            let commands = [self.command];
            let submits = [vk::SubmitInfo::default().command_buffers(&commands)];
            d.queue_submit(self.queue, &submits, self.fence)?;
            if let Err(err) = d.wait_for_fences(&[self.fence], true, 30_000_000_000) {
                // A timed-out dispatch may still own buffers: retain its resources until process exit.
                self.device.abandoned.store(true, Ordering::Relaxed);
                bail!("Vulkan dispatch failed ({err:?}); restart the app");
            }
        }
        output.read_f32(outputs)
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        if self.device.abandoned.load(Ordering::Relaxed) {
            return;
        }
        unsafe {
            let d = &self.device.raw;
            d.destroy_pipeline(self.qmv, None);
            d.destroy_pipeline(self.dense, None);
            d.destroy_pipeline_layout(self.layout, None);
            d.destroy_descriptor_pool(self.descriptor_pool, None);
            d.destroy_descriptor_set_layout(self.set_layout, None);
            d.destroy_fence(self.fence, None);
            d.destroy_command_pool(self.pool, None);
        }
    }
}
