//! Synchronous batch-one compute; no subgroup-size or FP16 extension requirement.
use crate::codec::{self, Codebook};
use anyhow::{Context, Result, bail, ensure};
use ash::{Entry, vk};
use std::{
    ffi::{CStr, CString},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

const CODEBOOK_BYTES: usize = 3 * 65536 * 4;

fn codebook_offset(codebook: u32, state: u16) -> Option<usize> {
    if codebook >= 3 {
        return None;
    }
    Some((codebook as usize * 65536 + usize::from(state)) * 4)
}

fn codebook_table() -> Vec<u8> {
    let mut table = vec![0; CODEBOOK_BYTES];
    for cb in [Codebook::Default, Codebook::Mcg, Codebook::Mul1] {
        for state in 0..=u16::MAX {
            let offset = codebook_offset(cb as u32, state).expect("valid codebook");
            table[offset..offset + 4]
                .copy_from_slice(&codec::decode_codeword(state, cb).to_f32().to_le_bytes());
        }
    }
    table
}

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
    codebooks: Option<Buffer>,
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
            codebooks: None,
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
            let bindings: Vec<_> = (0..4)
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
                descriptor_count: 4,
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
        gpu.codebooks = Some(gpu.buffer(&codebook_table())?);
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
        let params = [
            x.len() as u32,
            outputs as u32,
            if bits == 0 { codebook } else { bits },
            codebook,
        ];
        self.compute(
            if bits == 0 { self.dense } else { self.qmv },
            weights,
            x,
            params,
            outputs,
        )
    }

    fn compute(
        &mut self,
        pipeline: vk::Pipeline,
        weights: &Buffer,
        x: &[f32],
        params: [u32; 4],
        outputs: usize,
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
            "Invalid compute buffer dimensions"
        );
        let input = self.input.as_ref().context("GPU input unavailable")?;
        let output = self.output.as_ref().context("GPU output unavailable")?;
        let codebooks = self
            .codebooks
            .as_ref()
            .context("GPU codebooks unavailable")?;
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
                [vk::DescriptorBufferInfo::default()
                    .buffer(codebooks.raw)
                    .range(codebooks.size as u64)],
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
            d.cmd_bind_pipeline(self.command, vk::PipelineBindPoint::COMPUTE, pipeline);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codebook_table_has_all_finite_values_and_checked_offsets() {
        let table = codebook_table();
        assert_eq!(table.len(), CODEBOOK_BYTES);
        assert!(
            table
                .chunks_exact(4)
                .all(|b| f32::from_le_bytes(b.try_into().unwrap()).is_finite())
        );
        assert_eq!(codebook_offset(0, 0), Some(0));
        assert_eq!(codebook_offset(2, u16::MAX), Some(CODEBOOK_BYTES - 4));
        assert_eq!(codebook_offset(3, 0), None);
        assert_eq!(codebook_offset(u32::MAX, u16::MAX), None);
    }

    #[test]
    #[ignore = "requires Vulkan; exhaustive EXL3 procedural codebook parity"]
    fn codebooks_match_cpu_bit_exact_for_all_196608_values() {
        let mut gpu = Gpu::new().unwrap();
        let pipeline = gpu
            .pipeline(include_bytes!(concat!(env!("OUT_DIR"), "/codebook.spv")))
            .unwrap();
        let reference = gpu
            .pipeline(include_bytes!(concat!(
                env!("OUT_DIR"),
                "/codebook_reference.spv"
            )))
            .unwrap();
        let dummy = gpu.buffer(&[0; 4]).unwrap();
        for cb in [Codebook::Default, Codebook::Mcg, Codebook::Mul1] {
            for offset in [0, 32768] {
                let values = gpu
                    .compute(
                        pipeline,
                        &dummy,
                        &[0.],
                        [offset, 32768, 0, cb as u32],
                        32768,
                    )
                    .unwrap();
                let procedural = gpu
                    .compute(
                        reference,
                        &dummy,
                        &[0.],
                        [offset, 32768, 0, cb as u32],
                        32768,
                    )
                    .unwrap();
                assert_eq!(values.len(), 32768);
                assert_eq!(procedural.len(), values.len());
                for (i, actual) in values.iter().enumerate() {
                    let state = (offset as usize + i) as u16;
                    let expected = codec::decode_codeword(state, cb).to_f32();
                    assert_eq!(
                        actual.to_bits(),
                        procedural[i].to_bits(),
                        "GPU procedural/lookup {cb:?} state={state}"
                    );
                    assert_eq!(
                        actual.to_bits(),
                        expected.to_bits(),
                        "{cb:?} state={state}: {actual} != {expected}"
                    );
                }
            }
        }
        unsafe {
            gpu.device.raw.destroy_pipeline(pipeline, None);
            gpu.device.raw.destroy_pipeline(reference, None);
        };
    }

    #[test]
    #[ignore = "explicit projection microbenchmark; not a phone performance claim"]
    fn benchmark_codebook_lookup_against_procedural() {
        use std::time::Instant;
        let mut gpu = Gpu::new().unwrap();
        let reference = gpu
            .pipeline(include_bytes!(concat!(
                env!("OUT_DIR"),
                "/qmv_reference.spv"
            )))
            .unwrap();
        for (rows, cols) in [
            (2048usize, 2048usize),
            (2048, 6144),
            (2048, 7680),
            (7680, 2048),
        ] {
            let packed: Vec<u8> = (0..rows * cols / 8)
                .flat_map(|i| {
                    (i as u32)
                        .wrapping_mul(0x9e3779b9)
                        .wrapping_add(177)
                        .to_le_bytes()
                })
                .collect();
            let buffer = gpu.buffer(&packed).unwrap();
            let x: Vec<_> = (0..rows)
                .map(|i| ((i * 13 % 17) as f32 - 8.) / 64.)
                .collect();
            for cb in 0..3 {
                let params = [rows as u32, cols as u32, 4, cb];
                let expected = gpu.compute(reference, &buffer, &x, params, cols).unwrap();
                assert_eq!(expected.len(), cols);
                assert!(expected.iter().any(|v| v.abs() > 0.01));
                let check = |values: &[f32]| {
                    assert_eq!(values.len(), cols);
                    for (i, (actual, expected)) in values.iter().zip(&expected).enumerate() {
                        assert!(actual.is_finite());
                        assert_eq!(
                            actual.to_bits(),
                            expected.to_bits(),
                            "rows={rows} cols={cols} cb={cb} output={i}"
                        );
                    }
                };
                // Correctness precedes timing; both paths use the same packed resident buffer.
                check(&gpu.multiply(&buffer, &x, cols, 4, cb).unwrap());
                for _ in 0..2 {
                    check(&gpu.compute(reference, &buffer, &x, params, cols).unwrap());
                    check(&gpu.multiply(&buffer, &x, cols, 4, cb).unwrap());
                }
                let mut times = [vec![], vec![]];
                for pair in 0..7 {
                    for lookup in if pair % 2 == 0 {
                        [false, true]
                    } else {
                        [true, false]
                    } {
                        let start = Instant::now();
                        let values = if lookup {
                            gpu.multiply(&buffer, &x, cols, 4, cb).unwrap()
                        } else {
                            gpu.compute(reference, &buffer, &x, params, cols).unwrap()
                        };
                        let elapsed = start.elapsed().as_secs_f64() * 1000.;
                        assert!(elapsed.is_finite() && elapsed > 0.);
                        check(&values);
                        times[usize::from(lookup)].push(elapsed);
                    }
                }
                assert!(times.iter().all(|t| t.len() == 7));
                eprintln!(
                    "LOOKUP_BENCH {}",
                    serde_json::json!({
                        "status":"validated_microbenchmark", "device":gpu.name,
                        "rows":rows, "cols":cols, "bits":4, "codebook":cb,
                        "warmups":2, "pairs":7, "parity":"bit_exact",
                        "procedural_ms":times[0], "lookup_ms":times[1],
                        "limits":"software/device projection timing, not full-model or S24 throughput"
                    })
                );
            }
        }
        unsafe { gpu.device.raw.destroy_pipeline(reference, None) };
    }
}

#[cfg(kani)]
mod proofs {
    use super::*;

    #[kani::proof]
    fn codebook_offsets_are_aligned_bounded_and_reject_invalid_ids() {
        let cb: u32 = kani::any();
        let state: u16 = kani::any();
        let offset = codebook_offset(cb, state);
        assert_eq!(offset.is_some(), cb < 3);
        if let Some(offset) = offset {
            assert_eq!(offset % 4, 0);
            assert!(offset + 4 <= CODEBOOK_BYTES);
            assert_eq!(offset / 4 / 65536, cb as usize);
            assert_eq!(offset / 4 % 65536, usize::from(state));
        }
        kani::cover!(cb == 2 && state == u16::MAX);
        kani::cover!(cb == 0 && state == 0);
        kani::cover!(cb == u32::MAX && offset.is_none());
    }
}
