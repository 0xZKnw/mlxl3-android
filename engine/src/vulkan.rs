//! Synchronous batch-one compute; no subgroup-size or FP16 extension requirement.
use crate::codec::{self, Codebook};
use anyhow::{Context, Result, bail, ensure};
use ash::{Entry, vk};
use std::{
    cell::Cell,
    collections::BTreeMap,
    ffi::{CStr, CString},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};

#[derive(Clone, Copy, Default, serde::Serialize)]
pub struct Profile {
    pub upload_ns: u64,
    pub record_submit_ns: u64,
    pub wait_ns: u64,
    pub readback_ns: u64,
    pub projection_cpu_ns: u64,
    pub model_ns: u64,
    pub head_ns: u64,
}

pub(crate) fn elapsed_ns(start: Instant) -> u64 {
    start.elapsed().as_nanos().try_into().unwrap_or(u64::MAX)
}

fn choose_memory_type(flags: &[u32], allowed: u32) -> Option<usize> {
    let required =
        (vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT).as_raw();
    let mut fallback = None;
    for (i, flags) in flags.iter().take(32).enumerate() {
        if allowed & (1u32 << i) != 0 && flags & required == required {
            if flags & vk::MemoryPropertyFlags::DEVICE_LOCAL.as_raw() != 0 {
                return Some(i);
            }
            if fallback.is_none() {
                fallback = Some(i);
            }
        }
    }
    fallback
}

const CODEBOOK_BYTES: usize = 3 * 65536 * 2;

fn codebook_offset(codebook: u32, state: u16) -> Option<usize> {
    if codebook >= 3 {
        return None;
    }
    Some((codebook as usize * 65536 + usize::from(state)) * 2)
}

fn codebook_table() -> Vec<u8> {
    let mut table = vec![0; CODEBOOK_BYTES];
    for cb in [Codebook::Default, Codebook::Mcg, Codebook::Mul1] {
        for state in 0..=u16::MAX {
            let offset = codebook_offset(cb as u32, state).expect("valid codebook");
            table[offset..offset + 2]
                .copy_from_slice(&codec::decode_codeword(state, cb).to_bits().to_le_bytes());
        }
    }
    table
}

const MAX_DENSE_CHUNKS: usize = 128;

fn exl3_groups(outputs: usize) -> Option<u32> {
    ((16..=32768).contains(&outputs) && outputs.is_multiple_of(16)).then_some((outputs / 4) as u32)
}

fn prefer_tiled(scalar: [f64; 4], tiled: [f64; 4], parity: bool) -> bool {
    let median = |mut times: [f64; 4]| {
        if !times.iter().all(|t| t.is_finite() && *t > 0. && *t < 30.) {
            return None;
        }
        times.sort_by(f64::total_cmp);
        (times[3] <= times[0] * 1.5).then_some((times[1] + times[2]) / 2.)
    };
    parity
        && scalar
            .iter()
            .zip(tiled)
            .filter(|(a, b)| *b < **a * 0.95)
            .count()
            >= 3
        && matches!((median(scalar), median(tiled)), (Some(a), Some(b)) if b < a * 0.95)
}

fn dense_batch_output_count(rows: &[usize]) -> Option<usize> {
    if rows.is_empty() || rows.len() > MAX_DENSE_CHUNKS {
        return None;
    }
    let mut total = 0usize;
    for &row in rows {
        if row == 0 || row > 32768 {
            return None;
        }
        total = total.checked_add(row)?;
        if total > 128000 {
            return None;
        }
    }
    Some(total)
}

struct Dispatch<'a> {
    weights: &'a Buffer,
    output: &'a Buffer,
    params: [u32; 4],
    outputs: usize,
    groups: u32,
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
    mapped: Option<*mut std::ffi::c_void>,
}
// Buffers move with the serialized engine; the raw mapping deliberately keeps them !Sync.
unsafe impl Send for Buffer {}
impl Buffer {
    fn with_mapping<T>(&self, action: impl FnOnce(*mut std::ffi::c_void) -> T) -> Result<T> {
        ensure!(
            !self.device.abandoned.load(Ordering::Relaxed),
            "GPU timed out; restart the app"
        );
        if let Some(ptr) = self.mapped {
            return Ok(action(ptr));
        }
        unsafe {
            let ptr = self.device.raw.map_memory(
                self.memory,
                0,
                self.size as u64,
                vk::MemoryMapFlags::empty(),
            )?;
            let result = action(ptr);
            self.device.raw.unmap_memory(self.memory);
            Ok(result)
        }
    }
    pub fn write(&self, bytes: &[u8]) -> Result<()> {
        ensure!(
            bytes.len() <= self.size,
            "GPU buffer write exceeds allocation"
        );
        ensure!(
            !self.device.abandoned.load(Ordering::Relaxed),
            "GPU timed out; restart the app"
        );
        self.with_mapping(|ptr| unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.cast(), bytes.len());
        })?;
        Ok(())
    }
    fn read_f32(&self, count: usize) -> Result<Vec<f32>> {
        ensure!(
            count.checked_mul(4).is_some_and(|n| n <= self.size),
            "GPU buffer read exceeds allocation"
        );
        let mut output = vec![0.; count];
        self.with_mapping(|ptr| unsafe {
            std::ptr::copy_nonoverlapping(ptr.cast::<f32>(), output.as_mut_ptr(), count);
        })?;
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
                if self.mapped.is_some() {
                    self.device.raw.unmap_memory(self.memory);
                }
                self.device.raw.destroy_buffer(self.raw, None);
                self.device.raw.free_memory(self.memory, None);
            }
        }
    }
}

pub struct Gpu {
    device: Arc<Device>,
    properties: vk::PhysicalDeviceProperties,
    queue: vk::Queue,
    pool: vk::CommandPool,
    command: vk::CommandBuffer,
    fence: vk::Fence,
    set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptors: Vec<vk::DescriptorSet>,
    layout: vk::PipelineLayout,
    qmv: vk::Pipeline,
    qmv_tiled: [vk::Pipeline; 8],
    tuned: BTreeMap<(usize, usize, u32, u32), bool>,
    dense: vk::Pipeline,
    input: Option<Buffer>,
    output: Option<Buffer>,
    codebooks: Option<Buffer>,
    batch_outputs: Vec<Buffer>,
    submissions: Cell<u64>,
    profile: Cell<Profile>,
    used_memory_types: Cell<u32>,
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
            properties: props,
            queue,
            pool: vk::CommandPool::null(),
            command: vk::CommandBuffer::null(),
            fence: vk::Fence::null(),
            set_layout: vk::DescriptorSetLayout::null(),
            descriptor_pool: vk::DescriptorPool::null(),
            descriptors: vec![],
            layout: vk::PipelineLayout::null(),
            qmv: vk::Pipeline::null(),
            qmv_tiled: [vk::Pipeline::null(); 8],
            tuned: BTreeMap::new(),
            dense: vk::Pipeline::null(),
            input: None,
            output: None,
            codebooks: None,
            batch_outputs: vec![],
            submissions: Cell::new(0),
            profile: Cell::new(Profile::default()),
            used_memory_types: Cell::new(0),
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
                descriptor_count: (4 * MAX_DENSE_CHUNKS) as u32,
            }];
            gpu.descriptor_pool = d.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(MAX_DENSE_CHUNKS as u32)
                    .pool_sizes(&sizes),
                None,
            )?;
            let layouts = [gpu.set_layout; MAX_DENSE_CHUNKS];
            gpu.descriptors = d.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(gpu.descriptor_pool)
                    .set_layouts(&layouts),
            )?;
            let ranges = [vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .size(16)];
            gpu.layout = d.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&layouts[..1])
                    .push_constant_ranges(&ranges),
                None,
            )?;
        }
        gpu.qmv = gpu.pipeline(include_bytes!(concat!(env!("OUT_DIR"), "/qmv.spv")))?;
        gpu.dense = gpu.pipeline(include_bytes!(concat!(env!("OUT_DIR"), "/dense.spv")))?;
        gpu.input = Some(gpu.scratch_buffer()?);
        gpu.output = Some(gpu.scratch_buffer()?);
        gpu.codebooks = Some(gpu.buffer(&codebook_table())?);
        Ok(gpu)
    }

    fn pipeline(&self, bytes: &[u8]) -> Result<vk::Pipeline> {
        self.specialized_pipeline(bytes, None)
    }

    fn specialized_pipeline(&self, bytes: &[u8], bits: Option<u32>) -> Result<vk::Pipeline> {
        let code = ash::util::read_spv(&mut std::io::Cursor::new(bytes))?;
        unsafe {
            let module = self
                .device
                .raw
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)?;
            let data = bits.unwrap_or(4).to_ne_bytes();
            let entries = [vk::SpecializationMapEntry::default()
                .constant_id(0)
                .offset(0)
                .size(4)];
            let specialization = vk::SpecializationInfo::default()
                .map_entries(&entries)
                .data(&data);
            let mut stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(c"main");
            if bits.is_some() {
                stage = stage.specialization_info(&specialization);
            }
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
            let flags: Vec<_> = self.device.memory.memory_types
                [..self.device.memory.memory_type_count as usize]
                .iter()
                .map(|t| t.property_flags.as_raw())
                .collect();
            let ty = choose_memory_type(&flags, req.memory_type_bits);
            let Some(ty) = ty else {
                d.destroy_buffer(raw, None);
                bail!("No host-visible coherent Vulkan memory");
            };
            let memory = match d.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(ty as u32),
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
                mapped: None,
            };
            d.bind_buffer_memory(raw, memory, 0)?;
            buffer.write(bytes)?;
            self.allocated += size;
            self.used_memory_types
                .set(self.used_memory_types.get() | (1 << ty));
            Ok(buffer)
        }
    }

    fn scratch_buffer(&mut self) -> Result<Buffer> {
        let mut buffer = self.buffer(&vec![0; 32768 * 4])?;
        buffer.mapped = Some(unsafe {
            self.device.raw.map_memory(
                buffer.memory,
                0,
                buffer.size as u64,
                vk::MemoryMapFlags::empty(),
            )?
        });
        Ok(buffer)
    }

    pub(crate) fn record_profile(&self, update: impl FnOnce(&mut Profile)) {
        let mut profile = self.profile.get();
        update(&mut profile);
        self.profile.set(profile);
    }

    pub(crate) fn take_profile(&self) -> Profile {
        self.profile.replace(Profile::default())
    }

    pub fn diagnostics(&self) -> serde_json::Value {
        let types: Vec<_> = (0..self.device.memory.memory_type_count)
            .filter(|i| self.used_memory_types.get() & (1 << i) != 0)
            .map(|i| {
                let t = self.device.memory.memory_types[i as usize];
                serde_json::json!({"index":i,"flags":t.property_flags.as_raw(),"heap":t.heap_index})
            })
            .collect();
        serde_json::json!({"device":self.name,"device_type":self.properties.device_type.as_raw(),
            "api_version":self.properties.api_version,"driver_version":self.properties.driver_version,
            "vendor_id":self.properties.vendor_id,"device_id":self.properties.device_id,
            "logical_buffer_bytes":self.allocated,"codebook_bytes":CODEBOOK_BYTES,"memory_types":types,
            "kernel":"scalar_reference","automatic_calibration":false,"scratch_mapping":"persistent"})
    }

    fn validate_projection(
        &self,
        weights: &Buffer,
        x: &[f32],
        outputs: usize,
        bits: u32,
        codebook: u32,
    ) -> Result<[u32; 4]> {
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
        Ok([
            x.len() as u32,
            outputs as u32,
            if bits == 0 { codebook } else { bits },
            codebook,
        ])
    }

    pub fn submission_count(&self) -> u64 {
        self.submissions.get()
    }

    fn tiled_pipeline(&mut self, bits: u32) -> Result<vk::Pipeline> {
        ensure!((1..=8).contains(&bits), "Invalid specialized EXL3 bitrate");
        let index = bits as usize - 1;
        if self.qmv_tiled[index] == vk::Pipeline::null() {
            self.qmv_tiled[index] = self.specialized_pipeline(
                include_bytes!(concat!(env!("OUT_DIR"), "/qmv_tiled.spv")),
                Some(bits),
            )?;
        }
        Ok(self.qmv_tiled[index])
    }

    pub fn prepare_exl3(
        &mut self,
        weights: &Buffer,
        inputs: usize,
        outputs: usize,
        bits: u32,
        cb: u32,
    ) -> Result<()> {
        ensure!(
            (1..=32768).contains(&inputs) && (1..=8).contains(&bits),
            "Invalid EXL3 tuning dimensions"
        );
        let key = (inputs, outputs, bits, cb);
        if self.tuned.contains_key(&key) {
            return Ok(());
        }
        let x: Vec<_> = (0..inputs)
            .map(|i| ((i * 13 % 17) as f32 - 8.) / 64.)
            .collect();
        self.validate_projection(weights, &x, outputs, bits, cb)?;
        let scalar = self.multiply_kernel(weights, &x, outputs, bits, cb, false)?;
        let candidate = self.multiply_kernel(weights, &x, outputs, bits, cb, true);
        let candidate = match candidate {
            Ok(values) => values,
            Err(err) if self.device.abandoned.load(Ordering::Relaxed) => return Err(err),
            Err(_) => {
                self.tuned.insert(key, false);
                return Ok(());
            }
        };
        let equal = |a: &[f32], b: &[f32]| {
            a.len() == outputs
                && b.len() == outputs
                && a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits())
        };
        if !equal(&scalar, &candidate) {
            self.tuned.insert(key, false);
            return Ok(());
        }
        let mut times = [[0.; 2]; 4];
        let mut parity = true;
        for (pair, samples) in times.iter_mut().enumerate() {
            for tiled in if pair % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            } {
                let start = Instant::now();
                let values = self.multiply_kernel(weights, &x, outputs, bits, cb, tiled)?;
                samples[usize::from(tiled)] = start.elapsed().as_secs_f64();
                parity &= equal(&scalar, &values);
            }
        }
        self.tuned.insert(
            key,
            prefer_tiled(times.map(|p| p[0]), times.map(|p| p[1]), parity),
        );
        Ok(())
    }

    pub fn tuned_shapes(&self) -> (usize, usize) {
        (
            self.tuned.values().filter(|v| **v).count(),
            self.tuned.len(),
        )
    }

    pub fn multiply(
        &mut self,
        weights: &Buffer,
        x: &[f32],
        outputs: usize,
        bits: u32,
        codebook: u32,
    ) -> Result<Vec<f32>> {
        self.multiply_kernel(weights, x, outputs, bits, codebook, false)
    }

    fn multiply_kernel(
        &mut self,
        weights: &Buffer,
        x: &[f32],
        outputs: usize,
        bits: u32,
        codebook: u32,
        tiled: bool,
    ) -> Result<Vec<f32>> {
        let params = self.validate_projection(weights, x, outputs, bits, codebook)?;
        if tiled {
            let groups = exl3_groups(outputs).context("Invalid tiled EXL3 output shape")?;
            let pipeline = self.tiled_pipeline(bits)?;
            return self.compute_groups(pipeline, weights, x, params, outputs, groups);
        }
        self.compute(
            if bits == 0 { self.dense } else { self.qmv },
            weights,
            x,
            params,
            outputs,
        )
    }

    pub fn multiply_dense_chunks(
        &mut self,
        chunks: &[(Buffer, usize)],
        x: &[f32],
        dtype: u32,
    ) -> Result<Vec<f32>> {
        if chunks.len() == 1 {
            return self.multiply(&chunks[0].0, x, chunks[0].1, 0, dtype);
        }
        let rows: Vec<_> = chunks.iter().map(|(_, rows)| *rows).collect();
        let total = dense_batch_output_count(&rows).context("Invalid dense chunk batch")?;
        let params = chunks
            .iter()
            .map(|(weights, rows)| self.validate_projection(weights, x, *rows, 0, dtype))
            .collect::<Result<Vec<_>>>()?;
        while self.batch_outputs.len() < chunks.len() {
            let output = self.scratch_buffer()?;
            self.batch_outputs.push(output);
        }
        let jobs: Vec<_> = chunks
            .iter()
            .zip(&params)
            .zip(&self.batch_outputs)
            .map(|(((weights, rows), params), output)| Dispatch {
                weights,
                output,
                params: *params,
                outputs: *rows,
                groups: *rows as u32,
            })
            .collect();
        self.execute(self.dense, x, &jobs)?;
        let started = Instant::now();
        let mut result = Vec::with_capacity(total);
        for job in jobs {
            result.extend(job.output.read_f32(job.outputs)?);
        }
        self.record_profile(|p| p.readback_ns = p.readback_ns.saturating_add(elapsed_ns(started)));
        Ok(result)
    }

    fn compute(
        &mut self,
        pipeline: vk::Pipeline,
        weights: &Buffer,
        x: &[f32],
        params: [u32; 4],
        outputs: usize,
    ) -> Result<Vec<f32>> {
        self.compute_groups(pipeline, weights, x, params, outputs, outputs as u32)
    }

    fn compute_groups(
        &mut self,
        pipeline: vk::Pipeline,
        weights: &Buffer,
        x: &[f32],
        params: [u32; 4],
        outputs: usize,
        groups: u32,
    ) -> Result<Vec<f32>> {
        let output = self.output.as_ref().context("GPU output unavailable")?;
        self.execute(
            pipeline,
            x,
            &[Dispatch {
                weights,
                output,
                params,
                outputs,
                groups,
            }],
        )?;
        let started = Instant::now();
        let result = output.read_f32(outputs);
        self.record_profile(|p| p.readback_ns = p.readback_ns.saturating_add(elapsed_ns(started)));
        result
    }

    fn execute(&self, pipeline: vk::Pipeline, x: &[f32], jobs: &[Dispatch<'_>]) -> Result<()> {
        ensure!(
            !self.device.abandoned.load(Ordering::Relaxed),
            "GPU timed out; restart the app"
        );
        ensure!(
            !x.is_empty()
                && x.len() <= 32768
                && !jobs.is_empty()
                && jobs.len() <= self.descriptors.len()
                && jobs.iter().all(|job| job.outputs > 0
                    && job.outputs <= 32768
                    && job.groups > 0
                    && job.groups <= self.max_groups),
            "Invalid compute buffer dimensions"
        );
        let input = self.input.as_ref().context("GPU input unavailable")?;
        let codebooks = self
            .codebooks
            .as_ref()
            .context("GPU codebooks unavailable")?;
        let bytes = unsafe {
            std::slice::from_raw_parts(x.as_ptr().cast::<u8>(), std::mem::size_of_val(x))
        };
        let started = Instant::now();
        input.write(bytes)?;
        self.record_profile(|p| p.upload_ns = p.upload_ns.saturating_add(elapsed_ns(started)));
        let recording = Instant::now();
        unsafe {
            let d = &self.device.raw;
            d.reset_command_buffer(self.command, vk::CommandBufferResetFlags::empty())?;
            d.reset_fences(&[self.fence])?;
            d.begin_command_buffer(
                self.command,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            d.cmd_bind_pipeline(self.command, vk::PipelineBindPoint::COMPUTE, pipeline);
            for (index, job) in jobs.iter().enumerate() {
                // Each dispatch owns a descriptor set and output until the shared fence completes.
                let infos = [
                    [vk::DescriptorBufferInfo::default()
                        .buffer(input.raw)
                        .range(input.size as u64)],
                    [vk::DescriptorBufferInfo::default()
                        .buffer(job.weights.raw)
                        .range(job.weights.size as u64)],
                    [vk::DescriptorBufferInfo::default()
                        .buffer(job.output.raw)
                        .range(job.output.size as u64)],
                    [vk::DescriptorBufferInfo::default()
                        .buffer(codebooks.raw)
                        .range(codebooks.size as u64)],
                ];
                let descriptor = self.descriptors[index];
                let writes: Vec<_> = infos
                    .iter()
                    .enumerate()
                    .map(|(binding, info)| {
                        vk::WriteDescriptorSet::default()
                            .dst_set(descriptor)
                            .dst_binding(binding as u32)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .buffer_info(info)
                    })
                    .collect();
                d.update_descriptor_sets(&writes, &[]);
                d.cmd_bind_descriptor_sets(
                    self.command,
                    vk::PipelineBindPoint::COMPUTE,
                    self.layout,
                    0,
                    &[descriptor],
                    &[],
                );
                let push = std::slice::from_raw_parts(job.params.as_ptr().cast::<u8>(), 16);
                d.cmd_push_constants(
                    self.command,
                    self.layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    push,
                );
                d.cmd_dispatch(self.command, job.groups, 1, 1);
            }
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
            self.submissions.set(self.submissions.get().wrapping_add(1));
            self.record_profile(|p| {
                p.record_submit_ns = p.record_submit_ns.saturating_add(elapsed_ns(recording))
            });
            let waiting = Instant::now();
            let waited = d.wait_for_fences(&[self.fence], true, 30_000_000_000);
            self.record_profile(|p| p.wait_ns = p.wait_ns.saturating_add(elapsed_ns(waiting)));
            if let Err(err) = waited {
                // Pending work may still own buffers; retain resources until process exit.
                self.device.abandoned.store(true, Ordering::Relaxed);
                bail!("Vulkan dispatch failed ({err:?}); restart the app");
            }
        }
        Ok(())
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
            for pipeline in self.qmv_tiled {
                d.destroy_pipeline(pipeline, None);
            }
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
    fn memory_selection_preserves_requirements_and_prefers_local() {
        assert_eq!(choose_memory_type(&[6, 7, 7], 7), Some(1));
        assert_eq!(
            choose_memory_type(&[140, 142, 142, u32::MAX], u32::MAX),
            Some(3)
        );
        assert_eq!(choose_memory_type(&[6, 7], 1), Some(0));
        assert_eq!(choose_memory_type(&[1, 2, 4], u32::MAX), None);
        assert_eq!(choose_memory_type(&[], u32::MAX), None);
        assert_eq!(choose_memory_type(&[7; 32], 1 << 31), Some(31));
        assert_eq!(choose_memory_type(&[7; 33], 0), None);
    }

    #[test]
    #[ignore = "requires Vulkan; persistent scratch mappings and checked copies"]
    fn persistent_scratch_mapping_survives_reuse_and_errors() {
        let mut gpu = Gpu::new().unwrap();
        let buffer = gpu.scratch_buffer().unwrap();
        let address = buffer.mapped.unwrap();
        for values in [[1f32, -2., 3.], [4., 5., 6.]] {
            let bytes: Vec<_> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
            buffer.write(&bytes).unwrap();
            assert_eq!(buffer.read_f32(3).unwrap(), values);
            assert_eq!(buffer.mapped, Some(address));
        }
        assert!(buffer.write(&vec![0; buffer.size + 1]).is_err());
        assert!(buffer.read_f32(usize::MAX).is_err());
        assert!(buffer.read_f32(buffer.size / 4 + 1).is_err());
        buffer.write(&1f32.to_ne_bytes()).unwrap();
        assert_eq!(buffer.read_f32(1).unwrap(), [1.]);
    }

    #[test]
    fn tiled_selection_requires_parity_stability_and_a_clear_gain() {
        let scalar = [0.010, 0.0101, 0.0102, 0.0103];
        let faster = [0.008, 0.0081, 0.0082, 0.0083];
        assert!(prefer_tiled(scalar, faster, true));
        assert!(!prefer_tiled(scalar, faster, false));
        assert!(!prefer_tiled(scalar, scalar, true));
        assert!(!prefer_tiled(faster, scalar, true));
        assert!(!prefer_tiled(scalar, [0.01, 0.01, 0.01, 0.01], true));
        assert!(!prefer_tiled(
            [0.01; 4],
            [0.0075, 0.0075, 0.0105, 0.0105],
            true
        ));
        for bad in [0., -1., f64::NAN, f64::INFINITY, 30., f64::MAX] {
            let mut invalid = faster;
            invalid[2] = bad;
            assert!(!prefer_tiled(scalar, invalid, true));
            assert!(!prefer_tiled(invalid, faster, true));
        }
        assert!(!prefer_tiled(scalar, [0.001, 0.008, 0.008, 0.008], true));
        assert!(!prefer_tiled([0.001, 0.01, 0.01, 0.01], faster, true));
        assert_eq!(exl3_groups(0), None);
        assert_eq!(exl3_groups(15), None);
        assert_eq!(exl3_groups(16), Some(4));
        assert_eq!(exl3_groups(32768), Some(8192));
        assert_eq!(exl3_groups(32769), None);
        assert_eq!(exl3_groups(usize::MAX), None);
    }

    #[test]
    #[ignore = "requires Vulkan; specialized four-output kernel parity and tuning wiring"]
    fn tiled_qmv_matches_scalar_all_bitrates_and_codebooks() {
        let mut gpu = Gpu::new().unwrap();
        for (rows, cols) in [(128, 128), (256, 384), (2048, 512)] {
            let x: Vec<_> = (0..rows)
                .map(|i| ((i * 13 % 17) as f32 - 8.) / 64.)
                .collect();
            for bits in 1..=8 {
                let states: Vec<_> = (0..rows * cols)
                    .map(|i| ((i * 177 + i / 7 * 313 + 512) & 65535) as u16)
                    .collect();
                let bytes: Vec<_> = codec::pack(&states, bits)
                    .unwrap()
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect();
                let buffer = gpu.buffer(&bytes).unwrap();
                for cb in 0..3 {
                    let expected = gpu
                        .multiply_kernel(&buffer, &x, cols, bits as u32, cb, false)
                        .unwrap();
                    let actual = gpu
                        .multiply_kernel(&buffer, &x, cols, bits as u32, cb, true)
                        .unwrap();
                    assert_eq!(actual.len(), cols);
                    assert!(actual.iter().any(|v| v.abs() > 0.01));
                    for (a, b) in actual.iter().zip(&expected) {
                        assert_eq!(a.to_bits(), b.to_bits(), "{rows}x{cols}, K{bits}, cb{cb}");
                    }
                    if (rows, cols, bits, cb) == (128, 128, 4, 0) {
                        gpu.prepare_exl3(&buffer, rows, cols, bits as u32, cb)
                            .unwrap();
                        assert_eq!(gpu.tuned_shapes().1, 1);
                        let before = gpu.submission_count();
                        gpu.prepare_exl3(&buffer, rows, cols, bits as u32, cb)
                            .unwrap();
                        assert_eq!(gpu.submission_count(), before);
                        gpu.tuned.insert((rows, cols, bits as u32, cb), true);
                        assert_eq!(
                            gpu.multiply(&buffer, &x, cols, bits as u32, cb).unwrap(),
                            expected
                        );
                        assert!(
                            gpu.prepare_exl3(&buffer, usize::MAX, cols, bits as u32, cb)
                                .is_err()
                        );
                        assert!(
                            gpu.multiply_kernel(&buffer, &x[..127], cols, bits as u32, cb, true)
                                .is_err()
                        );
                        assert!(gpu.multiply_kernel(&buffer, &x, cols, 0, cb, true).is_err());
                    }
                }
            }
        }
    }

    #[test]
    fn dense_batch_bounds_reject_empty_zero_large_and_over_budget() {
        assert_eq!(dense_batch_output_count(&[]), None);
        assert_eq!(dense_batch_output_count(&[0]), None);
        assert_eq!(dense_batch_output_count(&[usize::MAX]), None);
        assert_eq!(dense_batch_output_count(&[1; 128]), Some(128));
        assert_eq!(dense_batch_output_count(&[1; 129]), None);
        assert_eq!(
            dense_batch_output_count(&[32768, 32768, 32768, 29696]),
            Some(128000)
        );
        assert_eq!(dense_batch_output_count(&[32768; 4]), None);
    }

    #[test]
    fn codebook_table_has_all_finite_values_and_checked_offsets() {
        let table = codebook_table();
        assert_eq!(table.len(), CODEBOOK_BYTES);
        let (values, tail) = table.as_chunks::<2>();
        assert!(tail.is_empty());
        assert!(
            values
                .iter()
                .all(|b| half::f16::from_bits(u16::from_le_bytes(*b)).is_finite())
        );
        assert_eq!(codebook_offset(0, 0), Some(0));
        assert_eq!(codebook_offset(2, u16::MAX), Some(CODEBOOK_BYTES - 2));
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
    #[kani::unwind(7)]
    fn memory_selection_checks_flags_and_prefers_local_for_four_types() {
        let flags: [u32; 4] = kani::any();
        let allowed: u32 = kani::any();
        let index = choose_memory_type(&flags, allowed);
        let mut eligible = false;
        for i in 0..4 {
            let valid = allowed & (1 << i) != 0 && flags[i] & 6 == 6;
            eligible |= valid;
            if valid {
                assert!(index.is_some());
                let selected = index.unwrap();
                assert!(selected < 4 && allowed & (1 << selected) != 0 && flags[selected] & 6 == 6);
                if flags[i] & 1 != 0 {
                    assert!(flags[selected] & 1 != 0);
                }
            }
        }
        assert_eq!(index.is_some(), eligible);
        kani::cover!(index == Some(3) && flags[0] & 6 == 6 && flags[0] & 1 == 0);
        kani::cover!(index.is_none());
    }

    #[kani::proof]
    fn tiled_groups_cover_exactly_valid_outputs() {
        let outputs: usize = kani::any();
        let groups = exl3_groups(outputs);
        let valid = outputs >= 16 && outputs <= 32768 && outputs % 16 == 0;
        assert_eq!(groups.is_some(), valid);
        if let Some(groups) = groups {
            assert!(groups >= 4 && groups <= 8192);
            assert_eq!(groups as usize * 4, outputs);
        }
        kani::cover!(outputs == 16 && groups == Some(4));
        kani::cover!(outputs == 32768 && groups == Some(8192));
    }

    #[kani::proof]
    #[kani::unwind(6)]
    fn dense_batch_count_is_bounded_for_four_symbolic_chunks() {
        let rows: [usize; 4] = kani::any();
        let len: usize = kani::any();
        kani::assume(len <= 4);
        let rows = &rows[..len];
        let actual = dense_batch_output_count(rows);
        let mut reference = 0u128;
        let mut valid = len > 0;
        for &row in rows {
            reference += row as u128;
            valid &= row > 0 && row <= 32768;
        }
        valid &= reference <= 128000;
        assert_eq!(actual.is_some(), valid);
        if let Some(total) = actual {
            assert_eq!(total as u128, reference);
            assert!(total <= 128000);
        }
        kani::cover!(len == 0 && actual.is_none());
        kani::cover!(len == 4 && actual == Some(128000));
    }

    #[kani::proof]
    fn codebook_offsets_are_aligned_bounded_and_reject_invalid_ids() {
        let cb: u32 = kani::any();
        let state: u16 = kani::any();
        let offset = codebook_offset(cb, state);
        assert_eq!(offset.is_some(), cb < 3);
        if let Some(offset) = offset {
            assert_eq!(offset % 2, 0);
            assert!(offset + 2 <= CODEBOOK_BYTES);
            assert_eq!(offset / 2 / 65536, cb as usize);
            assert_eq!(offset / 2 % 65536, usize::from(state));
        }
        kani::cover!(cb == 2 && state == u16::MAX);
        kani::cover!(cb == 0 && state == 0);
        kani::cover!(cb == u32::MAX && offset.is_none());
    }
}
