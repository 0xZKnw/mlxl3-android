//! Synchronous batch-one compute; no subgroup-size or FP16 extension requirement.
use crate::codec::{self, Codebook};
use anyhow::{Context, Result, bail, ensure};
use ash::{Entry, vk};
use std::{
    cell::{Cell, RefCell},
    ffi::{CStr, CString},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU32, Ordering},
    },
    time::Instant,
};

pub use crate::profiling::Profile;
use crate::profiling::ProjectionMeta;

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
static REQUESTED_SPLIT_K: AtomicU32 = AtomicU32::new(4);

pub fn set_split_k(value: u32) -> bool {
    if ![1, 2, 4, 8].contains(&value) {
        return false;
    }
    REQUESTED_SPLIT_K.store(value, Ordering::Relaxed);
    true
}

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
const MAX_EXL3_MATRICES: usize = 24 * 8 + 1;
const MAX_DESCRIPTOR_SETS: usize = 2 * MAX_DENSE_CHUNKS + 3 * MAX_EXL3_MATRICES + 48;
const QUERY_COUNT: u32 = 4 * (MAX_EXL3_MATRICES as u32 + 1);

fn projection_queries(index: usize) -> Option<u32> {
    (index < MAX_EXL3_MATRICES).then(|| (index as u32 + 1) * 4)
}

fn valid_group_size(count: usize) -> bool {
    (1..=3).contains(&count)
}

fn exl3_groups(outputs: usize) -> Option<u32> {
    ((16..=32768).contains(&outputs) && outputs.is_multiple_of(16)).then_some((outputs / 16) as u32)
}

fn dense_groups(outputs: usize) -> Option<u32> {
    (1..=32768)
        .contains(&outputs)
        .then(|| outputs.div_ceil(4) as u32)
}

fn convolution_state_len(hidden: usize) -> Option<usize> {
    (1..=32768 / 3).contains(&hidden).then(|| 1 + hidden * 14)
}

fn can_force_wave32(min: u32, max: u32, stages: u32, max_subgroups: u32) -> bool {
    min > 0 && min <= 32 && max >= 32 && stages & 32 != 0 && max_subgroups > 0
}

fn wave_kernel_label(split: u32, cooperative: u32) -> &'static str {
    match (split, cooperative) {
        (1, 4) => "wave32_split1_cooperative_k4",
        (2, 4) => "wave32_split2_cooperative_k4",
        (4, 4) => "wave32_split4_cooperative_k4",
        (8, 4) => "wave32_split8_cooperative_k4",
        (1, 5) => "wave32_split1_cooperative_k5",
        (2, 5) => "wave32_split2_cooperative_k5",
        (4, 5) => "wave32_split4_cooperative_k5",
        (8, 5) => "wave32_split8_cooperative_k5",
        (1, _) => "wave32_split1_native_f16",
        (2, _) => "wave32_split2_native_f16",
        (8, _) => "wave32_split8_native_f16",
        _ => "wave32_split4_native_f16",
    }
}

pub(crate) struct MlpPlan {
    gate: Exl3Matrix,
    up: Exl3Matrix,
    down: Exl3Matrix,
    activate: vk::CommandBuffer,
}

pub(crate) struct ConvPlan {
    input: Exl3Matrix,
    output: Exl3Matrix,
    state: Buffer,
    convolve: vk::CommandBuffer,
}

pub(crate) struct DenseHeadPlan {
    chunks: Vec<(Buffer, usize)>,
    outputs: Vec<Buffer>,
    input_len: usize,
    total: usize,
    command: vk::CommandBuffer,
}

pub(crate) struct DenseGatePlan {
    device: Arc<Device>,
    _weights: Buffer,
    input_len: usize,
    outputs: usize,
    command: vk::CommandBuffer,
}

fn timestamp_delta(start: u64, end: u64, bits: u32) -> Option<u64> {
    match bits {
        1..=63 => Some(end.wrapping_sub(start) & ((1u64 << bits) - 1)),
        64 => Some(end.wrapping_sub(start)),
        _ => None,
    }
}

#[derive(Clone, Copy)]
pub struct Exl3Shape {
    pub input: usize,
    pub output: usize,
    pub rows: usize,
    pub cols: usize,
    pub bits: u32,
    pub codebook: u32,
}
impl Exl3Shape {
    fn valid(self) -> bool {
        self.input > 0
            && self.output > 0
            && self.rows <= 32768
            && self.cols <= 32768
            && self.input <= self.rows
            && self.rows - self.input < 128
            && self.output <= self.cols
            && self.cols - self.output < 128
            && self.rows.is_multiple_of(128)
            && self.cols.is_multiple_of(128)
            && (1..=8).contains(&self.bits)
            && self.codebook < 3
    }
}

pub struct Exl3Matrix {
    weights: Buffer,
    _scales: Buffer,
    output: Buffer,
    shape: Exl3Shape,
    command: vk::CommandBuffer,
    query_first: u32,
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
    input: &'a Buffer,
    weights: &'a Buffer,
    output: &'a Buffer,
    params: [u32; 4],
    outputs: usize,
    groups: u32,
    groups_y: u32,
    pipeline: vk::Pipeline,
    barrier_after: bool,
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
    capabilities: serde_json::Value,
    amd_info: Option<ash::amd::shader_info::Device>,
    compiler_stats: Vec<serde_json::Value>,
    queue: vk::Queue,
    pool: vk::CommandPool,
    command: vk::CommandBuffer,
    host_readback: vk::CommandBuffer,
    queue_start: vk::CommandBuffer,
    fence: vk::Fence,
    set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptors: Vec<vk::DescriptorSet>,
    layout: vk::PipelineLayout,
    qmv: vk::Pipeline,
    qmv_stream: [vk::Pipeline; 8],
    native_float16: bool,
    native_verified: bool,
    native_self_test: serde_json::Value,
    native_kernels: [bool; 8],
    wave_kernels: [bool; 8],
    force_wave32: bool,
    allow_wave32: bool,
    allow_cooperative_k4: bool,
    cooperative_k4_active: bool,
    cooperative_k4_self_test: &'static str,
    cooperative_k5_active: bool,
    cooperative_k5_self_test: &'static str,
    split_k: u32,
    native_pipeline_errors: Vec<String>,
    hadamard: vk::Pipeline,
    prepared: usize,
    queries: vk::QueryPool,
    timestamp_bits: u32,
    dense: vk::Pipeline,
    reduce: vk::Pipeline,
    swiglu: vk::Pipeline,
    convolution: vk::Pipeline,
    partials: Option<Buffer>,
    input: Option<Buffer>,
    rotated: Option<Buffer>,
    output: Option<Buffer>,
    codebooks: Option<Buffer>,
    batch_outputs: Vec<Buffer>,
    gate_input: Option<Buffer>,
    gate_output: Option<Buffer>,
    submissions: Cell<u64>,
    profile: RefCell<Profile>,
    projection_meta: Vec<ProjectionMeta>,
    used_memory_types: Cell<u32>,
    pub name: String,
    pub max_range: usize,
    max_groups: u32,
    allocated: usize,
}

impl Gpu {
    pub fn new() -> Result<Self> {
        Self::new_with_native(true)
    }

    fn new_with_native(allow_native: bool) -> Result<Self> {
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
                chosen = Some((
                    physical,
                    family as u32,
                    props,
                    families[family].timestamp_valid_bits,
                ));
                break;
            }
        }
        let (physical, family, props, timestamp_bits) =
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
        let mut subgroup = vk::PhysicalDeviceSubgroupProperties::default();
        let mut properties2 = vk::PhysicalDeviceProperties2::default().push_next(&mut subgroup);
        unsafe {
            instance
                .raw
                .get_physical_device_properties2(physical, &mut properties2)
        };
        let supported_extensions: Vec<_> = device_extensions
            .iter()
            .map(|e| {
                unsafe { CStr::from_ptr(e.extension_name.as_ptr()) }
                    .to_string_lossy()
                    .into_owned()
            })
            .filter(|n| {
                matches!(
                    n.as_str(),
                    "VK_KHR_shader_float16_int8"
                        | "VK_KHR_cooperative_matrix"
                        | "VK_NV_cooperative_matrix"
                        | "VK_EXT_subgroup_size_control"
                        | "VK_KHR_shader_subgroup_extended_types"
                        | "VK_KHR_shader_float_controls"
                        | "VK_KHR_pipeline_executable_properties"
                        | "VK_AMD_shader_info"
                        | "VK_EXT_memory_budget"
                )
            })
            .collect();
        let mut float16 = vk::PhysicalDeviceShaderFloat16Int8Features::default();
        if props.api_version >= vk::API_VERSION_1_2
            || supported_extensions
                .iter()
                .any(|s| s == "VK_KHR_shader_float16_int8")
        {
            let mut features2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut float16);
            unsafe {
                instance
                    .raw
                    .get_physical_device_features2(physical, &mut features2)
            };
        }
        let has_extension = |name: &str| {
            device_extensions.iter().any(|e| {
                unsafe { CStr::from_ptr(e.extension_name.as_ptr()) }.to_bytes() == name.as_bytes()
            })
        };
        let mut controls = vk::PhysicalDeviceFloatControlsProperties::default();
        let mut driver = vk::PhysicalDeviceDriverProperties::default();
        if props.api_version >= vk::API_VERSION_1_2 || has_extension("VK_KHR_shader_float_controls")
        {
            let mut p = vk::PhysicalDeviceProperties2::default().push_next(&mut controls);
            unsafe {
                instance
                    .raw
                    .get_physical_device_properties2(physical, &mut p)
            };
        }
        if props.api_version >= vk::API_VERSION_1_2 || has_extension("VK_KHR_driver_properties") {
            let mut p = vk::PhysicalDeviceProperties2::default().push_next(&mut driver);
            unsafe {
                instance
                    .raw
                    .get_physical_device_properties2(physical, &mut p)
            };
        }
        let mut sizes = vk::PhysicalDeviceSubgroupSizeControlProperties::default();
        let mut size_features = vk::PhysicalDeviceSubgroupSizeControlFeatures::default();
        if props.api_version >= vk::API_VERSION_1_3 || has_extension("VK_EXT_subgroup_size_control")
        {
            let mut p = vk::PhysicalDeviceProperties2::default().push_next(&mut sizes);
            unsafe {
                instance
                    .raw
                    .get_physical_device_properties2(physical, &mut p)
            };
            let mut f = vk::PhysicalDeviceFeatures2::default().push_next(&mut size_features);
            unsafe { instance.raw.get_physical_device_features2(physical, &mut f) };
        }
        let capabilities = serde_json::json!({"subgroup_size":subgroup.subgroup_size,
            "subgroup_operations":subgroup.supported_operations.as_raw(),
            "subgroup_stages":subgroup.supported_stages.as_raw(),
            "shader_float16_supported":float16.shader_float16 == vk::TRUE,
            "optional_extensions":supported_extensions,"extension_count":device_extensions.len(),
            "subgroup_size_control":{"supported":size_features.subgroup_size_control == vk::TRUE,
                "min":sizes.min_subgroup_size,"max":sizes.max_subgroup_size,
                "required_stages":sizes.required_subgroup_size_stages.as_raw(),
                "max_workgroup_subgroups":sizes.max_compute_workgroup_subgroups},
            "float_controls":{"rte_float16":controls.shader_rounding_mode_rte_float16 == vk::TRUE,
                "rtz_float16":controls.shader_rounding_mode_rtz_float16 == vk::TRUE,
                "denorm_preserve_float16":controls.shader_denorm_preserve_float16 == vk::TRUE,
                "denorm_flush_float16":controls.shader_denorm_flush_to_zero_float16 == vk::TRUE},
            "driver":{"id":driver.driver_id.as_raw(),"name":unsafe {CStr::from_ptr(driver.driver_name.as_ptr())}.to_string_lossy(),
                "info":unsafe {CStr::from_ptr(driver.driver_info.as_ptr())}.to_string_lossy()}});
        let portability = device_extensions.iter().any(|e| unsafe { CStr::from_ptr(e.extension_name.as_ptr()) } == ash::khr::portability_subset::NAME);
        let native_float16 = allow_native
            && float16.shader_float16 == vk::TRUE
            && supported_extensions
                .iter()
                .any(|s| s == "VK_KHR_shader_float16_int8");
        let mut extension_names = vec![];
        if portability {
            extension_names.push(ash::khr::portability_subset::NAME.as_ptr());
        }
        if native_float16 {
            extension_names.push(c"VK_KHR_shader_float16_int8".as_ptr());
        }
        let amd_info_enabled = has_extension("VK_AMD_shader_info");
        if amd_info_enabled {
            extension_names.push(c"VK_AMD_shader_info".as_ptr());
        }
        let force_wave32 = has_extension("VK_EXT_subgroup_size_control")
            && size_features.subgroup_size_control == vk::TRUE
            && can_force_wave32(
                sizes.min_subgroup_size,
                sizes.max_subgroup_size,
                sizes.required_subgroup_size_stages.as_raw(),
                sizes.max_compute_workgroup_subgroups,
            );
        if force_wave32 {
            extension_names.push(c"VK_EXT_subgroup_size_control".as_ptr());
        }
        let mut enabled_sizes =
            vk::PhysicalDeviceSubgroupSizeControlFeatures::default().subgroup_size_control(true);
        let mut device_info = vk::DeviceCreateInfo::default().queue_create_infos(&queues);
        if force_wave32 {
            device_info = device_info.push_next(&mut enabled_sizes);
        }
        device_info = device_info.enabled_extension_names(&extension_names);
        let mut enabled_float16 =
            vk::PhysicalDeviceShaderFloat16Int8Features::default().shader_float16(true);
        if native_float16 {
            device_info = device_info.push_next(&mut enabled_float16);
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
        let amd_info = amd_info_enabled
            .then(|| ash::amd::shader_info::Device::new(&device._instance.raw, &device.raw));
        let requested_split_k = REQUESTED_SPLIT_K.load(Ordering::Relaxed);
        let mut gpu = Self {
            device,
            properties: props,
            capabilities,
            amd_info,
            compiler_stats: vec![],
            queue,
            pool: vk::CommandPool::null(),
            command: vk::CommandBuffer::null(),
            host_readback: vk::CommandBuffer::null(),
            queue_start: vk::CommandBuffer::null(),
            fence: vk::Fence::null(),
            set_layout: vk::DescriptorSetLayout::null(),
            descriptor_pool: vk::DescriptorPool::null(),
            descriptors: vec![],
            layout: vk::PipelineLayout::null(),
            qmv: vk::Pipeline::null(),
            qmv_stream: [vk::Pipeline::null(); 8],
            native_float16,
            native_verified: false,
            native_self_test: serde_json::json!({"status":if allow_native {"unsupported"} else {"disabled"},"checked_values":0}),
            native_kernels: [false; 8],
            wave_kernels: [false; 8],
            force_wave32,
            allow_wave32: (force_wave32 || (1..=32).contains(&subgroup.subgroup_size))
                && props.limits.max_compute_work_group_count[1] >= 4,
            allow_cooperative_k4: (force_wave32 || subgroup.subgroup_size == 32)
                && subgroup
                    .supported_stages
                    .contains(vk::ShaderStageFlags::COMPUTE)
                && subgroup
                    .supported_operations
                    .contains(vk::SubgroupFeatureFlags::SHUFFLE)
                && props.limits.max_compute_work_group_count[1] >= 4,
            cooperative_k4_active: false,
            cooperative_k4_self_test: "not_run",
            cooperative_k5_active: false,
            cooperative_k5_self_test: "not_run",
            split_k: if props.limits.max_compute_work_group_count[1] >= requested_split_k
                && (requested_split_k <= 4 || props.limits.max_storage_buffer_range >= 32768 * 32 * 4) {
                requested_split_k
            } else {
                4
            },
            native_pipeline_errors: vec![],
            hadamard: vk::Pipeline::null(),
            prepared: 0,
            queries: vk::QueryPool::null(),
            timestamp_bits,
            dense: vk::Pipeline::null(),
            reduce: vk::Pipeline::null(),
            swiglu: vk::Pipeline::null(),
            convolution: vk::Pipeline::null(),
            partials: None,
            input: None,
            rotated: None,
            output: None,
            codebooks: None,
            batch_outputs: vec![],
            gate_input: None,
            gate_output: None,
            submissions: Cell::new(0),
            profile: RefCell::new(Profile::default()),
            projection_meta: vec![],
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
            let commands = d.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(gpu.pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(3),
            )?;
            gpu.command = commands[0];
            gpu.host_readback = commands[1];
            gpu.queue_start = commands[2];
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
                descriptor_count: (4 * MAX_DESCRIPTOR_SETS) as u32,
            }];
            gpu.descriptor_pool = d.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(MAX_DESCRIPTOR_SETS as u32)
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
            if timestamp_delta(0, 0, timestamp_bits).is_some()
                && props.limits.timestamp_period.is_finite()
                && props.limits.timestamp_period > 0.
            {
                gpu.queries = d
                    .create_query_pool(
                        &vk::QueryPoolCreateInfo::default()
                            .query_type(vk::QueryType::TIMESTAMP)
                            .query_count(QUERY_COUNT),
                        None,
                    )
                    .unwrap_or(vk::QueryPool::null());
            }
        }
        unsafe {
            let d = &gpu.device.raw;
            d.begin_command_buffer(gpu.queue_start, &vk::CommandBufferBeginInfo::default())?;
            if gpu.queries != vk::QueryPool::null() {
                d.cmd_reset_query_pool(gpu.queue_start, gpu.queries, 0, 4);
                d.cmd_write_timestamp(
                    gpu.queue_start,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    gpu.queries,
                    0,
                );
            }
            d.end_command_buffer(gpu.queue_start)?;
            d.begin_command_buffer(gpu.host_readback, &vk::CommandBufferBeginInfo::default())?;
            let barriers = [vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)];
            d.cmd_pipeline_barrier(
                gpu.host_readback,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &barriers,
                &[],
                &[],
            );
            if gpu.queries != vk::QueryPool::null() {
                d.cmd_write_timestamp(
                    gpu.host_readback,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    gpu.queries,
                    1,
                );
            }
            d.end_command_buffer(gpu.host_readback)?;
        }

        gpu.qmv = gpu.pipeline(include_bytes!(concat!(env!("OUT_DIR"), "/qmv.spv")))?;
        gpu.dense = gpu.pipeline(include_bytes!(concat!(env!("OUT_DIR"), "/dense.spv")))?;
        gpu.reduce = gpu.pipeline(include_bytes!(concat!(env!("OUT_DIR"), "/reduce.spv")))?;
        gpu.swiglu = gpu.pipeline(include_bytes!(concat!(env!("OUT_DIR"), "/swiglu.spv")))?;
        gpu.convolution =
            gpu.pipeline(include_bytes!(concat!(env!("OUT_DIR"), "/convolution.spv")))?;
        let partial_bytes = 32768 * 4 * gpu.split_k.max(4) as usize * 4;
        gpu.partials = Some(gpu.buffer(&vec![0; partial_bytes])?);
        gpu.capture_stats("dense_four_rows".into(), gpu.dense);
        gpu.hadamard = gpu.pipeline(include_bytes!(concat!(env!("OUT_DIR"), "/hadamard.spv")))?;
        gpu.input = Some(gpu.scratch_buffer()?);
        gpu.rotated = Some(gpu.scratch_buffer()?);
        gpu.output = Some(gpu.scratch_buffer()?);
        gpu.codebooks = Some(gpu.buffer(&codebook_table())?);
        if native_float16 {
            let started = Instant::now();
            match gpu.check_native_codebooks() {
                Ok(()) => {
                    gpu.native_verified = true;
                    gpu.native_self_test = serde_json::json!({"status":"verified_bit_exact","checked_values":393216,"unique_values":196608,"lanes_tested":2});
                }
                Err(err) => {
                    if gpu.device.abandoned.load(Ordering::Relaxed) {
                        return Err(err);
                    }
                    gpu.native_self_test =
                        serde_json::json!({"status":"fallback","error":err.to_string()});
                }
            }
            gpu.native_self_test["elapsed_ns"] = elapsed_ns(started).into();
            // Startup validation is reported separately from generation counters.
            gpu.take_profile();
            gpu.submissions.set(0);
        }
        Ok(gpu)
    }

    fn check_native_codebooks(&mut self) -> Result<()> {
        let pipeline = self.pipeline(include_bytes!(concat!(
            env!("OUT_DIR"),
            "/codebook_native.spv"
        )))?;
        let result = (|| {
            let dummy = self.buffer(&[0; 4])?;
            for cb in [Codebook::Default, Codebook::Mcg, Codebook::Mul1] {
                for offset in [0, 32768] {
                    for lane in 0..2 {
                        let values = self.compute_groups(
                            pipeline,
                            &dummy,
                            &[0.],
                            [offset, 32768, lane, cb as u32],
                            32768,
                            128,
                        )?;
                        ensure!(values.len() == 32768, "Incomplete native codebook check");
                        for (i, actual) in values.iter().enumerate() {
                            let expected =
                                codec::decode_codeword((offset as usize + i) as u16, cb).to_f32();
                            ensure!(
                                actual.to_bits() == expected.to_bits(),
                                "Native codebook {cb:?}/{}: {actual} != {expected}",
                                offset as usize + i
                            );
                        }
                    }
                }
            }
            Ok(())
        })();
        if !self.device.abandoned.load(Ordering::Relaxed) {
            unsafe { self.device.raw.destroy_pipeline(pipeline, None) };
        }
        result
    }

    fn capture_stats(&mut self, name: String, pipeline: vk::Pipeline) {
        let Some(api) = &self.amd_info else {
            return;
        };
        let mut info = vk::ShaderStatisticsInfoAMD::default();
        let mut size = std::mem::size_of_val(&info);
        let result = unsafe {
            (api.fp().get_shader_info_amd)(
                api.device(),
                pipeline,
                vk::ShaderStageFlags::COMPUTE,
                vk::ShaderInfoTypeAMD::STATISTICS,
                &mut size,
                (&mut info as *mut vk::ShaderStatisticsInfoAMD).cast(),
            )
        };
        let row = if result == vk::Result::SUCCESS && size == std::mem::size_of_val(&info) {
            serde_json::json!({"name":name,"status":"available", "vgprs":info.resource_usage.num_used_vgprs,
                "sgprs":info.resource_usage.num_used_sgprs,"lds_bytes":info.resource_usage.lds_usage_size_in_bytes,
                "scratch_bytes":info.resource_usage.scratch_mem_usage_in_bytes,
                "available_vgprs":info.num_available_vgprs,"available_sgprs":info.num_available_sgprs,
                "workgroup_size":info.compute_work_group_size})
        } else {
            serde_json::json!({"name":name,"status":"unavailable","result":format!("{result:?}"),"bytes":size})
        };
        self.compiler_stats.push(row);
    }

    fn pipeline(&self, bytes: &[u8]) -> Result<vk::Pipeline> {
        self.specialized_pipeline(bytes, None)
    }

    fn specialized_pipeline(&self, bytes: &[u8], bits: Option<u32>) -> Result<vk::Pipeline> {
        self.pipeline_with_subgroup(bytes, bits, false)
    }

    fn pipeline_with_subgroup(
        &self,
        bytes: &[u8],
        bits: Option<u32>,
        wave32: bool,
    ) -> Result<vk::Pipeline> {
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
            let mut size = vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default()
                .required_subgroup_size(32);
            if wave32 && self.force_wave32 {
                stage = stage.push_next(&mut size);
            }
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
        self.mapped_buffer(32768 * 4)
    }

    fn mapped_buffer(&mut self, size: usize) -> Result<Buffer> {
        let mut buffer = self.buffer(&vec![0; size])?;
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
        update(&mut self.profile.borrow_mut());
    }

    pub(crate) fn take_profile(&self) -> Profile {
        self.profile.take()
    }

    pub(crate) fn projection_metadata(&self) -> &[ProjectionMeta] {
        &self.projection_meta
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
            "kernel":if self.wave_kernels.iter().any(|n| *n) {"register_window_16_wave32_native_f16"} else if self.native_kernels.iter().any(|n| *n) {"register_window_16_native_f16_literal"} else {"register_window_16"},
            "native_float16_enabled":self.native_float16,"native_codebook_self_test":self.native_self_test,
            "native_codebook_arithmetic":"default/mcg: precise FP16 pair add; mul1: exact LUT; accumulation: FP32",
            "native_pipeline_errors":self.native_pipeline_errors,"native_bitrates":self.native_kernels.iter().enumerate().filter_map(|(i,n)| n.then_some(i+1)).collect::<Vec<_>>(),
            "projection":"gpu_resident_mlp_conv",
            "exl3_wave32_bitrates":self.wave_kernels.iter().enumerate().filter_map(|(i,v)|v.then_some(i+1)).collect::<Vec<_>>(),
            "subgroup_size_control_enabled":self.force_wave32,"dense_kernel":"four_rows_fp32",
            "cooperative_k4_supported":self.allow_cooperative_k4,"cooperative_k4_active":self.cooperative_k4_active,
            "cooperative_k4_self_test":self.cooperative_k4_self_test,
            "cooperative_k5_active":self.cooperative_k5_active,"cooperative_k5_self_test":self.cooperative_k5_self_test,
            "split_k":self.split_k,
            "compiler_statistics":{"extension_enabled":self.amd_info.is_some(),"pipelines":self.compiler_stats},
            "capabilities":self.capabilities,
            "limits":{"storage_buffer_bytes":self.max_range,"shared_memory_bytes":self.properties.limits.max_compute_shared_memory_size,
                "workgroup_invocations":self.properties.limits.max_compute_work_group_invocations,
                "workgroup_size":self.properties.limits.max_compute_work_group_size,
                "workgroup_count":self.properties.limits.max_compute_work_group_count,
                "max_allocations":self.properties.limits.max_memory_allocation_count},
            "memory_heaps":self.device.memory.memory_heaps[..self.device.memory.memory_heap_count as usize].iter()
                .map(|h|serde_json::json!({"bytes":h.size,"flags":h.flags.as_raw()})).collect::<Vec<_>>(),
            "host_visibility":"once_per_group","device_timing_scope":"device_queue_ns covers whole submission including final HOST_READ; matrix intervals exclude group barriers; split post-Hadamard includes ordered partial reduction",
            "prepared_matrices":self.prepared,"gpu_timestamps":self.queries != vk::QueryPool::null(),
            "timestamp_valid_bits":self.timestamp_bits,"timestamp_period_ns":self.properties.limits.timestamp_period,
            "automatic_calibration":false,"scratch_mapping":"persistent"})
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

    pub(crate) fn dense_timing_snapshot(&self) -> (u64, u64) {
        let profile = self.profile.borrow();
        (profile.device_other_ns, profile.device_samples)
    }

    pub(crate) fn record_head_timing(&self, before: (u64, u64)) {
        self.record_profile(|p| {
            p.device_head_ns = p
                .device_head_ns
                .saturating_add(p.device_other_ns.saturating_sub(before.0));
            p.device_head_samples = p
                .device_head_samples
                .saturating_add(p.device_samples.saturating_sub(before.1));
        });
    }

    fn verify_cooperative(&mut self, bits: u32, candidate: vk::Pipeline) -> Result<bool> {
        let reference_bytes: &[u8] = match bits {
            4 => include_bytes!(concat!(env!("OUT_DIR"), "/qmv_wave_4.spv")),
            5 => include_bytes!(concat!(env!("OUT_DIR"), "/qmv_wave_5.spv")),
            _ => anyhow::bail!("Unsupported cooperative self-test bitrate"),
        };
        let reference = self.pipeline_with_subgroup(reference_bytes, None, true)?;
        let result = (|| -> Result<bool> {
            let rows = 128usize;
            let cols = 128usize;
            let states: Vec<u16> = (0..rows * cols)
                .map(|i| (i.wrapping_mul(181).wrapping_add(i / 13 * 911 + 37)) as u16)
                .collect();
            let packed = codec::pack(&states, bits as usize)?;
            let bytes: Vec<u8> = packed.iter().flat_map(|v| v.to_le_bytes()).collect();
            let weights = self.buffer(&bytes)?;
            let x: Vec<f32> = (0..rows)
                .map(|i| ((i * 13 % 17) as f32 - 8.) / 64.)
                .collect();
            for codebook in 0..3 {
                let params = [rows as u32, cols as u32, bits, codebook];
                let expected = self.compute_groups(
                    reference,
                    &weights,
                    &x,
                    params,
                    cols * 4,
                    (cols / 16) as u32,
                )?;
                let actual = self.compute_groups(
                    candidate,
                    &weights,
                    &x,
                    params,
                    cols * 4,
                    (cols / 16) as u32,
                )?;
                if !expected
                    .iter()
                    .zip(&actual)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
                {
                    return Ok(false);
                }
            }
            Ok(true)
        })();
        if !self.device.abandoned.load(Ordering::Relaxed) {
            unsafe { self.device.raw.destroy_pipeline(reference, None) };
        }
        result
    }

    fn stream_pipeline(&mut self, bits: u32) -> Result<vk::Pipeline> {
        ensure!((1..=8).contains(&bits), "Invalid specialized EXL3 bitrate");
        let index = bits as usize - 1;
        if self.qmv_stream[index] == vk::Pipeline::null() {
            if (bits == 4 || bits == 5) && self.native_verified && self.allow_cooperative_k4 {
                let bytes: &[u8] = if bits == 4 {
                    include_bytes!(concat!(env!("OUT_DIR"), "/qmv_wave_4_cooperative.spv"))
                } else {
                    include_bytes!(concat!(env!("OUT_DIR"), "/qmv_wave_5_cooperative.spv"))
                };
                match self.pipeline_with_subgroup(bytes, None, true) {
                    Ok(pipeline) => {
                        let verified = match self.verify_cooperative(bits, pipeline) {
                            Ok(verified) => verified,
                            Err(err) => {
                                if self.device.abandoned.load(Ordering::Relaxed) {
                                    return Err(err);
                                }
                                self.native_pipeline_errors.push(format!("cooperative K{bits} self-test: {err}"));
                                false
                            }
                        };
                        if verified {
                            self.qmv_stream[index] = pipeline;
                            self.native_kernels[index] = true;
                            self.wave_kernels[index] = true;
                            if bits == 4 {
                                self.cooperative_k4_active = true;
                                self.cooperative_k4_self_test = "passed";
                            } else {
                                self.cooperative_k5_active = true;
                                self.cooperative_k5_self_test = "passed";
                            }
                            self.capture_stats(
                                format!("exl3_{bits}bit_wave32_cooperative"),
                                pipeline,
                            );
                            return Ok(pipeline);
                        }
                        if bits == 4 {
                            self.cooperative_k4_self_test = "mismatch_fallback";
                        } else {
                            self.cooperative_k5_self_test = "mismatch_fallback";
                        }
                        unsafe { self.device.raw.destroy_pipeline(pipeline, None) };
                        self.native_pipeline_errors.push(format!(
                            "cooperative K{bits} parity mismatch; using fallback"
                        ));
                    }
                    Err(err) => self
                        .native_pipeline_errors
                        .push(format!("cooperative K{bits}: {err}")),
                }
            }
            if self.native_verified && self.allow_wave32 {
                let binaries: [&[u8]; 8] = [
                    include_bytes!(concat!(env!("OUT_DIR"), "/qmv_wave_1.spv")),
                    include_bytes!(concat!(env!("OUT_DIR"), "/qmv_wave_2.spv")),
                    include_bytes!(concat!(env!("OUT_DIR"), "/qmv_wave_3.spv")),
                    include_bytes!(concat!(env!("OUT_DIR"), "/qmv_wave_4.spv")),
                    include_bytes!(concat!(env!("OUT_DIR"), "/qmv_wave_5.spv")),
                    include_bytes!(concat!(env!("OUT_DIR"), "/qmv_wave_6.spv")),
                    include_bytes!(concat!(env!("OUT_DIR"), "/qmv_wave_7.spv")),
                    include_bytes!(concat!(env!("OUT_DIR"), "/qmv_wave_8.spv")),
                ];
                match self.pipeline_with_subgroup(binaries[index], None, true) {
                    Ok(pipeline) => {
                        self.qmv_stream[index] = pipeline;
                        self.native_kernels[index] = true;
                        self.wave_kernels[index] = true;
                        self.capture_stats(format!("exl3_{bits}bit_wave32"), pipeline);
                        return Ok(pipeline);
                    }
                    Err(err) => self
                        .native_pipeline_errors
                        .push(format!("wave32 {bits}bit: {err}")),
                }
            }
            if self.native_verified {
                let binaries: [&[u8]; 8] = [
                    include_bytes!(concat!(env!("OUT_DIR"), "/qmv_native_1.spv")),
                    include_bytes!(concat!(env!("OUT_DIR"), "/qmv_native_2.spv")),
                    include_bytes!(concat!(env!("OUT_DIR"), "/qmv_native_3.spv")),
                    include_bytes!(concat!(env!("OUT_DIR"), "/qmv_native_4.spv")),
                    include_bytes!(concat!(env!("OUT_DIR"), "/qmv_native_5.spv")),
                    include_bytes!(concat!(env!("OUT_DIR"), "/qmv_native_6.spv")),
                    include_bytes!(concat!(env!("OUT_DIR"), "/qmv_native_7.spv")),
                    include_bytes!(concat!(env!("OUT_DIR"), "/qmv_native_8.spv")),
                ];
                match self.pipeline(binaries[index]) {
                    Ok(pipeline) => {
                        self.qmv_stream[index] = pipeline;
                        self.native_kernels[index] = true;
                        self.capture_stats(format!("exl3_{bits}bit_workgroup128"), pipeline);
                        return Ok(pipeline);
                    }
                    Err(err) => self
                        .native_pipeline_errors
                        .push(format!("{bits}bit: {err}")),
                }
            }
            self.qmv_stream[index] = self.specialized_pipeline(
                include_bytes!(concat!(env!("OUT_DIR"), "/qmv_stream.spv")),
                Some(bits),
            )?;
        }
        Ok(self.qmv_stream[index])
    }

    pub fn prepare_exl3(
        &mut self,
        weights: Buffer,
        shape: Exl3Shape,
        su: &[f32],
        sv: &[f32],
    ) -> Result<Exl3Matrix> {
        self.prepare_named_exl3(weights, shape, su, sv, "unnamed_projection")
    }

    pub(crate) fn prepare_named_exl3(
        &mut self,
        weights: Buffer,
        shape: Exl3Shape,
        su: &[f32],
        sv: &[f32],
        name: &str,
    ) -> Result<Exl3Matrix> {
        ensure!(shape.valid(), "Invalid EXL3 projection shape");
        ensure!(self.prepared < MAX_EXL3_MATRICES, "Too many EXL3 matrices");
        ensure!(
            Arc::ptr_eq(&weights.device, &self.device),
            "Weights belong to another GPU"
        );
        ensure!(
            weights.size == shape.rows / 8 * shape.cols * shape.bits as usize,
            "Weight buffer does not match EXL3 shape"
        );
        ensure!(
            su.len() == shape.rows
                && sv.len() == shape.cols
                && su
                    .iter()
                    .chain(sv)
                    .all(|v| v.is_finite() && crate::math::half(*v) == *v),
            "EXL3 scales must be finite FP16 values with the padded shape"
        );
        ensure!(
            shape.rows / 128 <= self.max_groups as usize
                && shape.cols / 16 <= self.max_groups as usize,
            "EXL3 dispatch exceeds GPU limit"
        );
        let pipeline = self.stream_pipeline(shape.bits)?;
        let bytes: Vec<_> = su.iter().chain(sv).flat_map(|v| v.to_ne_bytes()).collect();
        let scales = self.buffer(&bytes)?;
        let result = self.mapped_buffer(shape.cols * 4)?;
        let query_first =
            projection_queries(self.prepared).context("EXL3 query range exhausted")?;
        let input = self.input.as_ref().context("GPU input unavailable")?;
        let rotated = self
            .rotated
            .as_ref()
            .context("GPU rotation buffer unavailable")?;
        let split = self.wave_kernels[shape.bits as usize - 1];
        let output = if split {
            self.partials.as_ref()
        } else {
            self.output.as_ref()
        }
        .context("GPU output unavailable")?;
        let params = [shape.rows as u32, shape.cols as u32, 0, shape.input as u32];
        let jobs = [
            Dispatch {
                input,
                weights: &scales,
                output: rotated,
                params,
                outputs: shape.rows,
                groups: (shape.rows / 128) as u32,
                groups_y: 1,
                pipeline: self.hadamard,
                barrier_after: true,
            },
            Dispatch {
                input: rotated,
                weights: &weights,
                output,
                params: [params[0], params[1], shape.bits | (self.split_k << 8), shape.codebook],
                outputs: shape.cols,
                groups: exl3_groups(shape.cols).context("Invalid EXL3 groups")?,
                groups_y: if split { self.split_k } else { 1 },
                pipeline,
                barrier_after: true,
            },
            Dispatch {
                input: output,
                weights: &scales,
                output: &result,
                params: [
                    params[0],
                    params[1],
                    if split { 2 | (self.split_k << 8) } else { 1 },
                    shape.output as u32,
                ],
                outputs: shape.cols,
                groups: (shape.cols / 128) as u32,
                groups_y: 1,
                pipeline: self.hadamard,
                barrier_after: false,
            },
        ];
        let command = unsafe {
            self.device.raw.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )?
        }[0];
        let sets = unsafe {
            self.device.raw.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(self.descriptor_pool)
                    .set_layouts(&[self.set_layout; 3]),
            )?
        };
        // Recorded once; the serialized engine waits for the fence before any scratch reuse.
        self.record_jobs(
            command,
            &sets,
            &jobs,
            vk::CommandBufferUsageFlags::empty(),
            Some(query_first),
        )?;
        self.projection_meta.push(ProjectionMeta {
            name: name.into(),
            input: shape.input,
            output: shape.output,
            rows: shape.rows,
            cols: shape.cols,
            bits: shape.bits,
            codebook: shape.codebook,
            packed_bytes: weights.size as u64,
            kernel: if split {
                let cooperative = if shape.bits == 4 && self.cooperative_k4_active { 4 }
                    else if shape.bits == 5 && self.cooperative_k5_active { 5 } else { 0 };
                wave_kernel_label(self.split_k, cooperative)
            } else if self.native_kernels[shape.bits as usize - 1] && shape.codebook != 2 {
                "native_f16_literal"
            } else if self.native_kernels[shape.bits as usize - 1] {
                "lut_literal"
            } else {
                "lut_f32"
            },
        });
        self.prepared += 1;
        Ok(Exl3Matrix {
            weights,
            _scales: scales,
            output: result,
            shape,
            command,
            query_first,
        })
    }

    pub fn project_exl3(&mut self, matrix: &Exl3Matrix, x: &[f32]) -> Result<Vec<f32>> {
        self.project_exl3_batch(&[matrix], x)?
            .pop()
            .context("Missing EXL3 output")
    }

    pub fn project_exl3_batch(
        &mut self,
        matrices: &[&Exl3Matrix],
        x: &[f32],
    ) -> Result<Vec<Vec<f32>>> {
        ensure!(
            valid_group_size(matrices.len()),
            "EXL3 group requires 1–3 matrices"
        );
        ensure!(
            matrices
                .iter()
                .all(|m| Arc::ptr_eq(&m.weights.device, &self.device)),
            "Matrix belongs to another GPU",
        );
        ensure!(
            matrices.iter().all(|m| x.len() == m.shape.input) && x.iter().all(|v| v.is_finite()),
            "Invalid EXL3 input"
        );
        ensure!(
            matrices
                .iter()
                .enumerate()
                .all(|(i, m)| matrices[..i].iter().all(|other| other.command != m.command)),
            "Duplicate matrix in EXL3 group"
        );
        self.upload(x)?;
        let recording = Instant::now();
        let commands: Vec<_> = matrices.iter().map(|m| m.command).collect();
        self.submit_graph(&commands, recording)?;
        for matrix in matrices {
            self.collect_timestamps(Some(matrix.query_first));
        }
        let started = Instant::now();
        let result = matrices
            .iter()
            .map(|m| m.output.read_f32(m.shape.output))
            .collect();
        self.record_profile(|p| p.readback_ns = p.readback_ns.saturating_add(elapsed_ns(started)));
        result
    }

    pub(crate) fn prepare_dense_gate(
        &mut self,
        weights: Buffer,
        input_len: usize,
        outputs: usize,
        dtype: u32,
    ) -> Result<DenseGatePlan> {
        let params = self.validate_projection(&weights, &vec![0.; input_len], outputs, 0, dtype)?;
        if self.gate_input.is_none() {
            let input = self.scratch_buffer()?;
            let output = self.scratch_buffer()?;
            self.gate_input = Some(input);
            self.gate_output = Some(output);
        }
        let command = self.prepare_operator(Dispatch {
            input: self.gate_input.as_ref().context("Gate input unavailable")?,
            weights: &weights,
            output: self
                .gate_output
                .as_ref()
                .context("Gate output unavailable")?,
            params,
            outputs,
            groups: dense_groups(outputs).context("Invalid gate groups")?,
            groups_y: 1,
            pipeline: self.dense,
            barrier_after: false,
        })?;
        Ok(DenseGatePlan {
            device: self.device.clone(),
            _weights: weights,
            input_len,
            outputs,
            command,
        })
    }

    pub(crate) fn forward_dense_gate(&self, gate: &DenseGatePlan, x: &[f32]) -> Result<Vec<f32>> {
        ensure!(
            Arc::ptr_eq(&gate.device, &self.device)
                && x.len() == gate.input_len
                && x.iter().all(|v| v.is_finite()),
            "Invalid gate input"
        );
        let input = self.gate_input.as_ref().context("Gate input unavailable")?;
        let output = self
            .gate_output
            .as_ref()
            .context("Gate output unavailable")?;
        let bytes = unsafe {
            std::slice::from_raw_parts(x.as_ptr().cast::<u8>(), std::mem::size_of_val(x))
        };
        let started = Instant::now();
        input.write(bytes)?;
        self.record_profile(|p| p.upload_ns = p.upload_ns.saturating_add(elapsed_ns(started)));
        self.submit_graph(&[gate.command], Instant::now())?;
        self.collect_gate_interval();
        output.read_f32(gate.outputs)
    }

    /// Submit independent Q/K/V projections and a dense attention gate together.
    /// The gate has its own FP16-rounded input buffer; EXL3 still receives x unchanged.
    pub(crate) fn project_exl3_with_dense_gate(
        &mut self,
        matrices: &[&Exl3Matrix],
        gate: &DenseGatePlan,
        x: &[f32],
    ) -> Result<(Vec<Vec<f32>>, Vec<f32>)> {
        ensure!(valid_group_size(matrices.len()), "Invalid QKV group size");
        ensure!(
            matrices
                .iter()
                .all(|m| Arc::ptr_eq(&m.weights.device, &self.device) && x.len() == m.shape.input)
                && Arc::ptr_eq(&gate.device, &self.device)
                && x.len() == gate.input_len
                && x.iter().all(|v| v.is_finite()),
            "Invalid grouped attention input or device"
        );
        ensure!(
            matrices
                .iter()
                .enumerate()
                .all(|(i, m)| matrices[..i].iter().all(|other| other.command != m.command)),
            "Duplicate matrix in QKV group"
        );
        let started = Instant::now();
        let rounded: Vec<f32> = x.iter().copied().map(crate::math::half).collect();
        self.record_profile(|p| {
            p.projection_cpu_ns = p.projection_cpu_ns.saturating_add(elapsed_ns(started))
        });
        ensure!(
            rounded.iter().all(|v| v.is_finite()),
            "Invalid rounded gate input"
        );
        self.upload(x)?;
        let input = self.gate_input.as_ref().context("Gate input unavailable")?;
        let output = self
            .gate_output
            .as_ref()
            .context("Gate output unavailable")?;
        let bytes = unsafe {
            std::slice::from_raw_parts(
                rounded.as_ptr().cast::<u8>(),
                std::mem::size_of_val(rounded.as_slice()),
            )
        };
        let started = Instant::now();
        input.write(bytes)?;
        self.record_profile(|p| p.upload_ns = p.upload_ns.saturating_add(elapsed_ns(started)));
        let recording = Instant::now();
        let mut commands: Vec<_> = matrices.iter().map(|m| m.command).collect();
        commands.push(gate.command);
        self.submit_graph(&commands, recording)?;
        for matrix in matrices {
            self.collect_timestamps(Some(matrix.query_first));
        }
        self.collect_gate_interval();
        let started = Instant::now();
        let qkv = matrices
            .iter()
            .map(|m| m.output.read_f32(m.shape.output))
            .collect::<Result<Vec<_>>>()?;
        let gate = output.read_f32(gate.outputs)?;
        self.record_profile(|p| p.readback_ns = p.readback_ns.saturating_add(elapsed_ns(started)));
        Ok((qkv, gate))
    }

    fn collect_gate_interval(&self) {
        if self.queries == vk::QueryPool::null() {
            return;
        }
        let mut ticks = [0u64; 2];
        let started = Instant::now();
        let result = unsafe {
            self.device.raw.get_query_pool_results(
                self.queries,
                2,
                &mut ticks,
                vk::QueryResultFlags::TYPE_64,
            )
        };
        self.record_profile(|p| {
            p.query_read_ns = p.query_read_ns.saturating_add(elapsed_ns(started));
            match result {
                Ok(()) => {
                    if let Some(delta) = timestamp_delta(ticks[0], ticks[1], self.timestamp_bits) {
                        let ns = (delta as f64 * f64::from(self.properties.limits.timestamp_period))
                            as u64;
                        p.device_compute_ns = p.device_compute_ns.saturating_add(ns);
                        p.device_other_ns = p.device_other_ns.saturating_add(ns);
                        p.device_attention_gate_ns = p.device_attention_gate_ns.saturating_add(ns);
                        p.device_attention_gate_samples =
                            p.device_attention_gate_samples.saturating_add(1);
                        p.device_samples = p.device_samples.saturating_add(1);
                    }
                }
                Err(err) => {
                    p.query_read_failures = p.query_read_failures.saturating_add(1);
                    p.last_query_error = Some(format!("attention gate: {err:?}"));
                }
            }
        });
    }

    fn submit_graph(&self, commands: &[vk::CommandBuffer], recording: Instant) -> Result<()> {
        let mut batch = Vec::with_capacity(commands.len() + 2);
        batch.push(self.queue_start);
        batch.extend_from_slice(commands);
        batch.push(self.host_readback);
        self.submit(&batch, recording)?;
        self.collect_interval(0, None);
        Ok(())
    }

    fn collect_interval(&self, first: u32, operator: Option<bool>) {
        if self.queries == vk::QueryPool::null() {
            return;
        }
        let mut ticks = [0u64; 2];
        let started = Instant::now();
        let result = unsafe {
            self.device.raw.get_query_pool_results(
                self.queries,
                first,
                &mut ticks,
                vk::QueryResultFlags::TYPE_64,
            )
        };
        self.record_profile(|p| {
            p.query_read_ns = p.query_read_ns.saturating_add(elapsed_ns(started));
            match result {
                Ok(()) => {
                    if let Some(delta) = timestamp_delta(ticks[0], ticks[1], self.timestamp_bits) {
                        let ns = (delta as f64 * f64::from(self.properties.limits.timestamp_period))
                            as u64;
                        if let Some(conv) = operator {
                            p.device_compute_ns = p.device_compute_ns.saturating_add(ns);
                            p.device_samples = p.device_samples.saturating_add(1);
                            if conv {
                                p.device_convolution_ns =
                                    p.device_convolution_ns.saturating_add(ns);
                            } else {
                                p.device_swiglu_ns = p.device_swiglu_ns.saturating_add(ns);
                            }
                        } else {
                            p.device_queue_ns = p.device_queue_ns.saturating_add(ns);
                            p.device_queue_samples = p.device_queue_samples.saturating_add(1);
                        }
                    }
                }
                Err(err) => {
                    p.query_read_failures = p.query_read_failures.saturating_add(1);
                    p.last_query_error = Some(format!("interval {first}: {err:?}"));
                }
            }
        });
    }

    fn prepare_operator(&self, job: Dispatch<'_>) -> Result<vk::CommandBuffer> {
        let command = unsafe {
            self.device.raw.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )?
        }[0];
        let sets = unsafe {
            self.device.raw.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(self.descriptor_pool)
                    .set_layouts(&[self.set_layout]),
            )?
        };
        self.record_jobs(
            command,
            &sets,
            &[job],
            vk::CommandBufferUsageFlags::empty(),
            Some(2),
        )?;
        Ok(command)
    }

    pub(crate) fn prepare_mlp(
        &self,
        gate: Exl3Matrix,
        up: Exl3Matrix,
        down: Exl3Matrix,
    ) -> Result<MlpPlan> {
        ensure!(
            [&gate, &up, &down]
                .iter()
                .all(|m| Arc::ptr_eq(&m.weights.device, &self.device)),
            "MLP belongs to another GPU"
        );
        ensure!(
            gate.shape.input == up.shape.input
                && gate.shape.output == up.shape.output
                && down.shape.input == gate.shape.output
                && down.shape.output == gate.shape.input,
            "Invalid MLP projection shapes"
        );
        let size = gate.shape.output;
        let activate = self.prepare_operator(Dispatch {
            input: &gate.output,
            weights: &up.output,
            output: self.input.as_ref().context("GPU input unavailable")?,
            params: [size as u32, 0, 0, 0],
            outputs: size,
            groups: size.div_ceil(128) as u32,
            groups_y: 1,
            pipeline: self.swiglu,
            barrier_after: true,
        })?;
        Ok(MlpPlan {
            gate,
            up,
            down,
            activate,
        })
    }

    pub(crate) fn forward_mlp(&self, plan: &MlpPlan, x: &[f32]) -> Result<Vec<f32>> {
        ensure!(
            Arc::ptr_eq(&plan.gate.weights.device, &self.device)
                && x.len() == plan.gate.shape.input
                && x.iter().all(|v| v.is_finite()),
            "Invalid MLP input"
        );
        self.upload(x)?;
        self.submit_graph(
            &[
                plan.gate.command,
                plan.up.command,
                plan.activate,
                plan.down.command,
            ],
            Instant::now(),
        )?;
        for m in [&plan.gate, &plan.up, &plan.down] {
            self.collect_timestamps(Some(m.query_first));
        }
        self.collect_interval(2, Some(false));
        let started = Instant::now();
        let result = plan.down.output.read_f32(plan.down.shape.output);
        self.record_profile(|p| p.readback_ns = p.readback_ns.saturating_add(elapsed_ns(started)));
        result
    }

    pub(crate) fn prepare_conv(
        &mut self,
        input: Exl3Matrix,
        output: Exl3Matrix,
        weight: &[f32],
    ) -> Result<ConvPlan> {
        ensure!(
            [&input, &output]
                .iter()
                .all(|m| Arc::ptr_eq(&m.weights.device, &self.device)),
            "Convolution belongs to another GPU"
        );
        let h = output.shape.input;
        ensure!(
            input.shape.input == h
                && input.shape.output == h * 3
                && output.shape.output == h
                && weight.len() == h * 7
                && weight.iter().all(|v| v.is_finite()),
            "Invalid convolution shape"
        );
        let mut values =
            vec![0f32; convolution_state_len(h).context("Invalid convolution workspace")?];
        for (dst, src) in values[1..1 + h * 7].iter_mut().zip(weight) {
            *dst = crate::math::half(*src);
        }
        ensure!(
            values.iter().all(|v| v.is_finite()),
            "Non-finite FP16 convolution weights"
        );
        let bytes: Vec<_> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let state = self.buffer(&bytes)?;
        let convolve = self.prepare_operator(Dispatch {
            input: &input.output,
            weights: &state,
            output: self.input.as_ref().context("GPU input unavailable")?,
            params: [h as u32, 0, 0, 0],
            outputs: h,
            groups: h.div_ceil(128) as u32,
            groups_y: 1,
            pipeline: self.convolution,
            barrier_after: true,
        })?;
        Ok(ConvPlan {
            input,
            output,
            state,
            convolve,
        })
    }

    pub(crate) fn forward_conv(
        &self,
        plan: &ConvPlan,
        x: &[f32],
        position: usize,
    ) -> Result<Vec<f32>> {
        ensure!(
            Arc::ptr_eq(&plan.input.weights.device, &self.device)
                && x.len() == plan.input.shape.input
                && position < 2048
                && x.iter().all(|v| v.is_finite()),
            "Invalid convolution input or position"
        );
        // The previous fence completed; only the position header is host-written.
        plan.state.write(&(position as u32).to_ne_bytes())?;
        self.upload(x)?;
        self.submit_graph(
            &[plan.input.command, plan.convolve, plan.output.command],
            Instant::now(),
        )?;
        for m in [&plan.input, &plan.output] {
            self.collect_timestamps(Some(m.query_first));
        }
        self.collect_interval(2, Some(true));
        let started = Instant::now();
        let result = plan.output.output.read_f32(plan.output.shape.output);
        self.record_profile(|p| p.readback_ns = p.readback_ns.saturating_add(elapsed_ns(started)));
        result
    }

    pub fn multiply(
        &mut self,
        weights: &Buffer,
        x: &[f32],
        outputs: usize,
        bits: u32,
        codebook: u32,
    ) -> Result<Vec<f32>> {
        let mut params = self.validate_projection(weights, x, outputs, bits, codebook)?;
        if bits == 0 {
            return self.compute_groups(
                self.dense,
                weights,
                x,
                params,
                outputs,
                dense_groups(outputs).context("Invalid dense groups")?,
            );
        }
        let groups = exl3_groups(outputs).context("Invalid EXL3 output shape")?;
        let pipeline = self.stream_pipeline(bits)?;
        if !self.wave_kernels[bits as usize - 1] {
            return self.compute_groups(pipeline, weights, x, params, outputs, groups);
        }
        params[2] |= self.split_k << 8;
        let output = self.output.as_ref().context("GPU output unavailable")?;
        let partials = self.partials.as_ref().context("GPU partials unavailable")?;
        self.execute(
            x,
            &[
                Dispatch {
                    input: self.input.as_ref().context("GPU input unavailable")?,
                    weights,
                    output: partials,
                    params,
                    outputs,
                    groups,
                    groups_y: self.split_k,
                    pipeline,
                    barrier_after: true,
                },
                Dispatch {
                    input: partials,
                    weights,
                    output,
                    params,
                    outputs,
                    groups: outputs.div_ceil(128) as u32,
                    groups_y: 1,
                    pipeline: self.reduce,
                    barrier_after: false,
                },
            ],
        )?;
        output.read_f32(outputs)
    }

    pub(crate) fn prepare_dense_head(
        &mut self,
        chunks: Vec<(Buffer, usize)>,
        input_len: usize,
        dtype: u32,
    ) -> Result<DenseHeadPlan> {
        let rows: Vec<_> = chunks.iter().map(|(_, rows)| *rows).collect();
        let total = dense_batch_output_count(&rows).context("Invalid dense head chunks")?;
        let dummy = vec![0.0; input_len];
        let params = chunks
            .iter()
            .map(|(weights, rows)| self.validate_projection(weights, &dummy, *rows, 0, dtype))
            .collect::<Result<Vec<_>>>()?;
        let outputs = rows
            .iter()
            .map(|rows| self.mapped_buffer(rows * 4))
            .collect::<Result<Vec<_>>>()?;
        let layouts = vec![self.set_layout; chunks.len()];
        let command = unsafe {
            self.device.raw.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )?
        }[0];
        let sets = unsafe {
            self.device.raw.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(self.descriptor_pool)
                    .set_layouts(&layouts),
            )?
        };
        let input = self.input.as_ref().context("GPU input unavailable")?;
        let jobs: Vec<_> = chunks
            .iter()
            .zip(&params)
            .zip(&outputs)
            .map(|(((weights, rows), params), output)| Dispatch {
                input,
                weights,
                output,
                params: *params,
                outputs: *rows,
                groups: dense_groups(*rows).expect("validated dense head shape"),
                groups_y: 1,
                pipeline: self.dense,
                barrier_after: false,
            })
            .collect();
        self.record_jobs(
            command,
            &sets,
            &jobs,
            vk::CommandBufferUsageFlags::empty(),
            None,
        )?;
        Ok(DenseHeadPlan {
            chunks,
            outputs,
            input_len,
            total,
            command,
        })
    }

    pub(crate) fn forward_dense_head(&self, plan: &DenseHeadPlan, x: &[f32]) -> Result<Vec<f32>> {
        ensure!(
            x.len() == plan.input_len && x.iter().all(|v| v.is_finite()),
            "Invalid dense head input"
        );
        ensure!(
            plan.chunks
                .iter()
                .all(|(weights, _)| Arc::ptr_eq(&weights.device, &self.device)),
            "Dense head belongs to another GPU"
        );
        self.upload(x)?;
        self.submit(&[plan.command], Instant::now())?;
        self.collect_timestamps(None);
        let started = Instant::now();
        let mut result = Vec::with_capacity(plan.total);
        for ((_, rows), output) in plan.chunks.iter().zip(&plan.outputs) {
            result.extend(output.read_f32(*rows)?);
        }
        self.record_profile(|p| p.readback_ns = p.readback_ns.saturating_add(elapsed_ns(started)));
        Ok(result)
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
                input: self.input.as_ref().expect("initialized GPU input"),
                weights,
                output,
                params: *params,
                outputs: *rows,
                groups: dense_groups(*rows)
                    .context("Invalid dense groups")
                    .expect("validated dense shape"),
                groups_y: 1,
                pipeline: self.dense,
                barrier_after: false,
            })
            .collect();
        self.execute(x, &jobs)?;
        let started = Instant::now();
        let mut result = Vec::with_capacity(total);
        for job in jobs {
            result.extend(job.output.read_f32(job.outputs)?);
        }
        self.record_profile(|p| p.readback_ns = p.readback_ns.saturating_add(elapsed_ns(started)));
        Ok(result)
    }

    #[cfg(test)]
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
            x,
            &[Dispatch {
                input: self.input.as_ref().context("GPU input unavailable")?,
                weights,
                output,
                params,
                outputs,
                groups,
                groups_y: 1,
                pipeline,
                barrier_after: false,
            }],
        )?;
        let started = Instant::now();
        let result = output.read_f32(outputs);
        self.record_profile(|p| p.readback_ns = p.readback_ns.saturating_add(elapsed_ns(started)));
        result
    }

    fn upload(&self, x: &[f32]) -> Result<()> {
        ensure!(
            !self.device.abandoned.load(Ordering::Relaxed),
            "GPU timed out; restart the app"
        );
        let input = self.input.as_ref().context("GPU input unavailable")?;
        let bytes = unsafe {
            std::slice::from_raw_parts(x.as_ptr().cast::<u8>(), std::mem::size_of_val(x))
        };
        let started = Instant::now();
        input.write(bytes)?;
        self.record_profile(|p| p.upload_ns = p.upload_ns.saturating_add(elapsed_ns(started)));
        Ok(())
    }

    fn execute(&self, x: &[f32], jobs: &[Dispatch<'_>]) -> Result<()> {
        self.upload(x)?;
        let recording = Instant::now();
        unsafe {
            self.device
                .raw
                .reset_command_buffer(self.command, vk::CommandBufferResetFlags::empty())?;
        }
        self.record_jobs(
            self.command,
            &self.descriptors,
            jobs,
            vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
            None,
        )?;
        self.submit(&[self.command], recording)?;
        self.collect_timestamps(None);
        Ok(())
    }

    fn record_jobs(
        &self,
        command: vk::CommandBuffer,
        sets: &[vk::DescriptorSet],
        jobs: &[Dispatch<'_>],
        flags: vk::CommandBufferUsageFlags,
        phase_queries: Option<u32>,
    ) -> Result<()> {
        ensure!(
            !self.device.abandoned.load(Ordering::Relaxed),
            "GPU timed out; restart the app"
        );
        ensure!(
            !jobs.is_empty()
                && jobs.len() <= sets.len()
                && jobs.iter().all(|job| job.outputs > 0
                    && job.outputs <= 32768
                    && job.groups > 0
                    && job.groups <= self.max_groups
                    && job.groups_y > 0
                    && job.groups_y <= self.properties.limits.max_compute_work_group_count[1]),
            "Invalid compute buffer dimensions"
        );
        let codebooks = self
            .codebooks
            .as_ref()
            .context("GPU codebooks unavailable")?;
        ensure!(
            phase_queries.is_none()
                || jobs.len() == 3
                || (phase_queries == Some(2) && jobs.len() == 1),
            "Invalid EXL3 phase queries"
        );
        let query_first = phase_queries.unwrap_or(0);
        let query_count = if phase_queries.is_some() && phase_queries != Some(2) {
            4
        } else {
            2
        };
        ensure!(
            query_first <= QUERY_COUNT - query_count,
            "Invalid GPU query range"
        );
        unsafe {
            let d = &self.device.raw;
            d.begin_command_buffer(command, &vk::CommandBufferBeginInfo::default().flags(flags))?;
            if phase_queries.is_none() && self.queries != vk::QueryPool::null() {
                d.cmd_reset_query_pool(command, self.queries, 0, 2);
                d.cmd_write_timestamp(
                    command,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    self.queries,
                    0,
                );
            }
            // Scratch is shared between prepared command buffers in the same submission.
            let scratch = [vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)];
            d.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &scratch,
                &[],
                &[],
            );
            if phase_queries.is_some() && self.queries != vk::QueryPool::null() {
                d.cmd_reset_query_pool(command, self.queries, query_first, query_count);
                d.cmd_write_timestamp(
                    command,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    self.queries,
                    query_first,
                );
            }
            for (index, job) in jobs.iter().enumerate() {
                let infos = [job.input, job.weights, job.output, codebooks].map(|buffer| {
                    [vk::DescriptorBufferInfo::default()
                        .buffer(buffer.raw)
                        .range(buffer.size as u64)]
                });
                let descriptor = sets[index];
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
                d.cmd_bind_pipeline(command, vk::PipelineBindPoint::COMPUTE, job.pipeline);
                d.cmd_bind_descriptor_sets(
                    command,
                    vk::PipelineBindPoint::COMPUTE,
                    self.layout,
                    0,
                    &[descriptor],
                    &[],
                );
                let push = std::slice::from_raw_parts(job.params.as_ptr().cast::<u8>(), 16);
                d.cmd_push_constants(command, self.layout, vk::ShaderStageFlags::COMPUTE, 0, push);
                d.cmd_dispatch(command, job.groups, job.groups_y, 1);
                if job.barrier_after {
                    let barriers = [vk::MemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                        .dst_access_mask(
                            vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                        )];
                    d.cmd_pipeline_barrier(
                        command,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::DependencyFlags::empty(),
                        &barriers,
                        &[],
                        &[],
                    );
                }
                if query_count == 4 && index < 2 && self.queries != vk::QueryPool::null() {
                    d.cmd_write_timestamp(
                        command,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        self.queries,
                        query_first + index as u32 + 1,
                    );
                }
            }
            if phase_queries.is_none() {
                let barriers = [vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::HOST_READ)];
                d.cmd_pipeline_barrier(
                    command,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::HOST,
                    vk::DependencyFlags::empty(),
                    &barriers,
                    &[],
                    &[],
                );
            }
            if self.queries != vk::QueryPool::null() {
                d.cmd_write_timestamp(
                    command,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    self.queries,
                    query_first + query_count - 1,
                );
            }
            d.end_command_buffer(command)?;
        }
        Ok(())
    }

    fn submit(&self, commands: &[vk::CommandBuffer], recording: Instant) -> Result<()> {
        ensure!(
            !self.device.abandoned.load(Ordering::Relaxed),
            "GPU timed out; restart the app"
        );
        unsafe {
            let d = &self.device.raw;
            d.reset_fences(&[self.fence])?;
            let submits = [vk::SubmitInfo::default().command_buffers(commands)];
            if let Err(err) = d.queue_submit(self.queue, &submits, self.fence) {
                self.device.abandoned.store(true, Ordering::Relaxed);
                bail!("Vulkan submit failed ({err:?}); restart the app");
            }
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
    fn collect_timestamps(&self, phase_queries: Option<u32>) {
        if self.queries == vk::QueryPool::null() {
            return;
        }
        let mut ticks = [0u64; 4];
        let count = if phase_queries.is_some() { 4 } else { 2 };
        let started = Instant::now();
        // Fence is complete; never introduce an unbounded query WAIT on a broken driver.
        let result = unsafe {
            self.device.raw.get_query_pool_results(
                self.queries,
                phase_queries.unwrap_or(0),
                &mut ticks[..count],
                vk::QueryResultFlags::TYPE_64,
            )
        };
        self.record_profile(|p| {
            p.query_read_ns = p.query_read_ns.saturating_add(elapsed_ns(started))
        });
        if let Err(err) = result {
            self.record_profile(|p| {
                p.query_read_failures = p.query_read_failures.saturating_add(1);
                p.last_query_error = Some(format!("{err:?}"));
                if let Some(first) = phase_queries {
                    p.record_projection((first / 4 - 1) as usize, None);
                }
            });
            return;
        }
        let mut durations = [0u64; 3];
        for i in 0..count - 1 {
            let Some(delta) = timestamp_delta(ticks[i], ticks[i + 1], self.timestamp_bits) else {
                return;
            };
            durations[i] =
                (delta as f64 * f64::from(self.properties.limits.timestamp_period)) as u64;
        }
        self.record_profile(|p| {
            let total = durations.iter().fold(0u64, |sum, v| sum.saturating_add(*v));
            p.device_compute_ns = p.device_compute_ns.saturating_add(total);
            p.device_samples = p.device_samples.saturating_add(1);
            if let Some(first) = phase_queries {
                p.record_projection((first / 4 - 1) as usize, Some(durations));
                p.device_exl3_input_ns = p.device_exl3_input_ns.saturating_add(durations[0]);
                p.device_exl3_matvec_ns = p.device_exl3_matvec_ns.saturating_add(durations[1]);
                p.device_exl3_output_ns = p.device_exl3_output_ns.saturating_add(durations[2]);
                p.device_exl3_samples = p.device_exl3_samples.saturating_add(1);
            } else {
                p.device_other_ns = p.device_other_ns.saturating_add(total);
                p.device_queue_ns = p.device_queue_ns.saturating_add(total);
                p.device_queue_samples = p.device_queue_samples.saturating_add(1);
            }
        });
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
            for pipeline in self.qmv_stream {
                d.destroy_pipeline(pipeline, None);
            }
            d.destroy_pipeline(self.dense, None);
            d.destroy_pipeline(self.reduce, None);
            d.destroy_pipeline(self.swiglu, None);
            d.destroy_pipeline(self.convolution, None);
            d.destroy_pipeline(self.hadamard, None);
            d.destroy_query_pool(self.queries, None);
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
    #[ignore = "requires Vulkan; split-K numerical parity"]
    fn resident_split_k_variants_match_reference() {
        let mut gpu = Gpu::new().unwrap();
        let input: Vec<f32> = (0..128).map(|i| (i as f32 % 17. - 8.) / 32.).collect();
        for bits in [4usize, 5] {
            let states: Vec<u16> = (0..128 * 128).map(|i| (i * 37 + i / 7) as u16).collect();
            let packed = codec::pack(&states, bits).unwrap();
            let bytes: Vec<u8> = packed.iter().flat_map(|v| v.to_le_bytes()).collect();
            let weights = gpu.buffer(&bytes).unwrap();
            for codebook in 0..3 {
                gpu.split_k = 4;
                let baseline = gpu.multiply(&weights, &input, 128, bits as u32, codebook).unwrap();
                for split in [1, 2, 4, 8] {
                    gpu.split_k = split;
                    let actual = gpu.multiply(&weights, &input, 128, bits as u32, codebook).unwrap();
                    assert!(actual.iter().zip(&baseline).all(|(a, b)| (a - b).abs() <= 0.0001 + b.abs() * 0.0005), "split {split}, bits {bits}, codebook {codebook}");
                }
            }
        }
    }

    #[test]
    #[ignore = "requires Vulkan; prepared dense head parity and reuse"]
    fn resident_prepared_dense_head_matches_dynamic() {
        let mut gpu = Gpu::new().unwrap();
        let input = 128usize;
        let weights: Vec<u8> = (0..input * 9)
            .flat_map(|i| {
                half::f16::from_f32((i as f32 % 31. - 15.) / 32.)
                    .to_bits()
                    .to_le_bytes()
            })
            .collect();
        let chunks = vec![
            (gpu.buffer(&weights[..input * 4 * 2]).unwrap(), 4),
            (gpu.buffer(&weights[input * 4 * 2..]).unwrap(), 5),
        ];
        let x: Vec<f32> = (0..input).map(|i| (i as f32 - 63.) / 128.).collect();
        let expected = gpu.multiply_dense_chunks(&chunks, &x, 0).unwrap();
        let plan = gpu.prepare_dense_head(chunks, input, 0).unwrap();
        for _ in 0..3 {
            let before = gpu.submission_count();
            let actual = gpu.forward_dense_head(&plan, &x).unwrap();
            assert_eq!(gpu.submission_count() - before, 1);
            assert_eq!(actual, expected);
        }
        assert!(gpu.forward_dense_head(&plan, &[f32::NAN; 128]).is_err());
        assert!(gpu.forward_dense_head(&plan, &[0.; 127]).is_err());
    }

    #[test]
    #[ignore = "requires Vulkan; mixed dense attention gate and QKV"]
    fn resident_grouped_gate_matches_separate_submissions() {
        let mut gpu = Gpu::new().unwrap();
        let q = resident_matrix(&mut gpu, 128, 128, 1);
        let k = resident_matrix(&mut gpu, 128, 128, 2);
        let v = resident_matrix(&mut gpu, 128, 128, 3);
        let x: Vec<f32> = (0..128).map(|i| (i as f32 - 63.5) / 127.).collect();
        let gate_bytes: Vec<u8> = (0..128 * 16)
            .flat_map(|i| {
                half::f16::from_f32((i as f32 % 29. - 14.) / 32.)
                    .to_bits()
                    .to_le_bytes()
            })
            .collect();
        let weights = gpu.buffer(&gate_bytes).unwrap();
        let rounded: Vec<f32> = x.iter().copied().map(crate::math::half).collect();
        let expected_qkv = gpu.project_exl3_batch(&[&q, &k, &v], &x).unwrap();
        let expected_gate = gpu.multiply(&weights, &rounded, 16, 0, 0).unwrap();
        let plan = gpu.prepare_dense_gate(weights, 128, 16, 0).unwrap();
        let before = gpu.submission_count();
        let (actual_qkv, actual_gate) = gpu
            .project_exl3_with_dense_gate(&[&q, &k, &v], &plan, &x)
            .unwrap();
        assert_eq!(gpu.submission_count() - before, 1);
        assert_eq!(actual_qkv, expected_qkv);
        assert_eq!(actual_gate, expected_gate);
    }

    #[test]
    #[ignore = "requires Vulkan; independent prepared gate plans share scratch"]
    fn resident_gate_plans_preserve_weights_across_reuse() {
        let mut gpu = Gpu::new().unwrap();
        let x: Vec<f32> = (0..128).map(|i| (i as f32 - 64.) / 128.).collect();
        let mut plans = Vec::new();
        let mut expected = Vec::new();
        for dtype in 0..=1 {
            let bytes: Vec<u8> = (0..128 * 16)
                .flat_map(|i| {
                    let value = (i as f32 % 37. - 18.) / 64.;
                    (if dtype == 0 {
                        half::f16::from_f32(value).to_bits()
                    } else {
                        half::bf16::from_f32(value).to_bits()
                    })
                    .to_le_bytes()
                })
                .collect();
            let weights = gpu.buffer(&bytes).unwrap();
            expected.push(gpu.multiply(&weights, &x, 16, 0, dtype).unwrap());
            plans.push(gpu.prepare_dense_gate(weights, 128, 16, dtype).unwrap());
        }
        for _ in 0..3 {
            for (plan, expected) in plans.iter().zip(&expected) {
                assert_eq!(&gpu.forward_dense_gate(plan, &x).unwrap(), expected);
            }
        }
    }

    #[test]
    fn cooperative_k4_load_windows_cover_every_lane_and_wrap() {
        let mut requested = [0u32; 32];
        for lane in 0..32u32 {
            let bit = ((8 * lane + 1) * 4 + 256 * 4 - 16) % (256 * 4);
            let word = bit / 32;
            let shift = bit % 32;
            let loads = (shift + 7 * 4 + 16).div_ceil(32);
            assert_eq!(loads, 2);
            requested[word as usize] += 1;
            requested[((word + 1) % 32) as usize] += 1;
        }
        assert_eq!(requested, [2; 32]);
    }

    fn resident_matrix(gpu: &mut Gpu, input: usize, output: usize, seed: usize) -> Exl3Matrix {
        let packed = codec::pack(
            &(0..input * output)
                .map(|i| (i * 181 + i / 13 * 911 + seed) as u16)
                .collect::<Vec<_>>(),
            4,
        )
        .unwrap();
        let buffer = gpu
            .buffer(
                &packed
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        gpu.prepare_exl3(
            buffer,
            Exl3Shape {
                input,
                output,
                rows: input,
                cols: output,
                bits: 4,
                codebook: 1,
            },
            &vec![0.125; input],
            &vec![0.125; output],
        )
        .unwrap()
    }

    #[test]
    #[ignore = "requires Vulkan; fused operators against the sequential GPU/CPU boundary, including convolution ring wrap/reset"]
    fn resident_operators_preserve_numeric_contract_and_submit_once() {
        for allow_native in [true, false] {
            let mut gpu = Gpu::new_with_native(allow_native).unwrap();
            let gate = resident_matrix(&mut gpu, 128, 256, 37);
            let up = resident_matrix(&mut gpu, 128, 256, 911);
            let down = resident_matrix(&mut gpu, 256, 128, 733);
            let mlp = gpu.prepare_mlp(gate, up, down).unwrap();
            for repeat in 0..3 {
                let x: Vec<_> = (0..128)
                    .map(|i| {
                        crate::math::half(((i * 7 % 31) as f32 - 15.) / 16. + repeat as f32 / 64.)
                    })
                    .collect();
                let g = gpu.project_exl3(&mlp.gate, &x).unwrap();
                let u = gpu.project_exl3(&mlp.up, &x).unwrap();
                let activated: Vec<_> = g
                    .iter()
                    .zip(u)
                    .map(|(g, u)| crate::math::half(g / (1. + (-g).exp()) * u))
                    .collect();
                let expected = gpu.project_exl3(&mlp.down, &activated).unwrap();
                gpu.take_profile();
                let before = gpu.submission_count();
                let actual = gpu.forward_mlp(&mlp, &x).unwrap();
                assert_eq!(gpu.submission_count() - before, 1);
                assert_eq!(actual.len(), 128);
                assert!(expected.iter().any(|v| v.abs() > 0.001));
                for (a, b) in actual.iter().zip(&expected) {
                    assert!((a - b).abs() <= 0.0005 + b.abs() * 0.002, "{a} != {b}");
                }
                let p = gpu.take_profile();
                if gpu.queries != vk::QueryPool::null() {
                    assert_eq!(p.device_queue_samples, 1);
                    assert_eq!(p.device_exl3_samples, 3);
                    assert_eq!(p.device_samples, 4);
                    assert!(p.device_swiglu_ns > 0);
                }
            }
            let input = resident_matrix(&mut gpu, 128, 384, 49);
            let output = resident_matrix(&mut gpu, 128, 128, 83);
            let weight: Vec<_> = (0..128 * 7)
                .map(|i| ((i * 5 % 23) as f32 - 11.) / 32.)
                .collect();
            let conv = gpu.prepare_conv(input, output, &weight).unwrap();
            let mut history = std::collections::VecDeque::new();
            let mut first = None;
            for position in (0..11).chain(0..4) {
                if position == 0 {
                    history.clear();
                }
                let x: Vec<_> = (0..128)
                    .map(|i| {
                        crate::math::half(((i * 7 % 31) as f32 - 15.) / 16. + position as f32 / 64.)
                    })
                    .collect();
                let projected = gpu.project_exl3(&conv.input, &x).unwrap();
                history.push_back(
                    (0..128)
                        .map(|i| crate::math::half(projected[i] * projected[256 + i]))
                        .collect::<Vec<_>>(),
                );
                if history.len() > 7 {
                    history.pop_front();
                }
                let activated: Vec<_> = (0..128)
                    .map(|i| {
                        let mut sum = 0.;
                        for (t, bx) in history.iter().enumerate() {
                            sum += bx[i] * crate::math::half(weight[i * 7 + 7 - history.len() + t]);
                        }
                        crate::math::half(crate::math::half(sum) * projected[128 + i])
                    })
                    .collect();
                let expected = gpu.project_exl3(&conv.output, &activated).unwrap();
                let before = gpu.submission_count();
                let actual = gpu.forward_conv(&conv, &x, position).unwrap();
                assert_eq!(gpu.submission_count() - before, 1);
                assert_eq!(actual.len(), 128);
                assert!(expected.iter().any(|v| v.abs() > 0.001));
                assert_eq!(actual, expected, "convolution position {position}");
                if position == 0 {
                    if let Some(previous) = &first {
                        assert_eq!(&actual, previous, "reset must ignore old ring contents");
                    } else {
                        first = Some(actual);
                    }
                }
            }
            let before = gpu.submission_count();
            assert!(gpu.forward_mlp(&mlp, &[f32::NAN; 128]).is_err());
            assert!(gpu.forward_mlp(&mlp, &[0.; 127]).is_err());
            assert!(gpu.forward_conv(&conv, &[0.; 128], 2048).is_err());
            assert!(gpu.forward_conv(&conv, &[f32::INFINITY; 128], 0).is_err());
            assert_eq!(gpu.submission_count(), before);
            assert!(
                gpu.forward_mlp(&mlp, &[0.; 128])
                    .unwrap()
                    .iter()
                    .all(|v| *v == 0.)
            );
            let wrong = resident_matrix(&mut gpu, 128, 128, 1);
            let up = resident_matrix(&mut gpu, 128, 256, 2);
            let down = resident_matrix(&mut gpu, 256, 128, 3);
            assert!(gpu.prepare_mlp(wrong, up, down).is_err());
            let input = resident_matrix(&mut gpu, 128, 384, 1);
            let output = resident_matrix(&mut gpu, 128, 128, 2);
            assert!(
                gpu.prepare_conv(input, output, &[f32::NAN; 128 * 7])
                    .is_err()
            );
        }
    }

    #[test]
    #[ignore = "requires Vulkan; dense four-row kernel against prior kernel with odd inputs and row tails"]
    fn resident_dense_four_rows_match_reference_bits() {
        let mut gpu = Gpu::new().unwrap();
        let baseline = gpu
            .pipeline(include_bytes!(concat!(
                env!("OUT_DIR"),
                "/dense_reference.spv"
            )))
            .unwrap();
        for input in [3, 129, 2048] {
            let x: Vec<_> = (0..input)
                .map(|i| ((i * 7 % 31) as f32 - 15.) / 16.)
                .collect();
            for output in [1, 3, 4, 5, 17] {
                for dtype in 0..=1 {
                    let bytes: Vec<_> = (0..input * output)
                        .flat_map(|i| {
                            let v = ((i * 3 % 23) as f32 - 11.) / 32.;
                            (if dtype == 0 {
                                half::f16::from_f32(v).to_bits()
                            } else {
                                half::bf16::from_f32(v).to_bits()
                            })
                            .to_le_bytes()
                        })
                        .collect();
                    let weights = gpu.buffer(&bytes).unwrap();
                    let expected = gpu
                        .compute_groups(
                            baseline,
                            &weights,
                            &x,
                            [input as u32, output as u32, dtype, 0],
                            output,
                            output as u32,
                        )
                        .unwrap();
                    let actual = gpu.multiply(&weights, &x, output, 0, dtype).unwrap();
                    assert_eq!(
                        actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                    );
                }
            }
        }
        unsafe { gpu.device.raw.destroy_pipeline(baseline, None) };
    }

    #[test]
    #[ignore = "requires Vulkan; production MLP shapes against the prior EXL3 kernel"]
    fn resident_wave32_hot_shapes_preserve_reference_bits() {
        let mut gpu = Gpu::new().unwrap();
        for (rows, cols, bits) in [(2048, 7680, 4), (7680, 2048, 5)] {
            let packed = codec::pack(
                &(0..rows * cols)
                    .map(|i| (i * 181 + i / 13 * 911 + 37) as u16)
                    .collect::<Vec<_>>(),
                bits,
            )
            .unwrap();
            let weights = gpu
                .buffer(
                    &packed
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let x: Vec<_> = (0..rows)
                .map(|i| ((i * 13 % 17) as f32 - 8.) / 64.)
                .collect();
            let baseline = gpu
                .specialized_pipeline(
                    include_bytes!(concat!(env!("OUT_DIR"), "/qmv_stream.spv")),
                    Some(bits as u32),
                )
                .unwrap();
            let reference = gpu
                .compute_groups(
                    baseline,
                    &weights,
                    &x,
                    [rows as u32, cols as u32, bits as u32, 1],
                    cols,
                    (cols / 16) as u32,
                )
                .unwrap();
            let actual = gpu.multiply(&weights, &x, cols, bits as u32, 1).unwrap();
            assert!(
                gpu.wave_kernels[bits - 1],
                "test must exercise split4, not fallback"
            );
            assert_eq!(actual.len(), cols);
            assert!(actual.iter().any(|v| v.abs() > 0.01));
            assert_eq!(
                actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                reference.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
            unsafe { gpu.device.raw.destroy_pipeline(baseline, None) };
        }
    }

    #[test]
    #[ignore = "requires Vulkan shaderFloat16; native codec qualification and forced portable fallback"]
    fn native_codebooks_are_exhaustive_and_portable_fallback_is_usable() {
        let gpu = Gpu::new().unwrap();
        assert!(gpu.native_float16, "test requires shaderFloat16 support");
        assert!(gpu.native_verified, "{}", gpu.native_self_test);
        assert_eq!(gpu.native_self_test["checked_values"], 393216);
        assert_eq!(gpu.native_self_test["lanes_tested"], 2);
        assert_eq!(gpu.submission_count(), 0);
        let mut fallback = Gpu::new_with_native(false).unwrap();
        assert!(!fallback.native_float16 && !fallback.native_verified);
        let packed = codec::pack(&vec![1234; 128 * 128], 4).unwrap();
        let bytes: Vec<_> = packed.iter().flat_map(|v| v.to_le_bytes()).collect();
        let buffer = fallback.buffer(&bytes).unwrap();
        let actual = fallback
            .multiply(&buffer, &[0.125; 128], 128, 4, 1)
            .unwrap();
        assert_eq!(actual.len(), 128);
        assert!(actual.iter().all(|v| v.is_finite()));
        assert!(actual.iter().any(|v| v.abs() > 0.01));
        assert!(!fallback.native_kernels.iter().any(|v| *v));
    }
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
    fn stream_groups_and_timestamp_wrap_are_checked() {
        for (outputs, groups) in [
            (0, None),
            (15, None),
            (16, Some(1)),
            (32768, Some(2048)),
            (32769, None),
            (usize::MAX, None),
        ] {
            assert_eq!(exl3_groups(outputs), groups);
        }
        assert_eq!(timestamp_delta(254, 2, 8), Some(4));
        assert_eq!(timestamp_delta(u64::MAX, 2, 64), Some(3));
        assert_eq!(timestamp_delta(0, u64::MAX, 1), Some(1));
        assert_eq!(timestamp_delta(0, 0, 0), None);
        assert_eq!(timestamp_delta(0, 0, 65), None);
        assert_eq!(projection_queries(0), Some(4));
        assert_eq!(
            projection_queries(MAX_EXL3_MATRICES - 1),
            Some(QUERY_COUNT - 4)
        );
        assert_eq!(projection_queries(MAX_EXL3_MATRICES), None);
        assert_eq!(projection_queries(usize::MAX), None);
        for count in 0..=4 {
            assert_eq!(valid_group_size(count), count > 0 && count < 4);
        }
    }

    #[test]
    #[ignore = "requires Vulkan; independent projections share a submission without aliasing outputs or queries"]
    fn grouped_projections_match_sequential_bit_exact_and_validate_before_submit() {
        let mut gpu = Gpu::new().unwrap();
        let mut matrices = vec![];
        for (i, (output, cols, bits)) in [(128, 128, 4usize), (256, 256, 5), (383, 384, 8)]
            .into_iter()
            .enumerate()
        {
            let packed = codec::pack(
                &(0..256 * cols)
                    .map(|n| (n * 137 + 31 + i * 47) as u16)
                    .collect::<Vec<_>>(),
                bits,
            )
            .unwrap();
            let bytes: Vec<_> = packed.iter().flat_map(|v| v.to_le_bytes()).collect();
            let weights = gpu.buffer(&bytes).unwrap();
            let shape = Exl3Shape {
                input: 129,
                output,
                rows: 256,
                cols,
                bits: bits as u32,
                codebook: i as u32,
            };
            matrices.push(
                gpu.prepare_exl3(weights, shape, &[0.125; 256], &vec![-0.25; cols])
                    .unwrap(),
            );
        }
        assert!(
            matrices
                .windows(2)
                .all(|w| w[0].query_first + 4 == w[1].query_first)
        );
        for repeat in 0..3 {
            let x: Vec<_> = (0..129)
                .map(|i| ((i * 7 % 31) as f32 - 15.) / 64. + repeat as f32 / 16.)
                .collect();
            let expected: Vec<_> = matrices
                .iter()
                .map(|m| gpu.project_exl3(m, &x).unwrap())
                .collect();
            assert!(expected.iter().all(|v| v.iter().any(|f| f.abs() > 0.001)));
            for count in 1..=3 {
                let group: Vec<_> = matrices[..count].iter().collect();
                gpu.take_profile();
                let before = gpu.submission_count();
                let actual = gpu.project_exl3_batch(&group, &x).unwrap();
                assert_eq!(gpu.submission_count() - before, 1);
                assert_eq!(actual.len(), count);
                for (a, b) in actual.iter().zip(&expected) {
                    assert_eq!(
                        a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        b.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                    );
                }
                let p = gpu.take_profile();
                if gpu.queries != vk::QueryPool::null() {
                    assert_eq!(p.device_samples, count as u64);
                    assert_eq!(p.device_exl3_samples, count as u64);
                    assert_eq!(
                        p.device_compute_ns,
                        p.device_exl3_input_ns + p.device_exl3_matvec_ns + p.device_exl3_output_ns
                    );
                    assert!(
                        p.device_exl3_input_ns > 0
                            && p.device_exl3_matvec_ns > 0
                            && p.device_exl3_output_ns > 0
                    );
                    assert_eq!(p.device_other_ns, 0);
                }
            }
            assert_eq!(
                gpu.project_exl3_batch(&[&matrices[2], &matrices[0]], &x)
                    .unwrap(),
                [expected[2].clone(), expected[0].clone()]
            );
        }
        let before = gpu.submission_count();
        for group in [
            vec![],
            vec![&matrices[0], &matrices[0]],
            vec![&matrices[0]; 4],
        ] {
            assert!(gpu.project_exl3_batch(&group, &[0.; 129]).is_err());
        }
        assert!(
            gpu.project_exl3_batch(&[&matrices[0], &matrices[1]], &[0.; 128])
                .is_err()
        );
        assert!(
            gpu.project_exl3_batch(&[&matrices[0], &matrices[1]], &[f32::NAN; 129])
                .is_err()
        );
        let weights = gpu.buffer(&[0; 8192]).unwrap();
        let shape = Exl3Shape {
            input: 128,
            output: 128,
            rows: 128,
            cols: 128,
            bits: 4,
            codebook: 0,
        };
        let other_shape = gpu
            .prepare_exl3(weights, shape, &[1.; 128], &[1.; 128])
            .unwrap();
        assert!(
            gpu.project_exl3_batch(&[&matrices[0], &other_shape], &[0.; 129])
                .is_err()
        );
        let mut other_gpu = Gpu::new().unwrap();
        let weights = other_gpu.buffer(&[0; 8192]).unwrap();
        let foreign = other_gpu
            .prepare_exl3(weights, shape, &[1.; 128], &[1.; 128])
            .unwrap();
        assert!(
            gpu.project_exl3_batch(&[&matrices[0], &foreign], &[0.; 129])
                .is_err()
        );
        assert_eq!(gpu.submission_count(), before);
        assert!(
            gpu.project_exl3_batch(&[&matrices[0], &matrices[1]], &[0.; 129])
                .unwrap()
                .iter()
                .flatten()
                .all(|v| *v == 0.)
        );
    }

    #[test]
    #[ignore = "requires Vulkan; register-window kernel against independent CPU/f64 reconstruction"]
    fn stream_qmv_matches_cpu_all_bitrates_and_codebooks() {
        let mut gpu = Gpu::new().unwrap();
        for (rows, cols) in [(16, 32), (128, 128), (256, 384), (2048, 512)] {
            let x: Vec<_> = (0..rows)
                .map(|i| ((i * 13 % 17) as f32 - 8.) / 64.)
                .collect();
            for bits in 1..=8 {
                let states: Vec<_> = (0..rows * cols)
                    .map(|i| ((i * 177 + i / 7 * 313 + 512) & 65535) as u16)
                    .collect();
                let packed = codec::pack(&states, bits).unwrap();
                let bytes: Vec<_> = packed.iter().flat_map(|v| v.to_le_bytes()).collect();
                let buffer = gpu.buffer(&bytes).unwrap();
                let baseline = gpu
                    .specialized_pipeline(
                        include_bytes!(concat!(env!("OUT_DIR"), "/qmv_stream.spv")),
                        Some(bits as u32),
                    )
                    .unwrap();
                for cb in [Codebook::Default, Codebook::Mcg, Codebook::Mul1] {
                    let matrix = codec::reconstruct_inner(&packed, bits, cb, rows, cols).unwrap();
                    let actual = gpu
                        .multiply(&buffer, &x, cols, bits as u32, cb as u32)
                        .unwrap();
                    let reference = gpu
                        .compute_groups(
                            baseline,
                            &buffer,
                            &x,
                            [rows as u32, cols as u32, bits as u32, cb as u32],
                            cols,
                            (cols / 16) as u32,
                        )
                        .unwrap();
                    assert_eq!(
                        actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        reference.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        "native/lut {rows}x{cols}, K{bits}, cb{cb:?}"
                    );
                    assert_eq!(actual.len(), cols);
                    assert!(actual.iter().any(|v| v.abs() > 0.01));
                    for n in 0..cols {
                        let expected: f64 = (0..rows)
                            .map(|r| f64::from(x[r]) * matrix[r * cols + n].to_f64())
                            .sum();
                        assert!(
                            (f64::from(actual[n]) - expected).abs() <= 1e-4 + expected.abs() * 2e-5,
                            "{rows}x{cols}, K{bits}, cb{cb:?}, n{n}: {} != {expected}",
                            actual[n]
                        );
                    }
                }
                unsafe { gpu.device.raw.destroy_pipeline(baseline, None) };
            }
        }
    }

    #[test]
    #[ignore = "requires Vulkan; full GPU Hadamard/scales and prepared-command reuse"]
    fn prepared_projection_matches_cpu_reference_and_rejects_invalid_inputs() {
        let mut gpu = Gpu::new().unwrap();
        for (input, output, rows, cols) in
            [(1, 1, 128, 128), (127, 129, 128, 256), (256, 384, 256, 384)]
        {
            for bits in 1usize..=8 {
                let packed = codec::pack(
                    &(0..rows * cols)
                        .map(|i| (i * 177 + 37) as u16)
                        .collect::<Vec<_>>(),
                    bits,
                )
                .unwrap();
                let bytes: Vec<_> = packed.iter().flat_map(|v| v.to_le_bytes()).collect();
                for cb in [Codebook::Default, Codebook::Mcg, Codebook::Mul1] {
                    let su: Vec<_> = (0..rows)
                        .map(|i| crate::math::half(if i % 3 == 0 { -0.1234 } else { 0.2503 }))
                        .collect();
                    let sv: Vec<_> = (0..cols)
                        .map(|i| crate::math::half(if i % 5 == 0 { -0.2503 } else { 0.1234 }))
                        .collect();
                    let buffer = gpu.buffer(&bytes).unwrap();
                    let shape = Exl3Shape {
                        input,
                        output,
                        rows,
                        cols,
                        bits: bits as u32,
                        codebook: cb as u32,
                    };
                    let reference_buffer = gpu.buffer(&bytes).unwrap();
                    let matrix = gpu.prepare_exl3(buffer, shape, &su, &sv).unwrap();
                    for repeat in 0..2 {
                        let x: Vec<_> = (0..input)
                            .map(|i| {
                                ((i * 11 % 19) as f32 - 9.) * 0.031253 + repeat as f32 * 0.062513
                            })
                            .collect();
                        let mut xhat = vec![0.; rows];
                        for (i, value) in x.iter().enumerate() {
                            xhat[i] = crate::math::half(crate::math::half(*value) * su[i]);
                        }
                        crate::math::hadamard(&mut xhat).unwrap();
                        let mut expected = gpu
                            .compute(
                                gpu.qmv,
                                &reference_buffer,
                                &xhat,
                                [rows as u32, cols as u32, bits as u32, cb as u32],
                                cols,
                            )
                            .unwrap();
                        expected.iter_mut().for_each(|v| *v = crate::math::half(*v));
                        crate::math::hadamard(&mut expected).unwrap();
                        for (v, scale) in expected.iter_mut().zip(&sv) {
                            *v = crate::math::half(*v * *scale);
                        }
                        let before = gpu.submission_count();
                        let actual = gpu.project_exl3(&matrix, &x).unwrap();
                        assert_eq!(gpu.submission_count() - before, 1);
                        assert_eq!(actual.len(), output);
                        assert!(actual.iter().any(|v| v.abs() > 0.0001));
                        for (a, b) in actual.iter().zip(&expected) {
                            assert!(
                                a.is_finite() && (a - b).abs() <= 2e-4 + b.abs() * 2e-3,
                                "shape {input}/{rows} x {output}/{cols}, K{bits} cb{cb:?}: {a} != {b}"
                            );
                        }
                    }
                    let before = gpu.submission_count();
                    assert!(gpu.project_exl3(&matrix, &vec![0.; input + 1]).is_err());
                    assert!(gpu.project_exl3(&matrix, &vec![f32::NAN; input]).is_err());
                    assert_eq!(gpu.submission_count(), before);
                    assert!(
                        gpu.project_exl3(&matrix, &vec![0.; input])
                            .unwrap()
                            .iter()
                            .all(|v| *v == 0.)
                    );
                    assert!(gpu.project_exl3(&matrix, &vec![f32::MAX; input]).is_err());
                }
            }
        }
        let shape = Exl3Shape {
            input: 128,
            output: 128,
            rows: 128,
            cols: 128,
            bits: 4,
            codebook: 0,
        };
        for (su, sv) in [
            (vec![1.; 127], vec![1.; 128]),
            (vec![f32::NAN; 128], vec![1.; 128]),
            (vec![1.0002; 128], vec![1.; 128]),
            (vec![1.; 128], vec![1.; 129]),
        ] {
            let buffer = gpu.buffer(&[0; 8192]).unwrap();
            assert!(gpu.prepare_exl3(buffer, shape, &su, &sv).is_err());
        }
        for invalid in [
            Exl3Shape {
                rows: usize::MAX,
                ..shape
            },
            Exl3Shape { input: 0, ..shape },
            Exl3Shape {
                output: 129,
                ..shape
            },
            Exl3Shape { cols: 129, ..shape },
            Exl3Shape { bits: 9, ..shape },
            Exl3Shape {
                codebook: 3,
                ..shape
            },
        ] {
            let buffer = gpu.buffer(&[0; 8192]).unwrap();
            assert!(
                gpu.prepare_exl3(buffer, invalid, &[1.; 128], &[1.; 128])
                    .is_err()
            );
        }
        let buffer = gpu.buffer(&[0; 4]).unwrap();
        assert!(
            gpu.prepare_exl3(buffer, shape, &[1.; 128], &[1.; 128])
                .is_err()
        );
        let mut other_gpu = Gpu::new().unwrap();
        let buffer = other_gpu.buffer(&[0; 8192]).unwrap();
        assert!(
            gpu.prepare_exl3(buffer, shape, &[1.; 128], &[1.; 128])
                .is_err()
        );
        let buffer = other_gpu.buffer(&[0; 8192]).unwrap();
        let matrix = other_gpu
            .prepare_exl3(buffer, shape, &[1.; 128], &[1.; 128])
            .unwrap();
        assert!(gpu.project_exl3(&matrix, &[0.; 128]).is_err());
        gpu.prepared = MAX_EXL3_MATRICES;
        let buffer = gpu.buffer(&[0; 8192]).unwrap();
        assert!(
            gpu.prepare_exl3(buffer, shape, &[1.; 128], &[1.; 128])
                .is_err()
        );
        assert!(gpu.submission_count() > 0);
        let profile = gpu.take_profile();
        if gpu.queries != vk::QueryPool::null() {
            assert_eq!(profile.device_samples, gpu.submission_count());
            assert!(profile.device_compute_ns > 0);
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
                check(&gpu.compute(gpu.qmv, &buffer, &x, params, cols).unwrap());
                for _ in 0..2 {
                    check(&gpu.compute(reference, &buffer, &x, params, cols).unwrap());
                    check(&gpu.compute(gpu.qmv, &buffer, &x, params, cols).unwrap());
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
                            gpu.compute(gpu.qmv, &buffer, &x, params, cols).unwrap()
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
    #[kani::proof]
    fn dense_dispatch_and_wave32_capabilities_are_bounded() {
        let n: usize = kani::any();
        let groups = super::dense_groups(n);
        assert_eq!(groups.is_some(), n > 0 && n <= 32768);
        if let Some(g) = groups {
            assert!(g > 0 && g <= 8192);
            assert!((g as usize) * 4 >= n && ((g as usize) - 1) * 4 < n);
        }
        let hidden: usize = kani::any();
        let size = super::convolution_state_len(hidden);
        assert_eq!(size.is_some(), hidden > 0 && hidden <= 10922);
        if let Some(size) = size {
            assert!(size <= 152909);
            assert!(1 + 7 * hidden + 6 * hidden + hidden - 1 < size);
        }
        kani::cover!(hidden == 10922 && size == Some(152909));
        let min: u32 = kani::any();
        let max: u32 = kani::any();
        let stages: u32 = kani::any();
        let count: u32 = kani::any();
        let force = super::can_force_wave32(min, max, stages, count);
        assert_eq!(
            force,
            min >= 1 && min <= 32 && max >= 32 && stages & 32 == 32 && count > 0
        );
        kani::cover!(n == 32768 && groups == Some(8192));
        kani::cover!(n == usize::MAX && groups.is_none());
        kani::cover!(force && min == 32 && max == 64);
    }

    use super::*;
    #[kani::proof]
    fn query_ranges_and_group_counts_are_bounded() {
        let index: usize = kani::any();
        let count: usize = kani::any();
        let first = projection_queries(index);
        assert_eq!(first.is_some(), index < MAX_EXL3_MATRICES);
        if let Some(first) = first {
            assert!(first >= 4 && first % 4 == 0 && first + 3 < QUERY_COUNT);
            assert_eq!(first as usize / 4 - 1, index);
        }
        assert_eq!(valid_group_size(count), count >= 1 && count <= 3);
        kani::cover!(index == MAX_EXL3_MATRICES - 1 && first == Some(QUERY_COUNT - 4));
        kani::cover!(index == usize::MAX && first.is_none());
        kani::cover!(count == 3 && valid_group_size(count));
    }

    #[kani::proof]
    fn timestamp_wrapping_is_bounded_for_all_counter_widths() {
        let start: u64 = kani::any();
        let end: u64 = kani::any();
        let bits: u32 = kani::any();
        let delta = timestamp_delta(start, end, bits);
        assert_eq!(delta.is_some(), bits >= 1 && bits <= 64);
        if let Some(delta) = delta {
            if bits < 64 {
                assert!(delta < (1u64 << bits));
            } else {
                assert!(delta == end.wrapping_sub(start));
            }
            if end == start {
                assert!(delta == 0);
            }
        }
        kani::cover!(start == 255 && end == 1 && bits == 8 && delta == Some(2));
        kani::cover!(start == u64::MAX && end == 0 && bits == 64 && delta == Some(1));
        kani::cover!(bits > 64 && delta.is_none());
    }

    #[kani::proof]
    fn prepared_shapes_bound_all_buffer_and_dispatch_ranges() {
        let shape = Exl3Shape {
            input: kani::any(),
            output: kani::any(),
            rows: kani::any(),
            cols: kani::any(),
            bits: kani::any(),
            codebook: kani::any(),
        };
        if shape.valid() {
            assert!(shape.input > 0 && shape.input <= shape.rows && shape.rows <= 32768);
            assert!(shape.output > 0 && shape.output <= shape.cols && shape.cols <= 32768);
            assert!(shape.rows / 128 >= 1 && shape.cols / 128 >= 1);
            assert!(shape.rows % 128 == 0 && shape.cols % 128 == 0);
            assert!(shape.rows / 8 * shape.cols * shape.bits as usize <= 1024 * 1024 * 1024);
            assert!((shape.rows + shape.cols) * 4 <= 262144);
            assert!(shape.codebook <= 2);
        }
        kani::cover!(
            shape.valid() && shape.rows == 32768 && shape.cols == 32768 && shape.bits == 8
        );
        kani::cover!(shape.valid() && shape.input == 1 && shape.output == 1);
        kani::cover!(!shape.valid() && shape.input == usize::MAX);
    }

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
    fn stream_groups_cover_exactly_valid_outputs() {
        let outputs: usize = kani::any();
        let groups = exl3_groups(outputs);
        let valid = outputs >= 16 && outputs <= 32768 && outputs % 16 == 0;
        assert_eq!(groups.is_some(), valid);
        if let Some(groups) = groups {
            assert!(groups >= 1 && groups <= 2048);
            assert_eq!(groups as usize * 16, outputs);
        }
        kani::cover!(outputs == 16 && groups == Some(1));
        kani::cover!(outputs == 32768 && groups == Some(2048));
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
