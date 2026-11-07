use crate::{
    codec::{self, Codebook},
    math,
    vulkan::{Buffer, DenseGatePlan, DenseHeadPlan, Exl3Matrix, Exl3Shape, Gpu},
};
use anyhow::{Context, Result, ensure};
use half::{bf16, f16};
use memmap2::{Mmap, MmapOptions};
use safetensors::{Dtype, SafeTensors};
use std::{
    collections::BTreeMap,
    fs::{self, File},
    path::Path,
};

struct Tensor {
    shard: usize,
    start: usize,
    len: usize,
    shape: Vec<usize>,
    dtype: Dtype,
}

fn rounded_scales(values: Vec<f32>) -> Result<Vec<f32>> {
    let values: Vec<_> = values.into_iter().map(math::half).collect();
    ensure!(
        values.iter().all(|v| v.is_finite()),
        "EXL3 scales exceed FP16 range"
    );
    Ok(values)
}
pub struct Weights {
    shards: Vec<Mmap>,
    tensors: BTreeMap<String, Tensor>,
}
impl Weights {
    pub fn open(directory: &Path) -> Result<Self> {
        let mut files = fs::read_dir(directory)?
            .map(|e| e.map(|e| e.path()))
            .collect::<std::io::Result<Vec<_>>>()?;
        files.retain(|p| p.extension().is_some_and(|e| e == "safetensors"));
        files.sort();
        ensure!(
            !files.is_empty() && files.len() <= 32,
            "Expected 1–32 Safetensors shards"
        );
        let mut result = Self {
            shards: vec![],
            tensors: BTreeMap::new(),
        };
        let mut total = 0u64;
        for path in files {
            ensure!(
                fs::symlink_metadata(&path)?.file_type().is_file(),
                "Model shards must be regular files"
            );
            let file = File::open(&path)?;
            let size = file.metadata()?.len();
            total = total
                .checked_add(size)
                .context("Checkpoint size overflow")?;
            ensure!(
                (8..=4_294_967_296).contains(&size) && total <= 4_294_967_296,
                "Checkpoint exceeds the 4 GiB file budget"
            );
            // Imported model files are private, immutable app copies for this session.
            let map = unsafe { MmapOptions::new().map(&file)? };
            let header_size = u64::from_le_bytes(map[..8].try_into()?);
            ensure!(
                header_size <= 16 * 1024 * 1024,
                "Safetensors header is too large"
            );
            let parsed = SafeTensors::deserialize(&map).context("Invalid Safetensors shard")?;
            for (name, view) in parsed.tensors() {
                let tensor = Tensor {
                    shard: result.shards.len(),
                    start: view.data().as_ptr() as usize - map.as_ptr() as usize,
                    len: view.data().len(),
                    shape: view.shape().to_vec(),
                    dtype: view.dtype(),
                };
                ensure!(
                    result.tensors.insert(name.clone(), tensor).is_none(),
                    "Duplicate tensor {name}"
                );
            }
            result.shards.push(map);
        }
        Ok(result)
    }
    pub fn has(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }
    pub fn shape(&self, name: &str) -> Result<&[usize]> {
        Ok(&self.tensor(name)?.shape)
    }
    fn tensor(&self, name: &str) -> Result<&Tensor> {
        self.tensors
            .get(name)
            .with_context(|| format!("Missing tensor {name}"))
    }
    pub fn bytes(&self, name: &str) -> Result<&[u8]> {
        let t = self.tensor(name)?;
        Ok(&self.shards[t.shard][t.start..t.start + t.len])
    }
    pub fn floats(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        ensure!(self.shape(name)? == shape, "Wrong shape for {name}");
        let values = self.float_slice(name, self.bytes(name)?)?;
        ensure!(
            !values.is_empty() && values.iter().all(|v| v.is_finite()),
            "Invalid numeric tensor {name}"
        );
        Ok(values)
    }
    fn float_slice(&self, name: &str, bytes: &[u8]) -> Result<Vec<f32>> {
        let dtype = self.tensor(name)?.dtype;
        Ok(match dtype {
            Dtype::F16 | Dtype::BF16 => bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|pair| {
                    let bits = u16::from_le_bytes([pair[0], pair[1]]);
                    if dtype == Dtype::F16 {
                        f16::from_bits(bits).to_f32()
                    } else {
                        bf16::from_bits(bits).to_f32()
                    }
                })
                .collect(),
            Dtype::F32 => bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|p| f32::from_le_bytes(*p))
                .collect(),
            _ => anyhow::bail!("Unsupported numeric dtype for {name}"),
        })
    }
    pub fn embedding(&self, token: u32, vocab: usize, hidden: usize) -> Result<Vec<f32>> {
        let name = "model.embed_tokens.weight";
        ensure!(
            self.shape(name)? == [vocab, hidden] && (token as usize) < vocab,
            "Invalid embedding shape or token"
        );
        let size = match self.tensor(name)?.dtype {
            Dtype::F16 | Dtype::BF16 => 2,
            Dtype::F32 => 4,
            _ => anyhow::bail!("Unsupported embedding dtype"),
        };
        let start = token as usize * hidden * size;
        let result = self.float_slice(name, &self.bytes(name)?[start..start + hidden * size])?;
        ensure!(
            result.iter().all(|v| v.is_finite()),
            "Embedding contains non-finite values"
        );
        Ok(result)
    }
}

enum Storage {
    Exl3(Exl3Matrix),
    Head(DenseHeadPlan),
    Gate(DenseGatePlan),
    Dense {
        chunks: Vec<(Buffer, usize)>,
        dtype: u32,
    },
}
pub struct Linear {
    storage: Storage,
    input: usize,
    output: usize,
    is_head: bool,
}
impl Linear {
    pub(crate) fn exl3_ref(&self) -> Option<&Exl3Matrix> {
        if let Storage::Exl3(matrix) = &self.storage {
            Some(matrix)
        } else {
            None
        }
    }

    pub(crate) fn gate_ref(&self) -> Option<&DenseGatePlan> {
        if let Storage::Gate(plan) = &self.storage {
            Some(plan)
        } else {
            None
        }
    }
    pub(crate) fn prepare_gpu_greedy(&mut self, gpu: &mut Gpu, vocabulary: &[u32]) -> Result<bool> {
        if let Storage::Head(plan) = &mut self.storage {
            gpu.prepare_greedy_head(plan, vocabulary)?;
            return Ok(true);
        }
        Ok(false)
    }

    pub(crate) fn forward_gpu_greedy(&self, gpu: &mut Gpu, x: &[f32]) -> Result<u32> {
        ensure!(x.len() == self.input, "Linear input shape mismatch");
        let Storage::Head(plan) = &self.storage else {
            anyhow::bail!("GPU greedy requires a dense head")
        };
        let started = std::time::Instant::now();
        let rounded: Vec<f32> = x.iter().copied().map(math::half).collect();
        gpu.record_profile(|p| {
            p.projection_cpu_ns = p
                .projection_cpu_ns
                .saturating_add(crate::vulkan::elapsed_ns(started))
        });
        let before = gpu.dense_timing_snapshot();
        let token = gpu.forward_greedy_head(plan, &rounded)?;
        gpu.record_head_timing(before);
        Ok(token)
    }
    pub(crate) fn into_exl3(self) -> Result<Exl3Matrix> {
        match self.storage {
            Storage::Exl3(matrix) => Ok(matrix),
            Storage::Dense { .. } | Storage::Head(_) | Storage::Gate(_) => {
                anyhow::bail!("GPU resident operator requires EXL3 projections")
            }
        }
    }

    pub fn forward_group<const N: usize>(
        linears: [&Self; N],
        gpu: &mut Gpu,
        x: &[f32],
    ) -> Result<[Vec<f32>; N]> {
        ensure!(
            linears.iter().all(|l| l.input == x.len()),
            "Grouped linear input shape mismatch"
        );
        let matrices = linears
            .iter()
            .map(|l| match &l.storage {
                Storage::Exl3(matrix) => Ok(matrix),
                Storage::Dense { .. } | Storage::Head(_) | Storage::Gate(_) => {
                    anyhow::bail!("Grouped projections require EXL3 weights")
                }
            })
            .collect::<Result<Vec<_>>>()?;
        gpu.project_exl3_batch(&matrices, x)?
            .try_into()
            .map_err(|_| anyhow::anyhow!("Grouped linear output count mismatch"))
    }

    pub(crate) fn forward_qkv_gate(
        qkv: [&Self; 3],
        gate: &Self,
        gpu: &mut Gpu,
        x: &[f32],
    ) -> Result<([Vec<f32>; 3], Vec<f32>)> {
        ensure!(gate.input == x.len(), "Gate input shape mismatch");
        if let Storage::Gate(plan) = &gate.storage {
            let matrices = qkv
                .iter()
                .map(|l| match &l.storage {
                    Storage::Exl3(matrix) => Ok(matrix),
                    Storage::Dense { .. } | Storage::Head(_) | Storage::Gate(_) => {
                        anyhow::bail!("Grouped projections require EXL3 weights")
                    }
                })
                .collect::<Result<Vec<_>>>()?;
            let (values, mut gate_values) = gpu.project_exl3_with_dense_gate(&matrices, plan, x)?;
            gate_values.iter_mut().for_each(|v| *v = math::half(*v));
            ensure!(
                gate_values.len() == gate.output && gate_values.iter().all(|v| v.is_finite()),
                "Invalid gate output"
            );
            let values = values
                .try_into()
                .map_err(|_| anyhow::anyhow!("Grouped linear output count mismatch"))?;
            return Ok((values, gate_values));
        }
        let values = Self::forward_group(qkv, gpu, x)?;
        Ok((values, gate.forward(gpu, x)?))
    }

    pub fn load(
        w: &Weights,
        gpu: &mut Gpu,
        name: &str,
        input: usize,
        output: usize,
    ) -> Result<Self> {
        ensure!(
            input > 0 && input <= 32768 && output > 0 && output <= 128000,
            "Invalid projection dimensions"
        );
        let trellis = format!("{name}.trellis");
        ensure!(
            w.has(&trellis) || name == "lm_head" || name.ends_with(".gate_proj"),
            "Missing EXL3 projection: {name}"
        );
        let storage = if w.has(&trellis) {
            let shape = w.shape(&trellis)?;
            ensure!(
                shape.len() == 3 && shape[2].is_multiple_of(16),
                "Invalid EXL3 shape: {name}"
            );
            let bits = shape[2] / 16;
            codec::check_k(bits)?;
            let rows = shape[0].checked_mul(16).context("EXL3 row overflow")?;
            let cols = shape[1].checked_mul(16).context("EXL3 column overflow")?;
            ensure!(
                rows >= input
                    && rows - input < 128
                    && cols >= output
                    && cols - output < 128
                    && rows.is_multiple_of(128)
                    && cols.is_multiple_of(128)
                    && cols <= 32768,
                "Invalid EXL3 padding: {name}"
            );
            ensure!(
                matches!(w.tensor(&trellis)?.dtype, Dtype::U16 | Dtype::I16),
                "Invalid EXL3 storage dtype"
            );
            let scale = |primary: &str, legacy: &str, n| {
                let primary = format!("{name}.{primary}");
                let legacy = format!("{name}.{legacy}");
                w.floats(if w.has(&primary) { &primary } else { &legacy }, &[n])
            };
            let cb = if w.has(&format!("{name}.mul1")) {
                Codebook::Mul1
            } else if w.has(&format!("{name}.mcg")) {
                Codebook::Mcg
            } else {
                Codebook::Default
            };
            let su = rounded_scales(scale("suh", "su", rows)?)?;
            let sv = rounded_scales(scale("svh", "sv", cols)?)?;
            let buffer = gpu.buffer(w.bytes(&trellis)?)?;
            Storage::Exl3(gpu.prepare_named_exl3(
                buffer,
                Exl3Shape {
                    input,
                    output,
                    rows,
                    cols,
                    bits: bits as u32,
                    codebook: cb as u32,
                },
                &su,
                &sv,
                name,
            )?)
        } else {
            let key = format!("{name}.weight");
            ensure!(
                w.shape(&key)? == [output, input],
                "Wrong dense projection shape: {name}"
            );
            let dtype = match w.tensor(&key)?.dtype {
                Dtype::F16 => 0,
                Dtype::BF16 => 1,
                _ => anyhow::bail!("Dense weights must be F16/BF16: {name}"),
            };
            let bytes_per_row = input * 2;
            let chunk_rows = (gpu.max_range.min(64 * 1024 * 1024) / bytes_per_row).min(32768);
            ensure!(chunk_rows > 0, "Vulkan buffer limit is too small");
            let mut chunks = vec![];
            for bytes in w.bytes(&key)?.chunks(chunk_rows * bytes_per_row) {
                chunks.push((gpu.buffer(bytes)?, bytes.len() / bytes_per_row));
            }
            if name == "lm_head" {
                Storage::Head(gpu.prepare_dense_head(chunks, input, dtype)?)
            } else if name.ends_with(".self_attn.gate_proj") && chunks.len() == 1 {
                let (weights, rows) = chunks.pop().expect("one gate chunk");
                Storage::Gate(gpu.prepare_dense_gate(weights, input, rows, dtype)?)
            } else {
                Storage::Dense { chunks, dtype }
            }
        };
        ensure!(
            !w.has(&format!("{name}.bias")),
            "Biased projections are unsupported in LFM3-1.7B"
        );
        Ok(Self {
            storage,
            input,
            output,
            is_head: name == "lm_head",
        })
    }

    pub fn forward(&self, gpu: &mut Gpu, x: &[f32]) -> Result<Vec<f32>> {
        ensure!(x.len() == self.input, "Linear input shape mismatch");
        let mut cpu_started = std::time::Instant::now();
        let mut result = match &self.storage {
            // GPU post-Hadamard already enforces the FP16 boundary and finite output.
            Storage::Exl3(matrix) => return gpu.project_exl3(matrix, x),
            Storage::Gate(plan) => {
                let x: Vec<_> = x.iter().copied().map(math::half).collect();
                gpu.forward_dense_gate(plan, &x)?
            }
            Storage::Head(plan) => {
                let x: Vec<_> = x.iter().copied().map(math::half).collect();
                gpu.record_profile(|p| {
                    p.projection_cpu_ns = p
                        .projection_cpu_ns
                        .saturating_add(crate::vulkan::elapsed_ns(cpu_started))
                });
                let before = gpu.dense_timing_snapshot();
                let result = gpu.forward_dense_head(plan, &x)?;
                gpu.record_head_timing(before);
                cpu_started = std::time::Instant::now();
                result
            }
            Storage::Dense { chunks, dtype } => {
                let x: Vec<_> = x.iter().copied().map(math::half).collect();
                gpu.record_profile(|p| {
                    p.projection_cpu_ns = p
                        .projection_cpu_ns
                        .saturating_add(crate::vulkan::elapsed_ns(cpu_started))
                });
                let before = gpu.dense_timing_snapshot();
                let result = gpu.multiply_dense_chunks(chunks, &x, *dtype)?;
                if self.is_head {
                    gpu.record_head_timing(before);
                }
                cpu_started = std::time::Instant::now();
                result
            }
        };
        // Match the FP16 projection boundary used by EXL3 reference execution.
        result.iter_mut().for_each(|v| *v = math::half(*v));
        ensure!(
            result.len() == self.output && result.iter().all(|v| v.is_finite()),
            "Invalid linear output"
        );
        gpu.record_profile(|p| {
            p.projection_cpu_ns = p
                .projection_cpu_ns
                .saturating_add(crate::vulkan::elapsed_ns(cpu_started))
        });
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn immutable_scales_round_once_and_reject_overflow() {
        assert_eq!(
            rounded_scales(vec![1.0002, -1.0002, 65504., 1e-8]).unwrap(),
            [1., -1., 65504., 0.]
        );
        for value in [f32::MAX, f32::NAN, f32::INFINITY] {
            assert!(rounded_scales(vec![value]).is_err());
        }
    }
}
