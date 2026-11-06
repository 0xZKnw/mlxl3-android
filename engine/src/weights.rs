use crate::{
    codec::{self, Codebook},
    math,
    vulkan::{Buffer, Gpu},
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
    Exl3 {
        buffer: Buffer,
        input: usize,
        output: usize,
        bits: u32,
        cb: u32,
        su: Vec<f32>,
        sv: Vec<f32>,
    },
    Dense {
        chunks: Vec<(Buffer, usize)>,
        dtype: u32,
    },
}
pub struct Linear {
    storage: Storage,
    input: usize,
    output: usize,
}
impl Linear {
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
            let su = scale("suh", "su", rows)?;
            let sv = scale("svh", "sv", cols)?;
            let buffer = gpu.buffer(w.bytes(&trellis)?)?;
            Storage::Exl3 {
                buffer,
                input: rows,
                output: cols,
                bits: bits as u32,
                cb: cb as u32,
                su,
                sv,
            }
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
            Storage::Dense { chunks, dtype }
        };
        ensure!(
            !w.has(&format!("{name}.bias")),
            "Biased projections are unsupported in LFM3-1.7B"
        );
        Ok(Self {
            storage,
            input,
            output,
        })
    }

    pub fn forward(&self, gpu: &mut Gpu, x: &[f32]) -> Result<Vec<f32>> {
        ensure!(x.len() == self.input, "Linear input shape mismatch");
        let mut result = match &self.storage {
            Storage::Exl3 {
                buffer,
                input,
                output,
                bits,
                cb,
                su,
                sv,
            } => {
                let mut xhat = vec![0.; *input];
                for (i, value) in x.iter().enumerate() {
                    xhat[i] = math::half(math::half(*value) * math::half(su[i]));
                }
                math::hadamard(&mut xhat)?;
                let mut y = gpu.multiply(buffer, &xhat, *output, *bits, *cb)?;
                y.iter_mut().for_each(|v| *v = math::half(*v));
                math::hadamard(&mut y)?;
                for (v, scale) in y.iter_mut().zip(sv) {
                    *v = math::half(*v * math::half(*scale));
                }
                y.truncate(self.output);
                y
            }
            Storage::Dense { chunks, dtype } => {
                let x: Vec<_> = x.iter().copied().map(math::half).collect();
                gpu.multiply_dense_chunks(chunks, &x, *dtype)?
            }
        };
        // Match the FP16 projection boundary used by EXL3 reference execution.
        result.iter_mut().for_each(|v| *v = math::half(*v));
        ensure!(
            result.len() == self.output && result.iter().all(|v| v.is_finite()),
            "Invalid linear output"
        );
        Ok(result)
    }
}
