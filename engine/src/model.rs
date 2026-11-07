use crate::{
    math,
    vulkan::{
        AttentionLayerConfig, AttentionLayerPlan, ConvLayerPlan, ConvPlan, Gpu, LayerCopyPlan,
        MlpPlan, ResidentLayer,
    },
    weights::{Linear, Weights},
};
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use std::{
    collections::{BTreeMap, VecDeque},
    fs,
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
};

#[derive(Clone, Deserialize)]
pub struct Rope {
    pub rope_theta: f32,
    pub rope_type: String,
    #[serde(default = "one")]
    pub partial_rotary_factor: f32,
}
fn one() -> f32 {
    1.
}

#[derive(Clone, Deserialize)]
pub struct Config {
    model_type: String,
    architectures: Vec<String>,
    pub hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    pub vocab_size: usize,
    max_position_embeddings: usize,
    norm_eps: f32,
    #[serde(rename = "conv_L_cache")]
    conv_length: usize,
    conv_bias: bool,
    use_sandwich_norm: bool,
    tie_word_embeddings: bool,
    layer_types: Vec<String>,
    sliding_window: usize,
    rope_parameters: BTreeMap<String, Rope>,
    pub eos_token_id: u32,
    pub bos_token_id: u32,
    quantization_config: Quantization,
}

#[derive(Clone, Deserialize)]
struct Quantization {
    quant_method: String,
}
impl Config {
    pub fn read(path: &Path) -> Result<Self> {
        ensure!(
            fs::metadata(path)?.len() <= 1024 * 1024,
            "Config exceeds 1 MiB"
        );
        let bytes = fs::read(path)?;
        ensure!(bytes.len() <= 1024 * 1024, "Config exceeds 1 MiB");
        let config: Self = serde_json::from_slice(&bytes)?;
        config.validate()?;
        Ok(config)
    }
    fn validate(&self) -> Result<()> {
        ensure!(
            self.model_type == "lfm3" && self.architectures == ["Lfm3ForCausalLM"],
            "Only LFM3 text checkpoints are supported"
        );
        ensure!(
            self.quantization_config.quant_method == "exl3",
            "An EXL3 checkpoint is required"
        );
        ensure!(
            (1..=24).contains(&self.num_hidden_layers)
                && self.layer_types.len() == self.num_hidden_layers,
            "Invalid layer count"
        );
        ensure!(
            (128..=2048).contains(&self.hidden_size)
                && self.hidden_size.is_multiple_of(128)
                && (128..=7680).contains(&self.intermediate_size)
                && self.intermediate_size.is_multiple_of(128)
                && (1..=128000).contains(&self.vocab_size),
            "Unsupported LFM3 dimensions"
        );
        ensure!(
            self.head_dim == 128
                && self.num_attention_heads > 0
                && self.num_attention_heads <= 16
                && self.num_key_value_heads > 0
                && self.num_key_value_heads <= self.num_attention_heads
                && self
                    .num_attention_heads
                    .is_multiple_of(self.num_key_value_heads)
                && self.num_attention_heads * self.head_dim == self.hidden_size,
            "Invalid attention dimensions"
        );
        ensure!(
            self.conv_length == 7
                && !self.conv_bias
                && self.use_sandwich_norm
                && !self.tie_word_embeddings,
            "Unsupported LFM3 operator configuration"
        );
        ensure!(
            self.norm_eps.is_finite()
                && self.norm_eps > 0.
                && self.max_position_embeddings > 0
                && (1..=2048).contains(&self.sliding_window),
            "Invalid context or normalization parameters"
        );
        ensure!(
            (self.eos_token_id as usize) < self.vocab_size
                && (self.bos_token_id as usize) < self.vocab_size,
            "Invalid special token IDs"
        );
        for layer in &self.layer_types {
            ensure!(
                matches!(
                    layer.as_str(),
                    "conv" | "full_attention" | "sliding_attention"
                ),
                "Unsupported LFM3 layer: {layer}"
            );
            if layer != "conv" {
                let rope = self
                    .rope_parameters
                    .get(layer)
                    .context("Missing layer RoPE settings")?;
                ensure!(
                    rope.rope_type == "default"
                        && rope.rope_theta.is_finite()
                        && rope.rope_theta > 0.
                        && rope.partial_rotary_factor.is_finite()
                        && rope.partial_rotary_factor > 0.
                        && rope.partial_rotary_factor <= 1.,
                    "Unsupported RoPE settings"
                );
            }
        }
        Ok(())
    }
}

struct Conv {
    plan: ConvPlan,
}
struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    output: Linear,
    gate: Linear,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    keys: VecDeque<Vec<f32>>,
    values: VecDeque<Vec<f32>>,
    window: usize,
    rope: Rope,
}
enum Operator {
    Conv(Box<Conv>),
    Attention(Box<Attention>),
}
struct Layer {
    operator: Operator,
    operator_norm: Vec<f32>,
    post_operator_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    post_ffn_norm: Vec<f32>,
    mlp: MlpPlan,
    gpu_conv_layer: Option<ConvLayerPlan>,
    gpu_attention_layer: Option<AttentionLayerPlan>,
}

impl Layer {
    fn resident(&self) -> Option<ResidentLayer<'_>> {
        match &self.operator {
            Operator::Conv(conv) => self
                .gpu_conv_layer
                .as_ref()
                .map(|plan| ResidentLayer::Conv(plan, &conv.plan, &self.mlp)),
            Operator::Attention(attn) => Some(ResidentLayer::Attention(
                self.gpu_attention_layer.as_ref()?,
                [
                    attn.q.exl3_ref()?,
                    attn.k.exl3_ref()?,
                    attn.v.exl3_ref()?,
                    attn.output.exl3_ref()?,
                ],
                attn.gate.gate_ref()?,
                &self.mlp,
            )),
        }
    }
}
pub struct Model {
    pub config: Config,
    pub gpu: Gpu,
    weights: Weights,
    layers: Vec<Layer>,
    layer_links: Vec<Option<LayerCopyPlan>>,
    norm: Vec<f32>,
    head: Linear,
    position: usize,
    pub context_limit: usize,
}

fn normalize(gpu: &Gpu, x: &[f32], weight: &[f32], eps: f32) -> Result<Vec<f32>> {
    let started = std::time::Instant::now();
    let result = math::rms_norm(x, weight, eps);
    gpu.record_profile(|p| {
        p.normalization_ns = p
            .normalization_ns
            .saturating_add(crate::vulkan::elapsed_ns(started))
    });
    result
}

impl Model {
    pub fn diagnostics(&self) -> serde_json::Value {
        let mut cache_bytes = 0usize;
        for layer in &self.layers {
            cache_bytes += match &layer.operator {
                Operator::Conv(_) => 0,
                Operator::Attention(a) => a
                    .keys
                    .iter()
                    .chain(&a.values)
                    .map(|v| v.len() * 4)
                    .sum::<usize>(),
            };
        }
        serde_json::json!({"architecture":"lfm3","hidden_size":self.config.hidden_size,
            "intermediate_size":self.config.intermediate_size,"layers":self.layers.len(),
            "layer_types":self.config.layer_types,"attention_heads":self.config.num_attention_heads,
            "kv_heads":self.config.num_key_value_heads,"vocabulary":self.config.vocab_size,
            "context_limit":self.context_limit,"position":self.position,"cpu_cache_logical_bytes":cache_bytes,
            "gpu_convolution_cache_logical_bytes":self.config.layer_types.iter().filter(|k| *k == "conv").count()*self.config.hidden_size*7*4,
            "gpu_attention_layers":self.layers.iter().filter(|l| l.gpu_attention_layer.is_some()).count(),
            "gpu_attention_cache_logical_bytes":self.layers.iter().filter_map(|l| l.gpu_attention_layer.as_ref()).map(|p| p.cache_bytes()).sum::<usize>(),
            "prefill_batch_size":1,"sampling":"greedy"})
    }
    pub fn load(
        directory: &Path,
        context_limit: usize,
        cancel: &AtomicBool,
        progress: &mut impl FnMut(&str),
    ) -> Result<Self> {
        let config = Config::read(&directory.join("config.json"))?;
        ensure!(
            (1..=2048).contains(&context_limit) && context_limit <= config.max_position_embeddings,
            "Context must be between 1 and 2048 tokens"
        );
        progress("Checking EXL3 checkpoint…");
        let weights = Weights::open(directory)?;
        ensure!(
            weights.shape("model.embed_tokens.weight")? == [config.vocab_size, config.hidden_size],
            "Wrong embedding shape"
        );
        let mut gpu = Gpu::new()?;
        let mut layers = vec![];
        let h = config.hidden_size;
        for (i, kind) in config.layer_types.iter().enumerate() {
            ensure!(!cancel.load(Ordering::Relaxed), "Loading cancelled");
            progress(&format!(
                "Loading layer {}/{}…",
                i + 1,
                config.num_hidden_layers
            ));
            let p = format!("model.layers.{i}");
            let norm = |name: &str, n| weights.floats(&format!("{p}.{name}.weight"), &[n]);
            let operator = if kind == "conv" {
                let input =
                    Linear::load(&weights, &mut gpu, &format!("{p}.conv.in_proj"), h, h * 3)?
                        .into_exl3()?;
                let output = Linear::load(&weights, &mut gpu, &format!("{p}.conv.out_proj"), h, h)?
                    .into_exl3()?;
                let weight = weights.floats(
                    &format!("{p}.conv.conv.weight"),
                    &[h, 1, config.conv_length],
                )?;
                Operator::Conv(Box::new(Conv {
                    plan: gpu.prepare_conv(input, output, &weight)?,
                }))
            } else {
                let kv = config.num_key_value_heads * config.head_dim;
                Operator::Attention(Box::new(Attention {
                    q: Linear::load(&weights, &mut gpu, &format!("{p}.self_attn.q_proj"), h, h)?,
                    k: Linear::load(&weights, &mut gpu, &format!("{p}.self_attn.k_proj"), h, kv)?,
                    v: Linear::load(&weights, &mut gpu, &format!("{p}.self_attn.v_proj"), h, kv)?,
                    output: Linear::load(
                        &weights,
                        &mut gpu,
                        &format!("{p}.self_attn.o_proj"),
                        h,
                        h,
                    )?,
                    gate: Linear::load(
                        &weights,
                        &mut gpu,
                        &format!("{p}.self_attn.gate_proj"),
                        h,
                        config.num_attention_heads,
                    )?,
                    q_norm: norm("self_attn.q_norm", config.head_dim)?,
                    k_norm: norm("self_attn.k_norm", config.head_dim)?,
                    keys: VecDeque::new(),
                    values: VecDeque::new(),
                    window: if kind == "sliding_attention" {
                        config.sliding_window.min(context_limit)
                    } else {
                        context_limit
                    },
                    rope: config.rope_parameters[kind].clone(),
                }))
            };
            let w1 = Linear::load(
                &weights,
                &mut gpu,
                &format!("{p}.feed_forward.w1"),
                h,
                config.intermediate_size,
            )?
            .into_exl3()?;
            let w2 = Linear::load(
                &weights,
                &mut gpu,
                &format!("{p}.feed_forward.w2"),
                config.intermediate_size,
                h,
            )?
            .into_exl3()?;
            let w3 = Linear::load(
                &weights,
                &mut gpu,
                &format!("{p}.feed_forward.w3"),
                h,
                config.intermediate_size,
            )?
            .into_exl3()?;
            let operator_norm = norm("operator_norm", h)?;
            let post_operator_norm = norm("post_operator_norm", h)?;
            let ffn_norm = norm("ffn_norm", h)?;
            let post_ffn_norm = norm("post_ffn_norm", h)?;
            let mlp = gpu.prepare_mlp(w1, w3, w2)?;
            let gpu_conv_layer = if gpu.gpu_conv_layer_enabled() {
                if let Operator::Conv(conv) = &operator {
                    Some(gpu.prepare_conv_layer(
                        &conv.plan,
                        &mlp,
                        [
                            &operator_norm,
                            &post_operator_norm,
                            &ffn_norm,
                            &post_ffn_norm,
                        ],
                        config.norm_eps,
                    )?)
                } else {
                    None
                }
            } else {
                None
            };
            let gpu_attention_layer = if gpu.gpu_attention_layer_enabled() {
                if let Operator::Attention(attn) = &operator
                    && let (Some(q), Some(k), Some(v), Some(output), Some(gate)) = (
                        attn.q.exl3_ref(),
                        attn.k.exl3_ref(),
                        attn.v.exl3_ref(),
                        attn.output.exl3_ref(),
                        attn.gate.gate_ref(),
                    )
                {
                    Some(gpu.prepare_attention_layer(AttentionLayerConfig {
                        q,
                        k,
                        v,
                        output,
                        gate,
                        mlp: &mlp,
                        norms: [
                            &operator_norm,
                            &post_operator_norm,
                            &ffn_norm,
                            &post_ffn_norm,
                        ],
                        q_norm: &attn.q_norm,
                        k_norm: &attn.k_norm,
                        heads: config.num_attention_heads,
                        kv_heads: config.num_key_value_heads,
                        dim: config.head_dim,
                        window: attn.window,
                        eps: config.norm_eps,
                        theta: attn.rope.rope_theta,
                        fraction: attn.rope.partial_rotary_factor,
                    })?)
                } else {
                    None
                }
            } else {
                None
            };
            layers.push(Layer {
                operator,
                operator_norm,
                post_operator_norm,
                ffn_norm,
                post_ffn_norm,
                mlp,
                gpu_conv_layer,
                gpu_attention_layer,
            });
        }
        let mut layer_links = Vec::new();
        if gpu.layer_batch_size() > 1 {
            for pair in layers.windows(2) {
                let link = match (pair[0].resident(), pair[1].resident()) {
                    (Some(source), Some(target)) => {
                        Some(gpu.prepare_layer_copy(&source, &target, h)?)
                    }
                    _ => None,
                };
                layer_links.push(link);
            }
        }
        ensure!(!cancel.load(Ordering::Relaxed), "Loading cancelled");
        progress("Loading vocabulary projection…");
        let norm = weights.floats("model.embedding_norm.weight", &[h])?;
        let head = Linear::load(&weights, &mut gpu, "lm_head", h, config.vocab_size)?;
        Ok(Self {
            config,
            gpu,
            weights,
            layers,
            layer_links,
            norm,
            head,
            position: 0,
            context_limit,
        })
    }

    pub fn reset(&mut self) {
        self.position = 0;
        for layer in &mut self.layers {
            match &mut layer.operator {
                Operator::Conv(_) => {}
                Operator::Attention(attn) => {
                    attn.keys.clear();
                    attn.values.clear();
                }
            }
        }
    }

    pub(crate) fn configure_gpu_greedy(&mut self, vocabulary: &[u32]) -> Result<bool> {
        if self.gpu.gpu_greedy_enabled() {
            return self.head.prepare_gpu_greedy(&mut self.gpu, vocabulary);
        }
        Ok(false)
    }

    fn advance(&mut self, token: u32, cancel: &AtomicBool) -> Result<Vec<f32>> {
        ensure!(self.position < self.context_limit, "Context is full");
        let started = std::time::Instant::now();
        let mut hidden =
            self.weights
                .embedding(token, self.config.vocab_size, self.config.hidden_size)?;
        self.gpu.record_profile(|p| {
            p.embedding_ns = p
                .embedding_ns
                .saturating_add(crate::vulkan::elapsed_ns(started))
        });
        let mut index = 0;
        while index < self.layers.len() {
            ensure!(!cancel.load(Ordering::Relaxed), "Generation cancelled");
            if self.gpu.layer_batch_size() > 1 && self.layers[index].resident().is_some() {
                let mut end = index + 1;
                while end < self.layers.len()
                    && end - index < self.gpu.layer_batch_size()
                    && self.layer_links.get(end - 1).is_some_and(Option::is_some)
                    && self.layers[end].resident().is_some()
                {
                    end += 1;
                }
                if end - index > 1 {
                    let layers: Vec<_> = self.layers[index..end]
                        .iter()
                        .map(|l| l.resident().expect("resident layer checked"))
                        .collect();
                    let links: Vec<_> = self.layer_links[index..end - 1]
                        .iter()
                        .map(|l| l.as_ref().expect("resident link checked"))
                        .collect();
                    hidden =
                        self.gpu
                            .forward_resident_batch(&layers, &links, &hidden, self.position)?;
                    index = end;
                    continue;
                }
            }
            let layer = &mut self.layers[index];
            index += 1;
            if let (Some(plan), Operator::Conv(conv)) = (&layer.gpu_conv_layer, &layer.operator) {
                hidden = self.gpu.forward_conv_layer(
                    plan,
                    &conv.plan,
                    &layer.mlp,
                    &hidden,
                    self.position,
                )?;
                continue;
            }
            if let (Some(plan), Operator::Attention(attn)) =
                (&layer.gpu_attention_layer, &layer.operator)
            {
                let q = attn.q.exl3_ref().context("Missing resident Q")?;
                let k = attn.k.exl3_ref().context("Missing resident K")?;
                let v = attn.v.exl3_ref().context("Missing resident V")?;
                let output = attn.output.exl3_ref().context("Missing resident output")?;
                let gate = attn.gate.gate_ref().context("Missing resident gate")?;
                hidden = self.gpu.forward_attention_layer(
                    plan,
                    [q, k, v, output],
                    gate,
                    &layer.mlp,
                    &hidden,
                    self.position,
                )?;
                continue;
            }
            let normalized = normalize(
                &self.gpu,
                &hidden,
                &layer.operator_norm,
                self.config.norm_eps,
            )?;
            let output = match &mut layer.operator {
                Operator::Conv(conv) => {
                    self.gpu
                        .forward_conv(&conv.plan, &normalized, self.position)?
                }
                Operator::Attention(attn) => {
                    attn.forward(&mut self.gpu, &normalized, &self.config, self.position)?
                }
            };
            let output = normalize(
                &self.gpu,
                &output,
                &layer.post_operator_norm,
                self.config.norm_eps,
            )?;
            let started = std::time::Instant::now();
            for (h, value) in hidden.iter_mut().zip(output) {
                *h += value;
            }
            self.gpu.record_profile(|p| {
                p.activation_ns = p
                    .activation_ns
                    .saturating_add(crate::vulkan::elapsed_ns(started))
            });
            let normalized = normalize(&self.gpu, &hidden, &layer.ffn_norm, self.config.norm_eps)?;
            let down = self.gpu.forward_mlp(&layer.mlp, &normalized)?;
            let down = normalize(&self.gpu, &down, &layer.post_ffn_norm, self.config.norm_eps)?;
            let started = std::time::Instant::now();
            for (h, value) in hidden.iter_mut().zip(down) {
                *h += value;
            }
            self.gpu.record_profile(|p| {
                p.activation_ns = p
                    .activation_ns
                    .saturating_add(crate::vulkan::elapsed_ns(started))
            });
        }
        ensure!(
            hidden.iter().all(|v| v.is_finite()),
            "Non-finite hidden state"
        );
        self.position += 1;
        Ok(hidden)
    }

    pub fn forward(
        &mut self,
        token: u32,
        logits: bool,
        cancel: &AtomicBool,
    ) -> Result<Option<Vec<f32>>> {
        let hidden = self.advance(token, cancel)?;
        if logits {
            let hidden = normalize(&self.gpu, &hidden, &self.norm, self.config.norm_eps)?;
            let started = std::time::Instant::now();
            let result = self.head.forward(&mut self.gpu, &hidden)?;
            self.gpu.record_profile(|p| {
                p.head_ns = p.head_ns.saturating_add(crate::vulkan::elapsed_ns(started))
            });
            Ok(Some(result))
        } else {
            Ok(None)
        }
    }

    pub(crate) fn forward_gpu_greedy(&mut self, token: u32, cancel: &AtomicBool) -> Result<u32> {
        let hidden = self.advance(token, cancel)?;
        let hidden = normalize(&self.gpu, &hidden, &self.norm, self.config.norm_eps)?;
        let started = std::time::Instant::now();
        let result = self.head.forward_gpu_greedy(&mut self.gpu, &hidden)?;
        self.gpu.record_profile(|p| {
            p.head_ns = p.head_ns.saturating_add(crate::vulkan::elapsed_ns(started))
        });
        Ok(result)
    }
}

#[cfg(test)]
fn convolution(
    history: &VecDeque<Vec<f32>>,
    weight: &[f32],
    gate: &[f32],
    h: usize,
    width: usize,
) -> Result<Vec<f32>> {
    ensure!(
        width > 0
            && history.len() <= width
            && gate.len() == h
            && weight.len() == h * width
            && history.iter().all(|x| x.len() == h),
        "Invalid convolution shape"
    );
    let mut y = vec![0.; h];
    let padding = width - history.len();
    for dim in 0..h {
        let mut acc = 0.;
        for (t, bx) in history.iter().enumerate() {
            acc += bx[dim] * math::half(weight[dim * width + padding + t]);
        }
        y[dim] = math::half(math::half(acc) * gate[dim]);
    }
    Ok(y)
}

impl Attention {
    fn forward(
        &mut self,
        gpu: &mut Gpu,
        x: &[f32],
        c: &Config,
        position: usize,
    ) -> Result<Vec<f32>> {
        let ([mut q, mut k, v], gate) =
            Linear::forward_qkv_gate([&self.q, &self.k, &self.v], &self.gate, gpu, x)?;
        for head in q.chunks_exact_mut(c.head_dim) {
            head.copy_from_slice(&normalize(gpu, head, &self.q_norm, c.norm_eps)?);
        }
        for head in k.chunks_exact_mut(c.head_dim) {
            head.copy_from_slice(&normalize(gpu, head, &self.k_norm, c.norm_eps)?);
        }
        let started = std::time::Instant::now();
        math::rope(
            &mut q,
            c.head_dim,
            self.rope.rope_theta,
            self.rope.partial_rotary_factor,
            position,
        )?;
        math::rope(
            &mut k,
            c.head_dim,
            self.rope.rope_theta,
            self.rope.partial_rotary_factor,
            position,
        )?;
        gpu.record_profile(|p| {
            p.rope_ns = p.rope_ns.saturating_add(crate::vulkan::elapsed_ns(started))
        });
        let started = std::time::Instant::now();
        self.keys.push_back(k);
        self.values.push_back(v);
        if self.keys.len() > self.window {
            self.keys.pop_front();
            self.values.pop_front();
        }
        gpu.record_profile(|p| {
            p.attention_cpu_ns = p
                .attention_cpu_ns
                .saturating_add(crate::vulkan::elapsed_ns(started))
        });
        let started = std::time::Instant::now();
        let attended = attention(
            &q,
            &self.keys,
            &self.values,
            &gate,
            c.num_attention_heads,
            c.num_key_value_heads,
            c.head_dim,
        )?;
        gpu.record_profile(|p| {
            p.attention_cpu_ns = p
                .attention_cpu_ns
                .saturating_add(crate::vulkan::elapsed_ns(started))
        });
        self.output.forward(gpu, &attended)
    }
}

fn attention(
    q: &[f32],
    keys: &VecDeque<Vec<f32>>,
    values: &VecDeque<Vec<f32>>,
    gate: &[f32],
    heads: usize,
    kv_heads: usize,
    dim: usize,
) -> Result<Vec<f32>> {
    ensure!(
        heads > 0
            && kv_heads > 0
            && heads.is_multiple_of(kv_heads)
            && dim > 0
            && q.len() == heads * dim
            && gate.len() == heads
            && !keys.is_empty()
            && keys.len() == values.len()
            && keys.iter().chain(values).all(|v| v.len() == kv_heads * dim),
        "Invalid attention shapes"
    );
    let mut output = vec![0.; q.len()];
    for h in 0..heads {
        let kv = h / (heads / kv_heads);
        let q = &q[h * dim..(h + 1) * dim];
        let mut scores: Vec<_> = keys
            .iter()
            .map(|k| {
                q.iter()
                    .zip(&k[kv * dim..(kv + 1) * dim])
                    .map(|(q, k)| q * k)
                    .sum::<f32>()
                    / (dim as f32).sqrt()
            })
            .collect();
        math::softmax(&mut scores)?;
        let g = 1. / (1. + (-gate[h]).exp());
        for d in 0..dim {
            let sum: f32 = scores
                .iter()
                .zip(values)
                .map(|(s, v)| s * v[kv * dim + d])
                .sum();
            output[h * dim + d] = math::half(math::half(sum) * g);
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn causal_width_seven_convolution_includes_current_token() {
        let mut history = VecDeque::new();
        let weight = vec![1., 2., 3., 4., 5., 6., 7.];
        for t in 1..=10 {
            history.push_back(vec![t as f32]);
            if history.len() > 7 {
                history.pop_front();
            }
            let actual = convolution(&history, &weight, &[2.], 1, 7).unwrap();
            let expected: f32 = (0..7)
                .map(|lag| (t - lag).max(0) as f32 * (7 - lag) as f32 * 2.)
                .sum();
            assert_eq!(actual, [expected]);
        }
        history.clear();
        history.push_back(vec![1.]);
        assert_eq!(convolution(&history, &weight, &[2.], 1, 7).unwrap(), [14.]);
        assert!(convolution(&history, &weight, &[], 1, 7).is_err());
    }
    #[test]
    fn grouped_attention_head_gate_and_single_token_oracle() {
        let keys = VecDeque::from([vec![3., 4.]]);
        let values = VecDeque::from([vec![2., 6.]]);
        let output = attention(&[1., 2., 3., 4.], &keys, &values, &[0., 0.], 2, 1, 2).unwrap();
        assert_eq!(output, [1., 3., 1., 3.]);
        assert!(attention(&[], &keys, &values, &[], 0, 1, 2).is_err());
    }
}
