use crate::{
    math,
    vulkan::Gpu,
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
    input: Linear,
    output: Linear,
    weight: Vec<f32>,
    history: VecDeque<Vec<f32>>,
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
    w1: Linear,
    w2: Linear,
    w3: Linear,
}
pub struct Model {
    pub config: Config,
    pub gpu: Gpu,
    weights: Weights,
    layers: Vec<Layer>,
    norm: Vec<f32>,
    head: Linear,
    position: usize,
    pub context_limit: usize,
}

impl Model {
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
                Operator::Conv(Box::new(Conv {
                    input: Linear::load(
                        &weights,
                        &mut gpu,
                        &format!("{p}.conv.in_proj"),
                        h,
                        h * 3,
                    )?,
                    output: Linear::load(&weights, &mut gpu, &format!("{p}.conv.out_proj"), h, h)?,
                    weight: weights.floats(
                        &format!("{p}.conv.conv.weight"),
                        &[h, 1, config.conv_length],
                    )?,
                    history: VecDeque::new(),
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
            layers.push(Layer {
                operator,
                operator_norm: norm("operator_norm", h)?,
                post_operator_norm: norm("post_operator_norm", h)?,
                ffn_norm: norm("ffn_norm", h)?,
                post_ffn_norm: norm("post_ffn_norm", h)?,
                w1: Linear::load(
                    &weights,
                    &mut gpu,
                    &format!("{p}.feed_forward.w1"),
                    h,
                    config.intermediate_size,
                )?,
                w2: Linear::load(
                    &weights,
                    &mut gpu,
                    &format!("{p}.feed_forward.w2"),
                    config.intermediate_size,
                    h,
                )?,
                w3: Linear::load(
                    &weights,
                    &mut gpu,
                    &format!("{p}.feed_forward.w3"),
                    h,
                    config.intermediate_size,
                )?,
            });
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
                Operator::Conv(conv) => conv.history.clear(),
                Operator::Attention(attn) => {
                    attn.keys.clear();
                    attn.values.clear();
                }
            }
        }
    }

    pub fn forward(
        &mut self,
        token: u32,
        logits: bool,
        cancel: &AtomicBool,
    ) -> Result<Option<Vec<f32>>> {
        ensure!(self.position < self.context_limit, "Context is full");
        let mut hidden =
            self.weights
                .embedding(token, self.config.vocab_size, self.config.hidden_size)?;
        for layer in &mut self.layers {
            ensure!(!cancel.load(Ordering::Relaxed), "Generation cancelled");
            let normalized = math::rms_norm(&hidden, &layer.operator_norm, self.config.norm_eps)?;
            let output = match &mut layer.operator {
                Operator::Conv(conv) => conv.forward(&mut self.gpu, &normalized, &self.config)?,
                Operator::Attention(attn) => {
                    attn.forward(&mut self.gpu, &normalized, &self.config, self.position)?
                }
            };
            let output = math::rms_norm(&output, &layer.post_operator_norm, self.config.norm_eps)?;
            for (h, value) in hidden.iter_mut().zip(output) {
                *h += value;
            }
            let normalized = math::rms_norm(&hidden, &layer.ffn_norm, self.config.norm_eps)?;
            let gate = layer.w1.forward(&mut self.gpu, &normalized)?;
            let up = layer.w3.forward(&mut self.gpu, &normalized)?;
            let activated: Vec<_> = gate
                .iter()
                .zip(up)
                .map(|(g, u)| math::half(g / (1. + (-g).exp()) * u))
                .collect();
            let down = layer.w2.forward(&mut self.gpu, &activated)?;
            let down = math::rms_norm(&down, &layer.post_ffn_norm, self.config.norm_eps)?;
            for (h, value) in hidden.iter_mut().zip(down) {
                *h += value;
            }
        }
        ensure!(
            hidden.iter().all(|v| v.is_finite()),
            "Non-finite hidden state"
        );
        self.position += 1;
        if logits {
            let hidden = math::rms_norm(&hidden, &self.norm, self.config.norm_eps)?;
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
}

impl Conv {
    fn forward(&mut self, gpu: &mut Gpu, x: &[f32], c: &Config) -> Result<Vec<f32>> {
        let projected = self.input.forward(gpu, x)?;
        let h = c.hidden_size;
        let bx = projected[..h]
            .iter()
            .zip(&projected[2 * h..])
            .map(|(b, x)| math::half(b * x))
            .collect();
        self.history.push_back(bx);
        if self.history.len() > c.conv_length {
            self.history.pop_front();
        }
        let y = convolution(
            &self.history,
            &self.weight,
            &projected[h..2 * h],
            h,
            c.conv_length,
        )?;
        self.output.forward(gpu, &y)
    }
}

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
        let mut q = self.q.forward(gpu, x)?;
        let mut k = self.k.forward(gpu, x)?;
        let v = self.v.forward(gpu, x)?;
        for head in q.chunks_exact_mut(c.head_dim) {
            head.copy_from_slice(&math::rms_norm(head, &self.q_norm, c.norm_eps)?);
        }
        for head in k.chunks_exact_mut(c.head_dim) {
            head.copy_from_slice(&math::rms_norm(head, &self.k_norm, c.norm_eps)?);
        }
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
        self.keys.push_back(k);
        self.values.push_back(v);
        if self.keys.len() > self.window {
            self.keys.pop_front();
            self.values.pop_front();
        }
        let gate = self.gate.forward(gpu, x)?;
        let attended = attention(
            &q,
            &self.keys,
            &self.values,
            &gate,
            c.num_attention_heads,
            c.num_key_value_heads,
            c.head_dim,
        )?;
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
