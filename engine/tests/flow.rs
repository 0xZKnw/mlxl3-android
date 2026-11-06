use mlxl3_android::{Engine, Message, codec};
use serde_json::json;
use std::{
    collections::BTreeMap,
    fs,
    sync::atomic::{AtomicBool, Ordering},
};

struct Fixture {
    metadata: BTreeMap<String, serde_json::Value>,
    data: Vec<u8>,
}
impl Fixture {
    fn tensor(&mut self, key: String, shape: &[usize], dtype: &str, bytes: Vec<u8>) {
        let start = self.data.len();
        self.data.extend(bytes);
        self.metadata.insert(
            key,
            json!({"dtype":dtype,"shape":shape,"data_offsets":[start,self.data.len()]}),
        );
    }
    fn floats(&mut self, name: &str, shape: &[usize], values: &[f32]) {
        self.tensor(
            name.into(),
            shape,
            "F16",
            values
                .iter()
                .flat_map(|v| half::f16::from_f32(*v).to_bits().to_le_bytes())
                .collect(),
        );
    }
    fn exl3(&mut self, name: &str, input: usize, output: usize) {
        let states: Vec<_> = (0..input * output)
            .map(|i| ((i * 181 + i / 13 * 911 + 37) & 65535) as u16)
            .collect();
        let packed = codec::pack(&states, 4).unwrap();
        self.tensor(
            format!("{name}.trellis"),
            &[input / 16, output / 16, 64],
            "U16",
            packed.iter().flat_map(|v| v.to_le_bytes()).collect(),
        );
        self.tensor(format!("{name}.mcg"), &[], "BOOL", vec![1]);
        self.floats(&format!("{name}.suh"), &[input], &vec![0.125; input]);
        self.floats(&format!("{name}.svh"), &[output], &vec![0.125; output]);
    }
}

#[test]
#[ignore = "requires Vulkan; reduced nonzero LFM3 fixture, no real weights"]
fn full_engine_prefill_decode_reset_cancel_and_context_errors() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture {
        metadata: BTreeMap::new(),
        data: vec![],
    };
    let embedding: Vec<_> = (0..16 * 128)
        .map(|i| ((i * 7 % 31) as f32 - 15.) / 16.)
        .collect();
    f.floats("model.embed_tokens.weight", &[16, 128], &embedding);
    f.floats("model.embedding_norm.weight", &[128], &[1.; 128]);
    let head: Vec<_> = (0..16 * 128)
        .map(|i| {
            let row = i / 128;
            if row == 1 {
                0.
            } else if i % 128 == row % 128 {
                if row % 2 == 0 { 0.5 } else { -0.5 }
            } else {
                0.
            }
        })
        .collect();
    f.floats("lm_head.weight", &[16, 128], &head);
    for layer in 0..3 {
        let p = format!("model.layers.{layer}");
        for norm in [
            "operator_norm",
            "post_operator_norm",
            "ffn_norm",
            "post_ffn_norm",
        ] {
            f.floats(&format!("{p}.{norm}.weight"), &[128], &[1.; 128]);
        }
        for name in ["w1", "w2", "w3"] {
            f.exl3(&format!("{p}.feed_forward.{name}"), 128, 128);
        }
        if layer == 0 {
            f.exl3(&format!("{p}.conv.in_proj"), 128, 384);
            f.exl3(&format!("{p}.conv.out_proj"), 128, 128);
            f.floats(
                &format!("{p}.conv.conv.weight"),
                &[128, 1, 7],
                &vec![0.125; 128 * 7],
            );
        } else {
            for name in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                f.exl3(&format!("{p}.self_attn.{name}"), 128, 128);
            }
            for norm in ["q_norm", "k_norm"] {
                f.floats(&format!("{p}.self_attn.{norm}.weight"), &[128], &[1.; 128]);
            }
            f.floats(
                &format!("{p}.self_attn.gate_proj.weight"),
                &[1, 128],
                &[0.125; 128],
            );
        }
    }
    let mut header = serde_json::to_vec(&f.metadata).unwrap();
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    bytes.extend(f.data);
    fs::write(dir.path().join("model.safetensors"), bytes).unwrap();
    let mut config: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/lfm3-config.json")).unwrap();
    for (key, value) in [
        ("hidden_size", json!(128)),
        ("intermediate_size", json!(128)),
        ("vocab_size", json!(16)),
        ("num_hidden_layers", json!(3)),
        ("num_attention_heads", json!(1)),
        ("num_key_value_heads", json!(1)),
        (
            "layer_types",
            json!(["conv", "full_attention", "sliding_attention"]),
        ),
        ("sliding_window", json!(2)),
        ("bos_token_id", json!(0)),
        ("eos_token_id", json!(1)),
    ] {
        config[key] = value;
    }
    fs::write(dir.path().join("config.json"), config.to_string()).unwrap();
    let vocab: tokenizers::models::wordlevel::WordLevel =
        tokenizers::models::wordlevel::WordLevel::builder()
            .vocab(
                [
                    ("<|startoftext|>".into(), 0),
                    ("<|im_end|>".into(), 1),
                    ("<|im_start|>".into(), 2),
                    ("[UNK]".into(), 3),
                ]
                .into_iter()
                .collect(),
            )
            .unk_token("[UNK]".into())
            .build()
            .unwrap();
    let tokenizer = tokenizers::Tokenizer::new(vocab);
    tokenizer
        .save(dir.path().join("tokenizer.json"), false)
        .unwrap();
    let cancelled = AtomicBool::new(false);
    let mut engine = Engine::load(dir.path(), &cancelled, &mut |_| {}).unwrap();
    // Captured from the sequential 38f2d17 engine; allow cross-driver FP32 reduction noise.
    let reference: Vec<Vec<u32>> =
        serde_json::from_str(include_str!("fixtures/lfm3-reduced-logits.json")).unwrap();
    assert_eq!(reference.len(), 8);
    let mut last = vec![];
    for (index, token) in [3, 4, 5, 6, 7, 8, 9, 10].into_iter().enumerate() {
        let before = engine.model.gpu.submission_count();
        last = engine
            .model
            .forward(token, true, &cancelled)
            .unwrap()
            .unwrap();
        assert_eq!(engine.model.gpu.submission_count() - before, 22);
        assert_eq!(reference[index].len(), last.len());
        for (actual, bits) in last.iter().zip(&reference[index]) {
            let expected = f32::from_bits(*bits);
            assert!((actual - expected).abs() <= 0.0005 + expected.abs() * 0.002);
        }
        assert_eq!(last.len(), 16);
        assert!(last.iter().all(|v| v.is_finite()));
        assert!(last.iter().any(|v| v.abs() > 0.01));
    }
    engine.model.reset();
    let first = engine.model.forward(3, true, &cancelled).unwrap().unwrap();
    assert_ne!(first, last);
    engine.model.reset();
    assert_eq!(
        first,
        engine.model.forward(3, true, &cancelled).unwrap().unwrap()
    );
    let messages = [Message {
        role: "user".into(),
        content: "Hello".into(),
    }];
    let mut events = vec![];
    engine
        .generate(&messages, 4, &cancelled, &mut |event| {
            events.push(event);
            Ok(())
        })
        .unwrap();
    assert!(
        events
            .iter()
            .any(|e| e["type"] == "text" && !e["text"].as_str().unwrap().is_empty())
    );
    let done = events.last().unwrap();
    assert_eq!(done["type"], "done");
    assert_eq!(done["performance"]["gpu"]["automatic_calibration"], false);
    assert_eq!(done["performance"]["gpu"]["scratch_mapping"], "persistent");
    for phase in ["prefill", "decode"] {
        let p = &done["performance"][phase];
        assert!(p["wait_ns"].as_u64().unwrap() > 0);
        assert!(p["model_ns"].as_u64().unwrap() >= p["wait_ns"].as_u64().unwrap());
        assert!(p["head_ns"].as_u64().unwrap() > 0);
    }
    assert!(!done["performance"].to_string().contains("Hello"));
    assert_eq!(
        done["prefill_submissions"].as_u64().unwrap(),
        done["prompt_tokens"].as_u64().unwrap() * 21 + 1
    );
    assert_eq!(
        done["decode_submissions"].as_u64().unwrap(),
        done["decode_tokens"].as_u64().unwrap() * 22
    );
    cancelled.store(true, Ordering::Relaxed);
    assert!(
        engine
            .generate(&messages, 4, &cancelled, &mut |_| Ok(()))
            .is_err()
    );
    cancelled.store(false, Ordering::Relaxed);
    engine.model.context_limit = 1;
    assert!(
        engine
            .generate(&messages, 4, &cancelled, &mut |_| Ok(()))
            .is_err()
    );
}
