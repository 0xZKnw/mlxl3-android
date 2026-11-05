use mlxl3_android::{model::Config, weights::Weights};
use serde_json::json;
use std::{fs, path::Path};

fn shard(path: &Path, header: serde_json::Value, payload: &[u8]) {
    let mut metadata = serde_json::to_vec(&header).unwrap();
    while !metadata.len().is_multiple_of(8) {
        metadata.push(b' ');
    }
    let mut bytes = (metadata.len() as u64).to_le_bytes().to_vec();
    bytes.extend(metadata);
    bytes.extend(payload);
    fs::write(path, bytes).unwrap();
}

#[test]
fn official_lfm3_config_is_accepted_and_unsupported_contracts_fail() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("config.json");
    let config: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/lfm3-config.json")).unwrap();
    fs::write(&path, serde_json::to_vec(&config).unwrap()).unwrap();
    Config::read(&path).unwrap();
    for (key, value) in [
        ("model_type", json!("lfm2")),
        ("architectures", json!(["Lfm3VlForConditionalGeneration"])),
        ("hidden_size", json!(usize::MAX)),
        ("num_attention_heads", json!(0)),
        ("num_key_value_heads", json!(3)),
        ("conv_L_cache", json!(3)),
        ("norm_eps", json!(-1.0)),
        ("layer_types", json!(["conv"])),
        ("vocab_size", json!(0)),
        ("use_sandwich_norm", json!(false)),
    ] {
        let mut bad = config.clone();
        bad[key] = value;
        fs::write(&path, serde_json::to_vec(&bad).unwrap()).unwrap();
        assert!(Config::read(&path).is_err(), "accepted invalid {key}");
    }
}

#[test]
fn safetensors_loader_reads_values_and_rejects_corruption() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("model.safetensors");
    let header = json!({"v":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}});
    let bytes: Vec<_> = [2.5f32, -3.].iter().flat_map(|v| v.to_le_bytes()).collect();
    shard(&path, header.clone(), &bytes);
    let weights = Weights::open(directory.path()).unwrap();
    assert_eq!(weights.floats("v", &[2]).unwrap(), [2.5, -3.]);
    assert!(weights.floats("v", &[1]).is_err());
    assert!(weights.bytes("missing").is_err());
    drop(weights);
    shard(&path, header.clone(), &bytes[..4]);
    assert!(Weights::open(directory.path()).is_err());
    shard(
        &path,
        json!({"v":{"dtype":"F32","shape":[2],"data_offsets":[8,16]}}),
        &bytes,
    );
    assert!(Weights::open(directory.path()).is_err());
    fs::write(&path, u64::MAX.to_le_bytes()).unwrap();
    assert!(Weights::open(directory.path()).is_err());
    shard(&path, header.clone(), &bytes);
    shard(
        &directory.path().join("duplicate.safetensors"),
        header,
        &bytes,
    );
    assert!(Weights::open(directory.path()).is_err());
}

#[test]
fn embedding_rows_and_non_finite_values_are_checked() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("model.safetensors");
    let header =
        json!({"model.embed_tokens.weight":{"dtype":"F16","shape":[2,2],"data_offsets":[0,8]}});
    let payload: Vec<_> = [1., 2., 3., 4.]
        .iter()
        .flat_map(|v| half::f16::from_f32(*v).to_bits().to_le_bytes())
        .collect();
    shard(&path, header.clone(), &payload);
    let weights = Weights::open(directory.path()).unwrap();
    assert_eq!(weights.embedding(1, 2, 2).unwrap(), [3., 4.]);
    assert!(weights.embedding(2, 2, 2).is_err());
    drop(weights);
    let nan: Vec<_> = [f32::NAN; 4]
        .iter()
        .flat_map(|v| half::f16::from_f32(*v).to_bits().to_le_bytes())
        .collect();
    shard(&path, header, &nan);
    assert!(
        Weights::open(directory.path())
            .unwrap()
            .embedding(0, 2, 2)
            .is_err()
    );
}
