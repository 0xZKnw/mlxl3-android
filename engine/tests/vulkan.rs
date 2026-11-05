use half::f16;
use mlxl3_android::{
    codec::{self, Codebook},
    vulkan::Gpu,
};

#[test]
#[ignore = "requires a Vulkan 1.1 compute device; CI uses Mesa lavapipe"]
fn exl3_qmv_matches_cpu_reconstruction_all_bitrates_and_codebooks() {
    let mut gpu = Gpu::new().unwrap();
    eprintln!("Vulkan device: {}", gpu.name);
    for bits in 1..=8 {
        let source: Vec<u16> = (0..128 * 128)
            .map(|i| ((i * 177 + i / 7 * 313 + 512) & 65535) as u16)
            .collect();
        let packed = codec::pack(&source, bits).unwrap();
        let bytes: Vec<_> = packed.iter().flat_map(|v| v.to_le_bytes()).collect();
        let buffer = gpu.buffer(&bytes).unwrap();
        let x: Vec<_> = (0..128)
            .map(|i| ((i * 13 % 17) as f32 - 8.) / 64.)
            .collect();
        for cb in [Codebook::Default, Codebook::Mcg, Codebook::Mul1] {
            let matrix = codec::reconstruct_inner(&packed, bits, cb, 128, 128).unwrap();
            let actual = gpu
                .multiply(&buffer, &x, 128, bits as u32, cb as u32)
                .unwrap();
            assert_eq!(actual.len(), 128);
            assert!(actual.iter().any(|v| v.abs() > 0.01));
            for n in 0..128 {
                let expected: f64 = (0..128)
                    .map(|r| f64::from(x[r]) * f64::from(matrix[r * 128 + n].to_f32()))
                    .sum();
                assert!(
                    (f64::from(actual[n]) - expected).abs() <= 0.0001 + expected.abs() * 0.00002,
                    "K={bits} cb={cb:?} n={n}: {} != {expected}",
                    actual[n]
                );
            }
        }
        assert!(
            gpu.multiply(&buffer, &x[..127], 128, bits as u32, 0)
                .is_err()
        );
        assert!(gpu.multiply(&buffer, &x, 128, bits as u32, 3).is_err());
    }
}

#[test]
#[ignore = "requires a Vulkan 1.1 compute device; CI uses Mesa lavapipe"]
fn dense_gpu_checks_f16_bf16_and_invalid_shapes() {
    let mut gpu = Gpu::new().unwrap();
    let weights: Vec<_> = [1., 2., 3., 4., 5., 6.]
        .iter()
        .flat_map(|v| f16::from_f32(*v).to_bits().to_le_bytes())
        .collect();
    let buffer = gpu.buffer(&weights).unwrap();
    assert_eq!(
        gpu.multiply(&buffer, &[2., -1., 3.], 2, 0, 0).unwrap(),
        [9., 21.]
    );
    let weights: Vec<_> = [1., 2., 3., 4., 5., 6.]
        .iter()
        .flat_map(|v| half::bf16::from_f32(*v).to_bits().to_le_bytes())
        .collect();
    let buffer = gpu.buffer(&weights).unwrap();
    assert_eq!(
        gpu.multiply(&buffer, &[2., -1., 3.], 2, 0, 1).unwrap(),
        [9., 21.]
    );
    assert!(gpu.multiply(&buffer, &[f32::NAN; 3], 2, 0, 1).is_err());
    assert!(gpu.multiply(&buffer, &[0.; 3], 3, 0, 1).is_err());
    assert!(gpu.multiply(&buffer, &[], 2, 0, 0).is_err());
}
