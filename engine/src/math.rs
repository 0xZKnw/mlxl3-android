//! Small LFM3 operations on CPU; the bandwidth-heavy projections execute in Vulkan.
use anyhow::{Result, ensure};
use half::f16;

pub fn half(value: f32) -> f32 {
    f16::from_f32(value).to_f32()
}

pub fn hadamard(values: &mut [f32]) -> Result<()> {
    ensure!(
        !values.is_empty() && values.len().is_multiple_of(128),
        "Hadamard requires 128-element blocks"
    );
    for block in values.as_chunks_mut::<128>().0 {
        let mut stride = 1;
        while stride < 128 {
            for base in (0..128).step_by(stride * 2) {
                for offset in 0..stride {
                    let a = block[base + offset];
                    let b = block[base + offset + stride];
                    block[base + offset] = a + b;
                    block[base + offset + stride] = a - b;
                }
            }
            stride *= 2;
        }
        for x in block {
            *x = half(*x / 128.0f32.sqrt());
        }
    }
    Ok(())
}

pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Result<Vec<f32>> {
    ensure!(
        !x.is_empty() && x.len() == weight.len() && eps.is_finite() && eps > 0.,
        "Invalid RMSNorm shape or epsilon"
    );
    let scale = (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + eps)
        .sqrt()
        .recip();
    let result: Vec<_> = x
        .iter()
        .zip(weight)
        .map(|(x, w)| half(x * scale * half(*w)))
        .collect();
    ensure!(
        result.iter().all(|v| v.is_finite()),
        "RMSNorm produced non-finite values"
    );
    Ok(result)
}

pub fn rope(
    heads: &mut [f32],
    dim: usize,
    theta: f32,
    fraction: f32,
    position: usize,
) -> Result<()> {
    ensure!(
        dim > 0
            && dim.is_multiple_of(2)
            && heads.len().is_multiple_of(dim)
            && theta.is_finite()
            && theta > 0.
            && fraction.is_finite()
            && fraction > 0.
            && fraction <= 1.,
        "Invalid RoPE parameters"
    );
    let active = (fraction * dim as f32) as usize / 2;
    for head in heads.chunks_exact_mut(dim) {
        for j in 0..active {
            let angle = position as f32 / theta.powf((2 * j) as f32 / dim as f32);
            let (sin, cos) = angle.sin_cos();
            let a = head[j];
            let b = head[j + dim / 2];
            head[j] = half(a * cos - b * sin);
            head[j + dim / 2] = half(b * cos + a * sin);
        }
    }
    Ok(())
}

pub fn softmax(scores: &mut [f32]) -> Result<()> {
    ensure!(
        !scores.is_empty() && scores.iter().all(|x| x.is_finite()),
        "Invalid attention scores"
    );
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.;
    for score in scores.iter_mut() {
        *score = (*score - max).exp();
        sum += *score;
    }
    ensure!(sum.is_finite() && sum > 0., "Invalid softmax sum");
    for score in scores {
        *score /= sum;
    }
    Ok(())
}

pub fn context_fits(prompt: usize, generation: usize, limit: usize) -> bool {
    prompt > 0 && generation > 0 && prompt.checked_add(generation).is_some_and(|n| n <= limit)
}

pub fn greedy(logits: &[f32]) -> Result<u32> {
    ensure!(
        !logits.is_empty() && logits.len() <= 128000 && logits.iter().all(|v| v.is_finite()),
        "Invalid logits"
    );
    let mut best = 0;
    for i in 1..logits.len() {
        if logits[i] > logits[best] {
            best = i;
        }
    }
    Ok(best as u32)
}

pub fn greedy_in_vocabulary(logits: &[f32], vocabulary: &[u32]) -> Result<u32> {
    ensure!(
        !logits.is_empty() && logits.iter().all(|v| v.is_finite()) && !vocabulary.is_empty(),
        "Invalid vocabulary or logits"
    );
    let mut best = vocabulary[0];
    let mut score = *logits
        .get(best as usize)
        .ok_or_else(|| anyhow::anyhow!("Token ID exceeds logits"))?;
    for &id in &vocabulary[1..] {
        let candidate = *logits
            .get(id as usize)
            .ok_or_else(|| anyhow::anyhow!("Token ID exceeds logits"))?;
        if candidate > score {
            best = id;
            score = candidate;
        }
    }
    Ok(best)
}

#[cfg(kani)]
#[kani::proof]
fn context_budget_never_wraps() {
    let prompt: usize = kani::any();
    let generation: usize = kani::any();
    let limit: usize = kani::any();
    let accepted = context_fits(prompt, generation, limit);
    if accepted {
        assert!(prompt > 0 && generation > 0);
        assert!(prompt <= limit && generation <= limit - prompt);
    }
    kani::cover!(accepted);
    kani::cover!(prompt.checked_add(generation).is_none());
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn hadamard_matches_independent_sylvester_matrix() {
        let input: Vec<_> = (0..256)
            .map(|i| ((i * 31 % 47) as f32 - 23.) / 32.)
            .collect();
        let mut result = input.clone();
        hadamard(&mut result).unwrap();
        for (i, actual) in result.iter().enumerate() {
            let row = i % 128;
            let expected = (0..128)
                .map(|j| {
                    let sign = if (row & j).count_ones() % 2 == 0 {
                        1.
                    } else {
                        -1.
                    };
                    f64::from(input[(i / 128) * 128 + j]) * sign
                })
                .sum::<f64>()
                / 128f64.sqrt();
            assert!((actual - half(expected as f32)).abs() <= 0.002);
        }
        assert!(hadamard(&mut []).is_err());
        assert!(hadamard(&mut [0.; 127]).is_err());
    }
    #[test]
    fn partial_rope_uses_full_head_frequency_and_zero_tail() {
        let mut x = vec![1.; 8];
        rope(&mut x, 8, 10000., 0.5, 3).unwrap();
        assert!((x[1] - half(0.3f32.cos() - 0.3f32.sin())).abs() < 0.001);
        assert_eq!([x[2], x[3], x[6], x[7]], [1.; 4]);
        assert!(rope(&mut x, 0, 10000., 0.5, 0).is_err());
        assert!(rope(&mut x, 8, f32::NAN, 1., 0).is_err());
    }
    #[test]
    fn boundaries_and_numeric_failures() {
        assert!(context_fits(1, 1, 2));
        assert!(!context_fits(0, 1, 2));
        assert!(!context_fits(usize::MAX, 1, usize::MAX));
        assert!(!context_fits(1, 0, 2));
        let mut scores = [1000., 1001., 999.];
        softmax(&mut scores).unwrap();
        assert!((scores[1] - 0.66524096).abs() < 1e-6);
        assert!(softmax(&mut [f32::INFINITY]).is_err());
        assert_eq!(greedy(&[3., 3., 2.]).unwrap(), 0);
        assert!(greedy(&[]).is_err());
        assert!(greedy(&[f32::NAN]).is_err());
        assert_eq!(greedy_in_vocabulary(&[1., 2., 999.], &[0, 1]).unwrap(), 1);
        assert!(greedy_in_vocabulary(&[1.], &[2]).is_err());
        let norm = rms_norm(&[3., 4.], &[1., 2.], 1e-5).unwrap();
        assert!((norm[0] - 0.848528).abs() < 0.001);
        assert!((norm[1] - 2.262742).abs() < 0.002);
        assert!(rms_norm(&[], &[], 1e-5).is_err());
    }
}
