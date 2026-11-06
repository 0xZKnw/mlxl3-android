use serde_json::{Value, json};
use std::collections::BTreeMap;

pub const MAX_PROJECTIONS: usize = 193;

#[derive(Clone, serde::Serialize)]
pub struct ProjectionMeta {
    pub name: String,
    pub input: usize,
    pub output: usize,
    pub rows: usize,
    pub cols: usize,
    pub bits: u32,
    pub codebook: u32,
    pub kernel: &'static str,
    pub packed_bytes: u64,
}

#[derive(Clone, Copy, Default, serde::Serialize)]
pub struct ProjectionTiming {
    pub samples: u64,
    pub query_failures: u64,
    pub input_ns: u64,
    pub matvec_ns: u64,
    pub output_ns: u64,
    pub min_matvec_ns: u64,
    pub max_matvec_ns: u64,
}
impl ProjectionTiming {
    pub fn record(&mut self, durations: Option<[u64; 3]>) {
        let Some([input, matvec, output]) = durations else {
            self.query_failures = self.query_failures.saturating_add(1);
            return;
        };
        self.min_matvec_ns = if self.samples == 0 {
            matvec
        } else {
            self.min_matvec_ns.min(matvec)
        };
        self.max_matvec_ns = self.max_matvec_ns.max(matvec);
        self.samples = self.samples.saturating_add(1);
        self.input_ns = self.input_ns.saturating_add(input);
        self.matvec_ns = self.matvec_ns.saturating_add(matvec);
        self.output_ns = self.output_ns.saturating_add(output);
    }
    fn merge(&mut self, other: Self) {
        if other.samples > 0 {
            self.min_matvec_ns = if self.samples == 0 {
                other.min_matvec_ns
            } else {
                self.min_matvec_ns.min(other.min_matvec_ns)
            };
            self.max_matvec_ns = self.max_matvec_ns.max(other.max_matvec_ns);
        }
        self.samples = self.samples.saturating_add(other.samples);
        self.query_failures = self.query_failures.saturating_add(other.query_failures);
        self.input_ns = self.input_ns.saturating_add(other.input_ns);
        self.matvec_ns = self.matvec_ns.saturating_add(other.matvec_ns);
        self.output_ns = self.output_ns.saturating_add(other.output_ns);
    }
    fn json(self, packed_bytes: u64) -> Value {
        let mut result = json!(self);
        result["mean_matvec_ns"] = if self.samples == 0 {
            Value::Null
        } else {
            json!(self.matvec_ns as f64 / self.samples as f64)
        };
        // Logical compressed bytes, not measured DRAM traffic or physical bandwidth.
        result["logical_weight_gb_per_second"] = if self.samples == 0 || self.matvec_ns == 0 {
            Value::Null
        } else {
            json!(packed_bytes as f64 * self.samples as f64 / self.matvec_ns as f64)
        };
        if self.samples == 0 {
            result["min_matvec_ns"] = Value::Null;
            result["max_matvec_ns"] = Value::Null;
        }
        result
    }
}

#[derive(Clone, Default, serde::Serialize)]
pub struct Profile {
    pub upload_ns: u64,
    pub record_submit_ns: u64,
    pub wait_ns: u64,
    pub readback_ns: u64,
    pub projection_cpu_ns: u64,
    pub model_ns: u64,
    pub head_ns: u64,
    pub device_compute_ns: u64,
    pub device_queue_ns: u64,
    pub device_queue_samples: u64,
    pub device_swiglu_ns: u64,
    pub device_convolution_ns: u64,
    pub device_attention_gate_ns: u64,
    pub device_attention_gate_samples: u64,
    pub device_samples: u64,
    pub device_exl3_samples: u64,
    pub device_exl3_input_ns: u64,
    pub device_exl3_matvec_ns: u64,
    pub device_exl3_output_ns: u64,
    pub device_other_ns: u64,
    pub query_read_ns: u64,
    pub query_read_failures: u64,
    pub last_query_error: Option<String>,
    pub normalization_ns: u64,
    pub attention_cpu_ns: u64,
    pub convolution_cpu_ns: u64,
    pub rope_ns: u64,
    pub activation_ns: u64,
    pub embedding_ns: u64,
    #[serde(skip)]
    pub projections: Vec<ProjectionTiming>,
}

impl Profile {
    pub fn record_projection(&mut self, index: usize, durations: Option<[u64; 3]>) -> bool {
        if index >= MAX_PROJECTIONS {
            return false;
        }
        if self.projections.len() <= index {
            self.projections
                .resize(index + 1, ProjectionTiming::default());
        }
        self.projections[index].record(durations);
        true
    }

    pub fn report(&self, metadata: &[ProjectionMeta]) -> Value {
        let mut result = json!(self);
        let measured_host = [
            self.upload_ns,
            self.record_submit_ns,
            self.wait_ns,
            self.readback_ns,
            self.projection_cpu_ns,
            self.query_read_ns,
            self.normalization_ns,
            self.attention_cpu_ns,
            self.convolution_cpu_ns,
            self.rope_ns,
            self.activation_ns,
            self.embedding_ns,
        ]
        .into_iter()
        .fold(0u64, u64::saturating_add);
        result["host_unattributed_ns"] = json!(self.model_ns.saturating_sub(measured_host));
        let mut groups = BTreeMap::new();
        let mut slowest = vec![];
        for (i, meta) in metadata.iter().take(MAX_PROJECTIONS).enumerate() {
            let timing = self.projections.get(i).copied().unwrap_or_default();
            let key = (
                meta.input,
                meta.output,
                meta.rows,
                meta.cols,
                meta.bits,
                meta.codebook,
                meta.kernel,
            );
            let group = groups.entry(key).or_insert((
                0usize,
                ProjectionTiming::default(),
                meta.packed_bytes,
            ));
            group.0 += 1;
            group.1.merge(timing);
            if timing.samples > 0 || timing.query_failures > 0 {
                let mut row = json!(meta);
                row["timing"] = timing.json(meta.packed_bytes);
                slowest.push((timing.matvec_ns, i, row));
            }
        }
        slowest.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        result["slowest_projections"] =
            Value::Array(slowest.into_iter().take(8).map(|(_, _, row)| row).collect());
        result["projection_groups"] = Value::Array(groups.into_iter().map(|((input, output, rows, cols, bits, cb, kernel), (matrices, timing, bytes))| {
            json!({"input":input,"output":output,"padded_input":rows,"padded_output":cols,
                "bits":bits,"codebook":cb,"kernel":kernel,"matrices":matrices,
                "packed_bytes_per_matrix":bytes,"timing":timing.json(bytes)})
        }).collect());
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn meta(name: &str) -> ProjectionMeta {
        ProjectionMeta {
            name: name.into(),
            input: 128,
            output: 128,
            rows: 128,
            cols: 128,
            bits: 4,
            codebook: 1,
            kernel: "test",
            packed_bytes: 8192,
        }
    }
    #[test]
    fn reports_group_sums_top_order_missing_queries_and_reset() {
        let metadata: Vec<_> = (0..10)
            .map(|i| meta(&format!("model.layers.{i}.w1")))
            .collect();
        let mut profile = Profile::default();
        for i in 0..10 {
            assert!(profile.record_projection(i, Some([1, (i + 1) as u64 * 10, 2])));
        }
        assert!(profile.record_projection(9, None));
        let report = profile.report(&metadata);
        assert_eq!(report["projection_groups"].as_array().unwrap().len(), 1);
        let timing = &report["projection_groups"][0]["timing"];
        assert_eq!(timing["samples"], 10);
        assert_eq!(timing["query_failures"], 1);
        assert_eq!(timing["matvec_ns"], 550);
        assert_eq!(timing["min_matvec_ns"], 10);
        assert_eq!(timing["max_matvec_ns"], 100);
        assert_eq!(report["slowest_projections"].as_array().unwrap().len(), 8);
        assert_eq!(
            report["slowest_projections"][0]["name"],
            "model.layers.9.w1"
        );
        assert!(!profile.record_projection(usize::MAX, Some([1, 2, 3])));
        assert_eq!(profile.projections.len(), 10);
        let missing = Profile::default().report(&metadata);
        assert!(missing["projection_groups"][0]["timing"]["mean_matvec_ns"].is_null());
        assert!(
            missing["projection_groups"][0]["timing"]["logical_weight_gb_per_second"].is_null()
        );
        let before = std::mem::take(&mut profile);
        assert_eq!(before.projections.len(), 10);
        assert!(profile.projections.is_empty());
    }
    #[test]
    fn timings_saturate_without_losing_failures_or_fabricating_samples() {
        let mut timing = ProjectionTiming::default();
        timing.record(None);
        assert_eq!(timing.samples, 0);
        timing.record(Some([u64::MAX, 0, 1]));
        timing.record(Some([1, u64::MAX, u64::MAX]));
        assert_eq!(timing.input_ns, u64::MAX);
        assert_eq!(timing.output_ns, u64::MAX);
        assert_eq!(timing.min_matvec_ns, 0);
        assert_eq!(timing.max_matvec_ns, u64::MAX);
        assert_eq!(timing.query_failures, 1);
        assert_eq!(timing.samples, 2);
        assert!(
            timing.json(u64::MAX)["logical_weight_gb_per_second"]
                .as_f64()
                .unwrap()
                .is_finite()
        );
    }
}

#[cfg(kani)]
mod proofs {
    use super::*;
    #[kani::proof]
    fn timing_records_preserve_counts_bounds_and_saturating_totals() {
        let a: [u64; 3] = kani::any();
        let b: [u64; 3] = kani::any();
        let ok_a: bool = kani::any();
        let ok_b: bool = kani::any();
        let mut timing = ProjectionTiming::default();
        timing.record(if ok_a { Some(a) } else { None });
        timing.record(if ok_b { Some(b) } else { None });
        assert_eq!(timing.samples, u64::from(ok_a) + u64::from(ok_b));
        assert_eq!(timing.samples + timing.query_failures, 2);
        assert!(timing.min_matvec_ns <= timing.max_matvec_ns);
        if ok_a && ok_b {
            assert_eq!(timing.input_ns, a[0].saturating_add(b[0]));
            assert_eq!(timing.matvec_ns, a[1].saturating_add(b[1]));
            assert_eq!(timing.output_ns, a[2].saturating_add(b[2]));
            assert_eq!(timing.min_matvec_ns, a[1].min(b[1]));
            assert_eq!(timing.max_matvec_ns, a[1].max(b[1]));
        }
        kani::cover!(!ok_a && !ok_b && timing.samples == 0);
        kani::cover!(ok_a && ok_b && a[1] == u64::MAX && b[1] == 1 && timing.matvec_ns == u64::MAX);
    }
}
