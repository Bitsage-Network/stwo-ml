//! Integration test: verify `prove-model audit` CLI produces correct output.
//!
//! Creates a real inference log, builds a matching ONNX model via
//! `build_mlp_with_weights`, saves it to disk, then runs the audit
//! orchestrator directly (same code path as the CLI).

use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use stwo::core::fields::m31::M31;

use stwo_ml::audit::log::InferenceLog;
use stwo_ml::audit::orchestrator::run_audit_dry;
use stwo_ml::audit::replay::execute_forward_pass;
use stwo_ml::audit::types::{AuditRequest, InferenceLogEntry, ModelInfo};
use stwo_ml::aggregation::compute_io_commitment;
use stwo_ml::compiler::onnx::build_mlp_with_weights;
use stwo_ml::components::activation::ActivationType;
use stwo_ml::components::matmul::M31Matrix;

fn temp_dir() -> PathBuf {
    let d = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("stwo_ml_cli_audit_{}", d))
}

/// Full CLI-equivalent test: create log -> audit -> comprehensive report.
/// Verifies the report data is complete and well-structured.
#[test]
fn test_cli_audit_dry_run_output() {
    let dir = temp_dir();

    let model = build_mlp_with_weights(4, &[4], 2, ActivationType::ReLU, 42);
    let graph = &model.graph;
    let weights = &model.weights;

    // Create inference log with 5 entries
    let mut log = InferenceLog::new(&dir, "0x2", "0xabc", "test-mlp").unwrap();
    let base_ts = 1_700_000_000_000_000_000u64;

    for i in 0..5u64 {
        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 {
            input.set(0, j, M31::from(i as u32 * 4 + j as u32 + 1));
        }
        let output = execute_forward_pass(graph, &input, weights).unwrap();
        let io_commitment = compute_io_commitment(&input, &output);

        let input_data: Vec<u32> = input.data.iter().map(|m| m.0).collect();
        let (off, sz) = log.write_matrix(1, 4, &input_data).unwrap();

        let entry = InferenceLogEntry {
            inference_id: i,
            sequence_number: 0,
            model_id: "0x2".to_string(),
            weight_commitment: "0xabc".to_string(),
            model_name: "test-mlp".to_string(),
            num_layers: 3,
            input_tokens: vec![1, 2, 3, 4],
            output_tokens: vec![5, 6],
            matrix_offset: off,
            matrix_size: sz,
            input_rows: 1,
            input_cols: 4,
            output_rows: output.rows as u32,
            output_cols: output.cols as u32,
            io_commitment: format!("{:#066x}", io_commitment),
            layer_chain_commitment: "0x0".to_string(),
            prev_entry_hash: String::new(),
            entry_hash: String::new(),
            timestamp_ns: base_ts + i * 60_000_000_000,
            latency_ms: 42 + i % 10,
            gpu_device: "NVIDIA-H100".to_string(),
            tee_report_hash: "0x0".to_string(),
            task_category: Some("inference".to_string()),
            input_preview: Some(format!("What is {}+{}?", i + 1, i + 2)),
            output_preview: Some(format!("The answer is {}", i * 2 + 3)),
        };
        log.append(entry).unwrap();
    }

    let model_info = ModelInfo {
        model_id: "0x2".to_string(),
        name: "test-mlp".to_string(),
        architecture: "mlp".to_string(),
        parameters: format!("{}", model.metadata.num_parameters),
        layers: 3,
        weight_commitment: "0xabc".to_string(),
    };

    let request = AuditRequest {
        start_ns: 0,
        end_ns: u64::MAX,
        model_id: "0x2".to_string(),
        ..AuditRequest::default()
    };

    let t0 = std::time::Instant::now();
    let report = run_audit_dry(&log, graph, weights, request, model_info).unwrap();
    let elapsed = t0.elapsed();

    // ── Print the designed report (mirrors prove_model.rs output) ────────

    let w = 72usize;
    let inner = w - 2; // box inner width
    let box_border: String = std::iter::repeat('\u{2550}').take(inner).collect();
    let footer: String = std::iter::repeat('\u{2550}').take(w).collect();

    let section = |title: &str| -> String {
        let prefix = format!("\u{2500}\u{2500}\u{2500} {} ", title);
        let prefix_w = prefix.chars().count();
        let fill = w.saturating_sub(prefix_w);
        format!("{}{}", prefix, "\u{2500}".repeat(fill))
    };

    let section_info = |title: &str, info: &str| -> String {
        let prefix = format!("\u{2500}\u{2500}\u{2500} {} ", title);
        let suffix = format!(" {} \u{2500}\u{2500}\u{2500}", info);
        let prefix_w = prefix.chars().count();
        let suffix_w = suffix.chars().count();
        let fill = w.saturating_sub(prefix_w + suffix_w);
        format!("{}{}{}", prefix, "\u{2500}".repeat(fill.max(1)), suffix)
    };

    let bar = |value: u32, max: u32, bw: usize| -> String {
        if max == 0 { return "\u{2591}".repeat(bw); }
        let filled = ((value as f64 / max as f64) * bw as f64).round() as usize;
        let filled = filled.min(bw);
        let empty = bw - filled;
        format!("{}{}", "\u{2588}".repeat(filled), "\u{2591}".repeat(empty))
    };

    // Title
    eprintln!();
    eprintln!("\u{2554}{}\u{2557}", box_border);
    let title = "VERIFIABLE INFERENCE AUDIT REPORT";
    let pad = inner.saturating_sub(title.len());
    let lpad = pad / 2;
    let rpad = pad - lpad;
    eprintln!("\u{2551}{}{}{}\u{2551}", " ".repeat(lpad), title, " ".repeat(rpad));
    eprintln!("\u{255a}{}\u{255d}", box_border);

    eprintln!();
    eprintln!("  Audit       {}", report.audit_id);
    eprintln!("  Generated   {}", report.metadata.generated_at);

    // Model
    eprintln!();
    eprintln!("{}", section("MODEL"));
    eprintln!();
    eprintln!("  {:<56}ID  {}", report.model.name, report.model.model_id);
    eprintln!("  {} \u{00b7} {} parameters \u{00b7} {} layers",
        report.model.architecture, report.model.parameters, report.model.layers);
    eprintln!("  Weight  {}", report.model.weight_commitment);

    // Time window
    let duration_secs = report.time_window.duration_seconds;
    let dur_str = if duration_secs < 60 { format!("{}s", duration_secs) }
        else { format!("{}m {}s", duration_secs / 60, duration_secs % 60) };
    eprintln!();
    eprintln!("{}", section_info("TIME WINDOW", &dur_str));
    eprintln!();
    eprintln!("  {}  \u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}  {}",
        report.time_window.start, report.time_window.end);

    // Inference summary
    let s = &report.inference_summary;
    eprintln!();
    let inf_label = format!("{} inferences", s.total_inferences);
    eprintln!("{}", section_info("INFERENCE SUMMARY", &inf_label));
    eprintln!();
    eprintln!("  {:<38}Throughput  {:.1} tok/s",
        format!("Tokens    {} in \u{00b7} {} out", s.total_input_tokens, s.total_output_tokens),
        s.throughput_tokens_per_sec);
    let latency_str = format!("Latency   avg {}ms \u{00b7} p95 {}ms", s.avg_latency_ms, s.p95_latency_ms);
    if !s.categories.is_empty() {
        let mut cats: Vec<_> = s.categories.iter().collect();
        cats.sort_by(|a, b| b.1.cmp(a.1));
        let cat_str: Vec<String> = cats.iter().map(|(k, v)| format!("{} ({})", k, v)).collect();
        eprintln!("  {:<38}Categories  {}", latency_str, cat_str.join(", "));
    }

    // Semantic evaluation
    if let Some(ref sem) = report.semantic_evaluation {
        let sem_info = format!("{} \u{00b7} {}/{} checked",
            sem.method, sem.evaluated_count, s.total_inferences);
        eprintln!();
        eprintln!("{}", section_info("SEMANTIC EVALUATION", &sem_info));
        eprintln!();
        eprintln!("  Average Quality   {:.1}%", sem.avg_quality_score * 100.0);
        eprintln!();

        let max_bucket = sem.excellent_count.max(sem.good_count).max(sem.fair_count).max(sem.poor_count).max(1);
        let bw = 20;
        eprintln!("  Excellent  {}  {:>3}", bar(sem.excellent_count, max_bucket, bw), sem.excellent_count);
        eprintln!("  Good       {}  {:>3}", bar(sem.good_count, max_bucket, bw), sem.good_count);
        eprintln!("  Fair       {}  {:>3}", bar(sem.fair_count, max_bucket, bw), sem.fair_count);
        eprintln!("  Poor       {}  {:>3}", bar(sem.poor_count, max_bucket, bw), sem.poor_count);

        let det_total = sem.deterministic_pass + sem.deterministic_fail;
        if det_total > 0 {
            let pct = (sem.deterministic_pass as f64 / det_total as f64 * 100.0) as u32;
            eprintln!();
            eprintln!("  Deterministic   {} pass \u{00b7} {} fail ({}% pass rate)",
                sem.deterministic_pass, sem.deterministic_fail, pct);
        }

        if !sem.per_inference.is_empty() {
            let show_count = sem.per_inference.len().min(10);
            eprintln!();
            eprintln!("  Per-Inference ({}/{}):", show_count, sem.per_inference.len());
            eprintln!("  {:>4}  {:>5}  {:>7}  {:>5}  {}", "#", "Seq", "Score", "Det", "Status");
            eprintln!("  {:─>4}  {:─>5}  {:─>7}  {:─>5}  {:─>8}", "", "", "", "", "");

            for eval in sem.per_inference.iter().take(show_count) {
                let score_str = eval.semantic_score
                    .map(|sc| format!("{:.1}%", sc * 100.0))
                    .unwrap_or_else(|| "\u{2014}".to_string());
                let det_pass = eval.deterministic_checks.iter().filter(|c| c.passed).count();
                let det_total = eval.deterministic_checks.len();
                let status = if eval.deterministic_checks.iter().all(|c| c.passed) {
                    if eval.evaluation_proved { "proved" } else { "ok" }
                } else { "FAIL" };
                eprintln!("  {:>4}  {:>5}  {:>7}  {:>5}  {}",
                    eval.sequence, eval.sequence, score_str,
                    format!("{}/{}", det_pass, det_total), status);
            }
        }
    }

    // Proof & infrastructure
    let proof = &report.proof;
    let infra = &report.infrastructure;
    eprintln!();
    eprintln!("{}", section("PROOF & INFRASTRUCTURE"));
    eprintln!();
    let proof_str = if proof.proving_time_seconds > 0 {
        format!("Proof       {} \u{00b7} {}s", proof.mode, proof.proving_time_seconds)
    } else {
        format!("Proof       {}", proof.mode)
    };
    eprintln!("  {:<38}GPU         {}", proof_str, infra.gpu_device);
    eprintln!("  {:<38}TEE         {}",
        format!("Prover      stwo-ml v{}", infra.prover_version),
        if infra.tee_active { "active" } else { "inactive" });

    // Commitments
    eprintln!();
    eprintln!("{}", section("COMMITMENTS"));
    eprintln!();
    eprintln!("  IO root      {}", report.commitments.io_merkle_root);
    eprintln!("  Log root     {}", report.commitments.inference_log_merkle_root);
    eprintln!("  Weight       {}", report.commitments.weight_commitment);
    eprintln!("  Chain        {}", report.commitments.combined_chain_commitment);
    eprintln!("  Report hash  {}", report.commitments.audit_report_hash);

    // Inferences
    if !report.inferences.is_empty() {
        let show_count = report.inferences.len().min(5);
        let inf_info = format!("showing {} of {}", show_count, report.inferences.len());
        eprintln!();
        eprintln!("{}", section_info("INFERENCES", &inf_info));

        for entry in report.inferences.iter().take(show_count) {
            eprintln!();
            let cat = entry.category.as_deref().unwrap_or("-");
            let time_short = entry.timestamp.find('T')
                .map(|i| &entry.timestamp[i + 1..entry.timestamp.len().min(i + 9)])
                .unwrap_or(&entry.timestamp);
            eprintln!("  #{:<3} {}  {}ms  {}", entry.index, time_short, entry.latency_ms, cat);
            if let Some(ref preview) = entry.input_preview {
                eprintln!("       \u{25b8} {}", preview);
            }
            if let Some(ref preview) = entry.output_preview {
                eprintln!("       \u{25c2} {}", preview);
            }
        }
    }

    // Footer
    eprintln!();
    eprintln!("{}", footer);
    let time_str = format!("Completed in {:.2}s", elapsed.as_secs_f64());
    let file_str = "audit_report.json";
    let gap = w.saturating_sub(2 + time_str.len() + file_str.len());
    eprintln!("  {}{}{}", time_str, " ".repeat(gap), file_str);
    eprintln!("{}", footer);

    // ── Assertions ───────────────────────────────────────────────────────

    // Core report
    assert_eq!(report.inference_summary.total_inferences, 5);
    assert_eq!(report.model.model_id, "0x2");
    assert_eq!(report.model.architecture, "mlp");
    assert!(!report.commitments.io_merkle_root.is_empty());
    assert!(!report.commitments.combined_chain_commitment.is_empty());

    // Semantic evaluation
    let sem = report.semantic_evaluation.as_ref().unwrap();
    assert_eq!(sem.evaluated_count, 5);
    assert_eq!(sem.per_inference.len(), 5);
    assert!(sem.deterministic_pass + sem.deterministic_fail > 0);

    // Score distribution matches inferences with scores
    let dist_total = sem.excellent_count + sem.good_count + sem.fair_count + sem.poor_count;
    let scored_count = sem.per_inference.iter()
        .filter(|e| e.semantic_score.is_some())
        .count() as u32;
    assert_eq!(dist_total, scored_count);

    // Inference entries
    assert_eq!(report.inferences.len(), 5);
    for (i, entry) in report.inferences.iter().enumerate() {
        assert_eq!(entry.index, i as u32);
        assert_eq!(entry.input_tokens, 4);
        assert_eq!(entry.output_tokens, 2);
        assert!(entry.input_preview.is_some());
        assert!(entry.output_preview.is_some());
    }

    // Infrastructure
    assert!(!report.infrastructure.prover_version.is_empty());

    // Time window
    assert!(report.time_window.duration_seconds > 0);

    let _ = std::fs::remove_dir_all(&dir);
}
