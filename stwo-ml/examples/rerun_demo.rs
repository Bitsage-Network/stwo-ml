//! `rerun_demo` — stream a tiny MLP proof to the Rerun viewer.
//!
//! Builds a 4→16→16→4 MLP with random weights, proves it with the CPU GKR prover,
//! and streams all proof events (LayerStart/End, GKR claim, activations, STARK) to
//! a Rerun viewer in real time.
//!
//! Run with:
//! ```bash
//! cargo run --example rerun_demo -F proof-stream-rerun
//! ```
//! This spawns the Rerun viewer automatically.
//!
//! Or connect to an already-running viewer:
//! ```bash
//! rerun &
//! cargo run --example rerun_demo -F proof-stream-rerun -- rerun+http://127.0.0.1:9876/proxy
//! ```

#![feature(portable_simd)]

use stwo_ml::aggregation::prove_model_pure_gkr_auto;
use stwo_ml::compiler::onnx::build_mlp_with_weights;
use stwo_ml::components::activation::ActivationType;
use stwo_ml::components::matmul::M31Matrix;
use stwo_ml::gadgets::quantize::QuantStrategy;
use stwo_ml::gkr::prover::set_proof_sink;

fn main() {
    // First arg: viewer address ("spawn" | "file:out.rrd" | "host:port")
    // Second arg: optional .rrd save path alongside live view
    let rerun_addr = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "spawn".to_string());
    let save_path = std::env::args().nth(2); // e.g. "proof.rrd"
    eprintln!("proof-stream demo: connecting to Rerun via '{rerun_addr}'");
    if let Some(ref p) = save_path {
        eprintln!("  also saving replay to '{p}'");
    }

    // ── Build a 16→128→256→128→16 MLP — large enough to produce many
    //    SumcheckRound events (log2(256)=8 rounds per MatMul) so the wave
    //    animation has real content to show. ──────────────────────────────────
    let model = build_mlp_with_weights(16, &[128, 256, 128], 16, ActivationType::ReLU, 42);
    eprintln!(
        "Model: {} layers, input={:?}, output={:?}",
        model.graph.num_layers(),
        model.input_shape,
        model.graph.output_shape,
    );

    // ── Input ─────────────────────────────────────────────────────────────────
    use stwo::core::fields::m31::M31;
    let input = M31Matrix {
        data: (0..16).map(|i| M31::from(i * 50 + 1)).collect(),
        rows: 1,
        cols: 16,
    };

    // ── Install Rerun sink ────────────────────────────────────────────────────
    #[cfg(feature = "proof-stream-rerun")]
    let _guard = {
        match proof_stream::sink_from_str(&rerun_addr, "stwo-ml-proof-demo") {
            Ok(sink) => {
                eprintln!("[proof-stream] Rerun sink connected");
                Some(set_proof_sink(sink))
            }
            Err(e) => {
                eprintln!("[proof-stream] Warning: could not connect ({e}); running without viz");
                None
            }
        }
    };

    // ── Prove ─────────────────────────────────────────────────────────────────
    eprintln!("Proving...");
    let t0 = std::time::Instant::now();
    match prove_model_pure_gkr_auto(&model.graph, &input, &model.weights) {
        Ok(proof) => {
            let n_layer_proofs = proof
                .gkr_proof
                .as_ref()
                .map(|g| g.layer_proofs.len())
                .unwrap_or(0);
            eprintln!(
                "Proof complete in {:.2}s — {} matmul proofs",
                t0.elapsed().as_secs_f64(),
                n_layer_proofs,
            );
        }
        Err(e) => {
            eprintln!("Proving failed: {e}");
            std::process::exit(1);
        }
    }

    eprintln!("Done. Events were streamed to Rerun.");
}
