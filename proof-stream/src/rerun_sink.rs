//! `RerunSink` — streams proof events to a Rerun viewer via a background thread.
//!
//! Events are dispatched through a bounded crossbeam channel (capacity 8192) so
//! the prover hot path never blocks waiting for Rerun I/O. Events are silently
//! dropped if the background thread falls behind.

#[cfg(feature = "rerun")]
mod inner {
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::thread::{self, JoinHandle};
    use std::time::Duration;

    use crossbeam::channel::{self, Receiver, Sender};
    use rerun::{RecordingStream, RecordingStreamBuilder};

    use crate::events::{LayerKind, LogLevel, ProofEvent};
    use crate::sink::ProofEventSink;

    // ── Rerun entity path helpers ─────────────────────────────────────────────

    fn layer_node_path(layer_idx: usize) -> String {
        format!("gkr/walk/layer_{layer_idx}/node")
    }

    fn layer_poly_path(layer_idx: usize) -> String {
        format!("gkr/walk/layer_{layer_idx}/poly")
    }

    fn layer_claim_path(layer_idx: usize) -> String {
        format!("gkr/walk/layer_{layer_idx}/claim")
    }

    fn inference_output_path(layer_idx: usize) -> String {
        format!("inference/layer_{layer_idx}/output")
    }

    fn inference_mean_path(layer_idx: usize) -> String {
        format!("inference/stats/layer_{layer_idx}/mean")
    }

    fn inference_std_path(layer_idx: usize) -> String {
        format!("inference/stats/layer_{layer_idx}/std")
    }

    fn inference_attn_path(layer_idx: usize, head_idx: usize) -> String {
        format!("inference/layer_{layer_idx}/attn_head_{head_idx}")
    }

    fn gpu_util_path(device_id: usize) -> String {
        format!("gpu/device_{device_id}/util")
    }

    fn gpu_free_path(device_id: usize) -> String {
        format!("gpu/device_{device_id}/free_gb")
    }

    // ── Position / color helpers ──────────────────────────────────────────────

    fn kind_color(kind: LayerKind) -> rerun::Color {
        let [r, g, b] = kind.color_rgb();
        rerun::Color::from_rgb(r, g, b)
    }

    /// Vertical offset for each layer kind so the 3D view shows circuit "height".
    fn kind_y(kind: LayerKind) -> f32 {
        match kind {
            LayerKind::MatMul => 0.0,
            LayerKind::Activation => 1.8,
            LayerKind::LayerNorm | LayerKind::RMSNorm => -1.2,
            LayerKind::Attention => 3.0,
            LayerKind::Add => 0.8,
            LayerKind::Mul => 1.2,
            LayerKind::Embedding => -2.5,
            LayerKind::Dequantize | LayerKind::Quantize => -1.8,
            _ => 0.0,
        }
    }

    /// X spacing between consecutive layers.
    const X_STEP: f32 = 3.0;

    fn layer_x(layer_idx: usize) -> f32 {
        layer_idx as f32 * X_STEP
    }

    /// 3D position for a layer node — helix in YZ so nodes spiral in depth.
    /// Y = sin(i * 0.9) * 1.8,  Z = cos(i * 0.9) * 1.8
    fn helix_pos(layer_idx: usize) -> [f32; 3] {
        let angle = layer_idx as f32 * 0.9_f32;
        let r = 1.8_f32;
        [layer_x(layer_idx), angle.sin() * r, angle.cos() * r]
    }

    // ── Connection config ─────────────────────────────────────────────────────

    /// Where the Rerun SDK should send data.
    pub enum RerunConnection {
        /// Stream to a running `rerun` viewer over TCP.
        Tcp { addr: String },
        /// Write a `.rrd` file for later replay.
        File { path: PathBuf },
        /// Spawn a viewer subprocess and stream to it.
        Spawn,
    }

    impl RerunConnection {
        /// Parse from a user-facing string:
        /// - `"spawn"` → `Spawn`
        /// - `"file:<path>"` → `File`
        /// - anything else → `Tcp`
        pub fn from_str(s: &str) -> Self {
            if s == "spawn" {
                RerunConnection::Spawn
            } else if let Some(path) = s.strip_prefix("file:") {
                RerunConnection::File {
                    path: PathBuf::from(path),
                }
            } else {
                RerunConnection::Tcp { addr: s.to_owned() }
            }
        }
    }

    // ── Background worker ─────────────────────────────────────────────────────

    enum Msg {
        Event(ProofEvent),
        Flush,
        Stop,
    }

    fn build_recording_stream(
        conn: RerunConnection,
        app_id: &str,
    ) -> Result<RecordingStream, Box<dyn std::error::Error + Send + Sync>> {
        let builder = RecordingStreamBuilder::new(app_id);
        match conn {
            RerunConnection::Tcp { addr } => {
                // Normalize to rerun+http://host:port/proxy URL
                let addr_str = addr
                    .strip_prefix("tcp://")
                    .unwrap_or(addr.as_str());
                // If bare "host:port", wrap in gRPC proxy URL
                let url = if addr_str.starts_with("rerun+") {
                    addr_str.to_owned()
                } else {
                    format!("rerun+http://{addr_str}/proxy")
                };
                let stream = builder.connect_grpc_opts(url)?;
                Ok(stream)
            }
            RerunConnection::File { path } => {
                let stream = builder.save(path)?;
                Ok(stream)
            }
            RerunConnection::Spawn => {
                let stream = builder.spawn()?;
                Ok(stream)
            }
        }
    }

    fn dispatch_event(rec: &RecordingStream, event: &ProofEvent) {
        match event {
            ProofEvent::CircuitCompiled { nodes, .. } => {
                let positions: Vec<[f32; 3]> = nodes
                    .iter()
                    .map(|n| [layer_x(n.layer_idx), kind_y(n.kind), 0.0])
                    .collect();
                let colors: Vec<rerun::Color> = nodes.iter().map(|n| kind_color(n.kind)).collect();
                let radii: Vec<rerun::Radius> = nodes
                    .iter()
                    .map(|n| {
                        rerun::Radius::new_ui_points(4.0 + (n.trace_cost as f32).sqrt() * 0.4)
                    })
                    .collect();
                let labels: Vec<String> = nodes
                    .iter()
                    .map(|n| format!("{:?} [{}]", n.kind, n.layer_idx))
                    .collect();
                let _ = rec.log_static(
                    "circuit/nodes",
                    &rerun::Points3D::new(positions.clone())
                        .with_colors(colors)
                        .with_radii(radii)
                        .with_labels(labels),
                );

                // DAG edges: one LineStrip per node with predecessors
                let mut edge_strips: Vec<Vec<[f32; 3]>> = Vec::new();
                for n in nodes {
                    let to = [layer_x(n.layer_idx), kind_y(n.kind), 0.0];
                    for &from_idx in &n.input_layers {
                        if let Some(m) = nodes.iter().find(|m| m.layer_idx == from_idx) {
                            edge_strips.push(vec![
                                [layer_x(m.layer_idx), kind_y(m.kind), 0.0],
                                to,
                            ]);
                        }
                    }
                }
                if !edge_strips.is_empty() {
                    let _ = rec.log_static(
                        "circuit/edges",
                        &rerun::LineStrips3D::new(edge_strips)
                            .with_colors(vec![rerun::Color::from_unmultiplied_rgba(
                                0x44, 0x88, 0xcc, 0xaa,
                            )])
                            .with_radii(vec![rerun::Radius::new_ui_points(1.2)]),
                    );
                }
            }

            ProofEvent::LayerActivation {
                layer_idx,
                output_sample,
                stats,
                ..
            } => {
                let vals: Vec<f32> = output_sample
                    .iter()
                    .map(|&v| v as f32 / 0x7fff_ffff_u32 as f32)
                    .collect();
                let n = vals.len();
                let _ = rec.log(
                    inference_output_path(*layer_idx).as_str(),
                    &rerun::Tensor::new(rerun::TensorData::new(
                        vec![n as u64],
                        rerun::TensorBuffer::F32(vals.into()),
                    )),
                );
                let _ = rec.log(
                    inference_mean_path(*layer_idx).as_str(),
                    &rerun::Scalars::single(stats.mean as f64),
                );
                let _ = rec.log(
                    inference_std_path(*layer_idx).as_str(),
                    &rerun::Scalars::single(stats.std_dev as f64),
                );
            }

            ProofEvent::AttentionHeatmap {
                layer_idx,
                head_idx,
                scores,
                ..
            } => {
                let n = (scores.len() as f64).sqrt() as usize;
                let _ = rec.log(
                    inference_attn_path(*layer_idx, *head_idx).as_str(),
                    &rerun::Tensor::new(rerun::TensorData::new(
                        vec![n as u64, n as u64],
                        rerun::TensorBuffer::F32(scores.clone().into()),
                    )),
                );
            }

            ProofEvent::ProofStart {
                num_layers,
                input_shape,
                output_shape,
                backend,
                model_name,
            } => {
                // Pre-layout layer nodes on a helix so the 3D view has real depth.
                // LayerStart will update each node to its type color.
                let n = *num_layers;
                let positions: Vec<[f32; 3]> = (0..n).map(helix_pos).collect();
                let dim = rerun::Color::from_unmultiplied_rgba(72, 74, 92, 200);
                let labels: Vec<String> = (0..n).map(|i| format!("L{i}")).collect();
                let _ = rec.log_static(
                    "circuit/nodes",
                    &rerun::Points3D::new(positions.clone())
                        .with_colors(vec![dim; n])
                        .with_radii(vec![rerun::Radius::new_ui_points(5.0); n])
                        .with_labels(labels),
                );
                // Helix backbone rail connecting placeholder nodes
                let _ = rec.log_static(
                    "circuit/backbone",
                    &rerun::LineStrips3D::new(vec![positions])
                        .with_colors(vec![rerun::Color::from_unmultiplied_rgba(
                            60, 62, 90, 120,
                        )])
                        .with_radii(vec![rerun::Radius::new_ui_points(1.0)]),
                );

                let _ = rec.log("proof/progress", &rerun::Scalars::single(0.0_f64));
                let _ = rec.log(
                    "logs/proof",
                    &rerun::TextLog::new(format!(
                        "ProofStart: {backend}, {n} layers, \
                         input={input_shape:?} output={output_shape:?}, model={model_name:?}"
                    )),
                );
            }

            ProofEvent::LayerStart {
                layer_idx,
                kind,
                claim_value_approx,
                gpu_device,
                trace_cost,
                ..
            } => {
                let [hx, hy, hz] = helix_pos(*layer_idx);
                let ky = kind_y(*kind);
                let [r, g, b] = kind.color_rgb();
                let radius =
                    rerun::Radius::new_ui_points(6.0 + (*trace_cost as f32).sqrt() * 0.2);

                // Activate node — type color at helix position + kind elevation
                let _ = rec.log(
                    layer_node_path(*layer_idx).as_str(),
                    &rerun::Points3D::new(vec![[hx, hy + ky, hz]])
                        .with_colors(vec![rerun::Color::from_rgb(r, g, b)])
                        .with_radii(vec![radius])
                        .with_labels(vec![format!("{kind:?} [{layer_idx}]")]),
                );
                // Cyan scanning cursor above the active node
                let _ = rec.log(
                    "gkr/walk/cursor",
                    &rerun::Points3D::new(vec![[hx, hy + ky + 0.7, hz]])
                        .with_colors(vec![rerun::Color::from_rgb(0x00, 0xe5, 0xff)])
                        .with_radii(vec![rerun::Radius::new_ui_points(6.0)])
                        .with_labels(vec![format!("proving {kind:?}...")]),
                );
                let _ = rec.log("gkr/claim", &rerun::Scalars::single(*claim_value_approx as f64));
                let _ = rec.log(
                    "logs/gkr",
                    &rerun::TextLog::new(format!(
                        "-> [{layer_idx}] {kind:?}  claim={claim_value_approx:.6}  gpu={gpu_device:?}"
                    )),
                );
            }

            ProofEvent::SumcheckRound {
                layer_idx,
                round,
                total_rounds,
                poly_deg2,
                claim_value_approx,
                ..
            } => {
                // Update per-round claim scalar
                let _ = rec.log(
                    layer_claim_path(*layer_idx).as_str(),
                    &rerun::Scalars::single(*claim_value_approx as f64),
                );

                // Claim reduction tower: one vertical bar per sumcheck round,
                // shrinking as the claim compresses toward the input evaluation.
                // Bars fan out along X from the node; tallest = round 0 (full claim).
                {
                    let [hx, hy, hz] = helix_pos(*layer_idx);
                    let frac = *round as f32 / (*total_rounds).max(1) as f32;

                    // Bar height = normalised claim value (clamped to [0.1, 4.0])
                    let bar_h = (claim_value_approx.abs() * 4.0).clamp(0.1, 4.0);

                    // Fan bars out along X centred on the node; most recent at centre
                    let spread = *total_rounds as f32 * 0.38;
                    let x_off = (frac - 0.5) * spread;

                    // Vertical bar: base → top
                    let base = [hx + x_off, hy, hz];
                    let top  = [hx + x_off, hy + bar_h, hz];

                    // A small horizontal cap at the top to read claim level
                    let cap_w = 0.18_f32;
                    let cap_l = [hx + x_off - cap_w, hy + bar_h, hz];
                    let cap_r = [hx + x_off + cap_w, hy + bar_h, hz];

                    // Colour: blue (round 0, tall) → red (last round, short)
                    let r_col = (frac * 210.0) as u8 + 45;
                    let b_col = ((1.0 - frac) * 210.0) as u8 + 45;
                    let color = rerun::Color::from_rgb(r_col, 0x55, b_col);

                    let _ = rec.log(
                        format!("gkr/walk/layer_{layer_idx}/bar_{round}").as_str(),
                        &rerun::LineStrips3D::new(vec![
                            vec![base, top],       // vertical bar
                            vec![cap_l, cap_r],    // top cap
                        ])
                        .with_colors(vec![color])
                        .with_radii(vec![rerun::Radius::new_ui_points(2.2)]),
                    );
                }

                if *round == 0 || round + 1 == *total_rounds {
                    let _ = rec.log(
                        "logs/gkr",
                        &rerun::TextLog::new(format!(
                            "  [{layer_idx}] round {}/{total_rounds}  claim={claim_value_approx:.6}",
                            round + 1
                        )),
                    );
                }
            }

            ProofEvent::LayerEnd {
                layer_idx,
                kind,
                final_claim_value_approx,
                duration_ms,
                ..
            } => {
                let [hx, hy, hz] = helix_pos(*layer_idx);
                // Mark node proved — vivid green, settled at helix position
                let _ = rec.log(
                    layer_node_path(*layer_idx).as_str(),
                    &rerun::Points3D::new(vec![[hx, hy, hz]])
                        .with_colors(vec![rerun::Color::from_rgb(0x00, 0xe6, 0x76)])
                        .with_radii(vec![rerun::Radius::new_ui_points(6.0)])
                        .with_labels(vec![format!("{kind:?}")]),
                );
                // Grow the proof-progress trail along the helix
                let trail: Vec<[f32; 3]> = (0..=*layer_idx).map(helix_pos).collect();
                let _ = rec.log(
                    "gkr/walk/progress_trail",
                    &rerun::LineStrips3D::new(vec![trail])
                        .with_colors(vec![rerun::Color::from_unmultiplied_rgba(
                            0x00, 0xe6, 0x76, 0xcc,
                        )])
                        .with_radii(vec![rerun::Radius::new_ui_points(2.5)]),
                );
                let _ = rec.log(
                    "logs/gkr",
                    &rerun::TextLog::new(format!(
                        "[{layer_idx}] proved {kind:?} in {duration_ms}ms  claim={final_claim_value_approx:.6}"
                    )),
                );
            }

            ProofEvent::ProofComplete {
                duration_ms,
                num_layer_proofs,
                weight_binding_mode,
                ..
            } => {
                let _ = rec.log("proof/progress", &rerun::Scalars::single(1.0_f64));
                // Dismiss the scanning cursor (move far off-canvas)
                let _ = rec.log(
                    "gkr/walk/cursor",
                    &rerun::Points3D::new(vec![[0.0_f32, -30.0, -30.0]])
                        .with_colors(vec![rerun::Color::from_unmultiplied_rgba(0, 0, 0, 0)])
                        .with_radii(vec![rerun::Radius::new_ui_points(0.1)]),
                );
                let _ = rec.log(
                    "logs/proof",
                    &rerun::TextLog::new(format!(
                        "PROOF COMPLETE  {duration_ms}ms  {num_layer_proofs} layers  {weight_binding_mode}"
                    )),
                );
            }

            ProofEvent::WeightOpeningStart {
                weight_node_id,
                eval_point_len,
            } => {
                let _ = rec.log(
                    "logs/weights",
                    &rerun::TextLog::new(format!(
                        "WeightOpening node={weight_node_id}, eval_dim={eval_point_len}"
                    )),
                );
            }

            ProofEvent::WeightOpeningEnd {
                weight_node_id,
                duration_ms,
                commitment_hex,
            } => {
                let _ = rec.log(
                    "logs/weights",
                    &rerun::TextLog::new(format!(
                        "WeightOpening done node={weight_node_id}, {duration_ms}ms, commit={commitment_hex}"
                    )),
                );
            }

            ProofEvent::AggregatedBindingStart {
                num_claims,
                num_matrices,
            } => {
                let _ = rec.log(
                    "logs/proof",
                    &rerun::TextLog::new(format!(
                        "AggregatedBinding: {num_claims} claims, {num_matrices} matrices"
                    )),
                );
            }

            ProofEvent::AggregatedBindingEnd {
                duration_ms,
                estimated_calldata_felts,
            } => {
                let _ = rec.log("proof/binding_ms", &rerun::Scalars::single(*duration_ms as f64));
                let _ = rec.log(
                    "logs/proof",
                    &rerun::TextLog::new(format!(
                        "AggregatedBinding done: {duration_ms}ms, ~{estimated_calldata_felts} felts"
                    )),
                );
            }

            ProofEvent::GpuStatus {
                devices,
                matmul_done,
                matmul_total,
                layers_done,
                layers_total,
            } => {
                let positions: Vec<[f32; 3]> = devices
                    .iter()
                    .enumerate()
                    .map(|(i, _)| [i as f32 * 2.0, 0.0, 2.0])
                    .collect();
                let colors: Vec<rerun::Color> = devices
                    .iter()
                    .map(|d| {
                        let r = (d.utilization * 255.0) as u8;
                        let g = ((1.0 - d.utilization) * 200.0) as u8;
                        rerun::Color::from_rgb(r, g, 0x40)
                    })
                    .collect();
                let _ = rec.log(
                    "gpu/cluster",
                    &rerun::Points3D::new(positions)
                        .with_colors(colors)
                        .with_radii(vec![rerun::Radius::new_ui_points(10.0)]),
                );
                for d in devices {
                    let _ = rec.log(
                        gpu_util_path(d.device_id).as_str(),
                        &rerun::Scalars::single(d.utilization as f64),
                    );
                    if let Some(free) = d.free_memory_bytes {
                        let _ = rec.log(
                            gpu_free_path(d.device_id).as_str(),
                            &rerun::Scalars::single(free as f64 / 1e9),
                        );
                    }
                }
                if *layers_total > 0 {
                    let _ = rec.log(
                        "proof/progress",
                        &rerun::Scalars::single(*layers_done as f64 / *layers_total as f64),
                    );
                }
                let _ = rec.log(
                    "logs/proof",
                    &rerun::TextLog::new(format!(
                        "GPU: matmul {matmul_done}/{matmul_total}, layers {layers_done}/{layers_total}"
                    )),
                );
            }

            ProofEvent::StarkProofStart {
                num_activation_layers,
                num_add_layers,
                num_layernorm_layers,
            } => {
                let _ = rec.log(
                    "logs/stark",
                    &rerun::TextLog::new(format!(
                        "StarkProofStart: activations={num_activation_layers}, add={num_add_layers}, layernorm={num_layernorm_layers}"
                    )),
                );
            }

            ProofEvent::StarkProofEnd { duration_ms } => {
                let _ = rec.log(
                    "logs/stark",
                    &rerun::TextLog::new(format!("StarkProofEnd: {duration_ms}ms")),
                );
            }

            ProofEvent::Log { level, message } => {
                let level_str = match level {
                    LogLevel::Trace | LogLevel::Debug => rerun::TextLogLevel::TRACE,
                    LogLevel::Info => rerun::TextLogLevel::INFO,
                    LogLevel::Warn => rerun::TextLogLevel::WARN,
                    LogLevel::Error => rerun::TextLogLevel::ERROR,
                };
                let _ = rec.log(
                    "logs/proof",
                    &rerun::TextLog::new(message.as_str()).with_level(level_str),
                );
            }
        }
    }

    fn background_worker(
        rx: Receiver<Msg>,
        conn: RerunConnection,
        app_id: String,
        ready: Arc<AtomicBool>,
    ) {
        let rec = match build_recording_stream(conn, &app_id) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("[proof-stream] Failed to connect to Rerun: {e}");
                ready.store(true, Ordering::SeqCst);
                return;
            }
        };

        // Send blueprint before any data
        crate::blueprint::send_blueprint(&rec);

        ready.store(true, Ordering::SeqCst);

        for msg in &rx {
            match msg {
                Msg::Event(event) => dispatch_event(&rec, &event),
                Msg::Flush => {
                    let _ = rec.flush_blocking();
                }
                Msg::Stop => break,
            }
        }

        let _ = rec.flush_blocking();
    }

    // ── Public sink ───────────────────────────────────────────────────────────

    /// A `ProofEventSink` that forwards events to a Rerun viewer.
    pub struct RerunSink {
        tx: Sender<Msg>,
        _handle: Option<JoinHandle<()>>,
    }

    impl RerunSink {
        /// Connect to Rerun and start the background dispatch thread.
        pub fn connect(
            conn: RerunConnection,
            app_id: impl Into<String>,
        ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
            let (tx, rx) = channel::bounded::<Msg>(8192);
            let app_id = app_id.into();
            let ready = Arc::new(AtomicBool::new(false));
            let ready2 = Arc::clone(&ready);

            let handle = thread::Builder::new()
                .name("proof-stream-rerun".into())
                .spawn(move || background_worker(rx, conn, app_id, ready2))?;

            // Wait (briefly) for the connection to be established
            let deadline = std::time::Instant::now() + Duration::from_secs(10);
            while !ready.load(Ordering::SeqCst) && std::time::Instant::now() < deadline {
                thread::sleep(Duration::from_millis(20));
            }

            Ok(Self {
                tx,
                _handle: Some(handle),
            })
        }
    }

    impl ProofEventSink for RerunSink {
        #[inline]
        fn emit(&self, event: ProofEvent) {
            let _ = self.tx.try_send(Msg::Event(event));
        }

        fn flush(&self) {
            let _ = self.tx.send(Msg::Flush);
        }
    }

    impl Drop for RerunSink {
        fn drop(&mut self) {
            let _ = self.tx.send(Msg::Stop);
            if let Some(h) = self._handle.take() {
                let _ = h.join();
            }
        }
    }
}

// ── Re-exports ────────────────────────────────────────────────────────────────

#[cfg(feature = "rerun")]
pub use inner::{RerunConnection, RerunSink};

// ── Stubs when feature is disabled ────────────────────────────────────────────

#[cfg(not(feature = "rerun"))]
pub struct RerunSink;

#[cfg(not(feature = "rerun"))]
pub enum RerunConnection {
    Tcp { addr: String },
    File { path: std::path::PathBuf },
    Spawn,
}

#[cfg(not(feature = "rerun"))]
impl RerunConnection {
    pub fn from_str(s: &str) -> Self {
        if s == "spawn" {
            RerunConnection::Spawn
        } else if let Some(path) = s.strip_prefix("file:") {
            RerunConnection::File {
                path: std::path::PathBuf::from(path),
            }
        } else {
            RerunConnection::Tcp { addr: s.to_owned() }
        }
    }
}
