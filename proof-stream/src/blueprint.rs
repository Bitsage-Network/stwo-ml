//! Blueprint sender for the proof-stream visualization.
//!
//! Builds a declarative Rerun 0.29 blueprint with a 3-panel layout:
//!
//! ```text
//! ┌──────────────────────────┬───────────────────────────┐
//! │                          │  GKR Claim Descent        │
//! │  ZK Proof Walk (3D)      │  gkr/claim                │
//! │  circuit/** gkr/walk/**  │  gkr/walk/**/claim        │
//! │  gpu/cluster             │  proof/progress           │
//! │                          ├───────────────────────────┤
//! │                          │  Layer Stats & GPU        │
//! │                          │  inference/stats/**       │
//! │                          │  gpu/device_*/**          │
//! │                          │  proof/binding_ms         │
//! └──────────────────────────┴───────────────────────────┘
//! ```
//!
//! The blueprint is sent once at connection time before any proof events.

/// Send a structured blueprint to the Rerun viewer.
///
/// Creates a 2-column layout: the 3D GKR walk takes the left 60%, and two
/// stacked time-series charts (claim descent + layer stats) fill the right 40%.
/// Auto-layout and auto-views are disabled so Rerun doesn't create extra panels.
#[cfg(feature = "rerun")]
pub fn send_blueprint(rec: &rerun::RecordingStream) {
    use rerun::blueprint::{
        Blueprint, BlueprintActivation, ContainerLike, Horizontal, Spatial3DView, TimeSeriesView,
        Vertical,
    };

    // ── Right column: two stacked time-series charts ──────────────────────────

    // Top-right: GKR claim descending output→input as layers are proved
    let claim_chart: ContainerLike = TimeSeriesView::new("GKR Claim Descent")
        .with_origin("/")
        .with_contents([
            "gkr/claim",
            "gkr/walk/**/claim",
            "proof/progress",
        ])
        .into();

    // Bottom-right: per-layer stats, GPU utilisation, binding overhead
    let stats_chart: ContainerLike = TimeSeriesView::new("Layer Stats & GPU")
        .with_origin("/")
        .with_contents([
            "inference/stats/**",
            "gpu/device_*/**",
            "proof/binding_ms",
        ])
        .into();

    let right_col: ContainerLike = Vertical::new([claim_chart, stats_chart])
        .with_row_shares(vec![1.0_f32, 1.0_f32])
        .into();

    // ── Left: 3D GKR walk + circuit DAG + GPU cluster ────────────────────────

    let walk_3d: ContainerLike = Spatial3DView::new("ZK Proof Walk")
        .with_origin("/")
        .with_contents([
            "circuit/**",          // pre-layout nodes, backbone rail, DAG edges
            "gkr/walk/**",         // per-layer nodes, cursor, progress trail
            "gpu/cluster",         // GPU devices (when CUDA enabled)
        ])
        .into();

    // ── Root: horizontal split 60/40 ─────────────────────────────────────────

    let root = Horizontal::new([walk_3d, right_col])
        .with_column_shares(vec![3.0_f32, 2.0_f32]);

    let blueprint = Blueprint::new(root)
        .with_auto_layout(false)
        .with_auto_views(false);

    let _ = blueprint.send(
        rec,
        BlueprintActivation {
            make_active: true,
            make_default: false,
        },
    );

    // Emit a welcome TextDocument at the entity root so the user sees it in
    // the entity tree before any proof events arrive.
    let _ = rec.log_static(
        "logs/proof",
        &rerun::TextLog::new(
            "ZK Proof Stream ready — waiting for proof events.\n\
             \n\
             Entity layout:\n\
             • circuit/**          — 3D circuit DAG (color = layer kind)\n\
             • gkr/walk/**         — GKR layer nodes (green when proved)\n\
             • gkr/claim           — claim descending output→input\n\
             • gkr/walk/**/poly    — round polynomial [c0, c1, c2]\n\
             • inference/stats/**  — per-layer activation mean/std\n\
             • gpu/cluster         — GPU devices (color = utilisation)\n\
             • proof/progress      — overall proof progress 0→1",
        ),
    );
}

/// No-op stub when the `rerun` feature is disabled.
#[cfg(not(feature = "rerun"))]
pub fn send_blueprint(_rec: &()) {}
