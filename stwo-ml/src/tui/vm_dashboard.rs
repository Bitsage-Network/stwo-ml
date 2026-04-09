//! ObelyZK VM Dashboard — "Cipher Noir" real-time proving monitor.
//!
//! Layout:
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │  ╔═╗╔╗  ╔═╗╦  ╦ ╦╔═╗╦╔═    MODEL  Qwen3-14B   ◆ PROVING     │
//! │  ║ ║╠╩╗ ╠═ ║  ╚╦╝╔═╝╠╩╗    8×H100  5120 dim    14.2 tok/s   │
//! │  ╚═╝╚═╝ ╚═╝╩═╝ ╩ ╚═╝╩ ╩    ZK INFERENCE VM                   │
//! ├─────────────────────────────────────────────────────────────────┤
//! │ GPU WORKERS              │ PROVING QUEUE       │ SESSIONS       │
//! │ ▸ GPU 0 ████████░  95%  │ 12 queued           │ ses-a4f2 ✓ 42t │
//! │ ▸ GPU 1 ██████░░░  78%  │  3 proving          │ ses-b8c1 ▸ 18t │
//! │ ▸ GPU 2 ███░░░░░░  38%  │  0 failed           │ ses-d3e9   5t  │
//! │ · GPU 3                  │                     │                │
//! │ · GPU 4                  │ THROUGHPUT          │ COMMITMENTS    │
//! │ · GPU 5                  │ 14.2 tok/s proven   │ 0x3a8f…c2e1   │
//! │ · GPU 6                  │ 89.1 tok/s infer    │ 0xb4d2…91f0   │
//! │ · GPU 7                  │ 2.1 recursive/min   │ 0x7e20…a3b5   │
//! ├─────────────────────────────────────────────────────────────────┤
//! │ CONVERSATION STREAM                                             │
//! │ #0 YOU  What is ZKML?                                          │
//! │    AI   Zero-knowledge machine learning is a cryptographic...  │
//! │    ✓ 156 tokens  proof-a4f2  95.2s  0x312c…e16                 │
//! │ #1 YOU  How does it work?                                      │
//! │    AI   The prover executes the model forward pass in M31...   │
//! │    ▸ 234 tokens  proof-b8c1  proving...                        │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  ObelyZK VM  ◆ PROVING  12m 34s  0x1c20…0c7  Starknet Sepolia │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Paragraph, Wrap},
    Frame,
};

// ── Palette: Cipher Noir (shared with dashboard.rs) ────────────────

const BG:          Color = Color::Reset;
const LIME:        Color = Color::Indexed(118);
const LIME_DIM:    Color = Color::Indexed(70);
const EMERALD:     Color = Color::Indexed(48);
const VIOLET:      Color = Color::Indexed(73);
const WHITE:       Color = Color::Indexed(255);
const SILVER:      Color = Color::Indexed(249);
const SLATE:       Color = Color::Indexed(245);
const GHOST:       Color = Color::Indexed(240);
const AMBER:       Color = Color::Indexed(178);
const ORANGE:      Color = Color::Indexed(208);
const LILAC:       Color = Color::Indexed(141);
const CYAN:        Color = Color::Indexed(44);

const H_LINE: &str = "─";
const V_LINE: &str = "│";
const DOT: &str = "·";
const BLOCK_FULL: &str = "█";
const BLOCK_LOW:  &str = "░";
const ARROW_R: &str = "▸";
const CHECK: &str = "✓";
const CROSS: &str = "✗";
const DIAMOND: &str = "◆";

// ── VM Dashboard State ─────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct GpuWorkerState {
    pub device_id: usize,
    pub device_name: String,
    pub utilization: f32,      // 0.0 - 1.0
    pub current_job: Option<String>,
    pub memory_used_gb: f32,
    pub memory_total_gb: f32,
}

#[derive(Debug, Clone)]
pub struct SessionSummary {
    pub session_id: String,
    pub model_id: String,
    pub total_tokens: usize,
    pub tokens_proven: usize,
    pub turns: usize,
    pub status: SessionStatus,
    pub last_commitment: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SessionStatus {
    Active,
    Proving,
    Complete,
    Expired,
}

#[derive(Debug, Clone)]
pub struct ConversationTurnView {
    pub turn_index: usize,
    pub user_text: String,
    pub ai_text: String,
    pub num_tokens: usize,
    pub proof_id: Option<String>,
    pub prove_time_ms: Option<u64>,
    pub status: TurnProofStatus,
    pub commitment: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TurnProofStatus {
    Pending,
    Proving,
    Proven,
    OnChain,
    Failed,
}

#[derive(Debug, Clone)]
pub struct VmDashboardState {
    // Model
    pub model_name: String,
    pub model_params: String,
    pub d_model: usize,
    // GPU workers
    pub workers: Vec<GpuWorkerState>,
    // Queue
    pub queue_queued: usize,
    pub queue_proving: usize,
    pub queue_failed: usize,
    pub queue_completed: usize,
    // Throughput
    pub proven_tok_per_sec: f64,
    pub inference_tok_per_sec: f64,
    pub recursive_per_min: f64,
    // Sessions
    pub sessions: Vec<SessionSummary>,
    // Active conversation
    pub active_turns: Vec<ConversationTurnView>,
    // On-chain
    pub contract: String,
    pub network: String,
    pub on_chain_txs: usize,
    pub total_proven_tokens: usize,
    // Runtime
    pub uptime_secs: u64,
    pub status: VmStatus,
    pub frame_count: u64,
    // Logs
    pub logs: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum VmStatus {
    Starting,
    Ready,
    Proving,
    Idle,
    Error,
}

impl Default for VmDashboardState {
    fn default() -> Self {
        Self {
            model_name: String::new(),
            model_params: String::new(),
            d_model: 0,
            workers: Vec::new(),
            queue_queued: 0,
            queue_proving: 0,
            queue_failed: 0,
            queue_completed: 0,
            proven_tok_per_sec: 0.0,
            inference_tok_per_sec: 0.0,
            recursive_per_min: 0.0,
            sessions: Vec::new(),
            active_turns: Vec::new(),
            contract: String::new(),
            network: "Starknet Sepolia".into(),
            on_chain_txs: 0,
            total_proven_tokens: 0,
            uptime_secs: 0,
            status: VmStatus::Starting,
            frame_count: 0,
            logs: Vec::new(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Main render
// ═══════════════════════════════════════════════════════════════════

pub fn render(frame: &mut Frame, state: &VmDashboardState) {
    let area = frame.area();
    frame.render_widget(
        ratatui::widgets::Block::default().style(Style::default().bg(BG)),
        area,
    );

    let outer = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(5),   // Header
            Constraint::Length(1),   // Divider
            Constraint::Length(12),  // GPU + Queue + Sessions (3-col)
            Constraint::Length(1),   // Divider
            Constraint::Min(6),     // Conversation stream
            Constraint::Length(1),   // Divider
            Constraint::Length(2),   // Footer
        ])
        .split(area);

    render_header(frame, outer[0], state);
    render_divider(frame, outer[1], LIME_DIM);
    render_middle(frame, outer[2], state);
    render_divider(frame, outer[3], GHOST);
    render_conversation_stream(frame, outer[4], state);
    render_divider(frame, outer[5], GHOST);
    render_footer(frame, outer[6], state);
}

fn render_divider(frame: &mut Frame, area: Rect, color: Color) {
    let line = H_LINE.repeat(area.width as usize);
    frame.render_widget(
        Paragraph::new(Span::styled(line, Style::default().fg(color))),
        area,
    );
}

// ── Header ─────────────────────────────────────────────────────────

fn render_header(frame: &mut Frame, area: Rect, state: &VmDashboardState) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(30), Constraint::Min(20)])
        .split(area);

    // Logo
    let logo = vec![
        Line::from(Span::styled("", Style::default())),
        Line::from(Span::styled("  ╔═╗╔╗  ╔═╗╦  ╦ ╦╔═╗╦╔═", Style::default().fg(LIME))),
        Line::from(Span::styled("  ║ ║╠╩╗ ╠═ ║  ╚╦╝╔═╝╠╩╗", Style::default().fg(LIME))),
        Line::from(Span::styled("  ╚═╝╚═╝ ╚═╝╩═╝ ╩ ╚═╝╩ ╩", Style::default().fg(LIME_DIM))),
    ];
    frame.render_widget(Paragraph::new(logo), cols[0]);

    // Status info
    let (status_text, status_color) = match state.status {
        VmStatus::Starting => ("STARTING", AMBER),
        VmStatus::Ready => ("READY", LIME),
        VmStatus::Proving => ("PROVING", LIME),
        VmStatus::Idle => ("IDLE", GHOST),
        VmStatus::Error => ("ERROR", AMBER),
    };

    let pulse = if state.frame_count % 4 < 2 { LIME } else { LIME_DIM };
    let status_c = if state.status == VmStatus::Proving { pulse } else { status_color };

    let model_display = if state.model_name.is_empty() { "loading..." } else { &state.model_name };
    let gpu_count = state.workers.len();
    let gpu_label = if gpu_count == 1 { "GPU" } else { "GPUs" };

    let info = vec![
        Line::from(Span::styled("", Style::default())),
        Line::from(vec![
            Span::styled("  MODEL  ", Style::default().fg(SLATE)),
            Span::styled(model_display, Style::default().fg(WHITE).add_modifier(Modifier::BOLD)),
            Span::styled(format!("  {}", state.model_params), Style::default().fg(SILVER)),
        ]),
        Line::from(vec![
            Span::styled("  INFRA  ", Style::default().fg(SLATE)),
            Span::styled(format!("{gpu_count}×{gpu_label}"), Style::default().fg(CYAN)),
            Span::styled(format!("  d_model={}", state.d_model), Style::default().fg(GHOST)),
            Span::styled("  ZK INFERENCE VM", Style::default().fg(LIME_DIM)),
        ]),
        Line::from(vec![
            Span::styled("  STATUS ", Style::default().fg(SLATE)),
            Span::styled(
                format!("{DIAMOND} {status_text}"),
                Style::default().fg(status_c).add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!("  {:.1} tok/s proven", state.proven_tok_per_sec),
                Style::default().fg(EMERALD),
            ),
            Span::styled(
                format!("  {} tokens total", state.total_proven_tokens),
                Style::default().fg(GHOST),
            ),
        ]),
    ];
    frame.render_widget(Paragraph::new(info), cols[1]);
}

// ── Middle: GPU Workers | Proving Queue | Sessions ─────────────────

fn render_middle(frame: &mut Frame, area: Rect, state: &VmDashboardState) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(40),
            Constraint::Length(1),
            Constraint::Percentage(28),
            Constraint::Length(1),
            Constraint::Percentage(32),
        ])
        .split(area);

    render_gpu_workers(frame, cols[0], state);
    render_gutter(frame, cols[1]);
    render_queue_throughput(frame, cols[2], state);
    render_gutter(frame, cols[3]);
    render_sessions(frame, cols[4], state);
}

fn render_gutter(frame: &mut Frame, area: Rect) {
    let lines: Vec<Line> = (0..area.height)
        .map(|_| Line::from(Span::styled(V_LINE, Style::default().fg(GHOST))))
        .collect();
    frame.render_widget(Paragraph::new(lines), area);
}

fn render_gpu_workers(frame: &mut Frame, area: Rect, state: &VmDashboardState) {
    let mut lines = vec![
        Line::from(Span::styled(
            " GPU WORKERS",
            Style::default().fg(CYAN).add_modifier(Modifier::BOLD),
        )),
        Line::from(Span::styled("", Style::default())),
    ];

    let bar_width = (area.width as usize).saturating_sub(24);

    for w in &state.workers {
        let (icon, icon_color) = if w.current_job.is_some() {
            (ARROW_R, LIME)
        } else {
            (DOT, GHOST)
        };

        let filled = (w.utilization * bar_width as f32) as usize;
        let empty = bar_width.saturating_sub(filled);
        let bar_color = if w.utilization > 0.8 {
            EMERALD
        } else if w.utilization > 0.0 {
            LIME
        } else {
            GHOST
        };

        let pct_str = if w.utilization > 0.0 {
            format!("{:>3.0}%", w.utilization * 100.0)
        } else {
            "    ".into()
        };

        lines.push(Line::from(vec![
            Span::styled(format!(" {icon} "), Style::default().fg(icon_color)),
            Span::styled(format!("GPU {}", w.device_id), Style::default().fg(if w.current_job.is_some() { WHITE } else { SLATE })),
            Span::styled(" ", Style::default()),
            Span::styled(BLOCK_FULL.repeat(filled), Style::default().fg(bar_color)),
            Span::styled(BLOCK_LOW.repeat(empty), Style::default().fg(GHOST)),
            Span::styled(format!(" {pct_str}"), Style::default().fg(SLATE)),
        ]));

        // Memory line
        if w.memory_total_gb > 0.0 {
            lines.push(Line::from(vec![
                Span::styled("      ", Style::default()),
                Span::styled(
                    format!("{:.1}/{:.0}GB", w.memory_used_gb, w.memory_total_gb),
                    Style::default().fg(GHOST),
                ),
                if let Some(ref job) = w.current_job {
                    Span::styled(
                        format!("  {}", truncate(job, 14)),
                        Style::default().fg(LILAC),
                    )
                } else {
                    Span::styled("  idle", Style::default().fg(GHOST))
                },
            ]));
        }
    }

    if state.workers.is_empty() {
        lines.push(Line::from(Span::styled(
            " no GPU workers",
            Style::default().fg(GHOST),
        )));
    }

    frame.render_widget(Paragraph::new(lines), area);
}

fn render_queue_throughput(frame: &mut Frame, area: Rect, state: &VmDashboardState) {
    let mut lines = vec![
        Line::from(Span::styled(
            " PROVING QUEUE",
            Style::default().fg(LIME).add_modifier(Modifier::BOLD),
        )),
        Line::from(Span::styled("", Style::default())),
    ];

    // Queue stats
    lines.push(queue_stat_line("queued", state.queue_queued, SILVER));
    lines.push(queue_stat_line("proving", state.queue_proving, LIME));
    lines.push(queue_stat_line("completed", state.queue_completed, EMERALD));
    if state.queue_failed > 0 {
        lines.push(queue_stat_line("failed", state.queue_failed, AMBER));
    }

    lines.push(Line::from(Span::styled("", Style::default())));
    lines.push(Line::from(Span::styled(
        " THROUGHPUT",
        Style::default().fg(EMERALD).add_modifier(Modifier::BOLD),
    )));
    lines.push(Line::from(vec![
        Span::styled(" proven   ", Style::default().fg(SLATE)),
        Span::styled(
            format!("{:.1} tok/s", state.proven_tok_per_sec),
            Style::default().fg(EMERALD).add_modifier(Modifier::BOLD),
        ),
    ]));
    lines.push(Line::from(vec![
        Span::styled(" infer    ", Style::default().fg(SLATE)),
        Span::styled(
            format!("{:.1} tok/s", state.inference_tok_per_sec),
            Style::default().fg(SILVER),
        ),
    ]));
    lines.push(Line::from(vec![
        Span::styled(" on-chain ", Style::default().fg(SLATE)),
        Span::styled(
            format!("{:.1}/min", state.recursive_per_min),
            Style::default().fg(ORANGE),
        ),
    ]));

    frame.render_widget(Paragraph::new(lines), area);
}

fn queue_stat_line(label: &str, count: usize, color: Color) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!(" {:<10}", label), Style::default().fg(SLATE)),
        Span::styled(
            format!("{count}"),
            Style::default().fg(color).add_modifier(if count > 0 { Modifier::BOLD } else { Modifier::empty() }),
        ),
    ])
}

fn render_sessions(frame: &mut Frame, area: Rect, state: &VmDashboardState) {
    let mut lines = vec![
        Line::from(Span::styled(
            " SESSIONS",
            Style::default().fg(VIOLET).add_modifier(Modifier::BOLD),
        )),
        Line::from(Span::styled("", Style::default())),
    ];

    for s in state.sessions.iter().take(8) {
        let (icon, icon_color) = match s.status {
            SessionStatus::Active => (ARROW_R, LIME),
            SessionStatus::Proving => (ARROW_R, ORANGE),
            SessionStatus::Complete => (CHECK, EMERALD),
            SessionStatus::Expired => (DOT, GHOST),
        };
        let id_short = if s.session_id.len() > 10 {
            &s.session_id[..10]
        } else {
            &s.session_id
        };

        lines.push(Line::from(vec![
            Span::styled(format!(" {icon} "), Style::default().fg(icon_color)),
            Span::styled(id_short, Style::default().fg(LILAC)),
            Span::styled(
                format!(" {}t", s.total_tokens),
                Style::default().fg(if s.tokens_proven == s.total_tokens { EMERALD } else { SILVER }),
            ),
            Span::styled(
                format!(" {}T", s.turns),
                Style::default().fg(GHOST),
            ),
        ]));

        // Commitment line
        if let Some(ref c) = s.last_commitment {
            let short = if c.len() > 16 { &c[..16] } else { c };
            lines.push(Line::from(vec![
                Span::styled("     ", Style::default()),
                Span::styled(format!("{short}…"), Style::default().fg(VIOLET)),
            ]));
        }
    }

    if state.sessions.is_empty() {
        lines.push(Line::from(Span::styled(
            " no active sessions",
            Style::default().fg(GHOST),
        )));
    }

    frame.render_widget(Paragraph::new(lines), area);
}

// ── Conversation Stream ────────────────────────────────────────────

fn render_conversation_stream(frame: &mut Frame, area: Rect, state: &VmDashboardState) {
    let mut lines = vec![
        Line::from(Span::styled(
            " CONVERSATION STREAM",
            Style::default().fg(LIME).add_modifier(Modifier::BOLD),
        )),
        Line::from(Span::styled("", Style::default())),
    ];

    let col_width = area.width.saturating_sub(10) as usize;
    let max_text = col_width.min(60);

    for turn in &state.active_turns {
        let idx = format!("#{:<2}", turn.turn_index);

        // User line
        lines.push(Line::from(vec![
            Span::styled(format!(" {idx}"), Style::default().fg(GHOST)),
            Span::styled(" YOU ", Style::default().fg(LIME).add_modifier(Modifier::BOLD)),
            Span::styled(truncate(&turn.user_text, max_text), Style::default().fg(WHITE)),
        ]));

        // AI line
        lines.push(Line::from(vec![
            Span::styled("    ", Style::default()),
            Span::styled(" AI  ", Style::default().fg(EMERALD)),
            Span::styled(truncate(&turn.ai_text, max_text), Style::default().fg(SILVER)),
        ]));

        // Proof status line
        let (proof_icon, proof_color) = match turn.status {
            TurnProofStatus::Pending => (DOT, GHOST),
            TurnProofStatus::Proving => (ARROW_R, ORANGE),
            TurnProofStatus::Proven => (CHECK, EMERALD),
            TurnProofStatus::OnChain => (CHECK, ORANGE),
            TurnProofStatus::Failed => (CROSS, AMBER),
        };

        let time_str = turn.prove_time_ms.map(|t| {
            if t > 1000 { format!("{:.1}s", t as f64 / 1000.0) } else { format!("{t}ms") }
        }).unwrap_or_default();

        let proof_id_short = turn.proof_id.as_deref()
            .map(|p| if p.len() > 14 { &p[..14] } else { p })
            .unwrap_or("...");

        let commit_short = turn.commitment.as_deref()
            .map(|c| if c.len() > 14 { &c[..14] } else { c })
            .unwrap_or("");

        lines.push(Line::from(vec![
            Span::styled(format!("    {proof_icon} "), Style::default().fg(proof_color)),
            Span::styled(format!("{} tokens", turn.num_tokens), Style::default().fg(SLATE)),
            Span::styled(format!("  {proof_id_short}"), Style::default().fg(LILAC)),
            Span::styled(format!("  {time_str}"), Style::default().fg(EMERALD)),
            Span::styled(format!("  {commit_short}"), Style::default().fg(VIOLET)),
        ]));

        lines.push(Line::from(Span::styled("", Style::default())));
    }

    if state.active_turns.is_empty() {
        lines.push(Line::from(Span::styled(
            " awaiting conversation…",
            Style::default().fg(GHOST),
        )));
    }

    // Auto-scroll to show most recent
    let visible = area.height as usize;
    let offset = if lines.len() > visible { lines.len() - visible } else { 0 };
    let visible_lines: Vec<Line> = lines.into_iter().skip(offset).collect();

    frame.render_widget(Paragraph::new(visible_lines).wrap(Wrap { trim: false }), area);
}

// ── Footer ─────────────────────────────────────────────────────────

fn render_footer(frame: &mut Frame, area: Rect, state: &VmDashboardState) {
    let (status_text, status_color) = match state.status {
        VmStatus::Starting => ("STARTING", AMBER),
        VmStatus::Ready => ("READY", LIME),
        VmStatus::Proving => ("PROVING", LIME),
        VmStatus::Idle => ("IDLE", GHOST),
        VmStatus::Error => ("ERROR", AMBER),
    };

    let uptime = format_uptime(state.uptime_secs);
    let contract_short = if state.contract.len() > 20 {
        format!("{}…{}", &state.contract[..10], &state.contract[state.contract.len()-6..])
    } else if state.contract.is_empty() {
        "no contract".into()
    } else {
        state.contract.clone()
    };

    let spans = vec![
        Span::styled(" ", Style::default()),
        Span::styled(" ObelyZK VM ", Style::default().fg(BG).bg(LIME).add_modifier(Modifier::BOLD)),
        Span::styled("  ", Style::default()),
        Span::styled(status_text, Style::default().fg(status_color).add_modifier(Modifier::BOLD)),
        Span::styled(format!("  {uptime}"), Style::default().fg(ORANGE)),
        Span::styled("  ", Style::default()),
        Span::styled(&contract_short, Style::default().fg(VIOLET)),
        Span::styled("  ", Style::default()),
        Span::styled(&state.network, Style::default().fg(SLATE)),
        Span::styled(format!("  {} TXs", state.on_chain_txs), Style::default().fg(GHOST)),
        Span::styled("    ", Style::default()),
        Span::styled("q", Style::default().fg(LIME)),
        Span::styled(" exit  ", Style::default().fg(GHOST)),
        Span::styled("s", Style::default().fg(LIME)),
        Span::styled(" sessions  ", Style::default().fg(GHOST)),
        Span::styled("g", Style::default().fg(LIME)),
        Span::styled(" gpu", Style::default().fg(GHOST)),
    ];

    frame.render_widget(
        Paragraph::new(Line::from(spans)).style(Style::default().bg(BG)),
        area,
    );
}

// ── Helpers ────────────────────────────────────────────────────────

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        s.chars().take(max).collect::<String>() + "…"
    }
}

fn format_uptime(secs: u64) -> String {
    if secs < 60 {
        format!("{}s", secs)
    } else if secs < 3600 {
        format!("{}m {:02}s", secs / 60, secs % 60)
    } else {
        format!("{}h {:02}m", secs / 3600, (secs % 3600) / 60)
    }
}
