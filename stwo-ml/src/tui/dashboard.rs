//! ObelyZK Proof Dashboard — "Cipher Noir" aesthetic.
//!
//! Design: high-contrast dark field, surgical precision typography,
//! cryptographic data rendered as classified intel. Every element
//! communicates: this is real, this is verified, this cannot be faked.
//!
//! Layout: 3-column body (Pipeline 40% | Crypto 30% | Inference Log 30%)
//! with streaming on-chain verification steps.

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Paragraph, Wrap},
    Frame,
};

// ── Palette: Cipher Noir ────────────────────────────────────────────
//
// Dominant: near-black background with lime signal color.
// Hierarchy through brightness, not hue variety.
// Hashes get their own violet treatment — they're the proof.

// ANSI 256-color palette — universal terminal compatibility.
// No RGB — works in every terminal, no pink/magenta misinterpretation.

const BG:          Color = Color::Reset;                 // Terminal default
const BG_ACTIVE:   Color = Color::Indexed(236);          // Gray 8%

const LIME:        Color = Color::Indexed(118);           // Bright lime #87ff00
const LIME_DIM:    Color = Color::Indexed(70);            // Medium green #5faf00

const EMERALD:     Color = Color::Indexed(48);            // Bright cyan-green #00ff87
const VIOLET:      Color = Color::Indexed(73);            // Steel blue #5fafaf — for hashes

const WHITE:       Color = Color::Indexed(255);           // Bright white
const SILVER:      Color = Color::Indexed(249);           // Light gray
const SLATE:       Color = Color::Indexed(245);           // Medium gray
const GHOST:       Color = Color::Indexed(240);           // Dark gray — borders

const RED:         Color = Color::Indexed(178);           // Gold/amber #d7af00

const ORANGE:      Color = Color::Indexed(208);           // Orange — on-chain TX highlights
const LILAC:       Color = Color::Indexed(141);           // Light purple — streaming step names

// ── Box-drawing characters for custom borders ───────────────────────
const H_LINE: &str = "─";
const V_LINE: &str = "│";
const DOT: &str = "·";
const BLOCK_FULL: &str = "█";
const BLOCK_LOW:  &str = "░";
const ARROW_R: &str = "▸";
const CHECK: &str = "✓";
const CROSS: &str = "✗";
const DIAMOND: &str = "◆";
const SHIELD: &str = "⊕";

// ── Streaming step status ───────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum StepStatus {
    Pending,
    Submitting,
    Confirmed,
    Failed,
}

#[derive(Debug, Clone)]
pub struct StreamingStep {
    pub name: String,
    pub status: StepStatus,
    pub tx_hash: Option<String>,
    pub block: Option<u64>,
    pub felts: usize,
}

impl StreamingStep {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.into(),
            status: StepStatus::Pending,
            tx_hash: None,
            block: None,
            felts: 0,
        }
    }
}

/// The 6 default streaming verification step names.
pub fn default_streaming_steps() -> Vec<StreamingStep> {
    vec![
        StreamingStep::new("stream_init"),
        StreamingStep::new("output_mle"),
        StreamingStep::new("layers"),
        StreamingStep::new("weight_binding"),
        StreamingStep::new("input_mle"),
        StreamingStep::new("finalize"),
    ]
}

/// Dashboard state.
#[derive(Debug, Clone)]
pub struct DashboardState {
    pub model_name: String,
    pub model_params: String,
    pub model_layers: u32,
    pub num_turns: usize,
    pub tokens_in: usize,
    pub tokens_out: usize,
    pub step: PipelineStep,
    // 3-step pipeline progress
    pub capture_progress: f64,
    pub prove_progress: f64,
    pub onchain_progress: f64,
    // Timing
    pub capture_time: Option<f64>,
    pub prove_time: Option<f64>,
    pub onchain_time: Option<f64>,
    pub prove_time_secs: f64,
    pub elapsed_secs: u64,
    // Commitments
    pub weight_commitment: Option<String>,
    pub io_root: Option<String>,
    pub report_hash: Option<String>,
    // On-chain
    pub contract: String,
    pub network: String,
    pub verification_count: Option<u64>,
    pub tx_hash: Option<String>,
    pub deployer_address: Option<String>,
    pub total_felts: usize,
    pub gas_used: Option<String>,
    // Streaming verification
    pub streaming_steps: Vec<StreamingStep>,
    // Tamper detection
    pub tamper_io: Option<bool>,
    pub tamper_weight: Option<bool>,
    pub tamper_output: Option<bool>,
    // Inference log
    pub turns: Vec<(String, String)>,
    pub logs: Vec<String>,
    // Animation frame counter (for pulsing effects)
    pub frame_count: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PipelineStep {
    Idle,
    Capture,
    GkrProve,
    OnChain,
    Complete,
}

impl PipelineStep {
    pub fn as_u8(&self) -> u8 {
        match self {
            Self::Idle => 0,
            Self::Capture => 1,
            Self::GkrProve => 2,
            Self::OnChain => 3,
            Self::Complete => 4,
        }
    }
}

impl Default for DashboardState {
    fn default() -> Self {
        Self {
            model_name: "qwen2-0.5b".into(),
            model_params: "247,726,080".into(),
            model_layers: 169,
            num_turns: 0, tokens_in: 0, tokens_out: 0,
            step: PipelineStep::Idle,
            capture_progress: 0.0, prove_progress: 0.0, onchain_progress: 0.0,
            capture_time: None, prove_time: None, onchain_time: None,
            prove_time_secs: 0.0,
            elapsed_secs: 0,
            weight_commitment: None, io_root: None, report_hash: None,
            contract: "0x0121d1e9882967e03399f153d57fc208f3d9bce69adc48d9e12d424502a8c005".into(),
            network: "Starknet Sepolia".into(),
            verification_count: None,
            tx_hash: Some("0x2859e0605bd41bed34f65240e2e243cbfeb6c81f4dc60d7d431034f23fd2308".into()),
            deployer_address: None,
            total_felts: 0,
            gas_used: None,
            streaming_steps: default_streaming_steps(),
            tamper_io: None, tamper_weight: None, tamper_output: None,
            turns: Vec::new(), logs: Vec::new(),
            frame_count: 0,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Main render
// ═══════════════════════════════════════════════════════════════════════

pub fn render(frame: &mut Frame, state: &DashboardState) {
    let area = frame.area();
    frame.render_widget(
        ratatui::widgets::Block::default().style(Style::default().bg(BG)),
        area,
    );

    let outer = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(6),  // Header
            Constraint::Length(1),  // Divider
            Constraint::Min(10),   // Body
            Constraint::Length(1),  // Divider
            Constraint::Length(2),  // Footer
        ])
        .split(area);

    render_header(frame, outer[0], state);
    render_divider(frame, outer[1], LIME_DIM);
    render_body(frame, outer[2], state);
    render_divider(frame, outer[3], GHOST);
    render_footer(frame, outer[4], state);
}

fn render_divider(frame: &mut Frame, area: Rect, color: Color) {
    let line = H_LINE.repeat(area.width as usize);
    frame.render_widget(
        Paragraph::new(Span::styled(line, Style::default().fg(color))),
        area,
    );
}

// ── Header ──────────────────────────────────────────────────────────

fn render_header(frame: &mut Frame, area: Rect, state: &DashboardState) {
    let layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(46), Constraint::Min(20)])
        .split(area);

    // Logo — left side
    let logo = vec![
        Line::from(Span::styled("", Style::default())),
        Line::from(vec![
            Span::styled("  ╔═╗╔╗  ╔═╗╦  ╦ ╦╔═╗╦╔═", Style::default().fg(LIME)),
        ]),
        Line::from(vec![
            Span::styled("  ║ ║╠╩╗ ╠═ ║  ╚╦╝╔═╝╠╩╗", Style::default().fg(LIME)),
        ]),
        Line::from(vec![
            Span::styled("  ╚═╝╚═╝ ╚═╝╩═╝ ╩ ╚═╝╩ ╩", Style::default().fg(LIME_DIM)),
        ]),
        Line::from(vec![
            Span::styled("  VERIFIABLE ML INFERENCE", Style::default().fg(SLATE)),
        ]),
    ];

    frame.render_widget(
        Paragraph::new(logo).style(Style::default().bg(BG)),
        layout[0],
    );

    // Model info — right side
    let is_idle = state.step == PipelineStep::Idle;
    let is_complete = state.step == PipelineStep::Complete;
    let (status_text, status_color) = if is_complete {
        ("VERIFIED", EMERALD)
    } else if is_idle {
        ("IDLE", GHOST)
    } else {
        ("PROVING", LIME)
    };

    let elapsed_str = if !is_idle && !is_complete && state.elapsed_secs > 0 {
        format!("  {}s", state.elapsed_secs)
    } else {
        String::new()
    };

    let info = vec![
        Line::from(Span::styled("", Style::default())),
        Line::from(vec![
            Span::styled("  MODEL  ", Style::default().fg(SLATE)),
            Span::styled(&state.model_name, Style::default().fg(WHITE).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(vec![
            Span::styled("  ARCH   ", Style::default().fg(SLATE)),
            Span::styled(
                format!("{} params {} layers", state.model_params, state.model_layers),
                Style::default().fg(SILVER),
            ),
        ]),
        Line::from(vec![
            Span::styled("  STATUS ", Style::default().fg(SLATE)),
            Span::styled(
                format!("{DIAMOND} {status_text}"),
                Style::default().fg(status_color).add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                elapsed_str,
                Style::default().fg(ORANGE).add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!("  {} turns  {}→{} tokens",
                    state.num_turns, state.tokens_in, state.tokens_out),
                Style::default().fg(SLATE),
            ),
        ]),
        Line::from(vec![
            Span::styled("  ENGINE ", Style::default().fg(SLATE)),
            Span::styled("STWO ", Style::default().fg(LIME_DIM)),
            Span::styled("Circle STARK + GKR", Style::default().fg(GHOST)),
        ]),
    ];

    frame.render_widget(
        Paragraph::new(info).style(Style::default().bg(BG)),
        layout[1],
    );
}

// ── Body ────────────────────────────────────────────────────────────

fn render_body(frame: &mut Frame, area: Rect, state: &DashboardState) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(40),  // Pipeline
            Constraint::Length(1),        // Gutter
            Constraint::Percentage(30),  // Crypto
            Constraint::Length(1),        // Gutter
            Constraint::Percentage(30),  // Inference Log
        ])
        .split(area);

    render_pipeline(frame, cols[0], state);
    render_gutter(frame, cols[1]);
    render_crypto(frame, cols[2], state);
    render_gutter(frame, cols[3]);
    render_conversation(frame, cols[4], state);
}

fn render_gutter(frame: &mut Frame, area: Rect) {
    let lines: Vec<Line> = (0..area.height).map(|_| {
        Line::from(Span::styled(V_LINE, Style::default().fg(GHOST)))
    }).collect();
    frame.render_widget(Paragraph::new(lines), area);
}

// ── Pipeline column ─────────────────────────────────────────────────

pub fn render_pipeline(frame: &mut Frame, area: Rect, state: &DashboardState) {
    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),  // Section header
            Constraint::Length(1),  // Spacer
            Constraint::Length(3),  // Step 1: Capture
            Constraint::Length(3),  // Step 2: GKR Prove
            Constraint::Length(3),  // Step 3: On-Chain
            Constraint::Length(1),  // Spacer
            Constraint::Min(2),    // Coverage
        ])
        .split(area);

    // Section header
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled("  PROOF PIPELINE", Style::default().fg(LIME).add_modifier(Modifier::BOLD)),
        ])),
        layout[0],
    );

    let pulse_lime = if state.frame_count % 4 < 2 { LIME } else { LIME_DIM };

    render_step(frame, layout[2], 1, "CAPTURE",
        "M31 forward pass",
        state.capture_progress, state.capture_time,
        state.step.as_u8() >= PipelineStep::Capture.as_u8(),
        state.step == PipelineStep::Capture, pulse_lime);

    render_step(frame, layout[3], 2, "GKR PROVE",
        "Sumcheck + STARK + Binding",
        state.prove_progress, state.prove_time,
        state.step.as_u8() >= PipelineStep::GkrProve.as_u8(),
        state.step == PipelineStep::GkrProve, pulse_lime);

    render_step(frame, layout[4], 3, "ON-CHAIN",
        "6-step Starknet verification",
        state.onchain_progress, state.onchain_time,
        state.step.as_u8() >= PipelineStep::OnChain.as_u8(),
        state.step == PipelineStep::OnChain, pulse_lime);

    // Coverage stats
    let coverage = vec![
        Line::from(vec![
            Span::styled("  ", Style::default()),
            Span::styled("96", Style::default().fg(LIME).add_modifier(Modifier::BOLD)),
            Span::styled(" matmul ", Style::default().fg(SLATE)),
            Span::styled("24", Style::default().fg(LIME).add_modifier(Modifier::BOLD)),
            Span::styled(" silu ", Style::default().fg(SLATE)),
            Span::styled("49", Style::default().fg(LIME).add_modifier(Modifier::BOLD)),
            Span::styled(" rmsnorm", Style::default().fg(SLATE)),
        ]),
        Line::from(vec![
            Span::styled("  per turn · poseidon merkle · io binding", Style::default().fg(GHOST)),
        ]),
    ];
    frame.render_widget(Paragraph::new(coverage), layout[6]);
}

fn render_step(
    frame: &mut Frame, area: Rect,
    num: u8, name: &str, desc: &str,
    progress: f64, time: Option<f64>, active: bool,
    is_current: bool, pulse_color: Color,
) {
    let done = progress >= 1.0;
    let running = active && !done && progress > 0.0;

    let (icon_color, icon) = if done {
        (EMERALD, CHECK)
    } else if running || is_current {
        (if is_current { pulse_color } else { LIME }, ARROW_R)
    } else if active {
        (LIME_DIM, ARROW_R)
    } else {
        (GHOST, DOT)
    };

    let name_color = if done { EMERALD } else if active { WHITE } else { SLATE };
    let time_str = time.map(|t| format!(" {:.1}s", t)).unwrap_or_default();

    // Line 1: icon + number + name + desc + time
    let line1 = Line::from(vec![
        Span::styled(format!("  {icon} "), Style::default().fg(icon_color)),
        Span::styled(format!("{num}"), Style::default().fg(if active { LIME } else { GHOST })),
        Span::styled(" ", Style::default()),
        Span::styled(name, Style::default().fg(name_color).add_modifier(Modifier::BOLD)),
        Span::styled(format!("  {desc}"), Style::default().fg(GHOST)),
    ]);

    // Line 2: custom progress bar
    let bar_width = (area.width as usize).saturating_sub(6);
    let filled = ((progress * bar_width as f64) as usize).min(bar_width);
    let empty = bar_width.saturating_sub(filled);

    let bar_color = if done { EMERALD } else if is_current { pulse_color } else if active { LIME } else { GHOST };
    let _bar_bg = if active { BG_ACTIVE } else { BG };

    let bar_str = format!(
        "    {}{}{}",
        BLOCK_FULL.repeat(filled),
        if !done && active && filled < bar_width { BLOCK_LOW } else { "" },
        " ".repeat(empty.saturating_sub(if !done && active && filled < bar_width { 1 } else { 0 })),
    );

    let line2 = Line::from(vec![
        Span::styled(bar_str, Style::default().fg(bar_color)),
        Span::styled(time_str, Style::default().fg(EMERALD)),
    ]);

    frame.render_widget(
        Paragraph::new(vec![line1, line2]).style(Style::default().bg(BG)),
        area,
    );
}

// ── Crypto column ───────────────────────────────────────────────────

pub fn render_crypto(frame: &mut Frame, area: Rect, state: &DashboardState) {
    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),   // Commitments header
            Constraint::Length(1),   // Spacer
            Constraint::Length(3),   // Weight hash
            Constraint::Length(3),   // IO hash
            Constraint::Length(3),   // Report hash
            Constraint::Length(1),   // Spacer
            Constraint::Length(1),   // On-chain verification header
            Constraint::Length(1),   // Spacer
            Constraint::Min(4),     // Streaming steps + totals
        ])
        .split(area);

    // Commitments header
    frame.render_widget(
        Paragraph::new(Span::styled(" COMMITMENTS", Style::default().fg(VIOLET).add_modifier(Modifier::BOLD))),
        layout[0],
    );

    render_hash_field(frame, layout[2], "WEIGHT", &state.weight_commitment);
    render_hash_field(frame, layout[3], "IO", &state.io_root);
    render_hash_field(frame, layout[4], "REPORT", &state.report_hash);

    // On-chain verification header
    frame.render_widget(
        Paragraph::new(Span::styled(" ON-CHAIN VERIFICATION", Style::default().fg(EMERALD).add_modifier(Modifier::BOLD))),
        layout[6],
    );

    // Streaming steps + totals
    let mut lines: Vec<Line> = Vec::new();

    for step in &state.streaming_steps {
        let (icon, icon_color) = match step.status {
            StepStatus::Pending => (DOT, GHOST),
            StepStatus::Submitting => (ARROW_R, ORANGE),
            StepStatus::Confirmed => (CHECK, EMERALD),
            StepStatus::Failed => (CROSS, RED),
        };

        let name_color = match step.status {
            StepStatus::Pending => GHOST,
            StepStatus::Submitting => ORANGE,
            StepStatus::Confirmed => EMERALD,
            StepStatus::Failed => RED,
        };

        // Line 1: icon + step name
        let line1 = Line::from(vec![
            Span::styled(format!(" {icon} "), Style::default().fg(icon_color)),
            Span::styled(&step.name, Style::default().fg(LILAC)),
        ]);
        lines.push(line1);

        // Lines 2-3: TX hash split across two lines to fit in column
        if let Some(ref tx) = step.tx_hash {
            let chars: Vec<char> = tx.chars().collect();
            let mid = chars.len() / 2;
            let first: String = chars[..mid].iter().collect();
            let second: String = chars[mid..].iter().collect();
            lines.push(Line::from(vec![
                Span::styled("   ", Style::default()),
                Span::styled(first, Style::default().fg(EMERALD)),
            ]));
            lines.push(Line::from(vec![
                Span::styled("   ", Style::default()),
                Span::styled(second, Style::default().fg(EMERALD)),
            ]));
        }
    }

    // Spacer before totals
    lines.push(Line::from(Span::styled("", Style::default())));

    // Total felts
    if state.total_felts > 0 {
        lines.push(Line::from(vec![
            Span::styled(" felts    ", Style::default().fg(GHOST)),
            Span::styled(
                format!("{}", state.total_felts),
                Style::default().fg(ORANGE).add_modifier(Modifier::BOLD),
            ),
        ]));
    }

    // Gas cost
    if let Some(ref gas) = state.gas_used {
        lines.push(Line::from(vec![
            Span::styled(" gas      ", Style::default().fg(GHOST)),
            Span::styled(gas, Style::default().fg(SILVER)),
        ]));
    }

    // Tamper detection section
    lines.push(Line::from(Span::styled("", Style::default())));
    lines.push(Line::from(Span::styled(" INTEGRITY", Style::default().fg(RED).add_modifier(Modifier::BOLD))));

    lines.push(tamper_line("io commitment", state.tamper_io));
    lines.push(tamper_line("weight commitment", state.tamper_weight));
    lines.push(tamper_line("inference output", state.tamper_output));
    lines.push(Line::from(vec![
        Span::styled(format!(" {SHIELD} "), Style::default().fg(EMERALD)),
        Span::styled("56 adversarial tests", Style::default().fg(EMERALD)),
    ]));

    frame.render_widget(Paragraph::new(lines), layout[8]);
}

fn render_hash_field(frame: &mut Frame, area: Rect, label: &str, value: &Option<String>) {
    let placeholder = "waiting...";
    let hash = value.as_deref().unwrap_or(placeholder);

    // Show as much as possible — split across two lines if needed
    let width = area.width.saturating_sub(2) as usize;
    let label_width = 8;
    let hash_space = width.saturating_sub(label_width);

    let char_count = hash.chars().count();
    let lines = if char_count <= hash_space {
        vec![Line::from(vec![
            Span::styled(format!(" {label:<7} "), Style::default().fg(SLATE)),
            Span::styled(hash, Style::default().fg(VIOLET)),
        ])]
    } else {
        // Split hash across two lines using char boundaries
        let mid_chars = char_count / 2;
        let first: String = hash.chars().take(mid_chars).collect();
        let second: String = hash.chars().skip(mid_chars).collect();
        vec![
            Line::from(vec![
                Span::styled(format!(" {label:<7} "), Style::default().fg(SLATE)),
                Span::styled(first, Style::default().fg(VIOLET)),
            ]),
            Line::from(vec![
                Span::styled("         ", Style::default()),
                Span::styled(second, Style::default().fg(VIOLET)),
            ]),
        ]
    };
    frame.render_widget(Paragraph::new(lines), area);
}

// ── Conversation column ─────────────────────────────────────────────

fn render_conversation(frame: &mut Frame, area: Rect, state: &DashboardState) {
    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),  // Header
            Constraint::Length(1),  // Spacer
            Constraint::Min(4),    // Turns
        ])
        .split(area);

    frame.render_widget(
        Paragraph::new(Span::styled(" INFERENCE LOG", Style::default().fg(LIME).add_modifier(Modifier::BOLD))),
        layout[0],
    );

    let mut lines: Vec<Line> = Vec::new();

    for (i, (user, ai)) in state.turns.iter().enumerate() {
        let idx = format!("#{:<2}", i);
        lines.push(Line::from(vec![
            Span::styled(format!(" {idx}"), Style::default().fg(GHOST)),
            Span::styled(" YOU ", Style::default().fg(LIME).add_modifier(Modifier::BOLD)),
            Span::styled(truncate_str(user, 24), Style::default().fg(WHITE)),
        ]));
        lines.push(Line::from(vec![
            Span::styled("    ", Style::default()),
            Span::styled(" AI  ", Style::default().fg(EMERALD)),
            Span::styled(truncate_str(ai, 24), Style::default().fg(SLATE)),
        ]));
        lines.push(Line::from(Span::styled("", Style::default())));
    }

    if lines.is_empty() {
        lines.push(Line::from(Span::styled(" awaiting input…", Style::default().fg(GHOST))));
    }

    frame.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }), layout[2]);
}

// ── Footer ──────────────────────────────────────────────────────────

fn render_footer(frame: &mut Frame, area: Rect, state: &DashboardState) {
    let (status_text, status_color) = match state.step {
        PipelineStep::Idle =>     ("IDLE",      GHOST),
        PipelineStep::Capture =>  ("CAPTURING", LIME),
        PipelineStep::GkrProve => ("PROVING",   LIME),
        PipelineStep::OnChain =>  ("ON-CHAIN",  ORANGE),
        PipelineStep::Complete => ("VERIFIED",  EMERALD),
    };

    let elapsed_str = if state.elapsed_secs > 0 {
        format!(" {}s", state.elapsed_secs)
    } else {
        String::new()
    };

    // Truncate contract to ~20 chars
    let contract_short = truncate_hash(&state.contract, 20);

    let footer = Line::from(vec![
        Span::styled(" ", Style::default()),
        Span::styled(" ObelyZK ", Style::default().fg(BG).bg(LIME).add_modifier(Modifier::BOLD)),
        Span::styled("  ", Style::default()),
        Span::styled(status_text, Style::default().fg(status_color).add_modifier(Modifier::BOLD)),
        Span::styled(&elapsed_str, Style::default().fg(ORANGE)),
        Span::styled("  ", Style::default()),
        Span::styled(&contract_short, Style::default().fg(VIOLET)),
        Span::styled("  ", Style::default()),
        Span::styled(&state.network, Style::default().fg(SLATE)),
        Span::styled("              ", Style::default()),
        Span::styled("q", Style::default().fg(LIME)),
        Span::styled(" exit  ", Style::default().fg(GHOST)),
        Span::styled("p", Style::default().fg(LIME)),
        Span::styled(" prove", Style::default().fg(GHOST)),
    ]);

    frame.render_widget(
        Paragraph::new(footer).style(Style::default().bg(BG)),
        area,
    );
}

// ── Helpers ─────────────────────────────────────────────────────────

fn tamper_line(name: &str, result: Option<bool>) -> Line<'static> {
    match result {
        Some(true) => Line::from(vec![
            Span::styled(format!(" {CHECK} "), Style::default().fg(EMERALD)),
            Span::styled(name.to_string(), Style::default().fg(SILVER)),
            Span::styled(" REJECTED", Style::default().fg(EMERALD).add_modifier(Modifier::BOLD)),
        ]),
        Some(false) => Line::from(vec![
            Span::styled(format!(" {CROSS} "), Style::default().fg(RED)),
            Span::styled(name.to_string(), Style::default().fg(SILVER)),
            Span::styled(" LEAKED", Style::default().fg(RED).add_modifier(Modifier::BOLD)),
        ]),
        None => Line::from(vec![
            Span::styled(format!(" {DOT} "), Style::default().fg(GHOST)),
            Span::styled(name.to_string(), Style::default().fg(GHOST)),
        ]),
    }
}

fn truncate_hash(s: &str, max: usize) -> String {
    if s.len() <= max { return s.to_string(); }
    format!("{}…{}", &s[..max/2], &s[s.len()-6..])
}

fn truncate_str(s: &str, max: usize) -> String {
    if s.chars().count() <= max { return s.to_string(); }
    s.chars().take(max).collect::<String>() + "…"
}
