//! ObelyZK Interactive TUI — production-quality ML proving dashboard.
//!
//! Inspired by Couture Trace (AbdelStark/llm-provable-computer) aesthetic.
//! Custom sparklines, gauges, badges, Chart widgets, mood text, 3 themes.
//! All indexed colors for SSH/tmux compatibility.
//!
//! Architecture: sync event loop with mpsc channels for background proving.

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect, Margin},
    style::{Color, Modifier, Style},
    symbols,
    text::{Line, Span},
    widgets::{
        Block, Borders, Chart, Dataset, GraphType, Axis,
        List, ListItem, Paragraph, Tabs, Padding, Wrap,
    },
    Frame,
};

// ═══════════════════════════════════════════════════════════════════════
// THEME SYSTEM — 3 themes, all indexed colors for SSH
// ═══════════════════════════════════════════════════════════════════════

#[derive(Clone, Copy)]
pub struct Theme {
    pub bg: Color,
    pub panel: Color,
    pub panel_alt: Color,
    pub border: Color,
    pub text: Color,
    pub muted: Color,
    pub accent: Color,
    pub accent_soft: Color,
    pub accent_alt: Color,
    pub success: Color,
    pub danger: Color,
}

pub const THEME_NEON: Theme = Theme {
    bg:          Color::Indexed(233),   // near-black
    panel:       Color::Indexed(235),   // dark grey
    panel_alt:   Color::Indexed(237),   // slightly lighter
    border:      Color::Indexed(44),    // cyan
    text:        Color::Indexed(253),   // bright grey
    muted:       Color::Indexed(243),   // mid grey
    accent:      Color::Indexed(118),   // bright green
    accent_soft: Color::Indexed(70),    // dim green
    accent_alt:  Color::Indexed(208),   // orange
    success:     Color::Indexed(48),    // emerald
    danger:      Color::Indexed(196),   // red
};

pub const THEME_EMBER: Theme = Theme {
    bg:          Color::Indexed(233),
    panel:       Color::Indexed(235),
    panel_alt:   Color::Indexed(237),
    border:      Color::Indexed(208),   // orange borders
    text:        Color::Indexed(223),   // warm white
    muted:       Color::Indexed(180),   // warm grey
    accent:      Color::Indexed(209),   // coral
    accent_soft: Color::Indexed(178),   // amber
    accent_alt:  Color::Indexed(114),   // mint green
    success:     Color::Indexed(114),
    danger:      Color::Indexed(167),   // rust red
};

pub const THEME_VIOLET: Theme = Theme {
    bg:          Color::Indexed(233),
    panel:       Color::Indexed(235),
    panel_alt:   Color::Indexed(237),
    border:      Color::Indexed(141),   // violet borders
    text:        Color::Indexed(189),   // lavender white
    muted:       Color::Indexed(146),   // soft purple
    accent:      Color::Indexed(213),   // pink
    accent_soft: Color::Indexed(141),   // violet
    accent_alt:  Color::Indexed(81),    // sky blue
    success:     Color::Indexed(114),
    danger:      Color::Indexed(168),
};

pub const THEMES: &[Theme] = &[THEME_NEON, THEME_EMBER, THEME_VIOLET];
pub const THEME_NAMES: &[&str] = &["Neon", "Ember", "Violet"];

// ═══════════════════════════════════════════════════════════════════════
// CONSTANTS
// ═══════════════════════════════════════════════════════════════════════

const SPARK_GLYPHS: &[char] = &['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
const PULSE_CHARS: &[&str] = &["◢", "◣", "◤", "◥"];
pub const HISTORY_LIMIT: usize = 96;

// ═══════════════════════════════════════════════════════════════════════
// STATE
// ═══════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, PartialEq)]
pub enum Mode { Monitor, Prove, Chat, OnChain, Model }

impl Mode {
    pub fn titles() -> Vec<&'static str> { vec!["1:Monitor", "2:Prove", "3:Chat", "4:On-Chain", "5:Model"] }
    pub fn index(&self) -> usize { match self { Mode::Monitor=>0, Mode::Prove=>1, Mode::Chat=>2, Mode::OnChain=>3, Mode::Model=>4 } }
}

#[derive(Debug, Clone)]
pub struct ModelEntry { pub name: String, pub params: String, pub layers: usize, pub matmuls: usize, pub loaded: bool }

#[derive(Debug, Clone)]
pub struct LayerEvent { pub layer_idx: usize, pub layer_type: String, pub time_ms: u64, pub done: bool }

#[derive(Debug, Clone)]
pub struct OnChainTx { pub tx_hash: String, pub model: String, pub felts: usize, pub verified: bool }

#[derive(Debug, Clone)]
pub struct ChatMsg { pub role: String, pub content: String, pub proof_status: Option<String>, pub tx_hash: Option<String> }

pub struct InteractiveDashState {
    pub mode: Mode,
    pub theme_idx: usize,
    pub models: Vec<ModelEntry>,
    pub selected_model: usize,

    pub gpu_name: String,
    pub gpu_memory_gb: f32,
    pub gpu_temp_c: f32,
    pub gpu_util_pct: f32,
    pub gpu_power_w: f32,
    pub gpu_mem_used_gb: f32,
    pub gpu_temp_history: Vec<u64>,

    pub proving_active: bool,
    pub proving_layer: usize,
    pub proving_total_layers: usize,
    pub proving_matmul: usize,
    pub proving_total_matmuls: usize,
    pub proving_elapsed_secs: f64,
    pub proving_phase: String,

    pub throughput_history: Vec<u64>,
    pub gpu_util_history: Vec<u64>,
    pub layer_time_history: Vec<u64>,
    pub matmul_time_history: Vec<u64>,
    pub stark_history: Vec<u64>,

    pub current_tok_per_sec: f64,
    pub peak_tok_per_sec: f64,
    pub total_tokens_proven: usize,
    pub total_proofs: usize,

    pub stark_time_secs: Option<f64>,
    pub stark_felts: Option<usize>,

    // Model architecture details
    pub model_hidden_size: usize,
    pub model_num_heads: usize,
    pub model_ff_size: usize,
    pub model_vocab_size: usize,
    pub model_arch: String,

    // Weight details
    pub weight_total_params: String,
    pub weight_size_gb: f32,
    pub weight_shards: usize,
    pub weight_commitment: String,
    pub weight_cache_status: String,

    // Forward pass details
    pub fwd_node_current: usize,
    pub fwd_node_total: usize,
    pub fwd_elapsed_secs: f64,
    pub fwd_history: Vec<u64>,

    // Poseidon hashing stats
    pub poseidon_perms: usize,
    pub poseidon_history: Vec<u64>,

    // Sumcheck round details
    pub sumcheck_round: usize,
    pub sumcheck_total_rounds: usize,
    pub sumcheck_history: Vec<u64>,

    pub on_chain_txs: Vec<OnChainTx>,
    pub verification_count: usize,
    pub contract: String,
    pub network: String,

    pub chat_messages: Vec<ChatMsg>,
    pub input_buffer: String,
    pub input_cursor: usize,

    pub layer_events: Vec<LayerEvent>,

    pub uptime_secs: u64,
    pub tick: u64,
    pub tick_rate_ms: u64,
}

impl Default for InteractiveDashState {
    fn default() -> Self {
        // Seed ALL histories with wave patterns so sparklines never look empty
        let mut th = Vec::with_capacity(HISTORY_LIMIT);
        let mut gu = Vec::with_capacity(HISTORY_LIMIT);
        let mut lt = Vec::with_capacity(HISTORY_LIMIT);
        let mut mt = Vec::with_capacity(HISTORY_LIMIT);
        let mut st = Vec::with_capacity(HISTORY_LIMIT);
        let mut gt = Vec::with_capacity(HISTORY_LIMIT);
        let mut fh = Vec::with_capacity(HISTORY_LIMIT);
        let mut ph = Vec::with_capacity(HISTORY_LIMIT);
        let mut sh = Vec::with_capacity(HISTORY_LIMIT);
        for i in 0..HISTORY_LIMIT {
            let x = i as f64;
            th.push(((x * 0.15).sin().abs() * 6.0 + 1.0) as u64);
            gu.push((30.0 + (x * 0.08).sin().abs() * 50.0) as u64);
            lt.push((80.0 + (x * 0.12).cos().abs() * 350.0) as u64);
            mt.push((150.0 + (x * 0.18).sin().abs() * 200.0) as u64);
            st.push(((x * 0.05).sin().abs() * 3.0) as u64);
            gt.push((35.0 + (x * 0.06).sin().abs() * 10.0) as u64);
            fh.push(((x * 0.2).cos().abs() * 50.0 + 5.0) as u64);
            ph.push(((x * 0.3).sin().abs() * 800.0 + 100.0) as u64);
            sh.push(((x * 0.25).cos().abs() * 15.0 + 2.0) as u64);
        }

        Self {
            mode: Mode::Monitor,
            theme_idx: 0,
            models: vec![
                ModelEntry { name: "Qwen2.5-14B".into(), params: "14B".into(), layers: 337, matmuls: 192, loaded: true },
                ModelEntry { name: "GLM-4-9B".into(), params: "9B".into(), layers: 281, matmuls: 160, loaded: true },
                ModelEntry { name: "SmolLM2-135M".into(), params: "135M".into(), layers: 25, matmuls: 12, loaded: false },
            ],
            selected_model: 0,
            gpu_name: "NVIDIA H100 PCIe".into(),
            gpu_memory_gb: 80.0,
            gpu_temp_c: 38.0,
            gpu_util_pct: 12.0,
            gpu_power_w: 120.0,
            gpu_mem_used_gb: 28.5,
            gpu_temp_history: gt,
            proving_active: false,
            proving_layer: 0, proving_total_layers: 337,
            proving_matmul: 0, proving_total_matmuls: 192,
            proving_elapsed_secs: 0.0,
            proving_phase: "idle".into(),
            throughput_history: th,
            gpu_util_history: gu,
            layer_time_history: lt,
            matmul_time_history: mt,
            stark_history: st,
            current_tok_per_sec: 0.0,
            peak_tok_per_sec: 0.23,
            total_tokens_proven: 7,
            total_proofs: 5,
            stark_time_secs: Some(1.2),
            stark_felts: Some(946),
            model_hidden_size: 5120,
            model_num_heads: 40,
            model_ff_size: 13824,
            model_vocab_size: 152064,
            model_arch: "qwen2".into(),
            weight_total_params: "14.2B".into(),
            weight_size_gb: 29.5,
            weight_shards: 8,
            weight_commitment: "0x05e8dcc9bdf4ff44ae26...".into(),
            weight_cache_status: "warm".into(),
            fwd_node_current: 0,
            fwd_node_total: 337,
            fwd_elapsed_secs: 0.0,
            fwd_history: fh,
            poseidon_perms: 22771,
            poseidon_history: ph,
            sumcheck_round: 0,
            sumcheck_total_rounds: 2544,
            sumcheck_history: sh,
            on_chain_txs: vec![
                OnChainTx { tx_hash: "0x5ce1b41815e29a7b3dd0..".into(), model: "Qwen2.5-14B".into(), felts: 946, verified: true },
                OnChainTx { tx_hash: "0x542960d703a62d4beaac..".into(), model: "GLM-4-9B".into(), felts: 929, verified: true },
                OnChainTx { tx_hash: "0x677694b934d9bd6d8d2f..".into(), model: "Qwen2.5-14B".into(), felts: 892, verified: true },
            ],
            verification_count: 7,
            contract: "0x1c208a5fe731c0d03b098b524f274c537587ea1d43d903838cc4a2bf90c40c7".into(),
            network: "Starknet Sepolia".into(),
            chat_messages: Vec::new(),
            input_buffer: String::new(),
            input_cursor: 0,
            layer_events: vec![
                LayerEvent { layer_idx: 336, layer_type: "RMSNorm".into(), time_ms: 12, done: true },
                LayerEvent { layer_idx: 335, layer_type: "MatMul".into(), time_ms: 245, done: true },
                LayerEvent { layer_idx: 334, layer_type: "Add".into(), time_ms: 8, done: true },
                LayerEvent { layer_idx: 333, layer_type: "MatMul".into(), time_ms: 231, done: true },
                LayerEvent { layer_idx: 332, layer_type: "RMSNorm".into(), time_ms: 11, done: true },
                LayerEvent { layer_idx: 331, layer_type: "MatMul".into(), time_ms: 252, done: true },
                LayerEvent { layer_idx: 330, layer_type: "MatMul".into(), time_ms: 219, done: true },
                LayerEvent { layer_idx: 329, layer_type: "Add".into(), time_ms: 7, done: true },
            ],
            uptime_secs: 0,
            tick: 0,
            tick_rate_ms: 50,
        }
    }
}

impl InteractiveDashState {
    pub fn theme(&self) -> &Theme { &THEMES[self.theme_idx % THEMES.len()] }
}

// ═══════════════════════════════════════════════════════════════════════
// HELPERS — Custom widgets following Couture Trace patterns
// ═══════════════════════════════════════════════════════════════════════

/// Ring buffer push — FIFO, removes oldest when at capacity.
pub fn push_limited(buf: &mut Vec<u64>, val: u64) {
    if buf.len() >= HISTORY_LIMIT { buf.remove(0); }
    buf.push(val);
}

/// Consistent panel block with themed borders + title marker.
fn panel_block<'a>(title: &'a str, t: &Theme) -> Block<'a> {
    Block::default()
        .title(Line::from(vec![
            Span::styled("█", Style::default().fg(t.accent)),
            Span::styled(format!(" {} ", title), Style::default().fg(t.text).add_modifier(Modifier::BOLD)),
        ]))
        .borders(Borders::ALL)
        .border_set(symbols::border::PROPORTIONAL_TALL)
        .border_style(Style::default().fg(t.border))
        .style(Style::default().bg(t.panel))
        .padding(Padding::horizontal(1))
}

/// Badge pill: inverted color (fg text on bg color, BOLD).
fn badge(text: &str, fg: Color, bg: Color) -> Span<'static> {
    Span::styled(format!(" {} ", text), Style::default().fg(fg).bg(bg).add_modifier(Modifier::BOLD))
}

/// 8-level sparkline string from data using ▁▂▃▄▅▆▇█, sliding window.
fn sparkline_str(data: &[u64], width: usize) -> String {
    if data.is_empty() { return " ".repeat(width); }
    let start = data.len().saturating_sub(width);
    let window = &data[start..];
    let max = window.iter().copied().max().unwrap_or(1).max(1);
    window.iter().map(|&v| {
        let idx = ((v as f64 / max as f64) * 7.0).round() as usize;
        SPARK_GLYPHS[idx.min(7)]
    }).collect()
}

/// Gauge bar: 18-char ████████░░░░░░░░░░ with percentage.
fn gauge_line<'a>(label: &str, ratio: f64, t: &Theme) -> Line<'a> {
    let width = 18usize;
    let filled = (ratio.clamp(0.0, 1.0) * width as f64).round() as usize;
    let mut spans = vec![
        Span::styled(format!("{:<10}", label), Style::default().fg(t.muted)),
    ];
    for i in 0..width {
        spans.push(Span::styled(
            if i < filled { "█" } else { "░" },
            Style::default().fg(if i < filled { t.accent } else { t.panel_alt }),
        ));
    }
    spans.push(Span::styled(format!(" {:>3.0}%", ratio * 100.0), Style::default().fg(t.text)));
    Line::from(spans)
}

/// Timing bar: █ width proportional to ms value.
fn timing_bar(ms: u64, max_ms: u64, width: usize, color: Color) -> String {
    let filled = if max_ms > 0 { ((ms as f64 / max_ms as f64) * width as f64).round() as usize } else { 0 };
    "█".repeat(filled.min(width))
}

/// Pulse character based on tick.
fn pulse(tick: u64) -> &'static str { PULSE_CHARS[(tick / 3) as usize % PULSE_CHARS.len()] }

/// ML proving mood text — hardcoded narrative per phase.
fn proving_mood(phase: &str, layer_type: &str) -> &'static str {
    match (phase, layer_type) {
        ("forward", _) => "propagating activations through the transformer stack",
        ("gkr", "MatMul") => "proving matmul reductions with surgical precision",
        ("gkr", "RMSNorm") => "binding normalization through the Fiat-Shamir channel",
        ("gkr", "Add") => "folding residual connections into the proof transcript",
        ("gkr", "Activation") => "committing activation lookups via LogUp tables",
        ("stark", _) => "compressing the GKR transcript into a recursive STARK",
        ("on-chain", _) => "submitting the verified proof to Starknet",
        ("idle", _) => "awaiting the next inference request",
        _ => "processing layers through the proving pipeline",
    }
}

fn uptime_str(secs: u64) -> String {
    format!("{}:{:02}:{:02}", secs / 3600, (secs % 3600) / 60, secs % 60)
}

// ═══════════════════════════════════════════════════════════════════════
// MAIN RENDER
// ═══════════════════════════════════════════════════════════════════════

pub fn render_interactive(frame: &mut Frame, state: &InteractiveDashState) {
    let t = state.theme();
    let area = frame.area();

    // Clear background
    frame.render_widget(
        Block::default().style(Style::default().bg(t.bg)),
        area,
    );

    let outer = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(5),   // Hero section
            Constraint::Length(1),   // Tab bar
            Constraint::Min(10),    // Content area
            Constraint::Length(1),   // Footer
        ])
        .split(area);

    render_hero(frame, outer[0], state, t);
    render_tabs(frame, outer[1], state, t);

    match state.mode {
        Mode::Monitor => render_monitor(frame, outer[2], state, t),
        Mode::Prove => render_prove(frame, outer[2], state, t),
        Mode::Chat => render_chat(frame, outer[2], state, t),
        Mode::OnChain => render_onchain(frame, outer[2], state, t),
        Mode::Model => render_model(frame, outer[2], state, t),
    }

    render_footer(frame, outer[3], state, t);
}

// ═══════════════════════════════════════════════════════════════════════
// HERO — Model info + GPU + Status + Mood
// ═══════════════════════════════════════════════════════════════════════

fn render_hero(frame: &mut Frame, area: Rect, state: &InteractiveDashState, t: &Theme) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(55), Constraint::Percentage(45)])
        .split(area);

    let model = state.models.get(state.selected_model);
    let mname = model.map(|m| m.name.as_str()).unwrap_or("none");
    let p = pulse(state.tick);
    let phase_color = match state.proving_phase.as_str() {
        "gkr" => t.accent, "stark" => t.accent_alt, "on-chain" => t.accent_soft, "forward" => t.accent_alt, _ => t.muted,
    };
    let last_layer_type = state.layer_events.first().map(|e| e.layer_type.as_str()).unwrap_or("");
    let mood = proving_mood(&state.proving_phase, last_layer_type);

    // Left: title + model + mood
    let hero_lines = vec![
        Line::from(vec![
            Span::styled(format!(" {} ", p), Style::default().fg(t.bg).bg(t.accent)),
            Span::styled(" ObelyZK VM ", Style::default().fg(t.text).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(vec![
            Span::styled(format!("  {} ", mname), Style::default().fg(t.accent).add_modifier(Modifier::BOLD)),
            Span::styled(format!("• {} layers • {} matmuls",
                model.map(|m| m.layers).unwrap_or(0),
                model.map(|m| m.matmuls).unwrap_or(0),
            ), Style::default().fg(t.muted)),
        ]),
        Line::from(vec![
            Span::styled(format!("  {}", state.proving_phase.to_uppercase()), Style::default().fg(phase_color).add_modifier(Modifier::BOLD)),
            Span::styled(if state.proving_active { format!("  {:.1}s", state.proving_elapsed_secs) } else { String::new() }, Style::default().fg(t.accent_alt)),
        ]),
        Line::from(Span::styled(format!("  mood: {}", mood), Style::default().fg(t.muted).add_modifier(Modifier::ITALIC))),
    ];
    frame.render_widget(
        Paragraph::new(hero_lines).block(
            Block::default().borders(Borders::ALL).border_set(symbols::border::PROPORTIONAL_TALL)
                .border_style(Style::default().fg(t.border)).style(Style::default().bg(t.panel))
        ),
        cols[0],
    );

    // Right: status badges + GPU
    let status_lines = vec![
        Line::from(vec![
            badge(&state.proving_phase.to_uppercase(), t.bg, phase_color),
            Span::styled("  ", Style::default()),
            badge(THEME_NAMES[state.theme_idx % THEME_NAMES.len()], t.bg, t.border),
            Span::styled("  ", Style::default()),
            badge(&format!("{}ms", state.tick_rate_ms), t.bg, t.muted),
        ]),
        Line::from(vec![
            Span::styled(format!("  GPU {} ", state.gpu_name), Style::default().fg(t.text)),
            Span::styled(format!("{:.0}GB  {:.0}°C", state.gpu_memory_gb, state.gpu_temp_c), Style::default().fg(t.muted)),
        ]),
        Line::from(vec![
            Span::styled("  throughput ", Style::default().fg(t.muted)),
            Span::styled(format!("{:.1} tok/s", state.current_tok_per_sec), Style::default().fg(t.accent).add_modifier(Modifier::BOLD)),
            Span::styled("  proven ", Style::default().fg(t.muted)),
            Span::styled(format!("{}", state.total_tokens_proven), Style::default().fg(t.text)),
            Span::styled("  on-chain ", Style::default().fg(t.muted)),
            Span::styled(format!("{}", state.verification_count), Style::default().fg(t.accent_soft)),
        ]),
    ];
    frame.render_widget(
        Paragraph::new(status_lines).block(
            Block::default().borders(Borders::ALL).border_set(symbols::border::PROPORTIONAL_TALL)
                .border_style(Style::default().fg(t.border)).style(Style::default().bg(t.panel))
        ),
        cols[1],
    );
}

// ═══════════════════════════════════════════════════════════════════════
// TABS
// ═══════════════════════════════════════════════════════════════════════

fn render_tabs(frame: &mut Frame, area: Rect, state: &InteractiveDashState, t: &Theme) {
    let titles: Vec<Line> = Mode::titles().iter().map(|s| Line::from(*s)).collect();
    frame.render_widget(
        Tabs::new(titles)
            .select(state.mode.index())
            .highlight_style(Style::default().fg(t.bg).bg(t.accent).add_modifier(Modifier::BOLD))
            .style(Style::default().fg(t.muted).bg(t.bg))
            .divider(Span::styled(" │ ", Style::default().fg(t.border))),
        area,
    );
}

// ═══════════════════════════════════════════════════════════════════════
// MODE: MONITOR — Dense data-rich layout
// ═══════════════════════════════════════════════════════════════════════

fn render_monitor(frame: &mut Frame, area: Rect, state: &InteractiveDashState, t: &Theme) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(4),   // Top: dispatch + pulse + registers
            Constraint::Min(4),     // Middle: chart | layer bars + trace
            Constraint::Length(4),   // Sparklines row
            Constraint::Length(5),   // Bottom: GPU memory | proof artifact
        ])
        .split(area);

    // Top row: 3 panels
    let top_cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(34), Constraint::Percentage(33), Constraint::Percentage(33)])
        .split(rows[0]);

    // Dispatch panel
    let last_ev = state.layer_events.first();
    let dispatch_lines = vec![
        Line::from(vec![
            Span::styled("  next  ", Style::default().fg(t.muted)),
            Span::styled(format!("L{} {}", last_ev.map(|e| e.layer_idx.saturating_sub(1)).unwrap_or(0),
                last_ev.map(|e| e.layer_type.as_str()).unwrap_or("--")), Style::default().fg(t.accent)),
        ]),
        Line::from(vec![
            Span::styled("  latest ", Style::default().fg(t.muted)),
            Span::styled(format!("#{} {} {}ms",
                last_ev.map(|e| e.layer_idx).unwrap_or(0),
                last_ev.map(|e| e.layer_type.as_str()).unwrap_or("--"),
                last_ev.map(|e| e.time_ms).unwrap_or(0),
            ), Style::default().fg(t.text)),
        ]),
    ];
    frame.render_widget(Paragraph::new(dispatch_lines).block(panel_block("Dispatch", t)), top_cols[0]);

    // Pulse panel: progress + tempo
    let ratio = if state.proving_total_layers > 0 { state.proving_layer as f64 / state.proving_total_layers as f64 } else { 0.0 };
    let pulse_lines = vec![
        gauge_line("progress", ratio, t),
        Line::from(vec![
            Span::styled(format!("  tempo   {}ms/frame", state.tick_rate_ms), Style::default().fg(t.muted)),
        ]),
    ];
    frame.render_widget(Paragraph::new(pulse_lines).block(panel_block("Pulse", t)), top_cols[1]);

    // Registers panel: badge pills
    let reg_lines = vec![
        Line::from(vec![
            badge(&format!("{:.1} tok/s", state.current_tok_per_sec), t.bg, t.accent),
            Span::styled(" ", Style::default()),
            badge(&format!("peak {:.1}", state.peak_tok_per_sec), t.bg, t.accent_alt),
        ]),
        Line::from(vec![
            badge(&format!("{} proven", state.total_tokens_proven), t.bg, t.muted),
            Span::styled(" ", Style::default()),
            badge(&format!("{} on-chain", state.verification_count), t.bg, t.accent_soft),
        ]),
    ];
    frame.render_widget(Paragraph::new(reg_lines).block(panel_block("Registers", t)), top_cols[2]);

    // Middle: chart (55%) | layer spotlight + trace (45%)
    let mid_cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(55), Constraint::Percentage(45)])
        .split(rows[1]);

    // Chart: dual dataset (throughput + GPU util)
    let tp_points: Vec<(f64, f64)> = state.throughput_history.iter().enumerate()
        .map(|(i, &v)| (i as f64, v as f64)).collect();
    let gpu_points: Vec<(f64, f64)> = state.gpu_util_history.iter().enumerate()
        .map(|(i, &v)| (i as f64, v as f64 / 10.0)).collect();
    let max_y = state.throughput_history.iter().copied().max().unwrap_or(10).max(10) as f64;

    let datasets = vec![
        Dataset::default().name("tok/s").data(&tp_points)
            .graph_type(GraphType::Line).marker(symbols::Marker::Braille)
            .style(Style::default().fg(t.accent)),
        Dataset::default().name("gpu").data(&gpu_points)
            .graph_type(GraphType::Line).marker(symbols::Marker::Dot)
            .style(Style::default().fg(t.accent_alt)),
    ];

    let chart = Chart::new(datasets)
        .block(panel_block("Signal Drift", t))
        .x_axis(Axis::default()
            .bounds([0.0, HISTORY_LIMIT as f64])
            .labels(vec![
                Span::styled("0", Style::default().fg(t.muted)),
                Span::styled(format!("{}", HISTORY_LIMIT / 2), Style::default().fg(t.muted)),
                Span::styled(format!("{}", HISTORY_LIMIT), Style::default().fg(t.muted)),
            ]))
        .y_axis(Axis::default()
            .bounds([0.0, max_y])
            .labels(vec![
                Span::styled("0", Style::default().fg(t.muted)),
                Span::styled(format!("{:.0}", max_y), Style::default().fg(t.muted)),
            ]));
    frame.render_widget(chart, mid_cols[0]);

    // Right: layer spotlight + trace
    let right_rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(5), Constraint::Min(3)])
        .split(mid_cols[1]);

    // Layer spotlight: timing bars
    let max_ms = state.layer_events.iter().map(|e| e.time_ms).max().unwrap_or(1);
    let layer_items: Vec<ListItem> = state.layer_events.iter().take(3).map(|e| {
        let bar = timing_bar(e.time_ms, max_ms, 15, t.accent);
        let type_color = match e.layer_type.as_str() {
            "MatMul" => t.accent, "RMSNorm" => t.accent_soft, "Add" => t.muted, _ => t.text,
        };
        ListItem::new(Line::from(vec![
            Span::styled(format!("  {:<10}", e.layer_type), Style::default().fg(type_color)),
            Span::styled(format!("{:>4}ms ", e.time_ms), Style::default().fg(t.accent_alt)),
            Span::styled(bar, Style::default().fg(type_color)),
        ]))
    }).collect();
    frame.render_widget(List::new(layer_items).block(panel_block("Layer Spotlight", t)), right_rows[0]);

    // Live trace
    let trace_items: Vec<ListItem> = state.layer_events.iter().take(10).map(|e| {
        let sym = if e.done { "✓" } else { "▸" };
        let sc = if e.done { t.success } else { t.accent_alt };
        ListItem::new(Line::from(vec![
            Span::styled(format!("  {} ", sym), Style::default().fg(sc)),
            Span::styled(format!("L{:<4}", e.layer_idx), Style::default().fg(t.muted)),
            Span::styled(format!("{:<10}", e.layer_type), Style::default().fg(t.text)),
            Span::styled(format!("{:>4}ms", e.time_ms), Style::default().fg(t.accent_alt)),
        ]))
    }).collect();
    frame.render_widget(List::new(trace_items).block(panel_block("Live Trace", t)), right_rows[1]);

    // Sparklines row: 3 sparklines side by side
    let spark_cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(34), Constraint::Percentage(33), Constraint::Percentage(33)])
        .split(rows[2]);

    // Throughput sparkline
    let sw1 = spark_cols[0].width.saturating_sub(4) as usize;
    let s1 = sparkline_str(&state.throughput_history, sw1);
    frame.render_widget(
        Paragraph::new(vec![
            Line::from(Span::styled(format!("  {}", s1), Style::default().fg(t.accent))),
            Line::from(vec![
                Span::styled(format!("  {:.1} tok/s", state.current_tok_per_sec), Style::default().fg(t.accent).add_modifier(Modifier::BOLD)),
                Span::styled(format!("  peak {:.1}", state.peak_tok_per_sec), Style::default().fg(t.accent_alt)),
            ]),
        ]).block(panel_block("Throughput", t)),
        spark_cols[0],
    );

    // GPU utilization sparkline
    let sw2 = spark_cols[1].width.saturating_sub(4) as usize;
    let s2 = sparkline_str(&state.gpu_util_history, sw2);
    frame.render_widget(
        Paragraph::new(vec![
            Line::from(Span::styled(format!("  {}", s2), Style::default().fg(t.accent_alt))),
            Line::from(vec![
                Span::styled(format!("  {:.0}% util", state.gpu_util_pct), Style::default().fg(t.accent_alt).add_modifier(Modifier::BOLD)),
                Span::styled(format!("  {:.0}°C", state.gpu_temp_c), Style::default().fg(t.muted)),
            ]),
        ]).block(panel_block("GPU Load", t)),
        spark_cols[1],
    );

    // Layer timing sparkline
    let sw3 = spark_cols[2].width.saturating_sub(4) as usize;
    let s3 = sparkline_str(&state.layer_time_history, sw3);
    frame.render_widget(
        Paragraph::new(vec![
            Line::from(Span::styled(format!("  {}", s3), Style::default().fg(t.accent_soft))),
            Line::from(vec![
                Span::styled(format!("  {}ms/layer", state.layer_time_history.last().unwrap_or(&0)), Style::default().fg(t.accent_soft).add_modifier(Modifier::BOLD)),
            ]),
        ]).block(panel_block("Layer Time", t)),
        spark_cols[2],
    );

    // Bottom row: GPU Memory | Proof Artifact
    let bot_cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(rows[3]);

    // GPU Memory panel
    let gpu_mem_ratio: f64 = 0.35;
    let gpu_lines = vec![
        gauge_line("VRAM", gpu_mem_ratio, t),
        Line::from(vec![
            Span::styled(format!("  {:.0}GB / {:.0}GB", state.gpu_memory_gb as f64 * gpu_mem_ratio, state.gpu_memory_gb), Style::default().fg(t.text)),
            Span::styled(format!("  {}", state.gpu_name), Style::default().fg(t.muted)),
        ]),
        Line::from(vec![
            Span::styled("  weights ", Style::default().fg(t.muted)),
            Span::styled("loaded", Style::default().fg(t.success)),
            Span::styled("  cache ", Style::default().fg(t.muted)),
            Span::styled("warm", Style::default().fg(t.success)),
        ]),
    ];
    frame.render_widget(Paragraph::new(gpu_lines).block(panel_block("GPU Memory", t)), bot_cols[0]);

    // Proof Artifact panel
    let proof_lines = vec![
        Line::from(vec![
            Span::styled("  GKR proof  ", Style::default().fg(t.muted)),
            Span::styled(format!("{} layers", state.proving_total_layers), Style::default().fg(t.text)),
            Span::styled(format!("  {} matmuls", state.proving_total_matmuls), Style::default().fg(t.text)),
        ]),
        Line::from(vec![
            Span::styled("  STARK      ", Style::default().fg(t.muted)),
            Span::styled(format!("{} felts", state.stark_felts.unwrap_or(0)), Style::default().fg(t.accent_alt).add_modifier(Modifier::BOLD)),
            Span::styled(format!("  {:.1}s", state.stark_time_secs.unwrap_or(0.0)), Style::default().fg(t.text)),
        ]),
        Line::from(vec![
            Span::styled("  on-chain   ", Style::default().fg(t.muted)),
            badge(&format!("{} verified", state.verification_count), t.bg, t.success),
            Span::styled(format!("  {}", state.network), Style::default().fg(t.accent_soft)),
        ]),
    ];
    frame.render_widget(Paragraph::new(proof_lines).block(panel_block("Proof Artifact", t)), bot_cols[1]);
}

// ═══════════════════════════════════════════════════════════════════════
// MODE: PROVE
// ═══════════════════════════════════════════════════════════════════════

fn render_prove(frame: &mut Frame, area: Rect, state: &InteractiveDashState, t: &Theme) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(4), Constraint::Length(4), Constraint::Min(4)])
        .split(area);

    // GKR progress
    let ratio = if state.proving_total_layers > 0 { state.proving_layer as f64 / state.proving_total_layers as f64 } else { 0.0 };
    let gkr_lines = vec![
        gauge_line("GKR proof", ratio, t),
        Line::from(vec![
            Span::styled(format!("  Layer {}/{}", state.proving_layer, state.proving_total_layers), Style::default().fg(t.text)),
            Span::styled(format!("  Matmul {}/{}", state.proving_matmul, state.proving_total_matmuls), Style::default().fg(t.muted)),
            Span::styled(format!("  {:.1}s", state.proving_elapsed_secs), Style::default().fg(t.accent_alt)),
        ]),
    ];
    frame.render_widget(Paragraph::new(gkr_lines).block(panel_block("GKR Proof", t)), rows[0]);

    // STARK status
    let stark_lines = if let Some(time) = state.stark_time_secs {
        vec![Line::from(vec![
            badge("COMPRESSED", t.bg, t.success),
            Span::styled(format!("  {:.1}s ", time), Style::default().fg(t.accent_alt).add_modifier(Modifier::BOLD)),
            Span::styled(format!(" {} felts ", state.stark_felts.unwrap_or(0)), Style::default().fg(t.text)),
            badge("ON-CHAIN READY", t.bg, t.accent),
        ]),
        Line::from(Span::styled("  GKR (~46K felts) → recursive STARK (~950 felts) → Starknet TX", Style::default().fg(t.muted))),
        ]
    } else {
        vec![Line::from(vec![
            Span::styled(format!("  {} ", pulse(state.tick)), Style::default().fg(t.accent_alt)),
            Span::styled(if state.proving_active { "Compressing GKR proof..." } else { "Awaiting GKR completion" }, Style::default().fg(t.muted)),
        ])]
    };
    frame.render_widget(Paragraph::new(stark_lines).block(panel_block("Recursive STARK", t)), rows[1]);

    // Live trace with timing bars
    let max_ms = state.layer_events.iter().map(|e| e.time_ms).max().unwrap_or(1);
    let items: Vec<ListItem> = state.layer_events.iter().take(15).map(|e| {
        let sym = if e.done { "✓" } else { "▸" };
        let sc = if e.done { t.success } else { t.accent_alt };
        let bar = timing_bar(e.time_ms, max_ms, 20, t.accent);
        let tc = match e.layer_type.as_str() { "MatMul" => t.accent, "RMSNorm" => t.accent_soft, _ => t.muted };
        ListItem::new(Line::from(vec![
            Span::styled(format!("  {}", sym), Style::default().fg(sc)),
            Span::styled(format!(" L{:<4}", e.layer_idx), Style::default().fg(t.muted)),
            Span::styled(format!("{:<10}", e.layer_type), Style::default().fg(tc)),
            Span::styled(format!("{:>4}ms ", e.time_ms), Style::default().fg(t.accent_alt)),
            Span::styled(bar, Style::default().fg(tc)),
        ]))
    }).collect();
    frame.render_widget(List::new(items).block(panel_block("Live Proof Trace", t)), rows[2]);
}

// ═══════════════════════════════════════════════════════════════════════
// MODE: CHAT
// ═══════════════════════════════════════════════════════════════════════

fn render_chat(frame: &mut Frame, area: Rect, state: &InteractiveDashState, t: &Theme) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(6), Constraint::Length(3)])
        .split(area);

    // Messages
    let items: Vec<ListItem> = state.chat_messages.iter().map(|m| {
        let (label, lbg) = if m.role == "user" { ("YOU", t.accent) } else { ("AI", t.accent_alt) };
        let mut spans = vec![
            badge(label, t.bg, lbg),
            Span::styled(format!("  {}", m.content), Style::default().fg(t.text)),
        ];
        if let Some(ref st) = m.proof_status {
            let sc = match st.as_str() { "verified" => t.success, "proving" => t.accent_alt, _ => t.danger };
            spans.push(Span::styled(format!("  [{}]", st), Style::default().fg(sc)));
            if st == "proving" {
                spans.push(Span::styled(format!(" {} {:.0}s", pulse(state.tick), state.proving_elapsed_secs), Style::default().fg(t.accent_alt)));
            }
        }
        let mut lines = vec![Line::from(spans)];
        if let Some(ref tx) = m.tx_hash {
            lines.push(Line::from(vec![
                Span::styled("         TX: ", Style::default().fg(t.muted)),
                Span::styled(tx.clone(), Style::default().fg(t.accent_soft)),
            ]));
        }
        ListItem::new(lines)
    }).collect();

    let empty_msg = if state.chat_messages.is_empty() {
        vec![ListItem::new(Line::from(Span::styled(
            "  Type a message below to start a verifiable conversation...",
            Style::default().fg(t.muted),
        )))]
    } else { vec![] };

    let all_items = if state.chat_messages.is_empty() { empty_msg } else { items };
    frame.render_widget(List::new(all_items).block(panel_block("Verifiable Chat", t)), rows[0]);

    // Input
    let input = Paragraph::new(Line::from(vec![
        Span::styled(" › ", Style::default().fg(t.accent).add_modifier(Modifier::BOLD)),
        Span::styled(state.input_buffer.clone(), Style::default().fg(t.text)),
        Span::styled("█", Style::default().fg(t.accent)),
    ]))
    .block(Block::default()
        .borders(Borders::ALL).border_set(symbols::border::PROPORTIONAL_TALL)
        .border_style(Style::default().fg(t.accent_alt))
        .title(Span::styled(" Enter send · Esc cancel · proof runs in background ", Style::default().fg(t.muted)))
        .style(Style::default().bg(t.panel)));
    frame.render_widget(input, rows[1]);
}

// ═══════════════════════════════════════════════════════════════════════
// MODE: ON-CHAIN
// ═══════════════════════════════════════════════════════════════════════

fn render_onchain(frame: &mut Frame, area: Rect, state: &InteractiveDashState, t: &Theme) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(5), Constraint::Min(4)])
        .split(area);

    let cs = if state.contract.len() > 40 { format!("{}...{}", &state.contract[..20], &state.contract[state.contract.len()-8..]) } else { state.contract.clone() };
    let info = vec![
        Line::from(vec![ badge("CONTRACT", t.bg, t.accent_soft), Span::styled(format!("  {}", cs), Style::default().fg(t.text)) ]),
        Line::from(vec![ badge(&state.network, t.bg, t.border), Span::styled(format!("  {} verifications", state.verification_count), Style::default().fg(t.accent).add_modifier(Modifier::BOLD)) ]),
        Line::from(vec![
            Span::styled("  STARK: ", Style::default().fg(t.muted)),
            Span::styled(format!("{:.1}s compression", state.stark_time_secs.unwrap_or(0.0)), Style::default().fg(t.accent_alt)),
            Span::styled(format!("  {} felts/proof", state.stark_felts.unwrap_or(0)), Style::default().fg(t.text)),
        ]),
    ];
    frame.render_widget(Paragraph::new(info).block(panel_block("Starknet Verification", t)), rows[0]);

    let items: Vec<ListItem> = state.on_chain_txs.iter().rev().map(|tx| {
        let short = if tx.tx_hash.len() > 24 { &tx.tx_hash[..24] } else { &tx.tx_hash };
        ListItem::new(Line::from(vec![
            badge(if tx.verified {"VERIFIED"} else {"PENDING"}, t.bg, if tx.verified {t.success} else {t.accent_alt}),
            Span::styled(format!("  {}", short), Style::default().fg(t.accent_soft)),
            Span::styled(format!("  {}", tx.model), Style::default().fg(t.text)),
            Span::styled(format!("  {} felts", tx.felts), Style::default().fg(t.accent_alt)),
        ]))
    }).collect();
    frame.render_widget(List::new(items).block(panel_block("Verified Proofs", t)), rows[1]);
}

// ═══════════════════════════════════════════════════════════════════════
// FOOTER
// ═══════════════════════════════════════════════════════════════════════

fn render_footer(frame: &mut Frame, area: Rect, state: &InteractiveDashState, t: &Theme) {
    let sc = if state.proving_active { t.accent_alt } else { t.success };
    let st = if state.proving_active { "PROVING" } else { "READY" };
    let line = Line::from(vec![
        badge("ObelyZK", t.bg, t.accent), Span::styled("  ", Style::default()),
        Span::styled(st, Style::default().fg(sc).add_modifier(Modifier::BOLD)),
        Span::styled(format!("  {}  {}  {} TXs  ", uptime_str(state.uptime_secs), state.network, state.verification_count), Style::default().fg(t.muted)),
        Span::styled("1-5", Style::default().fg(t.accent)), Span::styled(" mode  ", Style::default().fg(t.muted)),
        Span::styled("t", Style::default().fg(t.accent)), Span::styled(" theme  ", Style::default().fg(t.muted)),
        Span::styled("q", Style::default().fg(t.accent)), Span::styled(" quit  ", Style::default().fg(t.muted)),
        Span::styled("+/-", Style::default().fg(t.accent)), Span::styled(" speed", Style::default().fg(t.muted)),
    ]);
    frame.render_widget(Paragraph::new(line).style(Style::default().bg(t.bg)), area);
}

// ═══════════════════════════════════════════════════════════════════════
// MODE: MODEL — Architecture + Weights + Inference Details
// ═══════════════════════════════════════════════════════════════════════

fn render_model(frame: &mut Frame, area: Rect, state: &InteractiveDashState, t: &Theme) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(area);

    // Left column
    let left_rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(8), Constraint::Length(7), Constraint::Min(4)])
        .split(cols[0]);

    // Architecture panel
    let model = state.models.get(state.selected_model);
    let mname = model.map(|m| m.name.as_str()).unwrap_or("none");
    let arch_lines = vec![
        Line::from(vec![
            badge(mname, t.bg, t.accent),
            Span::styled(format!("  {}", state.model_arch), Style::default().fg(t.accent_soft)),
        ]),
        Line::from(vec![
            Span::styled("  hidden_size  ", Style::default().fg(t.muted)),
            Span::styled(format!("{}", state.model_hidden_size), Style::default().fg(t.text).add_modifier(Modifier::BOLD)),
            Span::styled("  heads  ", Style::default().fg(t.muted)),
            Span::styled(format!("{}", state.model_num_heads), Style::default().fg(t.text).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(vec![
            Span::styled("  ff_size      ", Style::default().fg(t.muted)),
            Span::styled(format!("{}", state.model_ff_size), Style::default().fg(t.text).add_modifier(Modifier::BOLD)),
            Span::styled("  vocab  ", Style::default().fg(t.muted)),
            Span::styled(format!("{}", state.model_vocab_size), Style::default().fg(t.text).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(vec![
            Span::styled("  layers       ", Style::default().fg(t.muted)),
            Span::styled(format!("{}", model.map(|m| m.layers).unwrap_or(0)), Style::default().fg(t.accent).add_modifier(Modifier::BOLD)),
            Span::styled("  matmuls  ", Style::default().fg(t.muted)),
            Span::styled(format!("{}", model.map(|m| m.matmuls).unwrap_or(0)), Style::default().fg(t.accent).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(vec![
            Span::styled("  block: ", Style::default().fg(t.muted)),
            Span::styled("RMSNorm → Q/K/V → O → RMSNorm → gate×up → down → Add", Style::default().fg(t.accent_soft)),
        ]),
    ];
    frame.render_widget(Paragraph::new(arch_lines).block(panel_block("Architecture", t)), left_rows[0]);

    // Weight panel
    let weight_lines = vec![
        Line::from(vec![
            Span::styled("  params   ", Style::default().fg(t.muted)),
            Span::styled(state.weight_total_params.clone(), Style::default().fg(t.text).add_modifier(Modifier::BOLD)),
            Span::styled(format!("  ({:.1} GB, {} shards)", state.weight_size_gb, state.weight_shards), Style::default().fg(t.muted)),
        ]),
        gauge_line("loaded", 1.0, t),
        Line::from(vec![
            Span::styled("  commitment ", Style::default().fg(t.muted)),
            Span::styled(state.weight_commitment.clone(), Style::default().fg(t.accent_soft)),
        ]),
        Line::from(vec![
            Span::styled("  cache      ", Style::default().fg(t.muted)),
            badge(&state.weight_cache_status.to_uppercase(), t.bg,
                if state.weight_cache_status == "warm" { t.success } else { t.accent_alt }),
            Span::styled("  Poseidon Merkle roots cached to disk", Style::default().fg(t.muted)),
        ]),
    ];
    frame.render_widget(Paragraph::new(weight_lines).block(panel_block("Weights", t)), left_rows[1]);

    // Poseidon hashing sparkline
    let pw = left_rows[2].width.saturating_sub(4) as usize;
    let ps = sparkline_str(&state.poseidon_history, pw);
    let pos_lines = vec![
        Line::from(Span::styled(format!("  {}", ps), Style::default().fg(t.accent_alt))),
        Line::from(vec![
            Span::styled(format!("  {} perms total", state.poseidon_perms), Style::default().fg(t.accent_alt).add_modifier(Modifier::BOLD)),
            Span::styled("  Fiat-Shamir channel", Style::default().fg(t.muted)),
        ]),
    ];
    frame.render_widget(Paragraph::new(pos_lines).block(panel_block("Poseidon Hashing", t)), left_rows[2]);

    // Right column
    let right_rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(6), Constraint::Length(5), Constraint::Min(4)])
        .split(cols[1]);

    // GPU detail panel
    let gpu_mem_ratio = state.gpu_mem_used_gb as f64 / state.gpu_memory_gb as f64;
    let gw = right_rows[0].width.saturating_sub(4) as usize;
    let gs = sparkline_str(&state.gpu_temp_history, gw);
    let gpu_lines = vec![
        Line::from(vec![
            badge(&state.gpu_name, t.bg, t.accent_alt),
        ]),
        gauge_line("VRAM", gpu_mem_ratio, t),
        Line::from(vec![
            Span::styled(format!("  {:.1}GB / {:.0}GB  ", state.gpu_mem_used_gb, state.gpu_memory_gb), Style::default().fg(t.text)),
            Span::styled(format!("{:.0}W  {:.0}°C", state.gpu_power_w, state.gpu_temp_c), Style::default().fg(t.muted)),
        ]),
        Line::from(Span::styled(format!("  {}", gs), Style::default().fg(t.accent_alt))),
    ];
    frame.render_widget(Paragraph::new(gpu_lines).block(panel_block("GPU Detail", t)), right_rows[0]);

    // Sumcheck detail
    let sc_ratio = if state.sumcheck_total_rounds > 0 { state.sumcheck_round as f64 / state.sumcheck_total_rounds as f64 } else { 0.0 };
    let scw = right_rows[1].width.saturating_sub(4) as usize;
    let scs = sparkline_str(&state.sumcheck_history, scw);
    let sc_lines = vec![
        gauge_line("rounds", sc_ratio, t),
        Line::from(vec![
            Span::styled(format!("  {}/{}", state.sumcheck_round, state.sumcheck_total_rounds), Style::default().fg(t.text)),
            Span::styled("  M31/QM31 field arithmetic", Style::default().fg(t.muted)),
        ]),
        Line::from(Span::styled(format!("  {}", scs), Style::default().fg(t.accent))),
    ];
    frame.render_widget(Paragraph::new(sc_lines).block(panel_block("Sumcheck Rounds", t)), right_rows[1]);

    // Forward pass detail
    let fwd_ratio = if state.fwd_node_total > 0 { state.fwd_node_current as f64 / state.fwd_node_total as f64 } else { 0.0 };
    let fw = right_rows[2].width.saturating_sub(4) as usize;
    let fs = sparkline_str(&state.fwd_history, fw);
    let fwd_lines = vec![
        gauge_line("nodes", fwd_ratio, t),
        Line::from(vec![
            Span::styled(format!("  {}/{}", state.fwd_node_current, state.fwd_node_total), Style::default().fg(t.text)),
            Span::styled(format!("  {:.1}s elapsed", state.fwd_elapsed_secs), Style::default().fg(t.accent_alt)),
        ]),
        Line::from(Span::styled(format!("  {}", fs), Style::default().fg(t.accent_soft))),
    ];
    frame.render_widget(Paragraph::new(fwd_lines).block(panel_block("Forward Pass", t)), right_rows[2]);
}
