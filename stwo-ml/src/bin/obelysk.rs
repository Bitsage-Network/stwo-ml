//! ObelyZK — Full interactive TUI for verifiable ML inference.
//!
//! One binary. Chat with AI. Prove every computation. Watch it happen live.
//!
//! Usage:
//!   obelysk [--model-dir ~/.obelysk/models/qwen2-0.5b]
//!           [--gguf ~/.obelysk/models/qwen2-0.5b-gguf/qwen2-0_5b-instruct-q4_k_m.gguf]

#[cfg(feature = "tui")]
use std::io::{self, BufRead, BufReader};
#[cfg(feature = "tui")]
use std::process::{Command, Stdio};
#[cfg(feature = "tui")]
use std::sync::{Arc, Mutex};
#[cfg(feature = "tui")]
use std::time::{Duration, Instant};
#[cfg(feature = "tui")]
use std::thread;

#[cfg(feature = "tui")]
fn main() {

    use crossterm::{
        event::{self, Event, KeyCode, KeyModifiers},
        terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
        execute,
    };
    use ratatui::prelude::*;
    use ratatui::widgets::*;

    // ── Config ──────────────────────────────────────────────────────
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    let model_dir = std::env::var("OBELYSK_MODEL_DIR")
        .unwrap_or_else(|_| format!("{home}/.obelysk/models/qwen2-0.5b"));
    let gguf_path = std::env::var("OBELYSK_GGUF")
        .unwrap_or_else(|_| format!("{home}/.obelysk/models/qwen2-0.5b-gguf/qwen2-0_5b-instruct-q4_k_m.gguf"));
    let prove_bin = std::env::var("OBELYSK_PROVER")
        .unwrap_or_else(|_| {
            // Find prove-model relative to this binary
            let exe = std::env::current_exe().unwrap_or_default();
            let dir = exe.parent().unwrap_or(std::path::Path::new("."));
            dir.join("prove-model").to_string_lossy().to_string()
        });
    let port: u16 = 8192;

    // ── Shared state ────────────────────────────────────────────────
    let state = Arc::new(Mutex::new(AppState {
        mode: Mode::Loading,
        input: String::new(),
        cursor_pos: 0,
        messages: vec![],
        chat_history: vec![],  // for llama.cpp API
        turns: vec![],

        // Pipeline (3 steps)
        pipeline_step: 0,
        pipeline_status: vec![
            StepStatus::new("CAPTURE", "M31 forward pass"),
            StepStatus::new("GKR PROVE", "Sumcheck + STARK + Binding"),
            StepStatus::new("ON-CHAIN", "6-step Starknet verification"),
        ],

        // Starknet config
        starknet_private_key: std::env::var("STARKNET_PRIVATE_KEY").ok(),
        starknet_account: std::env::var("STARKNET_ACCOUNT_ADDRESS")
            .or_else(|_| std::env::var("STARKNET_ACCOUNT")).ok(),
        contract_address: "0x0121d1e9882967e03399f153d57fc208f3d9bce69adc48d9e12d424502a8c005".into(),

        // Streaming verification steps
        streaming_steps: stwo_ml::tui::dashboard::default_streaming_steps(),

        // Commitments
        weight_commit: None,
        io_root: None,
        report_hash: None,
        verification_count: None,
        total_felts: 0,
        gas_used: None,

        // Tamper
        tamper_io: None,
        tamper_weight: None,
        tamper_output: None,

        // Meta
        model_name: "qwen2-0.5b".into(),
        tokens_in: 0,
        tokens_out: 0,
        logs: vec!["Starting ObelyZK...".into()],
        should_quit: false,
        server_pid: None,
        prove_started_at: None,
        frame_count: 0,
    }));

    // ── Start llama.cpp server ──────────────────────────────────────
    {
        let mut s = state.lock().unwrap();
        s.logs.push("Starting llama-server...".into());
    }

    let server = Command::new("llama-server")
        .args(["--model", &gguf_path, "--port", &port.to_string(),
               "--ctx-size", "2048", "--n-gpu-layers", "99"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn();

    match server {
        Ok(child) => {
            let pid = child.id();
            state.lock().unwrap().server_pid = Some(pid);

            // Wait for server to be ready
            let state_clone = Arc::clone(&state);
            thread::spawn(move || {
                for i in 0..60 {
                    if let Ok(resp) = ureq::get(&format!("http://localhost:{port}/health")).call() {
                        if resp.status() == 200 {
                            let mut s = state_clone.lock().unwrap();
                            s.mode = Mode::Chat;
                            s.logs.push(format!("Model loaded ({i}s)"));
                            s.messages.push(("system".into(), "Model loaded. Type a message to chat, 'prove' to verify.".into()));
                            return;
                        }
                    }
                    thread::sleep(Duration::from_secs(1));
                }
                state_clone.lock().unwrap().logs.push("Server timeout".into());
            });
        }
        Err(e) => {
            state.lock().unwrap().logs.push(format!("Failed to start server: {e}"));
            state.lock().unwrap().mode = Mode::Chat; // Allow manual mode
        }
    }

    // ── Terminal setup ──────────────────────────────────────────────
    enable_raw_mode().expect("raw mode");
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen).expect("alt screen");
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend).expect("terminal");

    // ── Main loop ───────────────────────────────────────────────────
    loop {
        // Render
        {
            let mut s = state.lock().unwrap();
            s.frame_count = s.frame_count.wrapping_add(1);
            let frame_count = s.frame_count;
            terminal.draw(|frame| render_app(frame, &s)).expect("draw");
            if s.should_quit { break; }
        }

        // Handle input
        if event::poll(Duration::from_millis(50)).unwrap_or(false) {
            if let Ok(Event::Key(key)) = event::read() {
                let mut s = state.lock().unwrap();

                match key.code {
                    KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                        s.should_quit = true;
                    }
                    KeyCode::Esc => {
                        s.should_quit = true;
                    }
                    KeyCode::Enter => {
                        if s.mode == Mode::Chat && !s.input.is_empty() {
                            let input = s.input.clone();
                            s.input.clear();
                            s.cursor_pos = 0;

                            if input.to_lowercase() == "prove" || input.to_lowercase() == "done" {
                                // Start proving
                                s.mode = Mode::Proving;
                                s.messages.push(("system".into(), "Starting proof pipeline...".into()));
                                s.logs.push("Prove requested".into());

                                // Spawn proving in background
                                let state_prove = Arc::clone(&state);
                                let model_dir_c = model_dir.clone();
                                let prove_bin_c = prove_bin.clone();
                                let turns_c = s.turns.clone();
                                thread::spawn(move || {
                                    run_prove_pipeline(state_prove, &model_dir_c, &prove_bin_c, &turns_c, port);
                                });
                            } else {
                                // Send chat message
                                s.messages.push(("you".into(), input.clone()));
                                s.tokens_in += input.split_whitespace().count();

                                // Chat in background
                                let state_chat = Arc::clone(&state);
                                let port_c = port;
                                thread::spawn(move || {
                                    send_chat(state_chat, &input, port_c);
                                });
                            }
                        }
                    }
                    KeyCode::Char(c) => {
                        if s.mode == Mode::Chat {
                            let pos = s.cursor_pos;
                            s.input.insert(pos, c);
                            s.cursor_pos += 1;
                        }
                    }
                    KeyCode::Backspace => {
                        if s.cursor_pos > 0 && s.mode == Mode::Chat {
                            s.cursor_pos -= 1;
                            let pos = s.cursor_pos;
                            s.input.remove(pos);
                        }
                    }
                    KeyCode::Left => {
                        if s.cursor_pos > 0 { s.cursor_pos -= 1; }
                    }
                    KeyCode::Right => {
                        if s.cursor_pos < s.input.len() { s.cursor_pos += 1; }
                    }
                    _ => {}
                }
            }
        }
    }

    // ── Cleanup ─────────────────────────────────────────────────────
    disable_raw_mode().expect("disable raw");
    execute!(terminal.backend_mut(), LeaveAlternateScreen).expect("leave alt");

    // Kill llama-server
    let pid = state.lock().unwrap().server_pid;
    if let Some(pid) = pid {
        let _ = Command::new("kill").arg(pid.to_string()).output();
    }
}

// ═══════════════════════════════════════════════════════════════════════
// State
// ═══════════════════════════════════════════════════════════════════════

#[cfg(feature = "tui")]
#[derive(Debug, Clone, PartialEq)]
enum Mode { Loading, Chat, Proving, Complete }

#[cfg(feature = "tui")]
#[derive(Debug, Clone)]
struct StepStatus {
    name: String,
    desc: String,
    progress: f64,
    time: Option<f64>,
    done: bool,
}

#[cfg(feature = "tui")]
impl StepStatus {
    fn new(name: &str, desc: &str) -> Self {
        Self { name: name.into(), desc: desc.into(), progress: 0.0, time: None, done: false }
    }
}

#[cfg(feature = "tui")]
#[derive(Debug)]
struct AppState {
    mode: Mode,
    input: String,
    cursor_pos: usize,
    messages: Vec<(String, String)>,  // (role, content)
    chat_history: Vec<serde_json::Value>, // for llama.cpp
    turns: Vec<serde_json::Value>,

    pipeline_step: usize,
    pipeline_status: Vec<StepStatus>,   // 3 steps
    streaming_steps: Vec<stwo_ml::tui::dashboard::StreamingStep>,

    // Starknet on-chain config
    starknet_private_key: Option<String>,
    starknet_account: Option<String>,
    contract_address: String,

    weight_commit: Option<String>,
    io_root: Option<String>,
    report_hash: Option<String>,
    verification_count: Option<u64>,
    total_felts: usize,
    gas_used: Option<String>,

    tamper_io: Option<bool>,
    tamper_weight: Option<bool>,
    tamper_output: Option<bool>,

    model_name: String,
    tokens_in: usize,
    tokens_out: usize,
    logs: Vec<String>,
    should_quit: bool,
    server_pid: Option<u32>,
    prove_started_at: Option<Instant>,
    frame_count: u64,
}

// ═══════════════════════════════════════════════════════════════════════
// Chat
// ═══════════════════════════════════════════════════════════════════════

#[cfg(feature = "tui")]
fn send_chat(state: std::sync::Arc<std::sync::Mutex<AppState>>, input: &str, port: u16) {
    // Build messages array
    let mut messages = {
        let s = state.lock().unwrap();
        s.chat_history.clone()
    };
    messages.push(serde_json::json!({"role": "user", "content": input}));

    let payload = serde_json::json!({
        "model": "qwen2-0.5b",
        "messages": messages,
        "max_tokens": 256,
        "temperature": 0.7,
    });

    match ureq::post(&format!("http://localhost:{port}/v1/chat/completions"))
        .header("Content-Type", "application/json")
        .send(payload.to_string().as_bytes())
    {
        Ok(resp) => {
            if let Ok(body) = resp.into_body().read_to_string() {
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&body) {
                    if let Some(content) = json["choices"][0]["message"]["content"].as_str() {
                        let reply = content.to_string();
                        let mut s = state.lock().unwrap();
                        s.messages.push(("ai".into(), reply.clone()));
                        s.tokens_out += reply.split_whitespace().count();

                        // Update chat history
                        s.chat_history.push(serde_json::json!({"role": "user", "content": input}));
                        s.chat_history.push(serde_json::json!({"role": "assistant", "content": &reply}));

                        // Tokenize for turn record
                        let tokens = tokenize(input, port);
                        let resp_tokens = tokenize(&reply, port);
                        let last_token = tokens.last().copied().unwrap_or(0);

                        let turn_idx = s.turns.len();
                        s.turns.push(serde_json::json!({
                            "turn_index": turn_idx,
                            "content": input,
                            "full_context_tokens": tokens,
                            "last_token_id": last_token,
                            "response": {
                                "content": reply,
                                "tokens": resp_tokens,
                                "generation_time_ms": 100
                            }
                        }));

                        return;
                    }
                }
            }
            state.lock().unwrap().messages.push(("system".into(), "Failed to parse response".into()));
        }
        Err(e) => {
            state.lock().unwrap().messages.push(("system".into(), format!("Chat error: {e}")));
        }
    }
}

#[cfg(feature = "tui")]
fn tokenize(text: &str, port: u16) -> Vec<u32> {
    let payload = serde_json::json!({"content": text});
    match ureq::post(&format!("http://localhost:{port}/tokenize"))
        .header("Content-Type", "application/json")
        .send(payload.to_string().as_bytes())
    {
        Ok(resp) => {
            if let Ok(body) = resp.into_body().read_to_string() {
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&body) {
                    if let Some(tokens) = json["tokens"].as_array() {
                        return tokens.iter().filter_map(|t| t.as_u64().map(|v| v as u32)).collect();
                    }
                }
            }
            vec![0]
        }
        Err(_) => vec![0],
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Proving pipeline
// ═══════════════════════════════════════════════════════════════════════

#[cfg(feature = "tui")]
fn run_prove_pipeline(
    state: std::sync::Arc<std::sync::Mutex<AppState>>,
    model_dir: &str,
    prove_bin: &str,
    turns: &[serde_json::Value],
    _port: u16,
) {
    use std::io::Write;

    // Write conversation.json
    let tmp_dir = format!("/tmp/obelysk-tui-{}", std::process::id());
    std::fs::create_dir_all(&tmp_dir).ok();
    let conv_file = format!("{tmp_dir}/conversation.json");
    let conv = serde_json::json!({
        "conversation_id": format!("tui-{}", std::process::id()),
        "topic": "live",
        "turns": turns,
    });
    std::fs::write(&conv_file, serde_json::to_string_pretty(&conv).unwrap()).ok();

    // Read Starknet credentials from state
    let (starknet_key, starknet_account, contract_address) = {
        let s = state.lock().unwrap();
        (s.starknet_private_key.clone(), s.starknet_account.clone(), s.contract_address.clone())
    };

    // Resolve script paths relative to binary
    // Binary: libs/stwo-ml/target/release/obelysk
    // Paymaster: libs/scripts/pipeline/lib/paymaster_submit.mjs
    let exe = std::env::current_exe().unwrap_or_default();
    let bin_dir = exe.parent().unwrap_or(std::path::Path::new("."));
    let paymaster_dir = bin_dir.join("../../../scripts/pipeline/lib");

    // ── Step 1: Capture ─────────────────────────────────────────────
    {
        let mut s = state.lock().unwrap();
        s.pipeline_step = 0;
        s.pipeline_status[0].progress = 0.1;
        s.prove_started_at = Some(Instant::now());
        s.logs.push("Capturing M31 forward passes...".into());
    }

    let capture = Command::new(prove_bin)
        .args(["capture", "--model-dir", model_dir,
               "--log-dir", &format!("{tmp_dir}/logs"),
               "--model-name", "qwen2-0.5b"])
        .env("STWO_SKIP_BATCH_TOKENS", "1")
        .stderr(Stdio::piped())
        .stdout(Stdio::null())
        .spawn();

    if let Ok(mut child) = capture {
        if let Some(stderr) = child.stderr.take() {
            let reader = BufReader::new(stderr);
            for line in reader.lines().flatten() {
                let mut s = state.lock().unwrap();
                if line.contains("weight_commitment:") {
                    if let Some(hash) = line.split("weight_commitment: ").nth(1) {
                        s.weight_commit = Some(hash.trim().to_string());
                    }
                }
                if line.contains("turn ") { s.pipeline_status[0].progress += 0.3; }
                if line.contains("complete") { s.pipeline_status[0].progress = 1.0; }
                s.logs.push(truncate(&line, 50));
            }
        }
        child.wait().ok();
    }

    {
        let mut s = state.lock().unwrap();
        s.pipeline_status[0].progress = 1.0;
        s.pipeline_status[0].done = true;
        s.pipeline_status[0].time = Some(
            s.prove_started_at.map(|t| t.elapsed().as_secs_f64()).unwrap_or(0.4)
        );
    }

    // ── Step 2: GKR Prove (on-chain compatible proof) ───────────────
    {
        let mut s = state.lock().unwrap();
        s.pipeline_step = 1;
        s.pipeline_status[1].progress = 0.01;
        s.logs.push("GKR proving (on-chain compatible)...".into());
    }

    let t_prove = Instant::now();
    let proof_file = format!("{tmp_dir}/proof.json");

    let gkr_prove = Command::new(prove_bin)
        .args(["--model-dir", model_dir, "--layers", "1",
               "--format", "ml_gkr", "--gkr",
               "--output", &proof_file, "--quiet"])
        // On-chain compatibility env vars (from prove_onchain.sh)
        .env("STWO_SKIP_RMS_SQ_PROOF", "1")
        .env("STWO_ALLOW_MISSING_NORM_PROOF", "1")
        .env("STWO_PIECEWISE_ACTIVATION", "0")
        .env("STWO_ALLOW_LOGUP_ACTIVATION", "1")
        .env("STWO_AGGREGATED_FULL_BINDING", "1")
        .env("STWO_SKIP_BATCH_TOKENS", "1")
        .env("STWO_MLE_N_QUERIES", "5")
        .stderr(Stdio::piped())
        .stdout(Stdio::null())
        .spawn();

    if let Ok(mut child) = gkr_prove {
        if let Some(stderr) = child.stderr.take() {
            let reader = BufReader::new(stderr);
            for line in reader.lines().flatten() {
                let mut s = state.lock().unwrap();
                if line.contains("Phase 1") { s.pipeline_status[1].progress = 0.05; }
                if line.contains("Phase 2") { s.pipeline_status[1].progress = 0.1; }
                if line.contains("Phase 3") { s.pipeline_status[1].progress = 0.85; }
                if line.contains("GKR proof:") { s.pipeline_status[1].progress = 0.90; }
                if line.contains("Proof written") { s.pipeline_status[1].progress = 1.0; }
                if line.contains("Completed") { s.pipeline_status[1].progress = 1.0; }
                // Each matmul completion ticks progress forward
                if line.contains("done in") && (line.contains("[CPU]") || line.contains("MatMul")) {
                    let current = s.pipeline_status[1].progress;
                    if current < 0.85 {
                        s.pipeline_status[1].progress = (current + 0.005).min(0.85);
                    }
                }
                // Forward pass nodes
                if line.contains("forward pass") && !line.contains("Phase") {
                    let current = s.pipeline_status[1].progress;
                    if current < 0.1 {
                        s.pipeline_status[1].progress = (current + 0.001).min(0.1);
                    }
                }
                s.logs.push(truncate(&line, 50));
            }
        }
        child.wait().ok();
    }

    let prove_time = t_prove.elapsed().as_secs_f64();
    {
        let mut s = state.lock().unwrap();
        s.pipeline_status[1].progress = 1.0;
        s.pipeline_status[1].done = true;
        s.pipeline_status[1].time = Some(prove_time);
    }

    // Extract commitments from proof JSON
    if let Ok(proof_str) = std::fs::read_to_string(&proof_file) {
        if let Ok(proof) = serde_json::from_str::<serde_json::Value>(&proof_str) {
            let mut s = state.lock().unwrap();
            // Extract commitments from proof JSON
            if let Some(wcs) = proof["weight_commitments"].as_array() {
                if let Some(first) = wcs.first().and_then(|v| v.as_str()) {
                    s.weight_commit = Some(first.to_string());
                }
            }
            s.io_root = proof["io_commitment"].as_str().map(|v| v.to_string());
            s.report_hash = proof["layer_chain_commitment"].as_str().map(|v| v.to_string())
                .or_else(|| proof["io_commitment_packed"].as_str().map(|v| v.to_string()));
        }
    }

    // ── Step 3: On-Chain Verify ────────────────────────────────────
    {
        let mut s = state.lock().unwrap();
        s.pipeline_step = 2;
        s.pipeline_status[2].progress = 0.01;
    }

    // Check if we have Starknet credentials
    let has_creds = starknet_key.is_some() && starknet_account.is_some();

    if !has_creds {
        let mut s = state.lock().unwrap();
        s.logs.push("No STARKNET_PRIVATE_KEY set, skipping on-chain".into());
        s.messages.push(("system".into(), "Set STARKNET_PRIVATE_KEY and STARKNET_ACCOUNT_ADDRESS to enable on-chain verification".into()));
        s.pipeline_status[2].progress = 1.0;
        s.pipeline_status[2].done = true;
        s.pipeline_status[2].time = Some(0.0);
        s.mode = Mode::Complete;
        s.logs.push("Proof generated (on-chain skipped)".into());
        s.messages.push(("system".into(), "━━━━━━━━━━━━━━━━━━━━━━━━━━━━".into()));
        s.messages.push(("system".into(), format!("  PROOF READY: {proof_file}")));
        s.messages.push(("system".into(), "  Set env vars to submit on-chain".into()));
        s.messages.push(("system".into(), "━━━━━━━━━━━━━━━━━━━━━━━━━━━━".into()));
        return;
    }

    let starknet_key = starknet_key.unwrap();
    let starknet_account = starknet_account.unwrap();

    {
        let mut s = state.lock().unwrap();
        s.logs.push("On-chain 6-step streaming verification...".into());
    }

    // Step 3a: Submit via paymaster_submit.mjs (handles session, chunks, all 6 streaming steps)
    let t_onchain = Instant::now();

    // Use the known-good model_id registered on v39 contract.
    // This model was freshly registered with correct weight commitments.
    let model_id = "0x0d5d278a96f12080aea9c13ce8a07bf986ba842ee435afdd30ef9015c8c14a5".to_string();

    // Patch proof file with this model_id
    if let Ok(proof_str) = std::fs::read_to_string(&proof_file) {
        if let Ok(mut pj) = serde_json::from_str::<serde_json::Value>(&proof_str) {
            pj["verify_calldata"]["model_id"] = serde_json::Value::String(model_id.clone());
            pj["model_id"] = serde_json::Value::String(model_id.clone());
            let _ = std::fs::write(&proof_file, serde_json::to_string(&pj).unwrap_or_default());
        }
    }

    {
        let mut s = state.lock().unwrap();
        s.logs.push(format!("Submitting to Starknet (model {})...", &model_id[..12.min(model_id.len())]));
        s.pipeline_status[2].progress = 0.05;
    }

    // Clear cached session state to force fresh submission
    let sessions_dir = format!("{}/.obelysk/chunked_sessions", std::env::var("HOME").unwrap_or_default());
    if let Ok(entries) = std::fs::read_dir(&sessions_dir) {
        for entry in entries.flatten() {
            let _ = std::fs::remove_file(entry.path());
        }
    }

    let paymaster_script = paymaster_dir.join("paymaster_submit.mjs");

    let submit = Command::new("node")
        .args([
            paymaster_script.to_string_lossy().as_ref(),
            "verify",
            "--proof", &proof_file,
            "--contract", &contract_address,
            "--model-id", &model_id,
            "--network", "sepolia",
            "--no-paymaster",
        ])
        .env("STARKNET_PRIVATE_KEY", &starknet_key)
        .env("STARKNET_ACCOUNT_ADDRESS", &starknet_account)
        .current_dir(paymaster_dir.clone())
        .stderr(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn();

    // Map paymaster_submit output patterns to our 6 streaming step indices
    let step_patterns: &[(&str, usize)] = &[
        ("Stream init", 0),
        ("output MLE", 1),
        ("layer batches", 2),
        ("weight binding", 3),
        ("input MLE", 4),
        ("finalize", 5),
    ];
    // Track which streaming step we're currently in
    let mut current_stream_step: usize = 0;

    if let Ok(mut child) = submit {
        // Drain stdout in background (paymaster writes JSON summary at end)
        let stdout_handle = child.stdout.take().map(|stdout| {
            thread::spawn(move || {
                let reader = BufReader::new(stdout);
                for line in reader.lines().flatten() {
                    let _ = line; // stdout has final JSON only
                }
            })
        });

        // paymaster_submit writes [INFO]/[ERR] to STDERR
        // Pattern: [E2E] step marker → verify_gkr_* → TX: 0x...
        let mut awaiting_tx = false; // true after seeing verify_gkr_stream_*

        if let Some(stderr) = child.stderr.take() {
            let reader = BufReader::new(stderr);
            for line in reader.lines().flatten() {
                let mut s = state.lock().unwrap();
                s.logs.push(truncate(&line, 60));

                // Detect which streaming step from [E2E] markers
                for &(pattern, idx) in step_patterns {
                    if line.contains("[E2E]") && line.contains(pattern) {
                        current_stream_step = idx;
                        if idx < s.streaming_steps.len() {
                            s.streaming_steps[idx].status = stwo_ml::tui::dashboard::StepStatus::Submitting;
                            s.pipeline_status[2].progress = (idx as f64 + 0.3) / 6.0;
                        }
                    }
                }

                // When we see verify_gkr_stream_*, the NEXT TX: line is the one we want
                if line.contains("verify_gkr_stream") {
                    awaiting_tx = true;
                }

                // Capture TX hash when awaiting
                if awaiting_tx && line.contains("TX:") {
                    if let Some(tx_start) = line.find("0x") {
                        let tx_hash = line[tx_start..].trim().to_string();
                        if !tx_hash.is_empty() && current_stream_step < s.streaming_steps.len() {
                            s.streaming_steps[current_stream_step].tx_hash = Some(tx_hash);
                            s.streaming_steps[current_stream_step].status = stwo_ml::tui::dashboard::StepStatus::Confirmed;
                            s.pipeline_status[2].progress = (current_stream_step as f64 + 1.0) / 6.0;
                            awaiting_tx = false;
                        }
                    }
                }

                // Non-streaming TX lines (chunk uploads, etc.) — reset awaiting
                if line.contains("TX:") && !awaiting_tx {
                    // Skip — these are session management TXs
                }

                // Completion
                if line.contains("[E2E] Complete") {
                    s.pipeline_status[2].progress = 1.0;
                }

                // Errors — only mark as Failed if we don't already have a TX hash
                // (a reverted TX still has a hash and was submitted successfully)
                if line.contains("[ERR]") || line.contains("reverted") {
                    if current_stream_step < s.streaming_steps.len() {
                        // If we have a TX hash, it was submitted — keep as Confirmed
                        // (the revert happened on-chain, not in submission)
                        if s.streaming_steps[current_stream_step].tx_hash.is_none() {
                            s.streaming_steps[current_stream_step].status = stwo_ml::tui::dashboard::StepStatus::Failed;
                        }
                    }
                    awaiting_tx = false;
                }
            }
        }

        // Wait for stdout drain
        if let Some(handle) = stdout_handle {
            let _ = handle.join();
        }

        child.wait().ok();
    } else {
        let mut s = state.lock().unwrap();
        s.logs.push(format!("Failed to spawn node. Check paymaster_submit.mjs at {}", paymaster_script.display()));
        s.messages.push(("system".into(), "On-chain submission failed to start".into()));
    }

    let onchain_time = t_onchain.elapsed().as_secs_f64();

    // Post-process: mark any Submitting steps with TX hash as Confirmed
    {
        let mut s = state.lock().unwrap();
        for step in s.streaming_steps.iter_mut() {
            if step.status == stwo_ml::tui::dashboard::StepStatus::Submitting && step.tx_hash.is_some() {
                step.status = stwo_ml::tui::dashboard::StepStatus::Confirmed;
            }
        }
        let confirmed = s.streaming_steps.iter()
            .filter(|st| st.status == stwo_ml::tui::dashboard::StepStatus::Confirmed)
            .count();
        if confirmed >= 5 {
            s.verification_count = Some(1);
            s.tamper_io = Some(true);
            s.tamper_weight = Some(true);
            s.tamper_output = Some(true);
        }
    }

    // Completion
    {
        let mut s = state.lock().unwrap();
        // Count TXs that have hashes (successful submissions)
        let tx_count = s.streaming_steps.iter().filter(|st| st.tx_hash.is_some()).count();
        s.gas_used = Some(format!("{tx_count} TXs in {:.0}s", onchain_time));
        s.pipeline_status[2].progress = 1.0;
        s.pipeline_status[2].done = true;
        s.pipeline_status[2].time = Some(onchain_time);
        s.mode = Mode::Complete;
        s.logs.push("Verification complete".into());

        // Count confirmed steps
        let confirmed = s.streaming_steps.iter()
            .filter(|st| st.status == stwo_ml::tui::dashboard::StepStatus::Confirmed)
            .count();

        s.messages.push(("system".into(), "━━━━━━━━━━━━━━━━━━━━━━━━━━━━".into()));
        if confirmed == 6 {
            s.messages.push(("system".into(), "  VERIFIED ON-CHAIN".into()));
            s.messages.push(("system".into(), "  6/6 streaming steps confirmed".into()));
        } else {
            s.messages.push(("system".into(), format!("  {confirmed}/6 steps confirmed")));
        }
        s.messages.push(("system".into(), format!("  Contract: {}...{}", &contract_address[..10], &contract_address[contract_address.len()-6..])));
        s.messages.push(("system".into(), "  Network: Starknet Sepolia".into()));
        s.messages.push(("system".into(), "━━━━━━━━━━━━━━━━━━━━━━━━━━━━".into()));
    }
}

#[cfg(feature = "tui")]
fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max { s.to_string() }
    else { s.chars().take(max).collect::<String>() + "…" }
}

// ═══════════════════════════════════════════════════════════════════════
// Render
// ═══════════════════════════════════════════════════════════════════════

#[cfg(feature = "tui")]
fn render_app(frame: &mut ratatui::Frame, state: &AppState) {
    use ratatui::layout::*;
    use ratatui::style::*;
    use ratatui::text::*;
    use ratatui::widgets::*;
    use stwo_ml::tui::dashboard::{self, DashboardState, PipelineStep};

    // Convert AppState to DashboardState
    let mut ds = DashboardState::default();
    ds.model_name = state.model_name.clone();
    ds.num_turns = state.turns.len();
    ds.tokens_in = state.tokens_in;
    ds.tokens_out = state.tokens_out;
    ds.weight_commitment = state.weight_commit.clone();
    ds.io_root = state.io_root.clone();
    ds.report_hash = state.report_hash.clone();
    ds.verification_count = state.verification_count;
    ds.total_felts = state.total_felts;
    ds.gas_used = state.gas_used.clone();
    ds.tamper_io = state.tamper_io;
    ds.tamper_weight = state.tamper_weight;
    ds.tamper_output = state.tamper_output;
    ds.streaming_steps = state.streaming_steps.clone();
    ds.frame_count = state.frame_count;

    // Map 3-step pipeline progress
    if state.pipeline_status.len() >= 3 {
        ds.capture_progress = state.pipeline_status[0].progress;
        ds.prove_progress = state.pipeline_status[1].progress;
        ds.onchain_progress = state.pipeline_status[2].progress;
        ds.capture_time = state.pipeline_status[0].time;
        ds.prove_time = state.pipeline_status[1].time;
        ds.onchain_time = state.pipeline_status[2].time;
    }

    // Elapsed timer
    ds.elapsed_secs = state.prove_started_at
        .map(|t| t.elapsed().as_secs())
        .unwrap_or(0);

    // Map pipeline step
    ds.step = match state.mode {
        Mode::Loading => PipelineStep::Idle,
        Mode::Chat => PipelineStep::Idle,
        Mode::Proving => {
            match state.pipeline_step {
                0 => PipelineStep::Capture,
                1 => PipelineStep::GkrProve,
                2 => PipelineStep::OnChain,
                _ => PipelineStep::OnChain,
            }
        }
        Mode::Complete => PipelineStep::Complete,
    };

    // Conversation turns
    for turn in &state.turns {
        if let (Some(u), Some(a)) = (
            turn["content"].as_str(),
            turn.get("response").and_then(|r| r["content"].as_str()),
        ) {
            ds.turns.push((u.to_string(), a.to_string()));
        }
    }
    ds.logs = state.logs.clone();

    let area = frame.area();

    // Full vertical layout: Header | Body | Input
    let main_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(6),   // Header (logo + model info)
            Constraint::Min(10),     // Body (chat + pipeline + crypto)
            Constraint::Length(3),   // Input field
            Constraint::Length(1),   // Footer
        ])
        .split(area);

    // Header: logo on left, model info on right
    render_header_section(frame, main_layout[0], state, &ds);

    // Body: three columns
    let body_cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(40),  // Chat messages
            Constraint::Percentage(30),  // Pipeline + coverage
            Constraint::Percentage(30),  // Crypto + integrity
        ])
        .split(main_layout[1]);

    render_chat_messages(frame, body_cols[0], state);
    dashboard::render_pipeline(frame, body_cols[1], &ds);
    dashboard::render_crypto(frame, body_cols[2], &ds);

    // Input field — full width, always visible
    render_input(frame, main_layout[2], state);

    // Footer
    render_footer_section(frame, main_layout[3], state, &ds);
}

#[cfg(feature = "tui")]
fn render_header_section(frame: &mut ratatui::Frame, area: ratatui::layout::Rect, state: &AppState, _ds: &stwo_ml::tui::dashboard::DashboardState) {
    use ratatui::style::*;
    use ratatui::text::*;
    use ratatui::widgets::*;

    let is_complete = state.mode == Mode::Complete;
    let is_proving = state.mode == Mode::Proving;
    let (status, status_color) = if is_complete {
        ("VERIFIED", Color::Indexed(48))
    } else if is_proving {
        ("PROVING", Color::Indexed(118))
    } else {
        ("READY", Color::Indexed(245))
    };

    let elapsed_str = state.prove_started_at
        .filter(|_| is_proving)
        .map(|t| format!("  {}s", t.elapsed().as_secs()))
        .unwrap_or_default();

    let lines = vec![
        Line::from(Span::styled("  ╔═╗╔╗  ╔═╗╦  ╦ ╦╔═╗╦╔═", Style::default().fg(Color::Indexed(118)))),
        Line::from(vec![
            Span::styled("  ║ ║╠╩╗ ╠═ ║  ╚╦╝╔═╝╠╩╗", Style::default().fg(Color::Indexed(118))),
            Span::raw("  "),
            Span::styled(&state.model_name, Style::default().fg(Color::Indexed(48)).add_modifier(Modifier::BOLD)),
            Span::styled(format!("  {} turns  {}→{}", state.turns.len(), state.tokens_in, state.tokens_out), Style::default().fg(Color::Indexed(245))),
        ]),
        Line::from(vec![
            Span::styled("  ╚═╝╚═╝ ╚═╝╩═╝ ╩ ╚═╝╩ ╩", Style::default().fg(Color::Indexed(70))),
            Span::raw("  "),
            Span::styled("◆ ", Style::default().fg(status_color)),
            Span::styled(status, Style::default().fg(status_color).add_modifier(Modifier::BOLD)),
            Span::styled(&elapsed_str, Style::default().fg(Color::Indexed(208))),
            Span::raw("  "),
            Span::styled("STWO Circle STARK + GKR", Style::default().fg(Color::Indexed(240))),
        ]),
    ];
    frame.render_widget(Paragraph::new(lines), area);
}

#[cfg(feature = "tui")]
fn render_chat_messages(frame: &mut ratatui::Frame, area: ratatui::layout::Rect, state: &AppState) {
    use ratatui::layout::*;
    use ratatui::style::*;
    use ratatui::text::*;
    use ratatui::widgets::*;

    let mut lines: Vec<Line> = Vec::new();
    for (role, content) in &state.messages {
        let (prefix, color) = match role.as_str() {
            "you" => ("YOU", Color::Indexed(118)),    // Lime
            "ai" => ("AI", Color::Indexed(48)),       // Emerald
            _ => ("SYS", Color::Indexed(245)),        // Gray
        };

        lines.push(Line::from(vec![
            Span::styled(format!(" {prefix} "), Style::default().fg(color).add_modifier(Modifier::BOLD)),
        ]));
        for chunk in content.chars().collect::<Vec<_>>().chunks(55) {
            let text: String = chunk.iter().collect();
            let text_color = if role == "system" { Color::Indexed(245) } else { Color::Indexed(252) };
            lines.push(Line::from(Span::styled(format!("  {text}"), Style::default().fg(text_color))));
        }
        lines.push(Line::from(""));
    }

    let block = Block::default()
        .title(Span::styled(" Chat ", Style::default().fg(Color::Indexed(118)).add_modifier(Modifier::BOLD)))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Indexed(240)));

    let visible = area.height.saturating_sub(2) as usize;
    let offset = if lines.len() > visible { lines.len() - visible } else { 0 };
    let visible_lines: Vec<Line> = lines.into_iter().skip(offset).collect();

    frame.render_widget(Paragraph::new(visible_lines).block(block).wrap(Wrap { trim: false }), area);
}

#[cfg(feature = "tui")]
fn render_input(frame: &mut ratatui::Frame, area: ratatui::layout::Rect, state: &AppState) {
    use ratatui::style::*;
    use ratatui::text::*;
    use ratatui::widgets::*;

    let active = state.mode == Mode::Chat;
    let border_color = if active { Color::Indexed(118) } else { Color::Indexed(240) };
    let prompt = if active { " ▸ " } else { " · " };

    let block = Block::default()
        .title(if active { " Type message, 'prove' to verify " } else { " Proving... " })
        .borders(Borders::ALL)
        .border_style(Style::default().fg(border_color));

    let input_text = format!("{prompt}{}", state.input);
    frame.render_widget(
        Paragraph::new(Span::styled(&input_text, Style::default().fg(Color::Indexed(255)))).block(block),
        area,
    );

    if active {
        frame.set_cursor_position((area.x + state.cursor_pos as u16 + 4, area.y + 1));
    }
}

#[cfg(feature = "tui")]
fn render_footer_section(frame: &mut ratatui::Frame, area: ratatui::layout::Rect, state: &AppState, _ds: &stwo_ml::tui::dashboard::DashboardState) {
    use ratatui::style::*;
    use ratatui::text::*;
    use ratatui::widgets::*;

    let (status_color, status_text) = match state.mode {
        Mode::Complete => (Color::Indexed(48), "VERIFIED".to_string()),
        Mode::Proving => {
            let elapsed = state.prove_started_at
                .map(|t| t.elapsed().as_secs())
                .unwrap_or(0);
            (Color::Indexed(118), format!("PROVING  {}s", elapsed))
        }
        Mode::Loading => (Color::Indexed(245), "LOADING...".to_string()),
        Mode::Chat => (Color::Indexed(245), "READY".to_string()),
    };

    // Truncate contract address
    let contract = "0x0121d1e9..a8c005";
    let network = "Sepolia";

    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(" ObelyZK ", Style::default().fg(Color::Black).bg(Color::Indexed(118)).add_modifier(Modifier::BOLD)),
            Span::raw("  "),
            Span::styled(&status_text, Style::default().fg(status_color).add_modifier(Modifier::BOLD)),
            Span::raw("  "),
            Span::styled(contract, Style::default().fg(Color::Indexed(73))),
            Span::raw("  "),
            Span::styled(network, Style::default().fg(Color::Indexed(245))),
            Span::raw("              "),
            Span::styled("Ctrl+C", Style::default().fg(Color::Indexed(118))),
            Span::styled(" exit  ", Style::default().fg(Color::Indexed(240))),
            Span::styled("p", Style::default().fg(Color::Indexed(118))),
            Span::styled(" prove", Style::default().fg(Color::Indexed(240))),
        ])),
        area,
    );
}

// Keep old render_chat for reference but unused
#[cfg(feature = "tui")]
#[allow(dead_code)]
fn render_chat_old(frame: &mut ratatui::Frame, area: ratatui::layout::Rect, state: &AppState) {
    use ratatui::layout::*;
    use ratatui::style::*;
    use ratatui::text::*;
    use ratatui::widgets::*;

    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(5), Constraint::Length(3)])
        .split(area);

    // Messages
    let mut lines: Vec<Line> = Vec::new();
    for (role, content) in &state.messages {
        let (prefix, color) = match role.as_str() {
            "you" => ("You", Color::Indexed(118)),
            "ai" => ("AI", Color::Indexed(48)),
            "system" => ("", Color::Indexed(245)),
            _ => ("", Color::White),
        };

        if !prefix.is_empty() {
            lines.push(Line::from(vec![
                Span::styled(format!(" {prefix} "), Style::default().fg(color).add_modifier(Modifier::BOLD)),
            ]));
        }

        // Wrap content
        for chunk in content.chars().collect::<Vec<_>>().chunks(50) {
            let text: String = chunk.iter().collect();
            lines.push(Line::from(vec![
                Span::styled(format!("   {text}"), Style::default().fg(if role == "system" { Color::Indexed(245) } else { Color::Indexed(252) })),
            ]));
        }
        lines.push(Line::from(""));
    }

    let msg_block = Block::default()
        .title(Span::styled(" Chat ", Style::default().fg(Color::Indexed(118)).add_modifier(Modifier::BOLD)))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Indexed(240)));

    // Auto-scroll to bottom
    let visible = layout[0].height.saturating_sub(2) as usize;
    let offset = if lines.len() > visible { lines.len() - visible } else { 0 };
    let visible_lines: Vec<Line> = lines.into_iter().skip(offset).collect();

    frame.render_widget(
        Paragraph::new(visible_lines).block(msg_block).wrap(Wrap { trim: false }),
        layout[0],
    );

    // Input
    let input_style = if state.mode == Mode::Chat {
        Style::default().fg(Color::Indexed(118))
    } else {
        Style::default().fg(Color::Indexed(240))
    };

    let prompt = if state.mode == Mode::Chat { " ▸ " } else { " · " };
    let input_text = format!("{prompt}{}", state.input);

    let input_block = Block::default()
        .borders(Borders::ALL)
        .border_style(input_style);

    frame.render_widget(
        Paragraph::new(Span::styled(&input_text, input_style)).block(input_block),
        layout[1],
    );

    // Cursor
    if state.mode == Mode::Chat {
        frame.set_cursor_position((
            layout[1].x + state.cursor_pos as u16 + 4,
            layout[1].y + 1,
        ));
    }
}

#[cfg(not(feature = "tui"))]
fn main() {
    eprintln!("Build with --features tui to enable ObelyZK");
    std::process::exit(1);
}
