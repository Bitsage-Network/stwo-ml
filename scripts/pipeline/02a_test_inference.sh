#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# Obelysk Pipeline — Step 2a: Model Inference & Chat Testing
# ═══════════════════════════════════════════════════════════════════════
#
# Installs llama.cpp (if needed), converts model to GGUF, and runs
# inference testing. Supports single-prompt test, interactive chat,
# and benchmark modes.
#
# Usage:
#   bash scripts/pipeline/02a_test_inference.sh --model-name qwen3-14b
#   bash scripts/pipeline/02a_test_inference.sh --model-dir ~/models/phi3 --chat
#   bash scripts/pipeline/02a_test_inference.sh --model-name phi3-mini --benchmark
#   bash scripts/pipeline/02a_test_inference.sh --model-name phi3-mini --prompt "Hello!"
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"

# ─── Defaults ────────────────────────────────────────────────────────

MODEL_NAME=""
MODEL_DIR=""
PROMPT="Hello, what is 2+2?"
DO_CHAT=false
DO_BENCHMARK=false
GPU_LAYERS=99
SKIP_INSTALL=false
SKIP_CONVERT=false
GGUF_PATH=""

# ─── Parse Arguments ─────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case $1 in
        --model-name)    MODEL_NAME="$2"; shift 2 ;;
        --model-dir)     MODEL_DIR="$2"; shift 2 ;;
        --prompt)        PROMPT="$2"; shift 2 ;;
        --chat)          DO_CHAT=true; shift ;;
        --benchmark)     DO_BENCHMARK=true; shift ;;
        --layers)        GPU_LAYERS="$2"; shift 2 ;;
        --skip-install)  SKIP_INSTALL=true; shift ;;
        --skip-convert)  SKIP_CONVERT=true; shift ;;
        --gguf)          GGUF_PATH="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Test model inference using llama.cpp."
            echo ""
            echo "Model source (pick one):"
            echo "  --model-name NAME   Load from ~/.obelysk/models/NAME/"
            echo "  --model-dir DIR     Path to model directory"
            echo ""
            echo "Modes:"
            echo "  (default)           Run a single inference test"
            echo "  --chat              Interactive chat mode (Ctrl+C to exit)"
            echo "  --benchmark         Run inference benchmark (tokens/sec)"
            echo ""
            echo "Options:"
            echo "  --prompt TEXT       Test prompt (default: 'Hello, what is 2+2?')"
            echo "  --layers N          GPU layers to offload (default: 99 = all)"
            echo "  --gguf PATH         Use pre-existing GGUF file (skip conversion)"
            echo "  --skip-install      Assume llama.cpp is already built"
            echo "  --skip-convert      Assume GGUF already exists in model dir"
            echo "  -h, --help          Show this help"
            exit 0
            ;;
        *) err "Unknown argument: $1"; exit 1 ;;
    esac
done

# ─── Resolve Model ──────────────────────────────────────────────────

init_obelysk_dir

if [[ -z "$MODEL_NAME" ]] && [[ -z "$MODEL_DIR" ]]; then
    MODEL_NAME=$(get_state "model_state.env" "CURRENT_MODEL" 2>/dev/null || echo "")
    if [[ -z "$MODEL_NAME" ]]; then
        err "No model specified. Use --model-name or --model-dir."
        exit 1
    fi
fi

if [[ -n "$MODEL_NAME" ]] && [[ -z "$MODEL_DIR" ]]; then
    MODEL_CONFIG="${OBELYSK_DIR}/models/${MODEL_NAME}/config.env"
    if [[ -f "$MODEL_CONFIG" ]]; then
        set -a; source "$MODEL_CONFIG"; set +a
    else
        # Default dir
        MODEL_DIR="${OBELYSK_DIR}/models/${MODEL_NAME}"
    fi
fi

check_dir "$MODEL_DIR" "Model directory not found: ${MODEL_DIR}" || exit 1

# ─── Display ─────────────────────────────────────────────────────────

banner
echo -e "${BOLD}  Inference Testing: ${MODEL_NAME:-$(basename "$MODEL_DIR")}${NC}"
echo ""
log "Model dir:  ${MODEL_DIR}"
log "Mode:       $(if [[ "$DO_CHAT" == "true" ]]; then echo "chat"; elif [[ "$DO_BENCHMARK" == "true" ]]; then echo "benchmark"; else echo "single prompt"; fi)"
log "GPU layers: ${GPU_LAYERS}"
echo ""

timer_start "inference"

# ═══════════════════════════════════════════════════════════════════════
# Step 1: Ensure llama.cpp is available
# ═══════════════════════════════════════════════════════════════════════

header "Step 1: llama.cpp"

LLAMA_DIR="${OBELYSK_DIR}/llama.cpp"
LLAMA_CLI=""

find_llama_bin() {
    local ll_dir="$1"
    local candidate
    for candidate in \
        "${ll_dir}/build/bin/llama-cli" \
        "${ll_dir}/build/bin/llama-run" \
        "${ll_dir}/build/bin/main" \
        "$(command -v llama-cli 2>/dev/null || echo "")" \
        "$(command -v llama-run 2>/dev/null || echo "")"; do
        if [[ -n "$candidate" ]] && [[ -f "$candidate" ]]; then
            echo "$candidate"
            return 0
        fi
    done
    for candidate in \
        "$(find "${ll_dir}/build" -name "llama-cli" -type f 2>/dev/null | head -1)" \
        "$(find "${ll_dir}/build" -name "llama-run" -type f 2>/dev/null | head -1)" \
        "$(find "${ll_dir}/build" -name "main" -type f 2>/dev/null | head -1)"; do
        if [[ -n "$candidate" ]] && [[ -f "$candidate" ]]; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

# Check state from setup
_STATE_BIN=$(get_state "setup_state.env" "LLAMA_BIN" 2>/dev/null || echo "")
if [[ -n "$_STATE_BIN" ]] && [[ -f "$_STATE_BIN" ]]; then
    LLAMA_CLI="$_STATE_BIN"
fi

# Check default locations / command PATH
if [[ -z "$LLAMA_CLI" ]]; then
    LLAMA_CLI="$(find_llama_bin "$LLAMA_DIR" || true)"
fi

if [[ -n "$LLAMA_CLI" ]]; then
    ok "llama.cpp found: ${LLAMA_CLI}"
elif [[ "$SKIP_INSTALL" == "true" ]]; then
    err "llama.cpp not found and --skip-install specified."
    err "Run 00_setup_gpu.sh first or remove --skip-install."
    exit 1
else
    log "llama.cpp not found — building..."

    # Ensure cmake
    if ! command -v cmake &>/dev/null; then
        if command -v apt-get &>/dev/null; then
            run_cmd sudo apt-get install -y -qq cmake g++ 2>&1 | tail -2
        elif command -v yum &>/dev/null; then
            run_cmd sudo yum install -y cmake3 gcc-c++ 2>/dev/null || \
                run_cmd sudo yum install -y cmake gcc-c++
        fi
    fi

    if [[ ! -d "$LLAMA_DIR" ]]; then
        run_cmd git clone --depth 1 https://github.com/ggerganov/llama.cpp "$LLAMA_DIR"
    fi

    # Detect CUDA for build
    CMAKE_ARGS=(-B "${LLAMA_DIR}/build")
    _GPU_CONFIG="${OBELYSK_DIR}/gpu_config.env"
    _CUDA_PATH=$(grep "^CUDA_PATH=" "$_GPU_CONFIG" 2>/dev/null | cut -d'=' -f2- || echo "")
    if [[ -n "$_CUDA_PATH" ]] && [[ -f "${_CUDA_PATH}/bin/nvcc" ]]; then
        CMAKE_ARGS+=(-DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER="${_CUDA_PATH}/bin/nvcc")
        log "Building with CUDA: ${_CUDA_PATH}"
    else
        CMAKE_ARGS+=(-DGGML_CUDA=OFF)
        log "Building CPU-only (no CUDA detected)"
        GPU_LAYERS=0
    fi

    (cd "$LLAMA_DIR" && cmake "${CMAKE_ARGS[@]}" 2>&1 | tail -3)
    (cd "$LLAMA_DIR" && cmake --build build --config Release -j"$(nproc 2>/dev/null || echo 4)" 2>&1 | tail -5)

    LLAMA_CLI="$(find_llama_bin "$LLAMA_DIR" || true)"
    if [[ -z "$LLAMA_CLI" ]]; then
        err "llama.cpp build failed — no runnable CLI binary found (llama-cli/llama-run/main)"
        exit 1
    fi
    ok "llama.cpp built: ${LLAMA_CLI}"
fi
echo ""

# ═══════════════════════════════════════════════════════════════════════
# Step 2: Convert model to GGUF (if needed)
# ═══════════════════════════════════════════════════════════════════════

header "Step 2: GGUF Conversion"

if [[ -n "$GGUF_PATH" ]] && [[ -f "$GGUF_PATH" ]]; then
    ok "Using provided GGUF: ${GGUF_PATH}"
elif [[ "$SKIP_CONVERT" == "true" ]]; then
    # Find existing GGUF
    GGUF_PATH=$(find "$MODEL_DIR" -name "*.gguf" -type f 2>/dev/null | head -1)
    if [[ -z "$GGUF_PATH" ]]; then
        err "No GGUF file found in ${MODEL_DIR} (--skip-convert specified)"
        exit 1
    fi
    ok "Found existing GGUF: ${GGUF_PATH}"
else
    # Check for existing GGUF
    GGUF_PATH=$(find "$MODEL_DIR" -name "*.gguf" -type f 2>/dev/null | head -1)

    if [[ -n "$GGUF_PATH" ]]; then
        ok "GGUF already exists: ${GGUF_PATH}"
    else
        log "Converting model to GGUF format (f16)..."

        CONVERT_SCRIPT="${LLAMA_DIR}/convert_hf_to_gguf.py"
        if [[ ! -f "$CONVERT_SCRIPT" ]]; then
            err "convert_hf_to_gguf.py not found at ${CONVERT_SCRIPT}"
            err "Ensure llama.cpp is cloned correctly."
            exit 1
        fi

        # Install conversion dependencies
        pip3 install --quiet sentencepiece transformers 2>/dev/null || \
            pip3 install --quiet --user sentencepiece transformers 2>/dev/null || true

        GGUF_PATH="${MODEL_DIR}/model-f16.gguf"

        if run_cmd python3 "$CONVERT_SCRIPT" "$MODEL_DIR" \
            --outtype f16 \
            --outfile "$GGUF_PATH" 2>&1 | tail -5; then
            if [[ -f "$GGUF_PATH" ]] && [[ -s "$GGUF_PATH" ]]; then
                GGUF_SIZE=$(du -h "$GGUF_PATH" | cut -f1)
                ok "GGUF created: ${GGUF_PATH} (${GGUF_SIZE})"
            else
                err "GGUF conversion produced empty file"
                exit 1
            fi
        else
            err "GGUF conversion failed"
            err "Check that the model format is supported by llama.cpp"
            exit 1
        fi
    fi
fi
echo ""

# ═══════════════════════════════════════════════════════════════════════
# Step 3: Run Inference
# ═══════════════════════════════════════════════════════════════════════

if [[ "$DO_CHAT" == "true" ]]; then
    # ─── Interactive Chat Mode ───────────────────────────────────────
    header "Interactive Chat"
    log "Starting chat with $(basename "$GGUF_PATH")..."
    log "GPU layers: ${GPU_LAYERS}"
    log "Press Ctrl+C to exit."
    echo ""

    "$LLAMA_CLI" \
        -m "$GGUF_PATH" \
        -ngl "$GPU_LAYERS" \
        --interactive \
        --color \
        -r "User:" \
        --in-prefix " " \
        -p "You are a helpful assistant.\n\nUser:"

elif [[ "$DO_BENCHMARK" == "true" ]]; then
    # ─── Benchmark Mode ──────────────────────────────────────────────
    header "Inference Benchmark"
    log "Running benchmark (512 tokens)..."
    echo ""

    BENCH_OUTPUT=$("$LLAMA_CLI" \
        -m "$GGUF_PATH" \
        -ngl "$GPU_LAYERS" \
        -p "Write a detailed essay about the history of mathematics." \
        -n 512 \
        --no-display-prompt \
        2>&1) || true

    # Parse timing info from llama.cpp output
    PROMPT_EVAL=$(echo "$BENCH_OUTPUT" | grep "prompt eval time" || echo "")
    GEN_EVAL=$(echo "$BENCH_OUTPUT" | grep "eval time" | grep -v "prompt" || echo "")
    TOKENS_SEC=$(echo "$GEN_EVAL" | grep -oP '[\d.]+\s+tokens per second' | head -1 || echo "")

    echo ""
    log "Benchmark Results:"
    if [[ -n "$PROMPT_EVAL" ]]; then
        log "  ${PROMPT_EVAL}"
    fi
    if [[ -n "$GEN_EVAL" ]]; then
        log "  ${GEN_EVAL}"
    fi
    if [[ -n "$TOKENS_SEC" ]]; then
        ok "Speed: ${TOKENS_SEC}"
    fi

else
    # ─── Single Prompt Test ──────────────────────────────────────────
    header "Inference Test"
    log "Prompt: \"${PROMPT}\""
    log "GPU layers: ${GPU_LAYERS}"
    echo ""

    TEST_OUTPUT=$("$LLAMA_CLI" \
        -m "$GGUF_PATH" \
        -ngl "$GPU_LAYERS" \
        -p "$PROMPT" \
        -n 128 \
        --no-display-prompt \
        2>&1) || true

    # Separate model output from timing
    MODEL_RESPONSE=$(echo "$TEST_OUTPUT" | grep -v "^llama_" | grep -v "^$" | grep -v "eval time" | head -20)
    TIMING=$(echo "$TEST_OUTPUT" | grep "eval time" || echo "")

    if [[ -n "$MODEL_RESPONSE" ]]; then
        echo ""
        echo -e "${BOLD}  Model Response:${NC}"
        echo "  ─────────────────────────────"
        echo "$MODEL_RESPONSE" | sed 's/^/  /'
        echo "  ─────────────────────────────"
        echo ""
        ok "Inference test passed — model generates text"
    else
        warn "Model produced empty output. Check model format and GPU compatibility."
    fi

    if [[ -n "$TIMING" ]]; then
        TOKENS_SEC=$(echo "$TIMING" | grep -oP '[\d.]+\s+tokens per second' | head -1 || echo "")
        if [[ -n "$TOKENS_SEC" ]]; then
            log "Speed: ${TOKENS_SEC}"
        fi
    fi
fi
echo ""

# ─── Save State ──────────────────────────────────────────────────────

ELAPSED=$(timer_elapsed "inference")

save_state "inference_state.env" \
    "INFERENCE_TESTED=true" \
    "INFERENCE_DATE=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "GGUF_PATH=${GGUF_PATH}" \
    "LLAMA_CLI=${LLAMA_CLI}" \
    "TOKENS_PER_SEC=${TOKENS_SEC:-unknown}" \
    "MODEL_NAME=${MODEL_NAME:-$(basename "$MODEL_DIR")}"

# ─── Summary ─────────────────────────────────────────────────────────

echo -e "${GREEN}${BOLD}"
echo "  ╔══════════════════════════════════════════════════════╗"
echo "  ║  INFERENCE TEST COMPLETE                             ║"
echo "  ╠══════════════════════════════════════════════════════╣"
printf "  ║  Model:       %-37s ║\n" "${MODEL_NAME:-$(basename "$MODEL_DIR")}"
printf "  ║  GGUF:        %-37s ║\n" "$(basename "${GGUF_PATH}")"
printf "  ║  Speed:       %-37s ║\n" "${TOKENS_SEC:-N/A}"
printf "  ║  Duration:    %-37s ║\n" "$(format_duration $ELAPSED)"
echo "  ╠══════════════════════════════════════════════════════╣"
echo "  ║                                                      ║"
echo "  ║  Next: ./03_prove.sh                                 ║"
echo "  ║                                                      ║"
echo "  ╚══════════════════════════════════════════════════════╝"
echo -e "${NC}"
