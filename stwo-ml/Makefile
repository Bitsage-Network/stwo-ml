# ═══════════════════════════════════════════════════════════════════
# ObelyZK — Developer Commands
#
# Quick reference:
#   make setup          Install all dependencies
#   make train          Train classifier on synthetic data
#   make train-ciro     Train classifier on CIRO real data
#   make build          Build prove-server (CPU)
#   make build-gpu      Build prove-server (CUDA)
#   make serve          Start prove-server locally
#   make test           Run all tests
#   make classify       Test classify endpoint
#   make e2e            Full end-to-end: train → build → serve → classify
#   make deploy         Deploy to A10G via SSH
#   make upload-hf      Upload trained model to HuggingFace
# ═══════════════════════════════════════════════════════════════════

.PHONY: setup train train-ciro build build-gpu serve test classify e2e deploy upload-hf clean

# ── Config ──────────────────────────────────────────────────────────
PROVER_PORT ?= 8080
GPU_HOST ?= $(shell echo $$GPU_HOST)
HF_REPO ?= obelyzk/transaction-classifier-v1
CIRO_URL ?= $(shell echo $$CIRO_URL)
CIRO_API_KEY ?= $(shell echo $$CIRO_API_KEY)

# ── Setup ───────────────────────────────────────────────────────────
setup:
	@echo "Installing Rust toolchain..."
	rustup install nightly-2025-07-14 2>/dev/null || true
	rustup default nightly-2025-07-14
	@echo "Installing Python dependencies..."
	pip3 install --user torch numpy scikit-learn safetensors huggingface-hub 2>/dev/null || \
		pip3 install --user --break-system-packages torch numpy scikit-learn safetensors huggingface-hub
	@echo "Installing Node dependencies..."
	cd ../sdk/typescript && npm install 2>/dev/null || true
	@echo "Setup complete."

# ── Training ────────────────────────────────────────────────────────
train:
	@echo "Training classifier on synthetic data..."
	cd training && python3 data_sources.py --output dataset_production.npz
	cd training && python3 train.py --dataset dataset_production.npz --epochs 200 --output-dir output
	@echo "Running adversarial evaluation..."
	cd training && python3 adversarial.py --model output/model.pt --scale 0.2
	@echo ""
	@echo "Copying trained weights to prover..."
	cp training/output/trained_weights.rs src/classifier/trained_weights.rs
	@echo "Done. Run 'make build' to compile with new weights."

train-ciro:
	@test -n "$(CIRO_URL)" || (echo "ERROR: Set CIRO_URL env var"; exit 1)
	@test -n "$(CIRO_API_KEY)" || (echo "ERROR: Set CIRO_API_KEY env var"; exit 1)
	@echo "Fetching labeled data from CIRO..."
	cd training && python3 -m ciro.client --url $(CIRO_URL) --api-key $(CIRO_API_KEY) --output dataset_ciro.npz
	@echo "Training on CIRO data..."
	cd training && python3 train.py --dataset dataset_ciro.npz --epochs 200 --output-dir output
	cd training && python3 adversarial.py --model output/model.pt --scale 0.2
	cp training/output/trained_weights.rs src/classifier/trained_weights.rs
	@echo "Done. Run 'make build' to compile with CIRO-trained weights."

# ── Build ───────────────────────────────────────────────────────────
build:
	@echo "Building prove-server (CPU)..."
	cargo build --release --bin prove-server --features server,model-loading
	@echo "Binary: target/release/prove-server"

build-gpu:
	@echo "Building prove-server (CUDA)..."
	cargo build --release --bin prove-server --features server,model-loading,cuda-runtime
	@echo "Binary: target/release/prove-server"

# ── Serve ───────────────────────────────────────────────────────────
serve:
	@echo "Starting prove-server on port $(PROVER_PORT)..."
	BIND_ADDR=0.0.0.0:$(PROVER_PORT) ./target/release/prove-server

serve-dev:
	@echo "Starting prove-server (debug) on port $(PROVER_PORT)..."
	BIND_ADDR=0.0.0.0:$(PROVER_PORT) cargo run --bin prove-server --features server,model-loading

# ── Test ────────────────────────────────────────────────────────────
test:
	@echo "Running Rust tests..."
	cargo test --lib --features std classifier -- --nocapture
	@echo ""
	@echo "Running TypeScript tests..."
	cd ../sdk/typescript && npx vitest run src/__tests__/firewall.test.ts src/__tests__/mcp-policy.test.ts
	@echo ""
	@echo "Running Cairo tests..."
	cd ../elo-cairo-verifier && snforge test test_recursive_verifier 2>/dev/null || echo "(Cairo tests skipped — snforge not installed)"

test-rust:
	cargo test --lib --features std -- --nocapture

test-ts:
	cd ../sdk/typescript && npx vitest run

test-cairo:
	cd ../elo-cairo-verifier && snforge test test_recursive_verifier

# ── Classify (test endpoint) ────────────────────────────────────────
classify:
	@echo "Testing classify endpoint (safe transfer)..."
	@curl -s -X POST http://localhost:$(PROVER_PORT)/api/v1/classify \
		-H "Content-Type: application/json" \
		-d '{"target":"0x049d36570d4e46f48e99674bd3fcc84644ddd6b96f7c741b1562b82f9e004dc7","value":"1000000000000000000","selector":"0xa9059cbb","target_verified":true,"target_interaction_count":200}' \
		| python3 -m json.tool 2>/dev/null || echo "(Server not running? Start with: make serve)"

classify-malicious:
	@echo "Testing classify endpoint (malicious flash loan)..."
	@curl -s -X POST http://localhost:$(PROVER_PORT)/api/v1/classify \
		-H "Content-Type: application/json" \
		-d '{"target":"0xbad0bad0bad0bad0bad0bad0bad0bad0bad0bad0","value":"100000000000000000000000","selector":"0x5cffe9de","target_verified":false,"agent_trust_score":80000,"agent_strikes":4,"tx_frequency":100}' \
		| python3 -m json.tool 2>/dev/null || echo "(Server not running?)"

# ── End-to-end ──────────────────────────────────────────────────────
e2e: train build
	@echo ""
	@echo "Starting prove-server for e2e test..."
	@BIND_ADDR=0.0.0.0:$(PROVER_PORT) ./target/release/prove-server &
	@sleep 3
	@$(MAKE) classify
	@$(MAKE) classify-malicious
	@echo ""
	@echo "Stopping server..."
	@pkill -f prove-server 2>/dev/null || true
	@echo "End-to-end test complete."

# ── Deploy ──────────────────────────────────────────────────────────
deploy:
	@test -n "$(GPU_HOST)" || (echo "ERROR: Set GPU_HOST env var"; exit 1)
	scripts/gpu-pipeline.sh --host $(GPU_HOST) --train

deploy-build:
	@test -n "$(GPU_HOST)" || (echo "ERROR: Set GPU_HOST env var"; exit 1)
	scripts/gpu-pipeline.sh --host $(GPU_HOST)

launch-gpu:
	scripts/gpu-pipeline.sh --launch --region us-west-2

# ── HuggingFace ─────────────────────────────────────────────────────
upload-hf:
	@echo "Uploading to HuggingFace: $(HF_REPO)..."
	cd training && python3 upload_hf.py --repo $(HF_REPO) --output-dir output

# ── SDK Build ───────────────────────────────────────────────────────
sdk:
	@echo "Building TypeScript SDK..."
	cd ../sdk/typescript && npm run build

# ── Clean ───────────────────────────────────────────────────────────
clean:
	cargo clean
	rm -rf training/output/
	rm -f training/dataset_*.npz

# ── Help ────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "ObelyZK Developer Commands"
	@echo "══════════════════════════════════════════════════"
	@echo ""
	@echo "  Setup:"
	@echo "    make setup          Install Rust, Python, Node deps"
	@echo ""
	@echo "  Training:"
	@echo "    make train          Train on synthetic data (local)"
	@echo "    make train-ciro     Train on CIRO real data"
	@echo ""
	@echo "  Build:"
	@echo "    make build          Build prove-server (CPU)"
	@echo "    make build-gpu      Build prove-server (CUDA)"
	@echo "    make sdk            Build TypeScript SDK"
	@echo ""
	@echo "  Run:"
	@echo "    make serve          Start prove-server"
	@echo "    make classify       Test safe classification"
	@echo "    make classify-malicious  Test malicious classification"
	@echo "    make e2e            Full pipeline: train → build → test"
	@echo ""
	@echo "  Test:"
	@echo "    make test           Run all tests (Rust + TS + Cairo)"
	@echo "    make test-rust      Rust tests only"
	@echo "    make test-ts        TypeScript tests only"
	@echo "    make test-cairo     Cairo snforge tests only"
	@echo ""
	@echo "  Deploy:"
	@echo "    make launch-gpu     Launch A10G on AWS"
	@echo "    make deploy         Train + build + test on GPU"
	@echo "    make upload-hf      Upload to HuggingFace"
	@echo ""
	@echo "  Cleanup:"
	@echo "    make clean          Remove build artifacts"
	@echo ""
