#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# ObelyZK GPU Pipeline — Train, Build, Deploy, Test on A10G
#
# One-command pipeline that handles the full classifier lifecycle:
#   1. SSH into GPU instance
#   2. Train classifier (if --train)
#   3. Build prove-server with trained weights
#   4. Run end-to-end classification test
#   5. Fetch trained weights back to local machine
#
# Usage:
#   # Full pipeline: train + build + test
#   ./gpu-pipeline.sh --host gpu.example.com --train
#
#   # Build + test only (using existing weights)
#   ./gpu-pipeline.sh --host gpu.example.com
#
#   # Just fetch weights from GPU
#   ./gpu-pipeline.sh --host gpu.example.com --fetch-weights
#
#   # Launch new A10G instance on AWS
#   ./gpu-pipeline.sh --launch --region us-west-2
#
# Environment:
#   GPU_HOST          — SSH host (or use --host)
#   GPU_USER          — SSH user (default: ubuntu)
#   GPU_KEY           — SSH key path (default: ~/.ssh/obelyzk-gpu.pem)
#   AWS_PROFILE       — AWS CLI profile for --launch
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Defaults ────────────────────────────────────────────────────────
GPU_HOST="${GPU_HOST:-}"
GPU_USER="${GPU_USER:-ubuntu}"
GPU_KEY="${GPU_KEY:-$HOME/.ssh/obelyzk-gpu.pem}"
REPO_PATH="/home/${GPU_USER}/bitsage-network/libs/stwo-ml"
TRAINING_DIR="${REPO_PATH}/training"
DO_TRAIN=false
DO_FETCH=false
DO_LAUNCH=false
REGION="us-west-2"
INSTANCE_TYPE="g5.xlarge"  # 1x A10G, 4 vCPUs, 16GB RAM

# ── Parse args ──────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --host)       GPU_HOST="$2"; shift 2 ;;
    --user)       GPU_USER="$2"; shift 2 ;;
    --key)        GPU_KEY="$2"; shift 2 ;;
    --train)      DO_TRAIN=true; shift ;;
    --fetch-weights) DO_FETCH=true; shift ;;
    --launch)     DO_LAUNCH=true; shift ;;
    --region)     REGION="$2"; shift 2 ;;
    --instance-type) INSTANCE_TYPE="$2"; shift 2 ;;
    *) echo "Unknown: $1"; exit 1 ;;
  esac
done

SSH_CMD="ssh -i ${GPU_KEY} -o StrictHostKeyChecking=no ${GPU_USER}@${GPU_HOST}"
SCP_CMD="scp -i ${GPU_KEY} -o StrictHostKeyChecking=no"

# ═══════════════════════════════════════════════════════════════════════
# Launch A10G instance on AWS
# ═══════════════════════════════════════════════════════════════════════

if $DO_LAUNCH; then
  echo "Launching ${INSTANCE_TYPE} in ${REGION}..."

  # Find latest Deep Learning AMI
  AMI_ID=$(aws ec2 describe-images \
    --region "$REGION" \
    --owners amazon \
    --filters "Name=name,Values=Deep Learning AMI GPU PyTorch*Ubuntu 22.04*" \
    --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
    --output text 2>/dev/null || echo "")

  if [ -z "$AMI_ID" ] || [ "$AMI_ID" = "None" ]; then
    # Fallback: Ubuntu 22.04 with NVIDIA drivers
    AMI_ID=$(aws ec2 describe-images \
      --region "$REGION" \
      --owners 099720109477 \
      --filters "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" \
      --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
      --output text)
  fi

  echo "AMI: ${AMI_ID}"

  # Launch
  INSTANCE_ID=$(aws ec2 run-instances \
    --region "$REGION" \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name obelyzk-gpu \
    --security-groups obelyzk-prover \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":200,"VolumeType":"gp3"}}]' \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=obelyzk-prover},{Key=Project,Value=obelyzk}]" \
    --query 'Instances[0].InstanceId' \
    --output text)

  echo "Instance: ${INSTANCE_ID}"
  echo "Waiting for running state..."

  aws ec2 wait instance-running --region "$REGION" --instance-ids "$INSTANCE_ID"

  GPU_HOST=$(aws ec2 describe-instances \
    --region "$REGION" \
    --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

  echo "Host: ${GPU_HOST}"
  echo "Waiting for SSH..."
  for i in $(seq 1 30); do
    if ssh -i "$GPU_KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=5 "${GPU_USER}@${GPU_HOST}" true 2>/dev/null; then
      echo "SSH ready."
      break
    fi
    sleep 10
  done

  echo ""
  echo "A10G instance launched:"
  echo "  Instance: ${INSTANCE_ID}"
  echo "  Host:     ${GPU_HOST}"
  echo "  Region:   ${REGION}"
  echo ""
  echo "Next: ./gpu-pipeline.sh --host ${GPU_HOST} --train"
  exit 0
fi

if [ -z "$GPU_HOST" ]; then
  echo "ERROR: --host required (or set GPU_HOST env var)"
  echo "       Use --launch to create a new A10G instance"
  exit 1
fi

echo "═══════════════════════════════════════════════════════"
echo "  ObelyZK GPU Pipeline"
echo "  Host: ${GPU_HOST}"
echo "═══════════════════════════════════════════════════════"

# ═══════════════════════════════════════════════════════════════════════
# Sync code to GPU
# ═══════════════════════════════════════════════════════════════════════

echo ""
echo "Syncing code to GPU..."
rsync -azP --delete \
  -e "ssh -i ${GPU_KEY} -o StrictHostKeyChecking=no" \
  --exclude 'target/' \
  --exclude 'node_modules/' \
  --exclude '.git/' \
  --exclude '*.log' \
  "$(dirname "$(dirname "$0")")/" \
  "${GPU_USER}@${GPU_HOST}:${REPO_PATH}/"
echo "Sync complete."

# ═══════════════════════════════════════════════════════════════════════
# Install dependencies (idempotent)
# ═══════════════════════════════════════════════════════════════════════

echo ""
echo "Installing dependencies..."
$SSH_CMD << 'DEPS'
set -euo pipefail

# Rust (if not installed)
if ! command -v rustup &>/dev/null; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
  source ~/.cargo/env
fi
source ~/.cargo/env
rustup default nightly-2025-07-14 2>/dev/null || rustup install nightly-2025-07-14

# Python deps for training
pip3 install torch numpy scikit-learn safetensors huggingface-hub 2>/dev/null || \
  pip3 install --user torch numpy scikit-learn safetensors huggingface-hub 2>/dev/null || true

# Check CUDA
if command -v nvidia-smi &>/dev/null; then
  echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
  echo "CUDA: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
else
  echo "WARNING: No CUDA detected. Training will use CPU."
fi

echo "Dependencies OK."
DEPS

# ═══════════════════════════════════════════════════════════════════════
# Train classifier (optional)
# ═══════════════════════════════════════════════════════════════════════

if $DO_TRAIN; then
  echo ""
  echo "═══════════════════════════════════════════════════════"
  echo "  Training classifier..."
  echo "═══════════════════════════════════════════════════════"

  $SSH_CMD << TRAIN
set -euo pipefail
cd ${TRAINING_DIR}

# Generate production dataset
python3 data_sources.py --output dataset_production.npz 2>&1

# Train with production settings
python3 train.py \
  --dataset dataset_production.npz \
  --epochs 200 \
  --lr 0.001 \
  --batch-size 256 \
  --output-dir output 2>&1

# Run adversarial evaluation
python3 adversarial.py --model output/model.pt --scale 0.2 2>&1

echo ""
echo "Training complete. Weights at: output/trained_weights.rs"
TRAIN

  # Copy trained weights back to local source tree
  echo ""
  echo "Fetching trained weights..."
  $SCP_CMD "${GPU_USER}@${GPU_HOST}:${TRAINING_DIR}/output/trained_weights.rs" \
    "$(dirname "$0")/../src/classifier/trained_weights.rs"
  $SCP_CMD "${GPU_USER}@${GPU_HOST}:${TRAINING_DIR}/output/weights_m31.json" \
    "$(dirname "$0")/../training/output/weights_m31.json"
  $SCP_CMD "${GPU_USER}@${GPU_HOST}:${TRAINING_DIR}/output/training_metadata.json" \
    "$(dirname "$0")/../training/output/training_metadata.json"
  echo "Weights synced to local source tree."
fi

# ═══════════════════════════════════════════════════════════════════════
# Build prove-server with CUDA
# ═══════════════════════════════════════════════════════════════════════

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Building prove-server (release + CUDA)..."
echo "═══════════════════════════════════════════════════════"

$SSH_CMD << BUILD
set -euo pipefail
source ~/.cargo/env
cd ${REPO_PATH}

# Detect CUDA features
FEATURES="server,model-loading"
if command -v nvidia-smi &>/dev/null; then
  FEATURES="\${FEATURES},cuda-runtime"
fi

echo "Features: \${FEATURES}"
cargo build --release --bin prove-server --features "\${FEATURES}" 2>&1 | tail -5

echo "Build complete: target/release/prove-server"
BUILD

# ═══════════════════════════════════════════════════════════════════════
# Run end-to-end test
# ═══════════════════════════════════════════════════════════════════════

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  End-to-end classification test..."
echo "═══════════════════════════════════════════════════════"

$SSH_CMD << 'E2E'
set -euo pipefail
cd /home/ubuntu/bitsage-network/libs/stwo-ml

# Start prove-server in background
BIND_ADDR=0.0.0.0:8080 ./target/release/prove-server &
SERVER_PID=$!
sleep 3

# Test health
echo "Health check..."
curl -s http://localhost:8080/health | python3 -m json.tool 2>/dev/null || echo "(health endpoint not available)"

# Test classify — safe transaction (ETH transfer to verified contract)
echo ""
echo "Test 1: Safe ETH transfer..."
curl -s -X POST http://localhost:8080/api/v1/classify \
  -H "Content-Type: application/json" \
  -d '{
    "target": "0x049d36570d4e46f48e99674bd3fcc84644ddd6b96f7c741b1562b82f9e004dc7",
    "value": "1000000000000000000",
    "selector": "0xa9059cbb",
    "target_verified": true,
    "target_interaction_count": 200,
    "agent_trust_score": 3000
  }' | python3 -m json.tool 2>/dev/null || echo "(classify returned non-JSON)"

# Test classify — suspicious transaction
echo ""
echo "Test 2: Suspicious max approval to unknown contract..."
curl -s -X POST http://localhost:8080/api/v1/classify \
  -H "Content-Type: application/json" \
  -d '{
    "target": "0xdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
    "value": "340282366920938463463374607431768211455",
    "selector": "0x095ea7b3",
    "target_verified": false,
    "agent_trust_score": 60000,
    "agent_strikes": 3
  }' | python3 -m json.tool 2>/dev/null || echo "(classify returned non-JSON)"

# Test classify — malicious flash loan
echo ""
echo "Test 3: Flash loan to unverified contract..."
curl -s -X POST http://localhost:8080/api/v1/classify \
  -H "Content-Type: application/json" \
  -d '{
    "target": "0xbad0bad0bad0bad0bad0bad0bad0bad0bad0bad0",
    "value": "100000000000000000000000",
    "selector": "0x5cffe9de",
    "target_verified": false,
    "target_has_source": false,
    "agent_trust_score": 80000,
    "agent_strikes": 4,
    "tx_frequency": 100,
    "unique_targets_24h": 50
  }' | python3 -m json.tool 2>/dev/null || echo "(classify returned non-JSON)"

# Stop server
kill $SERVER_PID 2>/dev/null || true
echo ""
echo "End-to-end test complete."
E2E

# ═══════════════════════════════════════════════════════════════════════
# Fetch weights (optional standalone)
# ═══════════════════════════════════════════════════════════════════════

if $DO_FETCH; then
  echo ""
  echo "Fetching weights from GPU..."
  $SCP_CMD "${GPU_USER}@${GPU_HOST}:${TRAINING_DIR}/output/trained_weights.rs" \
    "$(dirname "$0")/../src/classifier/trained_weights.rs"
  $SCP_CMD "${GPU_USER}@${GPU_HOST}:${TRAINING_DIR}/output/weights_m31.json" \
    "$(dirname "$0")/../training/output/weights_m31.json"
  $SCP_CMD "${GPU_USER}@${GPU_HOST}:${TRAINING_DIR}/output/model.pt" \
    "$(dirname "$0")/../training/output/model.pt"
  echo "Weights fetched."
fi

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Pipeline complete."
echo "═══════════════════════════════════════════════════════"
echo ""
echo "  Host:     ${GPU_HOST}"
echo "  Weights:  src/classifier/trained_weights.rs"
echo "  Server:   ssh ${GPU_USER}@${GPU_HOST} '${REPO_PATH}/target/release/prove-server'"
echo ""
echo "  Next steps:"
echo "    1. git add src/classifier/trained_weights.rs && git commit"
echo "    2. Start server on GPU: ssh ... 'BIND_ADDR=0.0.0.0:8080 prove-server'"
echo "    3. Configure MCP: PROVER_URL=http://${GPU_HOST}:8080"
echo "    4. Upload weights: python training/upload_hf.py --repo obelyzk/transaction-classifier-v1"
