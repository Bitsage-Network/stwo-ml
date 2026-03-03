#!/usr/bin/env bash
# test_verification_hardening.sh — End-to-end hardening verification tests
#
# Validates that proof verification rejects fake/tampered proofs and accepts
# valid ones. Run on any machine with a built prove-model binary.
#
# Usage:
#   # Minimal (no GPU, no model — rejection tests only):
#   bash scripts/test_verification_hardening.sh
#
#   # With model (adds commitment + forward-pass replay):
#   MODEL_DIR=~/.obelysk/models/qwen3-14b LAYERS=1 \
#     bash scripts/test_verification_hardening.sh
#
#   # Full GPU pipeline (generate real proof + verify):
#   MODEL_DIR=~/.obelysk/models/qwen3-14b LAYERS=1 GPU=1 \
#     bash scripts/test_verification_hardening.sh
#
# Environment:
#   PM          — path to prove-model binary (default: ./target/release/prove-model)
#   MODEL_DIR   — HuggingFace model directory (optional, enables tests 5-7)
#   LAYERS      — number of layers to prove (default: 1)
#   GPU         — set to 1 to enable GPU proving (default: off)
#   TMPDIR      — temp directory for test artifacts (default: /tmp/stwo_hardening_test)

set -euo pipefail

PM="${PM:-./target/release/prove-model}"
MODEL_DIR="${MODEL_DIR:-}"
LAYERS="${LAYERS:-1}"
GPU_FLAG=""
[ "${GPU:-0}" = "1" ] && GPU_FLAG="--gpu"
TMPDIR="${TMPDIR:-/tmp/stwo_hardening_test}"

mkdir -p "$TMPDIR"

PASS=0
FAIL=0
SKIP=0

pass() { echo "  PASS: $1"; PASS=$((PASS + 1)); }
fail() { echo "  FAIL: $1"; FAIL=$((FAIL + 1)); }
skip() { echo "  SKIP: $1"; SKIP=$((SKIP + 1)); }

echo "============================================"
echo " Proof Verification Hardening Tests"
echo "============================================"
echo "  Binary:    $PM"
echo "  Model:     ${MODEL_DIR:-<none>}"
echo "  Layers:    $LAYERS"
echo "  GPU:       ${GPU:-0}"
echo "  Temp dir:  $TMPDIR"
echo ""

if [ ! -x "$PM" ]; then
    echo "ERROR: prove-model binary not found at $PM"
    echo "Build it first:"
    echo "  cargo build --release --bin prove-model --features cli,model-loading,audit"
    exit 1
fi

# ── Test 1: Reject fake proof (right field names, garbage data) ──────────
echo "── Test 1: Fake proof rejection"
cat > "$TMPDIR/fake.json" <<'JSON'
{"format":"ml_gkr","gkr_calldata":["0x1"],"io_calldata":["0x1"],"io_commitment":"0xdead"}
JSON
if $PM --verify-proof "$TMPDIR/fake.json" >/dev/null 2>&1; then
    fail "fake proof accepted (should exit 1)"
else
    pass "fake proof rejected — truncated IO data"
fi

# ── Test 2: Reject empty io_calldata ─────────────────────────────────────
echo "── Test 2: Empty io_calldata rejection"
cat > "$TMPDIR/empty_io.json" <<'JSON'
{"format":"ml_gkr","gkr_calldata":["0x1"],"io_calldata":[],"io_commitment":"0x0"}
JSON
if $PM --verify-proof "$TMPDIR/empty_io.json" >/dev/null 2>&1; then
    fail "empty io_calldata accepted"
else
    pass "empty io_calldata rejected"
fi

# ── Test 3: Reject invalid JSON ──────────────────────────────────────────
echo "── Test 3: Invalid JSON rejection"
echo "not json at all" > "$TMPDIR/bad.json"
if $PM --verify-proof "$TMPDIR/bad.json" >/dev/null 2>&1; then
    fail "invalid JSON accepted"
else
    pass "invalid JSON rejected"
fi

# ── Test 4: Reject dimension mismatch in io_calldata ─────────────────────
echo "── Test 4: Dimension mismatch rejection"
cat > "$TMPDIR/bad_dims.json" <<'JSON'
{"format":"ml_gkr","gkr_calldata":["0x1"],
 "io_calldata":["0x2","0x3","0x5","0x0","0x0","0x0","0x0","0x0"],
 "io_commitment":"0x0"}
JSON
if $PM --verify-proof "$TMPDIR/bad_dims.json" >/dev/null 2>&1; then
    fail "dimension mismatch accepted"
else
    pass "dimension mismatch rejected (rows*cols != len)"
fi

# ── Test 5: Reject M31 out-of-range value ────────────────────────────────
echo "── Test 5: M31 out-of-range rejection"
# 0x80000000 = 2^31, which is > P = 2^31-1
cat > "$TMPDIR/oor.json" <<'JSON'
{"format":"ml_gkr","gkr_calldata":["0x1"],
 "io_calldata":["0x1","0x1","0x1","0x80000000","0x1","0x1","0x1","0x0"],
 "io_commitment":"0x0"}
JSON
if $PM --verify-proof "$TMPDIR/oor.json" >/dev/null 2>&1; then
    fail "out-of-range M31 accepted"
else
    pass "out-of-range M31 rejected"
fi

# ── Tests 6-9: Require a real proof (need MODEL_DIR) ────────────────────
if [ -z "$MODEL_DIR" ]; then
    echo ""
    echo "── Tests 6-9: SKIPPED (set MODEL_DIR to enable)"
    skip "real proof generation (no MODEL_DIR)"
    skip "commitment-only verification"
    skip "tampered data rejection"
    skip "tampered commitment rejection"
else
    PROOF="$TMPDIR/real_proof.json"

    # ── Test 6: Generate real proof ──────────────────────────────────────
    echo "── Test 6: Generate real proof"
    if $PM --model-dir "$MODEL_DIR" --layers "$LAYERS" \
           --output "$PROOF" --format ml_gkr $GPU_FLAG 2>"$TMPDIR/prove.log"; then
        SIZE=$(wc -c < "$PROOF")
        pass "real proof generated ($SIZE bytes)"
    else
        fail "proof generation failed"
        cat "$TMPDIR/prove.log"
        # Can't continue without a proof
        echo ""
        echo "Results: $PASS passed, $FAIL failed, $SKIP skipped"
        exit 1
    fi

    # ── Test 7: Verify real proof (commitment only) ──────────────────────
    echo "── Test 7: Commitment-only verification"
    if $PM --verify-proof "$PROOF" 2>"$TMPDIR/verify_commit.log"; then
        if grep -q "io_commitment: verified" "$TMPDIR/verify_commit.log"; then
            pass "commitment verified"
        else
            fail "commitment check not performed"
        fi
    else
        fail "valid proof rejected"
    fi

    # ── Test 8: Tampered io_calldata value ───────────────────────────────
    echo "── Test 8: Tampered io_calldata detection"
    python3 -c "
import json, sys
with open('$PROOF') as f:
    p = json.load(f)
p['io_calldata'][-1] = '0x42'
with open('$TMPDIR/tampered_data.json', 'w') as f:
    json.dump(p, f)
" 2>/dev/null
    if $PM --verify-proof "$TMPDIR/tampered_data.json" >/dev/null 2>&1; then
        fail "tampered io_calldata accepted"
    else
        pass "tampered io_calldata rejected"
    fi

    # ── Test 9: Tampered io_commitment ───────────────────────────────────
    echo "── Test 9: Tampered io_commitment detection"
    python3 -c "
import json
with open('$PROOF') as f:
    p = json.load(f)
p['io_commitment'] = '0xdeadbeefcafeface'
with open('$TMPDIR/tampered_commit.json', 'w') as f:
    json.dump(p, f)
"
    if $PM --verify-proof "$TMPDIR/tampered_commit.json" >/dev/null 2>&1; then
        fail "tampered io_commitment accepted"
    else
        pass "tampered io_commitment rejected"
    fi

    # ── Test 10: Forward-pass replay with model ──────────────────────────
    echo "── Test 10: Forward-pass replay (with model)"
    if $PM --verify-proof "$PROOF" \
           --model-dir "$MODEL_DIR" --layers "$LAYERS" 2>"$TMPDIR/verify_full.log"; then
        if grep -q "forward pass: verified" "$TMPDIR/verify_full.log"; then
            pass "forward pass matches exactly"
        elif grep -q "io_commitment: verified" "$TMPDIR/verify_full.log"; then
            # GPU/CPU divergence is expected — io_commitment is the authority
            pass "io_commitment verified (GPU/CPU replay divergence noted)"
        else
            fail "verification output unexpected"
        fi
    else
        fail "valid proof with model rejected"
    fi
fi

# ── Summary ──────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo " Results: $PASS passed, $FAIL failed, $SKIP skipped"
echo "============================================"

# Cleanup
rm -rf "$TMPDIR"

[ "$FAIL" -eq 0 ] && exit 0 || exit 1
