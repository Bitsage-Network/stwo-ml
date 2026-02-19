#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════════
# Obelysk Pipeline — Canonical Proof Calldata Validator
# ═══════════════════════════════════════════════════════════════════════
#
# Standalone Python script that validates proof calldata before on-chain
# submission. This is the CANONICAL implementation — the JS version in
# paymaster_submit.mjs should be kept in sync with this file.
#
# Usage:
#   python3 validate_proof.py <proof.json>
#   python3 validate_proof.py <proof.json> --out-dir /tmp/verify_payload
#   python3 validate_proof.py <proof.json> --out-dir /tmp/verify_payload --session-id 0x123
#
# Output (stdout, JSON):
#   {"entrypoint": "...", "calldata": [...], "model_id": "...", "valid": true}
#
# On failure: prints error to stderr, exits with code 1.
#
# Environment variables:
#   OBELYSK_MAX_GKR_CALLDATA_FELTS        Hard fail if calldata exceeds this (default: 300000)
#   OBELYSK_MAX_GKR_MODE4_CALLDATA_FELTS  Hard fail if v4/mode4 calldata exceeds this (default: 120000)
#   OBELYSK_MIN_GKR_MODE4_CALLDATA_FELTS  Hard fail if v4/mode4 calldata is below this (default: 1000)
#
# No external dependencies beyond Python 3 stdlib.
#
import argparse
import json
import os
import sys

# ─── Helpers ──────────────────────────────────────────────────────────

def fail(msg: str) -> None:
    """Print error to stderr and exit with code 1."""
    print(msg, file=sys.stderr)
    raise SystemExit(1)


def parse_positive_int_env(name: str, default: int) -> int:
    """Read a positive integer from an environment variable, or return default."""
    raw = os.environ.get(name, '').strip()
    if not raw:
        return default
    try:
        v = int(raw)
    except ValueError:
        fail(f'{name} must be a positive integer (got: {raw})')
    if v <= 0:
        fail(f'{name} must be a positive integer (got: {raw})')
    return v


def parse_nat(tok: str, label: str) -> int:
    """Parse a non-negative integer from a calldata token (supports 0x prefix)."""
    s = str(tok)
    try:
        v = int(s, 0)
    except Exception as e:
        fail(f'invalid {label}: {s} ({e})')
    if v < 0:
        fail(f'{label} must be >= 0 (got {v})')
    return v


# ─── Core Validation ─────────────────────────────────────────────────

ALLOWED_ENTRYPOINTS = {
    'verify_model_gkr',
    'verify_model_gkr_v2',
    'verify_model_gkr_v3',
    'verify_model_gkr_v4',
}

# weight_opening_mode string -> numeric mode ID
WEIGHT_MODE_MAP = {
    'Sequential': 0,
    'BatchedSubchannelV1': 1,
    'AggregatedTrustlessV2': 2,
    'AggregatedOpeningsV4Experimental': 3,
    'AggregatedOracleSumcheck': 4,
}


def validate_proof(proof_file: str, session_id: str = '') -> dict:
    """
    Validate a proof file's verify_calldata payload.

    Returns a dict with keys:
        entrypoint, calldata, model_id, valid, schema_version
    Calls fail() (exits) on any validation error.
    """
    # ── Read env-var limits ──
    max_gkr_calldata_felts = parse_positive_int_env(
        'OBELYSK_MAX_GKR_CALLDATA_FELTS', 300000
    )
    max_gkr_mode4_calldata_felts = parse_positive_int_env(
        'OBELYSK_MAX_GKR_MODE4_CALLDATA_FELTS', 120000
    )
    min_gkr_mode4_calldata_felts = parse_positive_int_env(
        'OBELYSK_MIN_GKR_MODE4_CALLDATA_FELTS', 1000
    )

    # ── Load proof JSON ──
    try:
        with open(proof_file, 'r', encoding='utf-8') as f:
            proof = json.load(f)
    except Exception as e:
        fail(f'Failed to read proof JSON: {e}')

    # ── Top-level verify_calldata object ──
    vc = proof.get('verify_calldata')
    if not isinstance(vc, dict):
        fail("Missing 'verify_calldata' object in proof file")

    # ── schema_version ──
    schema_version = vc.get('schema_version')
    if schema_version != 1:
        fail('verify_calldata.schema_version must be 1')

    # ── entrypoint ──
    entrypoint = vc.get('entrypoint')
    if not isinstance(entrypoint, str) or not entrypoint:
        fail("verify_calldata.entrypoint must be a non-empty string")
    if entrypoint not in ALLOWED_ENTRYPOINTS:
        mode = proof.get('weight_opening_mode', 'unknown')
        reason = vc.get('reason') or proof.get('soundness_gate_error') or 'unspecified'
        ready = bool(proof.get('submission_ready', False))
        fail(
            'Only verify_model_gkr / verify_model_gkr_v2 / verify_model_gkr_v3 / '
            'verify_model_gkr_v4 are supported in the hardened pipeline '
            f'(got: {entrypoint}, submission_ready={ready}, '
            f'weight_opening_mode={mode}, reason={reason})'
        )

    # ── calldata array ──
    calldata = vc.get('calldata')
    if not isinstance(calldata, list) or len(calldata) == 0:
        fail("verify_calldata.calldata must be a non-empty array")
    if any(str(v) == '__SESSION_ID__' for v in calldata):
        fail('verify_model_gkr(*) calldata must not include __SESSION_ID__ placeholder')

    resolved = [str(v) for v in calldata]

    # ── Global size bound ──
    if len(resolved) > max_gkr_calldata_felts:
        fail(
            f'calldata too large for hardened submit path: {len(resolved)} felts '
            f'(max {max_gkr_calldata_felts}). '
            'Likely legacy per-opening mode; use --submit to generate mode-4 aggregated proof.'
        )

    # ── submission_ready flag ──
    submission_ready = proof.get('submission_ready')
    if submission_ready is False:
        mode = proof.get('weight_opening_mode', 'unknown')
        reason = vc.get('reason') or proof.get('soundness_gate_error') or 'unspecified'
        fail(
            f'proof is marked submission_ready=false '
            f'(entrypoint={entrypoint}, weight_opening_mode={mode}, reason={reason})'
        )

    # ── weight_opening_mode validation ──
    weight_opening_mode = proof.get('weight_opening_mode')
    if entrypoint == 'verify_model_gkr':
        if weight_opening_mode is not None and str(weight_opening_mode) != 'Sequential':
            fail(
                f'{entrypoint} requires weight_opening_mode=Sequential '
                f'(got: {weight_opening_mode})'
            )
    elif entrypoint in ('verify_model_gkr_v2', 'verify_model_gkr_v3'):
        allowed_modes = {'Sequential', 'BatchedSubchannelV1'}
        if weight_opening_mode is not None and str(weight_opening_mode) not in allowed_modes:
            fail(
                f'{entrypoint} requires weight_opening_mode in {sorted(allowed_modes)} '
                f'(got: {weight_opening_mode})'
            )
    elif entrypoint == 'verify_model_gkr_v4':
        allowed_v4_modes = {'AggregatedOpeningsV4Experimental', 'AggregatedOracleSumcheck'}
        if weight_opening_mode is not None and str(weight_opening_mode) not in allowed_v4_modes:
            fail(
                f'{entrypoint} requires weight_opening_mode in {sorted(allowed_v4_modes)} '
                f'(got: {weight_opening_mode})'
            )

    # ── upload_chunks (must be empty for GKR) ──
    upload_chunks = vc.get('upload_chunks', [])
    if upload_chunks is None:
        upload_chunks = []
    if not isinstance(upload_chunks, list):
        fail('verify_calldata.upload_chunks must be an array')
    if len(upload_chunks) != 0:
        fail('verify_model_gkr(*) payload must not include upload_chunks')

    # ── Structural validation for v2/v3/v4 calldata ──
    if entrypoint in ('verify_model_gkr_v2', 'verify_model_gkr_v3', 'verify_model_gkr_v4'):
        # Layout:
        # model_id, raw_io_data, circuit_depth, num_layers, matmul_dims,
        # dequantize_bits, proof_data, weight_commitments, weight_binding_mode,
        # [weight_binding_data (v3/v4)], weight_openings...
        idx = 0
        idx += 1  # model_id
        if idx >= len(resolved):
            fail('v2 calldata truncated before raw_io length')
        raw_io_len = parse_nat(resolved[idx], 'raw_io_data length')
        idx += 1 + raw_io_len

        idx += 2  # circuit_depth, num_layers
        if idx >= len(resolved):
            fail('v2 calldata truncated before matmul_dims length')
        matmul_len = parse_nat(resolved[idx], 'matmul_dims length')
        idx += 1 + matmul_len

        if idx >= len(resolved):
            fail('v2 calldata truncated before dequantize_bits length')
        deq_len = parse_nat(resolved[idx], 'dequantize_bits length')
        idx += 1 + deq_len

        if idx >= len(resolved):
            fail('v2 calldata truncated before proof_data length')
        proof_data_len = parse_nat(resolved[idx], 'proof_data length')
        idx += 1 + proof_data_len

        if idx >= len(resolved):
            fail('v2 calldata truncated before weight_commitments length')
        wc_len = parse_nat(resolved[idx], 'weight_commitments length')
        idx += 1 + wc_len

        if idx >= len(resolved):
            fail('v2 calldata truncated before weight_binding_mode')
        weight_binding_mode = parse_nat(resolved[idx], 'weight_binding_mode')

        # Entrypoint-specific weight_binding_mode range checks
        if entrypoint == 'verify_model_gkr_v2' and weight_binding_mode not in (0, 1):
            fail(f'{entrypoint} requires weight_binding_mode in (0,1) (got {weight_binding_mode})')
        if entrypoint == 'verify_model_gkr_v4' and weight_binding_mode not in (3, 4):
            fail(f'{entrypoint} requires weight_binding_mode in (3, 4) (got {weight_binding_mode})')

        # Cross-check weight_opening_mode string vs numeric mode
        expected_mode = WEIGHT_MODE_MAP.get(str(weight_opening_mode)) if weight_opening_mode is not None else None
        if expected_mode is not None and weight_binding_mode != expected_mode:
            fail(
                f'{entrypoint} expected weight_binding_mode={expected_mode} '
                f'for weight_opening_mode={weight_opening_mode} (got {weight_binding_mode})'
            )

        # Fallback allowed-modes check when weight_opening_mode is absent
        if entrypoint == 'verify_model_gkr_v3':
            allowed_binding_modes = (0, 1, 2)
        elif entrypoint == 'verify_model_gkr_v4':
            allowed_binding_modes = (3, 4)
        else:
            allowed_binding_modes = (0, 1)
        if expected_mode is None and weight_binding_mode not in allowed_binding_modes:
            fail(
                f'{entrypoint} requires weight_binding_mode in {allowed_binding_modes} '
                f'(got {weight_binding_mode})'
            )

        # Cross-check artifact weight_binding_mode_id
        artifact_mode_id = proof.get('weight_binding_mode_id')
        if artifact_mode_id is not None:
            try:
                artifact_mode_id_i = int(str(artifact_mode_id), 0)
            except Exception as e:
                fail(f'invalid weight_binding_mode_id: {artifact_mode_id} ({e})')
            if artifact_mode_id_i != weight_binding_mode:
                fail(
                    f'weight_binding_mode_id mismatch: artifact={artifact_mode_id_i} '
                    f'calldata={weight_binding_mode}'
                )

        # v3/v4: weight_binding_data section
        if entrypoint in ('verify_model_gkr_v3', 'verify_model_gkr_v4'):
            idx += 1  # consume weight_binding_mode
            if idx >= len(resolved):
                fail(f'{entrypoint} calldata truncated before weight_binding_data length')
            weight_binding_data_len = parse_nat(resolved[idx], 'weight_binding_data length')
            idx += 1 + weight_binding_data_len

            if weight_binding_mode in (0, 1) and weight_binding_data_len != 0:
                fail(
                    f'{entrypoint} mode {weight_binding_mode} requires empty weight_binding_data '
                    f'(got len={weight_binding_data_len})'
                )
            if weight_binding_mode == 2 and weight_binding_data_len == 0:
                fail(f'{entrypoint} mode 2 requires non-empty weight_binding_data')
            if weight_binding_mode == 3 and weight_binding_data_len == 0:
                fail(f'{entrypoint} mode 3 requires non-empty weight_binding_data')
            if weight_binding_mode == 4 and weight_binding_data_len == 0:
                fail(f'{entrypoint} mode 4 (aggregated oracle sumcheck) requires non-empty weight_binding_data')

            # v4 mode 4: size bounds
            if entrypoint == 'verify_model_gkr_v4' and weight_binding_mode == 4:
                if len(resolved) > max_gkr_mode4_calldata_felts:
                    fail(
                        f'{entrypoint} mode 4 calldata unexpectedly large: {len(resolved)} felts '
                        f'(max {max_gkr_mode4_calldata_felts}). '
                        'This looks like non-aggregated payload; regenerate proof with --submit.'
                    )
                if len(resolved) < min_gkr_mode4_calldata_felts:
                    fail(
                        f'{entrypoint} mode 4 calldata unexpectedly small: {len(resolved)} felts '
                        f'(min {min_gkr_mode4_calldata_felts}).'
                    )

            # Cross-check weight_binding_data_calldata from artifact
            artifact_binding_data = proof.get('weight_binding_data_calldata')
            if isinstance(artifact_binding_data, list):
                if len(artifact_binding_data) != weight_binding_data_len:
                    fail(
                        f'weight_binding_data_calldata length mismatch: '
                        f'artifact={len(artifact_binding_data)} '
                        f'calldata={weight_binding_data_len}'
                    )
                start = idx - weight_binding_data_len
                calldata_binding_data = resolved[start:idx]
                for bind_i, (artifact_tok, calldata_tok) in enumerate(
                    zip(artifact_binding_data, calldata_binding_data)
                ):
                    try:
                        artifact_v = int(str(artifact_tok), 0)
                    except Exception as e:
                        fail(
                            f'invalid artifact weight_binding_data_calldata[{bind_i}]: '
                            f'{artifact_tok} ({e})'
                        )
                    try:
                        calldata_v = int(str(calldata_tok), 0)
                    except Exception as e:
                        fail(
                            f'invalid calldata weight_binding_data[{bind_i}]: '
                            f'{calldata_tok} ({e})'
                        )
                    if artifact_v != calldata_v:
                        fail(
                            f'weight_binding_data_calldata mismatch at index {bind_i}: '
                            f'artifact={artifact_tok} calldata={calldata_tok}'
                        )

    # ── Extract model_id (first element of calldata) ──
    model_id = resolved[0] if resolved else ''

    return {
        'entrypoint': entrypoint,
        'calldata': resolved,
        'model_id': model_id,
        'valid': True,
        'schema_version': schema_version,
        'upload_chunks': upload_chunks,
        'calldata_felts': len(resolved),
    }


# ─── File Output (--out-dir) ─────────────────────────────────────────

def write_output_files(result: dict, out_dir: str) -> None:
    """Write entrypoint.txt, calldata.txt, and chunks/ to out_dir."""
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, 'entrypoint.txt'), 'w', encoding='utf-8') as f:
        f.write(result['entrypoint'])

    with open(os.path.join(out_dir, 'calldata.txt'), 'w', encoding='utf-8') as f:
        f.write(' '.join(result['calldata']))

    chunks_dir = os.path.join(out_dir, 'chunks')
    os.makedirs(chunks_dir, exist_ok=True)

    upload_chunks = result.get('upload_chunks', [])
    with open(os.path.join(chunks_dir, 'count.txt'), 'w', encoding='utf-8') as f:
        f.write(str(len(upload_chunks)))

    for idx, chunk in enumerate(upload_chunks):
        if not isinstance(chunk, list):
            fail(f'verify_calldata.upload_chunks[{idx}] must be an array')
        with open(os.path.join(chunks_dir, f'chunk_{idx}.txt'), 'w', encoding='utf-8') as f:
            f.write(' '.join(str(v) for v in chunk))


# ─── CLI Entry Point ─────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Validate Obelysk proof calldata before on-chain submission.'
    )
    parser.add_argument(
        'proof_file',
        help='Path to the proof JSON file',
    )
    parser.add_argument(
        '--out-dir',
        default=None,
        help='Directory to write entrypoint.txt, calldata.txt, and chunks/ (optional)',
    )
    parser.add_argument(
        '--session-id',
        default='',
        help='Session ID (reserved for future use, currently unused)',
    )
    args = parser.parse_args()

    # Run validation (calls fail() on error, which exits with code 1)
    result = validate_proof(args.proof_file, args.session_id)

    # Optionally write output files
    if args.out_dir:
        write_output_files(result, args.out_dir)

    # JSON output to stdout (compact for piping, without internal fields)
    output = {
        'entrypoint': result['entrypoint'],
        'calldata': result['calldata'],
        'model_id': result['model_id'],
        'valid': result['valid'],
        'calldata_felts': result['calldata_felts'],
    }
    json.dump(output, sys.stdout)
    print()  # trailing newline


if __name__ == '__main__':
    main()
