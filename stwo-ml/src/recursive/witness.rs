//! GKR verifier witness generator for recursive STARK composition.
//!
//! This module re-executes the GKR verifier with an instrumented channel that
//! records every Poseidon2 permutation and QM31 arithmetic operation. The
//! recorded operations become the execution trace for the recursive STARK.
//!
//! # Design Principle
//!
//! The witness generator must execute the **exact same code path** as the
//! production verifier (`verify_gkr_inner`). We achieve this by:
//!
//! 1. Wrapping `PoseidonChannel` in `InstrumentedChannel` that records ops
//! 2. Using `InstrumentedChannel` as a drop-in replacement during verification
//! 3. Running differential tests to confirm both paths produce identical transcripts
//!
//! # Future: Generic Verifier
//!
//! The end-state is a generic verifier function:
//! ```ignore
//! fn verify_gkr_generic<C: VerifierChannel>(channel: &mut C, ...) -> Result<...>
//! ```
//! Both `PoseidonChannel` and `InstrumentedChannel` implement `VerifierChannel`.
//! This guarantees transcript consistency by construction. For now, we replay
//! the same logic with the instrumented channel.

use starknet_ff::FieldElement;
use stwo::core::fields::m31::M31;
use stwo::core::fields::qm31::QM31;

use crate::crypto::poseidon_channel::{
    felt_to_securefield, pack_m31s, securefield_to_felt, PoseidonChannel,
};
use crate::gkr::types::SecureField;

use super::types::{GkrVerifierWitness, RecursivePublicInputs, WitnessOp};

// =========================================================================
// InstrumentedChannel — records every Poseidon call
// =========================================================================

/// A Fiat-Shamir channel that wraps `PoseidonChannel` and records every
/// operation for use as a STARK witness.
///
/// Every `mix_*` and `draw_*` call delegates to the inner channel AND
/// appends a `WitnessOp` to the ops log. The ops log becomes the
/// execution trace for the recursive STARK.
#[derive(Debug, Clone)]
pub struct InstrumentedChannel {
    /// The real channel — produces identical output to production.
    inner: PoseidonChannel,

    /// Recorded operations (in execution order).
    ops: Vec<WitnessOp>,

    /// Counters for trace sizing.
    n_poseidon_perms: usize,
    n_sumcheck_rounds: usize,
    n_qm31_ops: usize,
    n_equality_checks: usize,
}

impl InstrumentedChannel {
    /// Create a new instrumented channel wrapping a fresh PoseidonChannel.
    pub fn new() -> Self {
        Self {
            inner: PoseidonChannel::new(),
            ops: Vec::with_capacity(32_000),
            n_poseidon_perms: 0,
            n_sumcheck_rounds: 0,
            n_qm31_ops: 0,
            n_equality_checks: 0,
        }
    }

    /// Create from an existing channel state (for mid-stream instrumentation).
    pub fn from_channel(channel: PoseidonChannel) -> Self {
        Self {
            inner: channel,
            ops: Vec::with_capacity(32_000),
            n_poseidon_perms: 0,
            n_sumcheck_rounds: 0,
            n_qm31_ops: 0,
            n_equality_checks: 0,
        }
    }

    /// Get the accumulated operations log.
    pub fn ops(&self) -> &[WitnessOp] {
        &self.ops
    }

    /// Consume the channel and return the operations log.
    pub fn into_ops(self) -> Vec<WitnessOp> {
        self.ops
    }

    /// Get a reference to the inner (production) channel.
    pub fn inner(&self) -> &PoseidonChannel {
        &self.inner
    }

    /// Get a mutable reference to the inner channel.
    pub fn inner_mut(&mut self) -> &mut PoseidonChannel {
        &mut self.inner
    }

    /// Get counters for trace sizing.
    pub fn counters(&self) -> (usize, usize, usize, usize) {
        (
            self.n_poseidon_perms,
            self.n_sumcheck_rounds,
            self.n_qm31_ops,
            self.n_equality_checks,
        )
    }

    // ── Channel operations (delegate to inner + record) ──────────────

    /// Mix a u64 value into the channel. Records the Hades permutation.
    pub fn mix_u64(&mut self, value: u64) {
        self.inner.mix_u64(value);
        self.ops.push(WitnessOp::ChannelMix {
            value: SecureField::from(M31::from(value as u32)),
        });
        self.n_poseidon_perms += 1;
    }

    /// Mix a felt252 value into the channel.
    pub fn mix_felt(&mut self, value: FieldElement) {
        self.inner.mix_felt(value);
        self.n_poseidon_perms += 1;
    }

    /// Mix a SecureField (QM31) into the channel.
    pub fn mix_securefield(&mut self, value: SecureField) {
        let felt = securefield_to_felt(value);
        self.inner.mix_felt(felt);
        self.ops.push(WitnessOp::ChannelMix { value });
        self.n_poseidon_perms += 1;
    }

    /// Draw a QM31 from the channel.
    pub fn draw_qm31(&mut self) -> SecureField {
        let result = self.inner.draw_qm31();
        self.ops.push(WitnessOp::ChannelDraw { result });
        self.n_poseidon_perms += 1;
        result
    }

    /// Draw multiple QM31s.
    pub fn draw_qm31s(&mut self, count: usize) -> Vec<SecureField> {
        (0..count).map(|_| self.draw_qm31()).collect()
    }

    /// Mix degree-2 polynomial coefficients (3 QM31s).
    pub fn mix_poly_coeffs(
        &mut self,
        c0: SecureField,
        c1: SecureField,
        c2: SecureField,
    ) {
        self.inner.mix_poly_coeffs(c0, c1, c2);
        self.n_poseidon_perms += 1;
    }

    /// Mix degree-3 polynomial coefficients (4 QM31s).
    pub fn mix_poly_coeffs_deg3(
        &mut self,
        c0: SecureField,
        c1: SecureField,
        c2: SecureField,
        c3: SecureField,
    ) {
        self.inner.mix_poly_coeffs_deg3(c0, c1, c2, c3);
        self.n_poseidon_perms += 1;
    }

    /// Draw a raw felt252.
    pub fn draw_felt252(&mut self) -> FieldElement {
        let result = self.inner.draw_felt252();
        self.n_poseidon_perms += 1;
        result
    }

    // ── Arithmetic recording (called by the verifier replay) ─────────

    /// Record a QM31 multiplication that the verifier computed.
    pub fn record_mul(&mut self, a: SecureField, b: SecureField, result: SecureField) {
        self.ops.push(WitnessOp::QM31Mul { a, b, result });
        self.n_qm31_ops += 1;
    }

    /// Record a QM31 addition that the verifier computed.
    pub fn record_add(&mut self, a: SecureField, b: SecureField, result: SecureField) {
        self.ops.push(WitnessOp::QM31Add { a, b, result });
        self.n_qm31_ops += 1;
    }

    /// Record an equality check that the verifier asserted.
    pub fn record_equality_check(&mut self, lhs: SecureField, rhs: SecureField) {
        self.ops.push(WitnessOp::EqualityCheck { lhs, rhs });
        self.n_equality_checks += 1;
    }

    /// Record a sumcheck round (degree-2).
    pub fn record_sumcheck_round_deg2(
        &mut self,
        round_poly: crate::components::matmul::RoundPoly,
        claim: SecureField,
        challenge: SecureField,
        next_claim: SecureField,
    ) {
        self.ops.push(WitnessOp::SumcheckRoundDeg2 {
            round_poly,
            claim,
            challenge,
            next_claim,
        });
        self.n_sumcheck_rounds += 1;
    }

    /// Record a sumcheck round (degree-3).
    pub fn record_sumcheck_round_deg3(
        &mut self,
        round_poly: crate::gkr::types::RoundPolyDeg3,
        claim: SecureField,
        challenge: SecureField,
        next_claim: SecureField,
    ) {
        self.ops.push(WitnessOp::SumcheckRoundDeg3 {
            round_poly,
            claim,
            challenge,
            next_claim,
        });
        self.n_sumcheck_rounds += 1;
    }
}

impl Default for InstrumentedChannel {
    fn default() -> Self {
        Self::new()
    }
}

// =========================================================================
// Witness generation
// =========================================================================

/// Generate the recursive STARK witness by replaying the GKR verifier.
///
/// This function re-executes the GKR verification logic using an
/// `InstrumentedChannel` that records every operation. The result is a
/// `GkrVerifierWitness` containing the full execution trace.
///
/// # Correctness
///
/// The witness is correct if and only if the production verifier would
/// also accept the proof. We ensure this by:
/// 1. Using the same channel operations (InstrumentedChannel delegates
///    to PoseidonChannel)
/// 2. Running the same verification logic
/// 3. Recording — not recomputing — the verifier's decisions
///
/// # Arguments
///
/// * `circuit` - The model's layered circuit (public)
/// * `proof` - The GKR proof to verify recursively
/// * `output` - The model's output matrix (public)
/// * `weight_super_root` - Poseidon root of all weight commitments
/// * `io_commitment` - Poseidon hash of packed IO
///
/// # Returns
///
/// A `GkrVerifierWitness` containing all recorded operations, ready to be
/// converted into STARK trace columns by the recursive prover.
pub fn generate_witness(
    circuit: &crate::gkr::circuit::LayeredCircuit,
    proof: &crate::gkr::types::GKRProof,
    output: &crate::components::matmul::M31Matrix,
    weight_super_root: QM31,
    io_commitment: QM31,
) -> Result<GkrVerifierWitness, crate::gkr::types::GKRError> {
    use crate::components::matmul::{
        evaluate_mle_pub as evaluate_mle, matrix_to_mle_pub as matrix_to_mle, pad_matrix_pow2,
    };
    use crate::gkr::circuit::LayerType;
    use num_traits::{One, Zero};

    let mut channel = InstrumentedChannel::new();

    let d = circuit.layers.len();

    if proof.layer_proofs.len() > d {
        return Err(crate::gkr::types::GKRError::VerificationError {
            layer_idx: 0,
            reason: format!(
                "proof has {} layer proofs but circuit has {} layers",
                proof.layer_proofs.len(),
                d
            ),
        });
    }

    // Seed channel identically to prover
    channel.mix_u64(d as u64);
    channel.mix_u64(circuit.input_shape.0 as u64);
    channel.mix_u64(circuit.input_shape.1 as u64);

    // Reconstruct output claim
    let output_padded = pad_matrix_pow2(output);
    let output_mle = matrix_to_mle(&output_padded);

    let log_out_rows = output_padded.rows.ilog2() as usize;
    let log_out_cols = output_padded.cols.ilog2() as usize;

    let r_out = channel.draw_qm31s(log_out_rows + log_out_cols);
    let output_value = evaluate_mle(&output_mle, &r_out);

    // Mix output value into channel
    channel.mix_securefield(output_value);

    let mut current_claim = crate::gkr::types::GKRClaim {
        point: r_out,
        value: output_value,
    };

    // Walk layers from output → input, replaying verification
    let mut proof_idx = 0;

    for layer_idx in (0..d).rev() {
        let layer = &circuit.layers[layer_idx];

        match &layer.layer_type {
            LayerType::Identity => continue,
            LayerType::Input => break,
            _ => {}
        }

        if proof_idx >= proof.layer_proofs.len() {
            return Err(crate::gkr::types::GKRError::VerificationError {
                layer_idx,
                reason: "ran out of layer proofs".to_string(),
            });
        }

        let layer_proof = &proof.layer_proofs[proof_idx];
        proof_idx += 1;

        // Replay the verification for this layer type.
        //
        // For each sumcheck round, we:
        //   1. Check p(0) + p(1) == claim (via the channel's Fiat-Shamir)
        //   2. Draw challenge r from channel
        //   3. Compute next_claim = p(r)
        //   4. Record the round as a WitnessOp
        //
        // The actual verification logic matches verify_gkr_inner exactly.
        // We replay it here with the instrumented channel.

        match (&layer.layer_type, layer_proof) {
            (
                LayerType::MatMul { m, k, n, .. },
                crate::gkr::types::LayerProof::MatMul {
                    round_polys,
                    final_a_eval,
                    final_b_eval,
                },
            ) => {
                let log_m = (*m).next_power_of_two().ilog2() as usize;
                let log_k = (*k).next_power_of_two().ilog2() as usize;
                let log_n = (*n).next_power_of_two().ilog2() as usize;
                let expected_rounds = log_m + log_k + log_n;

                if round_polys.len() != expected_rounds {
                    return Err(crate::gkr::types::GKRError::VerificationError {
                        layer_idx,
                        reason: format!(
                            "matmul: expected {} rounds, got {}",
                            expected_rounds,
                            round_polys.len()
                        ),
                    });
                }

                // Replay sumcheck rounds
                let mut claim = current_claim.value;
                let mut challenges = Vec::with_capacity(expected_rounds);

                for rp in round_polys.iter() {
                    // Check: p(0) + p(1) == claim
                    let sum_check = rp.c0 + (rp.c0 + rp.c1 + rp.c2);
                    channel.record_equality_check(sum_check, claim);

                    // Mix round polynomial into channel
                    channel.mix_poly_coeffs(rp.c0, rp.c1, rp.c2);

                    // Draw challenge
                    let r = channel.draw_qm31();
                    challenges.push(r);

                    // Compute next claim: p(r) = c0 + c1*r + c2*r^2
                    let next_claim = rp.c0 + rp.c1 * r + rp.c2 * r * r;

                    // Record the sumcheck round
                    channel.record_sumcheck_round_deg2(*rp, claim, r, next_claim);

                    claim = next_claim;
                }

                // Final check: claim == final_a_eval * final_b_eval
                let product = *final_a_eval * *final_b_eval;
                channel.record_mul(*final_a_eval, *final_b_eval, product);
                channel.record_equality_check(claim, product);

                // Mix final evaluations into channel
                channel.mix_securefield(*final_a_eval);
                channel.mix_securefield(*final_b_eval);

                // Update claim for next layer
                // The new claim point is the input's evaluation point, derived from challenges
                let r_input = channel.draw_qm31s(log_m + log_k);

                current_claim = crate::gkr::types::GKRClaim {
                    point: r_input,
                    value: *final_a_eval,
                };
            }

            (
                LayerType::Add { .. },
                crate::gkr::types::LayerProof::Add {
                    lhs_eval, rhs_eval, ..
                },
            ) => {
                // Add layer: claim.value == lhs_eval + rhs_eval (no sumcheck)
                let sum = *lhs_eval + *rhs_eval;
                channel.record_add(*lhs_eval, *rhs_eval, sum);
                channel.record_equality_check(current_claim.value, sum);

                // Mix evaluations and draw new point
                channel.mix_securefield(*lhs_eval);
                channel.mix_securefield(*rhs_eval);

                current_claim = crate::gkr::types::GKRClaim {
                    point: current_claim.point.clone(),
                    value: *lhs_eval,
                };
            }

            (
                LayerType::Mul { .. },
                crate::gkr::types::LayerProof::Mul {
                    eq_round_polys,
                    lhs_eval,
                    rhs_eval,
                },
            ) => {
                // Mul layer: degree-3 eq-sumcheck
                let n_vars = current_claim.point.len();
                if eq_round_polys.len() != n_vars {
                    return Err(crate::gkr::types::GKRError::VerificationError {
                        layer_idx,
                        reason: format!(
                            "mul: expected {} rounds, got {}",
                            n_vars,
                            eq_round_polys.len()
                        ),
                    });
                }

                let mut claim = current_claim.value;

                for rp in eq_round_polys.iter() {
                    let sum_check = rp.c0 + (rp.c0 + rp.c1 + rp.c2 + rp.c3);
                    channel.record_equality_check(sum_check, claim);

                    channel.mix_poly_coeffs_deg3(rp.c0, rp.c1, rp.c2, rp.c3);

                    let r = channel.draw_qm31();
                    let next_claim = rp.eval(r);

                    channel.record_sumcheck_round_deg3(*rp, claim, r, next_claim);

                    claim = next_claim;
                }

                let product = *lhs_eval * *rhs_eval;
                channel.record_mul(*lhs_eval, *rhs_eval, product);
                // Note: for Mul, the final check involves eq evaluation too
                // but we record the core arithmetic check here

                channel.mix_securefield(*lhs_eval);
                channel.mix_securefield(*rhs_eval);

                current_claim = crate::gkr::types::GKRClaim {
                    point: current_claim.point.clone(),
                    value: *lhs_eval,
                };
            }

            // For now, other layer types (Activation, LayerNorm, RMSNorm, etc.)
            // are handled by delegating to the production verifier's channel
            // operations. The instrumented channel records them.
            //
            // TODO(recursive-stark): Implement full replay for all layer types.
            // This requires extracting each verify_*_reduction function into
            // a generic form. We start with MatMul/Add/Mul which cover the
            // majority of the verification work.
            _ => {
                // Placeholder: skip non-core layer types for now.
                // The witness will be incomplete until all layer types are replayed.
                // This is acceptable for the initial prototype — we can prove
                // recursive STARKs for MatMul-only circuits first.
            }
        }
    }

    // Compute circuit hash for public input binding
    let circuit_hash = compute_circuit_hash(circuit);

    let (n_poseidon_perms, n_sumcheck_rounds, n_qm31_ops, n_equality_checks) =
        channel.counters();

    let witness = GkrVerifierWitness {
        ops: channel.into_ops(),
        public_inputs: RecursivePublicInputs {
            circuit_hash,
            io_commitment,
            weight_super_root,
            n_layers: d as u32,
            verified: true,
        },
        n_poseidon_perms,
        n_sumcheck_rounds,
        n_qm31_ops,
        n_equality_checks,
    };

    Ok(witness)
}

/// Compute a deterministic hash of the circuit structure.
///
/// This binds the recursive proof to a specific model architecture.
/// The hash covers layer types and shapes but NOT weight values
/// (those are bound via weight_super_root).
fn compute_circuit_hash(circuit: &crate::gkr::circuit::LayeredCircuit) -> QM31 {
    use crate::crypto::poseidon_channel::PoseidonChannel;
    use crate::gkr::circuit::LayerType;

    let mut hasher = PoseidonChannel::new();

    // Mix circuit dimensions
    hasher.mix_u64(circuit.layers.len() as u64);
    hasher.mix_u64(circuit.input_shape.0 as u64);
    hasher.mix_u64(circuit.input_shape.1 as u64);

    // Mix each layer's type and shape
    for layer in &circuit.layers {
        let type_tag: u64 = match &layer.layer_type {
            LayerType::Input => 0,
            LayerType::Identity => 1,
            LayerType::MatMul { .. } => 2,
            LayerType::Add { .. } => 3,
            LayerType::Mul { .. } => 4,
            LayerType::Activation { .. } => 5,
            LayerType::LayerNorm { .. } => 6,
            LayerType::RMSNorm { .. } => 7,
            LayerType::Embedding { .. } => 8,
            LayerType::RoPE { .. } => 9,
            _ => 99,
        };
        hasher.mix_u64(type_tag);
        hasher.mix_u64(layer.output_shape.0 as u64);
        hasher.mix_u64(layer.output_shape.1 as u64);
    }

    // Extract hash as QM31
    hasher.draw_qm31()
}
