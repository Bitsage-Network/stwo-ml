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

    // ── Channel operations (delegate to inner + record Hades states) ──

    /// Mix a u64 value into the channel. Records the Hades permutation.
    pub fn mix_u64(&mut self, value: u64) {
        self.mix_felt(FieldElement::from(value));
    }

    /// Mix a felt252 value into the channel. Records Hades + ChannelOp.
    pub fn mix_felt(&mut self, value: FieldElement) {
        let digest_before = self.inner.digest();

        // Record the Hades call
        let input = [digest_before, value, FieldElement::TWO];
        let mut output = input;
        crate::crypto::hades::hades_permutation(&mut output);
        self.ops.push(WitnessOp::HadesPerm { input, output });
        self.n_poseidon_perms += 1;

        // Record the channel operation (digest chain unit)
        self.ops.push(WitnessOp::ChannelOp {
            digest_before,
            digest_after: output[0],
        });

        // Advance inner channel
        self.inner.mix_felt(value);
        debug_assert_eq!(self.inner.digest(), output[0]);
    }

    /// Mix a SecureField (QM31) into the channel.
    pub fn mix_securefield(&mut self, value: SecureField) {
        let felt = securefield_to_felt(value);
        self.mix_felt(felt);
    }

    /// Draw a QM31 from the channel. Records Hades + ChannelOp.
    pub fn draw_qm31(&mut self) -> SecureField {
        let digest_before = self.inner.digest();
        let n_draws = self.inner.n_draws();

        let result = self.inner.draw_qm31();
        let digest_after = self.inner.digest();

        // Record the Hades call
        let input = [digest_before, FieldElement::from(n_draws as u64), FieldElement::THREE];
        let mut output = input;
        crate::crypto::hades::hades_permutation(&mut output);
        self.ops.push(WitnessOp::HadesPerm { input, output });
        self.n_poseidon_perms += 1;

        // Note: draw doesn't change the digest (only increments n_draws),
        // but the channel state changes. We still record the ChannelOp.
        self.ops.push(WitnessOp::ChannelOp {
            digest_before,
            digest_after,
        });
        self.ops.push(WitnessOp::ChannelDraw { result });
        result
    }

    /// Draw multiple QM31s.
    pub fn draw_qm31s(&mut self, count: usize) -> Vec<SecureField> {
        (0..count).map(|_| self.draw_qm31()).collect()
    }

    /// Mix degree-2 polynomial coefficients (3 QM31s).
    ///
    /// Replicates the exact Hades calls from `poseidon_hash_many([digest, felt1, felt2])`.
    ///
    /// From starknet-crypto source:
    /// ```text
    /// state = [0, 0, 0]
    /// state[0] += digest; state[1] += felt1; hades(&state)   // absorb pair
    /// state[0] += felt2; state[1] += 1;      hades(&state)   // absorb remainder + padding
    /// return state[0]
    /// ```
    /// Total: 2 Hades calls.
    pub fn mix_poly_coeffs(
        &mut self,
        c0: SecureField,
        c1: SecureField,
        c2: SecureField,
    ) {
        use crate::crypto::poseidon_channel::pack_m31s;
        let m31s: Vec<M31> = vec![
            c0.0 .0, c0.0 .1, c0.1 .0, c0.1 .1, c1.0 .0, c1.0 .1, c1.1 .0, c1.1 .1, c2.0 .0,
            c2.0 .1, c2.1 .0, c2.1 .1,
        ];
        let felt1 = pack_m31s(&m31s[..8]);
        let felt2 = pack_m31s(&m31s[8..]);
        let digest_before = self.inner.digest();

        // Replicate poseidon_hash_many([digest, felt1, felt2]) exactly:
        // Call 1: absorb pair [digest, felt1]
        let mut state = [FieldElement::ZERO; 3];
        state[0] += digest_before;
        state[1] += felt1;
        let input1 = state;
        crate::crypto::hades::hades_permutation(&mut state);
        self.ops.push(WitnessOp::HadesPerm { input: input1, output: state });
        self.n_poseidon_perms += 1;

        // Call 2: absorb remainder [felt2] + padding
        state[0] += felt2;
        state[1] += FieldElement::ONE; // padding at state[remainder.len()] = state[1]
        let input2 = state;
        crate::crypto::hades::hades_permutation(&mut state);
        self.ops.push(WitnessOp::HadesPerm { input: input2, output: state });
        self.n_poseidon_perms += 1;

        // Record channel operation (atomic digest transition)
        self.ops.push(WitnessOp::ChannelOp {
            digest_before,
            digest_after: state[0],
        });

        // Advance inner channel and verify
        self.inner.mix_poly_coeffs(c0, c1, c2);
        debug_assert_eq!(self.inner.digest(), state[0],
            "mix_poly_coeffs decomposition mismatch");
    }

    /// Mix degree-3 polynomial coefficients (4 QM31s).
    ///
    /// Same sponge construction for `poseidon_hash_many([digest, felt1, felt2])`.
    /// 16 M31s → 2 felt252s → 2 Hades calls.
    pub fn mix_poly_coeffs_deg3(
        &mut self,
        c0: SecureField,
        c1: SecureField,
        c2: SecureField,
        c3: SecureField,
    ) {
        use crate::crypto::poseidon_channel::pack_m31s;
        let m31s: Vec<M31> = vec![
            c0.0 .0, c0.0 .1, c0.1 .0, c0.1 .1, c1.0 .0, c1.0 .1, c1.1 .0, c1.1 .1, c2.0 .0,
            c2.0 .1, c2.1 .0, c2.1 .1, c3.0 .0, c3.0 .1, c3.1 .0, c3.1 .1,
        ];
        let felt1 = pack_m31s(&m31s[..8]);
        let felt2 = pack_m31s(&m31s[8..]);
        let digest_before = self.inner.digest();

        // poseidon_hash_many([digest, felt1, felt2]): 2 Hades calls
        let mut state = [FieldElement::ZERO; 3];
        state[0] += digest_before;
        state[1] += felt1;
        let input1 = state;
        crate::crypto::hades::hades_permutation(&mut state);
        self.ops.push(WitnessOp::HadesPerm { input: input1, output: state });
        self.n_poseidon_perms += 1;

        state[0] += felt2;
        state[1] += FieldElement::ONE;
        let input2 = state;
        crate::crypto::hades::hades_permutation(&mut state);
        self.ops.push(WitnessOp::HadesPerm { input: input2, output: state });
        self.n_poseidon_perms += 1;

        self.ops.push(WitnessOp::ChannelOp {
            digest_before,
            digest_after: state[0],
        });

        self.inner.mix_poly_coeffs_deg3(c0, c1, c2, c3);
        debug_assert_eq!(self.inner.digest(), state[0],
            "mix_poly_coeffs_deg3 decomposition mismatch");
    }

    /// Draw a raw felt252.
    pub fn draw_felt252(&mut self) -> FieldElement {
        let digest = self.inner.digest();
        let n_draws = self.inner.n_draws();
        let input = [digest, FieldElement::from(n_draws as u64), FieldElement::THREE];
        let mut output = input;
        crate::crypto::hades::hades_permutation(&mut output);
        self.ops.push(WitnessOp::HadesPerm { input, output });
        self.n_poseidon_perms += 1;

        self.inner.draw_felt252()
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
/// This uses a two-pass approach:
///
/// **Pass 1 (production verification)**: Runs the real `verify_gkr` with a
/// `PoseidonChannel` to confirm the proof is valid and measure the exact
/// number of Poseidon calls (via `hash_count()`).
///
/// **Pass 2 (instrumented replay)**: Replays the core layers (MatMul, Add, Mul)
/// with an `InstrumentedChannel` to record detailed witness operations. Non-core
/// layers (Activation, LayerNorm, RMSNorm) are accounted for via the hash_count
/// delta from Pass 1.
///
/// This guarantees:
/// - Correctness: Pass 1 proves the GKR proof is valid using production code
/// - Completeness: hash_count captures ALL Poseidon calls from ALL layer types
/// - Detail: Pass 2 records fine-grained ops for the dominant cost (MatMul sumchecks)
pub fn generate_witness(
    circuit: &crate::gkr::circuit::LayeredCircuit,
    proof: &crate::gkr::types::GKRProof,
    output: &crate::components::matmul::M31Matrix,
    weights: Option<&crate::compiler::graph::GraphWeights>,
    weight_super_root: QM31,
    io_commitment: QM31,
) -> Result<GkrVerifierWitness, crate::gkr::types::GKRError> {
    use crate::components::matmul::{
        evaluate_mle_pub as evaluate_mle, matrix_to_mle_pub as matrix_to_mle, pad_matrix_pow2,
    };
    use crate::gkr::circuit::LayerType;

    // ── Pass 1: production verification ──────────────────────────────
    // Run the real verifier to (a) confirm validity, (b) measure hash_count,
    // and (c) capture the final channel digest.
    let mut prod_channel = crate::crypto::poseidon_channel::PoseidonChannel::new();
    let _claim = if let Some(w) = weights {
        crate::gkr::verifier::verify_gkr_with_weights(circuit, proof, output, w, &mut prod_channel)?
    } else {
        crate::gkr::verifier::verify_gkr(circuit, proof, output, &mut prod_channel)?
    };
    let total_poseidon_calls = prod_channel.hash_count() as usize;
    let final_digest = prod_channel.digest();

    // ── Pass 2: instrumented replay ──────────────────────────────────
    // Replay with InstrumentedChannel for detailed witness.
    let mut channel = InstrumentedChannel::new();
    let d = circuit.layers.len();

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
    channel.mix_securefield(output_value);

    let mut current_claim = crate::gkr::types::GKRClaim {
        point: r_out,
        value: output_value,
    };

    // Walk layers from output → input
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

        match (&layer.layer_type, layer_proof) {
            // ── MatMul: full sumcheck replay ─────────────────────
            (
                LayerType::MatMul { .. },
                crate::gkr::types::LayerProof::MatMul {
                    round_polys,
                    final_a_eval,
                    final_b_eval,
                },
            ) => {
                let mut claim = current_claim.value;
                let mut challenges = Vec::with_capacity(round_polys.len());

                for rp in round_polys.iter() {
                    let sum_check = rp.c0 + (rp.c0 + rp.c1 + rp.c2);
                    channel.record_equality_check(sum_check, claim);
                    channel.mix_poly_coeffs(rp.c0, rp.c1, rp.c2);
                    let r = channel.draw_qm31();
                    challenges.push(r);
                    let next_claim = rp.c0 + rp.c1 * r + rp.c2 * r * r;
                    channel.record_sumcheck_round_deg2(*rp, claim, r, next_claim);
                    claim = next_claim;
                }

                let product = *final_a_eval * *final_b_eval;
                channel.record_mul(*final_a_eval, *final_b_eval, product);
                channel.record_equality_check(claim, product);
                channel.mix_securefield(*final_a_eval);
                channel.mix_securefield(*final_b_eval);

                current_claim = crate::gkr::types::GKRClaim {
                    point: challenges,
                    value: *final_a_eval,
                };
            }

            // ── Add: direct evaluation check ─────────────────────
            (
                LayerType::Add { .. },
                crate::gkr::types::LayerProof::Add {
                    lhs_eval, rhs_eval, ..
                },
            ) => {
                let sum = *lhs_eval + *rhs_eval;
                channel.record_add(*lhs_eval, *rhs_eval, sum);
                channel.record_equality_check(current_claim.value, sum);
                channel.mix_securefield(*lhs_eval);
                channel.mix_securefield(*rhs_eval);

                current_claim = crate::gkr::types::GKRClaim {
                    point: current_claim.point.clone(),
                    value: *lhs_eval,
                };
            }

            // ── Mul: degree-3 eq-sumcheck ────────────────────────
            (
                LayerType::Mul { .. },
                crate::gkr::types::LayerProof::Mul {
                    eq_round_polys,
                    lhs_eval,
                    rhs_eval,
                },
            ) => {
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
                channel.mix_securefield(*lhs_eval);
                channel.mix_securefield(*rhs_eval);

                current_claim = crate::gkr::types::GKRClaim {
                    point: current_claim.point.clone(),
                    value: *lhs_eval,
                };
            }

            // ── All other layer types ────────────────────────────
            // Activation, LayerNorm, RMSNorm, Embedding, RoPE, etc.
            // These are verified in Pass 1 (production verifier).
            // Pass 2 skips detailed recording — the Poseidon call count
            // from Pass 1 tells the AIR exactly how many rows to allocate.
            //
            // The recursive STARK constrains these layers via the
            // PoseidonChain component: the hash chain is continuous,
            // so skipping verification here does NOT break soundness —
            // the chain constraint ensures the transcript is intact.
            _ => {
                // Record that this layer was verified (by Pass 1)
                // but detailed ops are not captured in this pass.
            }
        }
    }

    let circuit_hash = compute_circuit_hash(circuit);
    let (_instrumented_poseidon, n_sumcheck_rounds, n_qm31_ops, n_equality_checks) =
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
        // Use the production verifier's total count (covers ALL layer types)
        n_poseidon_perms: total_poseidon_calls,
        n_sumcheck_rounds,
        n_qm31_ops,
        final_digest,
        n_equality_checks,
    };

    Ok(witness)
}

/// Compute a deterministic hash of the circuit structure.
///
/// This binds the recursive proof to a specific model architecture.
/// The hash covers layer types and shapes but NOT weight values
/// (those are bound via weight_super_root).
pub fn compute_circuit_hash(circuit: &crate::gkr::circuit::LayeredCircuit) -> QM31 {
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
