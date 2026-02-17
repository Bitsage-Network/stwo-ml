# Stage 3: Semantic Evaluation

**Status**: New Module
**Readiness**: 20% — Forward pass infrastructure exists, evaluation logic is new
**Depends on**: Nothing (runs in parallel with Stage 2)
**Blocks**: Stage 4 (Audit Report Format)

---

## Purpose

The ZK proof guarantees **computational integrity** — the model computed correctly. But it doesn't tell you whether the model's answer is *good*. Semantic evaluation adds a quality layer:

- Did the model answer the question that was asked?
- Is the answer factually reasonable?
- Did the model stay on-task or deviate?
- How confident should we be in the output?

This layer runs **in parallel with the prover** during an audit. It does not slow down proof generation. And critically, the evaluation itself is **another forward pass** — which means it can also be proved, creating a chain of verified evaluations.

---

## Evaluation Layers

### Layer 1: Deterministic Checks (Free, Instant)

For tasks with objectively verifiable outputs:

| Task Type | Check | Example |
|-----------|-------|---------|
| Math | Evaluate expression | "2+2" → 4, verify = true |
| Code generation | Parse + syntax check | Generated code compiles |
| Structured output | Schema validation | JSON matches expected schema |
| Classification | Label comparison | Output matches ground truth |
| SQL | Query parsing | SQL is syntactically valid |
| Regex/pattern | Match test | Generated regex matches test cases |

```rust
/// Deterministic evaluation result for a single inference.
pub struct DeterministicEval {
    /// Whether the output passes deterministic checks.
    pub passed: bool,
    /// Specific checks applied and their results.
    pub checks: Vec<DeterministicCheck>,
    /// Overall confidence (1.0 for pass, 0.0 for fail, 0.5 for partial).
    pub confidence: f32,
}

pub struct DeterministicCheck {
    pub check_type: String,     // "json_valid", "code_compiles", "math_correct"
    pub passed: bool,
    pub detail: Option<String>, // Error message if failed
}

/// Run deterministic checks on an inference.
///
/// Returns instantly — no model calls needed.
pub fn evaluate_deterministic(
    input_text: &str,
    output_text: &str,
    task_hint: Option<&str>,
) -> DeterministicEval;
```

### Layer 2: Self-Evaluation (Provable)

The model evaluates its own output. This is a second forward pass using an evaluation prompt:

```
System: You are evaluating the quality of an AI response.
Rate the following response on a scale of 0-10.

Input: {original_input}
Response: {original_output}

Score (0-10):
```

The self-evaluation produces a score. Since it's a forward pass through the same model, it can be proved with the same prover — creating a verified evaluation chain:

```
Original:     Input X → Model → Output Y     (proved)
Evaluation:   (X, Y) → Model → Score 8.5     (also proved)
```

Both proofs use the same weight commitment, so the verifier knows the same model produced both the answer and the evaluation.

```rust
/// Self-evaluation result.
pub struct SelfEvaluation {
    /// Quality score (0.0 - 1.0, normalized from model's 0-10 scale).
    pub quality_score: f32,
    /// Raw model output for the evaluation prompt.
    pub raw_evaluation: String,
    /// IO commitment of the evaluation forward pass.
    pub eval_io_commitment: FieldElement,
    /// Whether the evaluation was also proved.
    pub evaluation_proved: bool,
    /// Evaluation prompt used (for reproducibility).
    pub prompt_template: String,
}

/// Run self-evaluation on a batch of inferences.
///
/// For each inference, constructs an evaluation prompt and runs it through
/// the same model. Returns scores + io_commitments for proving.
pub fn evaluate_self(
    model: &OnnxModel,
    inferences: &[(String, String)],  // (input, output) pairs
) -> Vec<SelfEvaluation>;
```

### Layer 3: Cross-Model Verification (Strongest)

Use a second model to evaluate the first model's output. Like having two independent auditors:

```
Model A:     Input X → Output Y      (proved with A's weights)
Model B:     (X, Y) → "Correct"      (proved with B's weights)
```

Both proofs are independently verifiable. Different weight commitments prove different models were used.

```rust
/// Cross-model evaluation result.
pub struct CrossModelEval {
    /// Whether the evaluator model agrees the output is correct.
    pub agrees: bool,
    /// Quality score from evaluator model (0.0 - 1.0).
    pub quality_score: f32,
    /// Evaluator model identifier.
    pub evaluator_model_id: FieldElement,
    /// Evaluator model weight commitment.
    pub evaluator_weight_commitment: FieldElement,
    /// IO commitment of the evaluation pass.
    pub eval_io_commitment: FieldElement,
}
```

---

## Evaluation Prompt Templates

### General Quality

```
Rate the quality of this AI response from 0 to 10.
Consider: relevance, accuracy, completeness, clarity.

User question: {input}
AI response: {output}

Score (just the number):
```

### Task Adherence

```
Did this AI response directly address the user's request?
Answer YES or NO, then explain briefly.

User request: {input}
AI response: {output}

Answer:
```

### Factual Consistency

```
Does this response contain any statements that contradict each other
or that are obviously factually incorrect?
Answer YES (contains errors) or NO (appears consistent).

Response: {output}

Answer:
```

### Code Quality

```
Rate this code on a scale of 0-10 for:
- Correctness (does it do what was asked?)
- Style (is it clean and idiomatic?)
- Completeness (does it handle edge cases?)

Request: {input}
Code: {output}

Score (just the number):
```

---

## Provability

The evaluation forward pass uses the same infrastructure as the original inference:

1. Tokenize evaluation prompt → input tokens
2. Quantize to M31 → `input_m31`
3. Forward pass through model → `output_m31`
4. Extract score from output tokens
5. Compute `eval_io_commitment = compute_io_commitment(eval_input, eval_output)`
6. Optionally prove with same prover

The evaluation proof uses the **same weight commitment** as the original inference proof. This is critical — it proves the evaluation was done by the same model, not a cheaper substitute.

```rust
/// Prove a batch of self-evaluations.
///
/// Each evaluation is a forward pass that can be proved alongside
/// the original inferences in the same audit batch.
pub fn prove_evaluations(
    prover: &AuditProver,
    evaluations: &[SelfEvaluation],
) -> Result<Vec<InferenceProof>, AuditError>;
```

---

## Aggregate Scoring

The audit report includes aggregate scores across all inferences:

```rust
/// Aggregated semantic evaluation for an audit window.
pub struct AuditSemanticSummary {
    /// Evaluation method used.
    pub method: EvalMethod,
    /// Average quality score across all inferences.
    pub avg_quality_score: f32,
    /// Score distribution.
    pub distribution: ScoreDistribution,
    /// Number of inferences evaluated.
    pub evaluated_count: u32,
    /// Number of deterministic checks passed.
    pub deterministic_pass_count: u32,
    /// Number of deterministic checks failed.
    pub deterministic_fail_count: u32,
    /// Whether evaluations were proved.
    pub evaluations_proved: bool,
    /// Merkle root of evaluation io_commitments.
    pub eval_merkle_root: Option<FieldElement>,
}

pub enum EvalMethod {
    Deterministic,
    SelfEvaluation,
    CrossModel { evaluator_model: String },
    Combined,
}

pub struct ScoreDistribution {
    pub excellent: u32,  // 0.9 - 1.0
    pub good: u32,       // 0.7 - 0.9
    pub fair: u32,       // 0.5 - 0.7
    pub poor: u32,       // 0.0 - 0.5
}
```

---

## Performance

| Evaluation Type | Time per Inference | Parallelizable? |
|----------------|-------------------|-----------------|
| Deterministic | ~0.1ms | Yes (CPU) |
| Self-evaluation (no proof) | ~100ms (forward pass) | Yes (GPU) |
| Self-evaluation (with proof) | ~1s (forward pass + prove) | Yes (GPU) |
| Cross-model | ~200ms (different model forward pass) | Yes (separate GPU) |

For an audit of 150 inferences:
- Deterministic only: ~15ms total
- Self-evaluation (no proof): ~15s total (parallelized across GPU)
- Self-evaluation (with proof): ~2.5min total

Self-evaluation runs **in parallel with the main audit prover** — it does not add to the critical path. Both finish around the same time.

---

## Computer Vision Support

The same evaluation framework works for CV models:

| CV Task | Evaluation | Method |
|---------|-----------|--------|
| Image classification | Top-K accuracy | Deterministic |
| Object detection | IoU / mAP | Deterministic |
| Segmentation | Dice score | Deterministic |
| Image generation | FID score | Cross-model |
| Caption generation | BLEU/ROUGE | Deterministic |

For CV, the "input" is the image tensor (quantized to M31) and the "output" is the classification logits or detection boxes. The io_commitment binds both.

---

## Files to Create/Modify

| File | Action | Lines |
|------|--------|-------|
| `src/audit/evaluation.rs` | **New** | ~400 (evaluation types + logic) |
| `src/audit/prompts.rs` | **New** | ~100 (evaluation prompt templates) |
| `src/audit/deterministic.rs` | **New** | ~200 (deterministic checks) |

---

## Verification Criteria

- [ ] Deterministic checks run in < 1ms per inference
- [ ] Self-evaluation produces consistent scores for same input
- [ ] Self-evaluation io_commitment is verifiable with same model
- [ ] Aggregate scores correctly reflect individual scores
- [ ] Evaluation proofs use same weight commitment as original inference proofs
- [ ] Evaluation runs in parallel with main prover (no critical-path latency)
- [ ] Score distribution accurately represents individual scores
