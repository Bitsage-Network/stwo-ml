//! Types for the ZKML transaction classifier.
//!
//! A small MLP classifier scores transaction intent as safe/suspicious/malicious.
//! The inference is proven with GKR+STARK, binding the score to a cryptographic
//! proof that verifies on-chain.

use starknet_ff::FieldElement;

/// Raw transaction features before encoding to M31.
///
/// These map to the 48-feature input vector consumed by the classifier MLP.
/// The encoding scheme splits addresses and values into 31-bit chunks
/// for M31 field arithmetic.
#[derive(Debug, Clone)]
pub struct TransactionFeatures {
    /// Target contract address (felt252 / 251 bits).
    pub target: FieldElement,
    /// Transaction value as two u128 halves (u256 = [high, low]).
    pub value: [u128; 2],
    /// Function selector (first 4 bytes of calldata).
    pub selector: u32,
    /// First 8 calldata words (after selector). Pad with 0 if shorter.
    pub calldata_prefix: [u32; 8],
    /// Total calldata length in bytes.
    pub calldata_len: u32,
    /// Agent's current trust score (0-100000, from on-chain EMA).
    pub agent_trust_score: u32,
    /// Agent's current strike count.
    pub agent_strikes: u32,
    /// Agent age in blocks since registration.
    pub agent_age_blocks: u32,
    /// Target contract metadata flags.
    pub target_flags: TargetFlags,
    /// Value-derived features.
    pub value_features: ValueFeatures,
    /// Selector-derived features (common function signatures).
    pub selector_features: SelectorFeatures,
    /// Behavioral features (recent activity patterns).
    pub behavioral: BehavioralFeatures,
}

/// Target contract metadata.
#[derive(Debug, Clone, Default)]
pub struct TargetFlags {
    /// Contract is verified on block explorer.
    pub is_verified: bool,
    /// Contract is a proxy (delegatecall pattern).
    pub is_proxy: bool,
    /// Contract has published source code.
    pub has_source: bool,
    /// Number of previous interactions with this target.
    pub interaction_count: u32,
}

/// Value-derived features.
#[derive(Debug, Clone, Default)]
pub struct ValueFeatures {
    /// log2(value + 1), quantized to u32.
    pub log2_value: u32,
    /// value / agent_balance ratio (0-100000 fixed point).
    pub value_balance_ratio: u32,
    /// Whether this is a max uint256 approval.
    pub is_max_approval: bool,
    /// Whether value is zero.
    pub is_zero_value: bool,
}

/// Selector-derived features (common function signature detection).
#[derive(Debug, Clone, Default)]
pub struct SelectorFeatures {
    /// Matches ERC20 transfer / transferFrom.
    pub is_transfer: bool,
    /// Matches ERC20 approve / increaseAllowance.
    pub is_approve: bool,
    /// Matches common DEX swap selectors (Uniswap, Sushi, etc).
    pub is_swap: bool,
    /// Selector not in known-good list.
    pub is_unknown: bool,
}

/// Behavioral features from recent agent activity.
#[derive(Debug, Clone, Default)]
pub struct BehavioralFeatures {
    /// Transactions per hour (recent 24h).
    pub tx_frequency: u32,
    /// Unique target addresses in last 24h.
    pub unique_targets_24h: u32,
    /// Average transaction value in last 24h (quantized).
    pub avg_value_24h: u32,
    /// Maximum transaction value in last 24h (quantized).
    pub max_value_24h: u32,
}

/// Classifier decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    /// Score 0-40000: routine, safe to execute.
    Approve,
    /// Score 40001-70000: suspicious, needs human review.
    Escalate,
    /// Score 70001-100000: probable attack, reject.
    Block,
}

impl Decision {
    /// Derive decision from threat score using ENShell-compatible thresholds.
    pub fn from_score(score: u32) -> Self {
        match score {
            0..=40000 => Decision::Approve,
            40001..=70000 => Decision::Escalate,
            _ => Decision::Block,
        }
    }
}

impl std::fmt::Display for Decision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Decision::Approve => write!(f, "approve"),
            Decision::Escalate => write!(f, "escalate"),
            Decision::Block => write!(f, "block"),
        }
    }
}

/// Result of running the classifier on a transaction.
#[derive(Debug, Clone)]
pub struct ClassifierResult {
    /// Threat score (0-100000), ENShell-compatible scale.
    pub threat_score: u32,
    /// Decision based on threat score thresholds.
    pub decision: Decision,
    /// Raw output scores: [safe, suspicious, malicious].
    pub scores: [u32; 3],
    /// IO commitment: Poseidon(input || output).
    pub io_commitment: FieldElement,
    /// Policy commitment from the proof.
    pub policy_commitment: FieldElement,
    /// Proof hash (for on-chain verification lookup).
    pub proof_hash: FieldElement,
    /// Proving time in milliseconds.
    pub prove_time_ms: u64,
}

/// Classifier error.
#[derive(Debug, thiserror::Error)]
pub enum ClassifierError {
    #[error("encoding error: {0}")]
    EncodingError(String),
    #[error("proving error: {0}")]
    ProvingError(String),
    #[error("model error: {0}")]
    ModelError(String),
}

/// Thresholds for classifier decisions.
pub const APPROVE_THRESHOLD: u32 = 40_000;
pub const ESCALATE_THRESHOLD: u32 = 70_000;

/// Classifier model input dimension (must be power of 2 for MLE alignment).
pub const INPUT_DIM: usize = 64;
/// Number of used features (rest is zero-padded).
pub const NUM_FEATURES: usize = 48;
/// Number of output classes.
pub const NUM_CLASSES: usize = 3;
