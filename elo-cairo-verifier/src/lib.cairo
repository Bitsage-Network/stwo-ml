pub mod field;
pub mod channel;
pub mod types;
pub mod sumcheck;
pub mod mle;
// pub mod gkr;  // stripped for lean v18b (not imported by contract)
// Stripped for lean deploy — not used by GKR verifier contract:
// pub mod ml_air;
// pub mod logup;
pub mod layer_verifiers;
pub mod model_verifier;
// pub mod audit;           // stripped for lean v18b
// pub mod access_control;  // stripped for lean v18b
// pub mod view_key;        // stripped for lean v18b
pub mod vm31_merkle;
// pub mod vm31_verifier;   // stripped for lean v32
// pub mod vm31_pool;       // stripped for lean v32
pub mod aggregated_binding;
// Legacy recursive verifier (v0.1.1 Air trait interface — needs updating for v1.2.2)
// pub mod recursive_air;
// pub mod recursive_hades_air;
// pub mod recursive_verifier;

// General-purpose STWO verifier (v1.2.2 — uses verify_cairo from stwo_cairo_air)
pub mod general_stwo_verifier;
pub mod firewall;
pub mod registry;
pub mod verifier;
