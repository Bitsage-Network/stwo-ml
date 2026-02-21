//! Helpers for converting sumcheck round polynomials to `ProofEvent` types.

use crate::events::{RoundPolyDeg3Viz, RoundPolyViz, SecureFieldMirror};

/// Convert a raw `[u32; 4]` (the four limbs of a QM31/SecureField) to our
/// mirror type. The layout follows STWO's QM31: `(a + bi) + (c + di)j` where
/// each letter is an M31 stored as a u32.
pub fn sf_mirror(a: u32, b: u32, c: u32, d: u32) -> SecureFieldMirror {
    SecureFieldMirror { a, b, c, d }
}

/// Build a `RoundPolyViz` from six raw u32 limbs (three QM31 coefficients).
pub fn round_poly_viz(
    c0: [u32; 4],
    c1: [u32; 4],
    c2: [u32; 4],
) -> RoundPolyViz {
    RoundPolyViz {
        c0: sf_mirror(c0[0], c0[1], c0[2], c0[3]),
        c1: sf_mirror(c1[0], c1[1], c1[2], c1[3]),
        c2: sf_mirror(c2[0], c2[1], c2[2], c2[3]),
    }
}

/// Build a `RoundPolyDeg3Viz` from four raw u32 limbs (four QM31 coefficients).
pub fn round_poly_deg3_viz(
    c0: [u32; 4],
    c1: [u32; 4],
    c2: [u32; 4],
    c3: [u32; 4],
) -> RoundPolyDeg3Viz {
    RoundPolyDeg3Viz {
        c0: sf_mirror(c0[0], c0[1], c0[2], c0[3]),
        c1: sf_mirror(c1[0], c1[1], c1[2], c1[3]),
        c2: sf_mirror(c2[0], c2[1], c2[2], c2[3]),
        c3: sf_mirror(c3[0], c3[1], c3[2], c3[3]),
    }
}

/// Normalize a raw QM31 `.a` limb to [0,1] for claim value approximation.
pub fn claim_approx(qm31_a_limb: u32) -> f32 {
    qm31_a_limb as f32 / (0x7fff_ffff_u32 as f32)
}

