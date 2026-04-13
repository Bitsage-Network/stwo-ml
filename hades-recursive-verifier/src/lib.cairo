/// Hades Recursive Verifier
///
/// Verifies that a batch of Hades permutation (input, output) pairs are correct.
/// Each pair is checked by calling Cairo's native `hades_permutation` built-in
/// and asserting the result matches the claimed output.
///
/// The STWO Cairo prover (cairo-prove) generates a STARK proof that this program
/// executed correctly. This proof attests that ALL Hades permutations in the batch
/// were computed correctly -- with full 91-round S-box + MDS verification.
///
/// This is Level 1 of the two-level recursive STARK architecture:
///   Level 1: This proof (Hades correctness)
///   Level 2: Chain STARK (digest chain integrity + binds to Level 1)

use core::poseidon::hades_permutation;
use core::poseidon::poseidon_hash_span;

/// Verify all Hades permutation pairs and return a binding commitment.
///
/// Arguments (as felt252 array):
///   [0]        n_pairs: number of permutation pairs
///   [1..7)     pair 0: (in0, in1, in2, out0, out1, out2)
///   [7..13)    pair 1: ...
///   ...
///
/// The commitment is Poseidon hash of all (input, output) values,
/// used to bind this proof to the chain STARK.
#[executable]
fn main(input: Array<felt252>) -> Array<felt252> {
    let mut span = input.span();

    // Read number of pairs
    let n_pairs: u32 = (*span.pop_front().unwrap()).try_into().unwrap();

    // Verify each pair
    let mut commitment_data: Array<felt252> = array![];
    commitment_data.append(n_pairs.into());

    let mut i: u32 = 0;
    loop {
        if i >= n_pairs {
            break;
        }

        // Read input state
        let in0: felt252 = *span.pop_front().unwrap();
        let in1: felt252 = *span.pop_front().unwrap();
        let in2: felt252 = *span.pop_front().unwrap();

        // Read expected output state
        let expected_out0: felt252 = *span.pop_front().unwrap();
        let expected_out1: felt252 = *span.pop_front().unwrap();
        let expected_out2: felt252 = *span.pop_front().unwrap();

        // Execute Hades permutation (Cairo built-in: full 91-round verification)
        let (actual_out0, actual_out1, actual_out2) = hades_permutation(in0, in1, in2);

        // Assert correctness
        assert!(actual_out0 == expected_out0, "Hades output[0] mismatch at pair {}", i);
        assert!(actual_out1 == expected_out1, "Hades output[1] mismatch at pair {}", i);
        assert!(actual_out2 == expected_out2, "Hades output[2] mismatch at pair {}", i);

        // Accumulate for commitment
        commitment_data.append(in0);
        commitment_data.append(in1);
        commitment_data.append(in2);
        commitment_data.append(expected_out0);
        commitment_data.append(expected_out1);
        commitment_data.append(expected_out2);

        i += 1;
    };

    // Compute binding commitment: Poseidon hash of all verified pairs
    let commitment = poseidon_hash_span(commitment_data.span());

    // Return the commitment (used by chain STARK for binding)
    array![commitment, n_pairs.into()]
}
