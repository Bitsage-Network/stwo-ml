use core::dict::{Felt252Dict, Felt252DictEntryTrait, SquashedFelt252DictTrait};
use crate::channel::{Channel, ChannelTrait};
use crate::circle::CosetImpl;
use super::utils::{ArrayImpl, pow2};

/// An ordered set of query positions.
#[derive(Drop, Copy, Debug, PartialEq)]
pub struct Queries {
    /// Query positions sorted in ascending order.
    pub positions: Span<usize>,
    /// Size of the domain from which the queries were sampled.
    pub log_domain_size: u32,
}

#[generate_trait]
pub impl QueriesImpl of QueriesImplTrait {
    /// Returns an ascending list of query indices uniformly sampled over the range
    /// [0, 2^`log_query_size`).
    ///
    /// Panics if `log_domain_size` is >=32.
    fn generate(ref channel: Channel, log_domain_size: u32, n_queries: usize) -> Queries {
        // Sample unique query positions and sort them.
        // Uses a dict for deduplication but avoids squash().into_entries()
        // by tracking positions in a separate array.
        let mut positions_dict: Felt252Dict<felt252> = Default::default();
        let mut sampled_positions: Array<u32> = array![];
        let domain_size: NonZero<u32> = pow2(log_domain_size).try_into().unwrap();
        while sampled_positions.len() != n_queries {
            let mut random_words = channel.draw_u32s();
            for word in random_words {
                let (_, position) = DivRem::div_rem(*word, domain_size);
                // Check if already seen via dict (O(1) lookup).
                let (entry, prev) = positions_dict.entry(position.into());
                if prev == 0 {
                    // New position — mark as seen and record.
                    positions_dict = entry.finalize(1);
                    sampled_positions.append(position);
                } else {
                    // Already seen — skip (but still finalize entry).
                    positions_dict = entry.finalize(prev);
                }
                if sampled_positions.len() == n_queries {
                    break;
                }
            }
        }
        // Drop the dict without into_entries().
        let _squashed = positions_dict.squash();

        // Sort positions in ascending order (insertion sort, n_queries is small ~16-28).
        let mut sorted = sampled_positions;
        let len = sorted.len();
        // Simple selection sort for small arrays.
        let mut result: Array<u32> = array![];
        let mut used: Array<bool> = array![];
        let mut k: u32 = 0;
        while k < len {
            used.append(false);
            k += 1;
        };
        let mut i: u32 = 0;
        while i < len {
            let mut min_val: u32 = pow2(log_domain_size);
            let mut min_idx: u32 = 0;
            let mut j: u32 = 0;
            while j < len {
                if !*used[j] && *sorted[j] < min_val {
                    min_val = *sorted[j];
                    min_idx = j;
                };
                j += 1;
            };
            result.append(min_val);
            used = {
                let mut new_used: Array<bool> = array![];
                let mut m: u32 = 0;
                while m < len {
                    new_used.append(if m == min_idx { true } else { *used[m] });
                    m += 1;
                };
                new_used
            };
            i += 1;
        };

        Queries { positions: result.span(), log_domain_size }
    }

    fn len(self: @Queries) -> usize {
        (*self.positions).len()
    }

    /// Calculates the matching query indices in a folded domain (i.e each domain point is doubled)
    /// given `self` (the queries of the original domain) and the number of folds between domains.
    fn fold(self: Queries, n_folds: u32) -> Queries {
        Queries {
            positions: get_folded_query_positions(self.positions, n_folds),
            log_domain_size: self.log_domain_size - n_folds,
        }
    }
}

/// Returns a deduped list of folded query positions.
///
/// # Panics
///
/// Panics if query positions is empty.
pub fn get_folded_query_positions(mut query_positions: Span<usize>, n_folds: u32) -> Span<usize> {
    let folding_factor = pow2(n_folds);
    let mut prev_folded_position = *query_positions.pop_front().unwrap() / folding_factor;
    let mut folded_positions = array![prev_folded_position];

    for position in query_positions {
        let folded_position = *position / folding_factor;

        if folded_position != prev_folded_position {
            folded_positions.append(folded_position);
            prev_folded_position = folded_position;
        }
    }

    folded_positions.span()
}
