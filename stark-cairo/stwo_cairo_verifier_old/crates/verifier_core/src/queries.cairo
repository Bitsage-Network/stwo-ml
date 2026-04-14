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
        let domain_size: NonZero<u32> = pow2(log_domain_size).try_into().unwrap();
        let mut positions: Array<u32> = array![];
        let mut collected = 0_usize;
        while collected != n_queries {
            let mut random_words = channel.draw_u32s();
            for word in random_words {
                let (_, position) = DivRem::div_rem(*word, domain_size);
                positions.append(position);
                collected += 1;
                if collected == n_queries {
                    break;
                }
            }
        }

        // Sort positions using insertion sort (n_queries is small, typically 8-64).
        let mut sorted: Array<u32> = array![];
        let positions_span = positions.span();
        let mut i: usize = 0;
        while i < positions_span.len() {
            let val = *positions_span[i];
            // Find insertion point in sorted array
            let mut inserted = false;
            let mut new_sorted: Array<u32> = array![];
            let sorted_span = sorted.span();
            let mut j: usize = 0;
            while j < sorted_span.len() {
                let s = *sorted_span[j];
                if !inserted && val < s {
                    new_sorted.append(val);
                    inserted = true;
                }
                new_sorted.append(s);
                j += 1;
            };
            if !inserted {
                new_sorted.append(val);
            }
            sorted = new_sorted;
            i += 1;
        };

        // Deduplicate
        let mut deduped: Array<u32> = array![];
        let sorted_span = sorted.span();
        let mut k: usize = 0;
        while k < sorted_span.len() {
            let val = *sorted_span[k];
            if k == 0 || val != *sorted_span[k - 1] {
                deduped.append(val);
            }
            k += 1;
        };

        Queries { positions: deduped.span(), log_domain_size }
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
