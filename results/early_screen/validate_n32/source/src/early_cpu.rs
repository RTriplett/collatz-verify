//! Experimental CPU verification immediately after the GPU completion sieve.
//!
//! Unlike deferred tails, these states have not yet executed their first tail
//! double iteration. In particular, an initially small state must still execute
//! that iteration to preserve the reference checksum and cutoff convention.

use crate::arith::POW3;
use crate::tail::{step_until_below, TailContinuation};
use anyhow::{ensure, Result};
use std::sync::OnceLock;
use std::time::Instant;

/// Original scalar stepping, counting-sort plus scalar stepping, independent
/// affine lookup, initial shared lookup, and repeated shared transitions.
pub const SCALAR: u32 = 0;
pub const BUCKET_SCALAR: u32 = 1;
pub const TABLE: u32 = 2;
pub const GROUPED_TABLE: u32 = 3;
pub const REGROUPED_TABLE: u32 = 4;

#[derive(Debug, Default)]
pub struct EarlyResult {
    pub cases: u64,
    pub iters: u64,
    pub hash: u32,
    /// One origin for each flagged case, including duplicate input origins.
    pub flagged: Vec<u128>,
    pub grouping_seconds: f64,
    pub verification_seconds: f64,
    pub table_setup_seconds: f64,
    pub buckets: u64,
    pub jumps: u64,
    pub jumped_iterations: u64,
    pub scalar_iterations: u64,
    pub table_lookups: u64,
}

#[derive(Clone, Copy, Default)]
struct State {
    base: u128,
    number: u128,
}

#[derive(Clone, Copy, Default, Debug)]
struct Prefix {
    shift: u32,
    multiplier: u32,
    add: u32,
}

#[derive(Clone, Copy, Default, Debug)]
struct Row {
    count: usize,
    prefix: [Prefix; 3],
    // For y = Q*2^width + residue, every known endpoint/odd-run peak
    // is exactly a*Q+b. Separate extrema give conservative integer bounds.
    min_endpoint_a: u64,
    min_endpoint_b: u64,
    max_peak_a: u64,
    max_peak_b: u64,
}

static TABLES: [OnceLock<Vec<Row>>; 4] = [const { OnceLock::new() }; 4];

fn table_slot(width: u32) -> Result<usize> {
    ensure!(
        matches!(width, 8 | 10 | 12 | 14),
        "early CPU grouping width must be 8, 10, 12, or 14"
    );
    Ok(((width - 8) / 2) as usize)
}

fn known_ctz(value: u64, known: u32, cap: u32) -> Option<u32> {
    if known == 0 {
        return None;
    }
    let inspect = known.min(cap);
    let low = value & ((1u64 << inspect) - 1);
    if low != 0 {
        Some(low.trailing_zeros())
    } else if known >= cap {
        Some(cap)
    } else {
        None
    }
}

fn build_table(width: u32) -> Vec<Row> {
    (0..1u64 << width)
        .map(|residue| {
            // Q=1 avoids a zero synthetic state. Subtracting the Q coefficient
            // then gives the exact nonnegative additive term for every Q.
            let seed = (1u64 << width) + residue;
            let mut sample = seed;
            let (mut shift, mut multiplier) = (0u32, 1u32);
            let mut row = Row {
                min_endpoint_a: u64::MAX,
                min_endpoint_b: u64::MAX,
                ..Row::default()
            };
            while row.count < row.prefix.len() {
                let (mut next, mut d, mut p) = (sample, shift, multiplier);
                let (mut peak_a, mut peak_b) = (row.max_peak_a, row.max_peak_b);
                let mut known = true;
                for _ in 0..2 {
                    let Some(a) = known_ctz(next, width - d, 8) else {
                        known = false;
                        break;
                    };
                    let factor = POW3[a as usize] as u32;
                    next = (next >> a) * factor as u64;
                    d += a;
                    p *= factor;
                    let coefficient = (p as u64) << (width - d);
                    peak_a = peak_a.max(coefficient);
                    peak_b = peak_b.max(next - coefficient);
                    let Some(b) = known_ctz(next - 1, width - d, 31) else {
                        known = false;
                        break;
                    };
                    next = ((next - 1) >> b) + 1;
                    d += b;
                }
                if !known {
                    break;
                }
                let coefficient = (p as u64) << (width - d);
                row.min_endpoint_a = row.min_endpoint_a.min(coefficient);
                row.min_endpoint_b = row.min_endpoint_b.min(next - coefficient);
                row.max_peak_a = peak_a;
                row.max_peak_b = peak_b;
                row.prefix[row.count] = Prefix {
                    shift: d,
                    multiplier: p,
                    add: (next - p as u64 * (seed >> d)).try_into().unwrap(),
                };
                row.count += 1;
                sample = next;
                shift = d;
                multiplier = p;
            }
            row
        })
        .collect()
}

#[inline]
fn safe_jump(row: &Row, number: u128, base: u128, cutoff: u128, width: u32) -> bool {
    if row.count == 0 {
        return false;
    }
    let high = number >> width;
    // A failed checked bound merely selects ordinary exact stepping. No
    // intermediate cutoff peak or descent boundary is allowed to disappear.
    let Some(upper) = high
        .checked_mul(row.max_peak_a as u128)
        .and_then(|v| v.checked_add(row.max_peak_b as u128))
    else {
        return false;
    };
    if upper >= cutoff {
        return false;
    }
    // This lower product cannot overflow after the upper bound succeeded:
    // each endpoint is no greater than its preceding odd-run peak.
    high * row.min_endpoint_a as u128 + row.min_endpoint_b as u128 >= base
}

#[inline]
fn double_iteration(mut number: u128, cutoff: u128) -> (u128, bool) {
    let mut flagged = false;
    for _ in 0..2 {
        let a = ((number as u32) | (1 << 8)).trailing_zeros();
        number = POW3[a as usize] * (number >> a);
        if number >= cutoff {
            flagged = true;
            number = 2;
        }
        number -= 1;
        let b = ((number as u32) | (1 << 31)).trailing_zeros();
        number = (number >> b) + 1;
    }
    (number, flagged)
}

#[inline]
fn scalar(state: State, cutoff: u128, out: &mut EarlyResult) {
    let (hash, iters, flagged) = step_until_below(state.base, state.number, cutoff);
    out.hash ^= hash;
    out.iters += iters;
    out.scalar_iterations += iters;
    if flagged {
        out.flagged.push(state.base);
    }
}

#[inline]
fn advance(
    mut state: State,
    cutoff: u128,
    width: u32,
    row: &Row,
    out: &mut EarlyResult,
) -> Option<State> {
    if safe_jump(row, state.number, state.base, cutoff, width) {
        for prefix in &row.prefix[..row.count] {
            out.hash ^= ((state.number >> prefix.shift) as u32)
                .wrapping_mul(prefix.multiplier)
                .wrapping_add(prefix.add);
        }
        let last = row.prefix[row.count - 1];
        state.number = (state.number >> last.shift) * last.multiplier as u128 + last.add as u128;
        out.iters += row.count as u64;
        out.jumped_iterations += row.count as u64;
        out.jumps += 1;
        Some(state)
    } else {
        let (next, flagged) = double_iteration(state.number, cutoff);
        state.number = next;
        out.hash ^= next as u32;
        out.iters += 1;
        out.scalar_iterations += 1;
        if flagged {
            out.flagged.push(state.base);
        }
        (next >= state.base).then_some(state)
    }
}

fn affine(
    mut state: State,
    cutoff: u128,
    width: u32,
    table: &[Row],
    mut first: Option<&Row>,
    out: &mut EarlyResult,
) {
    let mask = table.len() - 1;
    loop {
        let row = if let Some(row) = first.take() {
            row
        } else {
            out.table_lookups += 1;
            &table[state.number as usize & mask]
        };
        let Some(next) = advance(state, cutoff, width, row, out) else {
            return;
        };
        state = next;
    }
}

/// Verify exact post-sieve states with one of the five experimental methods.
///
/// Modes 1 and 3 counting-sort bounded chunks by their *initial current*
/// iterate, not the original starting value. Mode 3 shares the initial row;
/// later transitions use per-state lookups. Mode 4 regroups unfinished states
/// after every shared transition, including all those sorting passes in time.
/// Table construction is cached and timed separately from steady-state work.
pub fn verify_states(
    states: &[TailContinuation],
    cutoff_bits: u32,
    width: u32,
    mode: u32,
) -> Result<EarlyResult> {
    verify_states_chunked(states, cutoff_bits, width, mode, 65_536)
}

/// The same verifier with an explicit bound on each grouping batch.
pub fn verify_states_chunked(
    states: &[TailContinuation],
    cutoff_bits: u32,
    width: u32,
    mode: u32,
    chunk_size: usize,
) -> Result<EarlyResult> {
    ensure!(
        mode <= REGROUPED_TABLE,
        "unknown early CPU verification mode"
    );
    ensure!(chunk_size > 0, "early CPU grouping chunk must not be empty");
    ensure!(
        (97..=123).contains(&cutoff_bits),
        "early CPU cutoff must be 97..=123"
    );
    let mut out = EarlyResult {
        cases: states.len() as u64,
        ..EarlyResult::default()
    };
    let slot = if mode != SCALAR {
        Some(table_slot(width)?)
    } else {
        None
    };
    let setup = Instant::now();
    let table = if mode >= TABLE {
        Some(
            TABLES[slot.unwrap()]
                .get_or_init(|| build_table(width))
                .as_slice(),
        )
    } else {
        None
    };
    out.table_setup_seconds = if table.is_some() {
        setup.elapsed().as_secs_f64()
    } else {
        0.0
    };
    let cutoff = 1u128 << cutoff_bits;
    let grouped = mode == BUCKET_SCALAR || mode == GROUPED_TABLE || mode == REGROUPED_TABLE;
    let mut counts = if grouped {
        vec![0usize; 1usize << width]
    } else {
        Vec::new()
    };
    let mut starts = counts.clone();
    let mut positions = counts.clone();
    let mut sorted = Vec::<State>::new();
    for chunk in states.chunks(chunk_size) {
        let prepare = Instant::now();
        let mut decoded: Vec<State> = chunk
            .iter()
            .map(|record| State {
                base: record.base(),
                number: record.number(),
            })
            .collect();
        ensure!(
            decoded
                .iter()
                .all(|s| s.base >= 3 && s.number > 0 && s.number < 1u128 << 123),
            "invalid early CPU state or arithmetic headroom"
        );
        if grouped {
            out.grouping_seconds += prepare.elapsed().as_secs_f64();
            while !decoded.is_empty() {
                let prepare = Instant::now();
                counts.fill(0);
                let mask = counts.len() - 1;
                for state in &decoded {
                    counts[state.number as usize & mask] += 1;
                }
                let mut next = 0;
                for ((start, position), &count) in
                    starts.iter_mut().zip(&mut positions).zip(&counts)
                {
                    *start = next;
                    *position = next;
                    next += count;
                }
                sorted.resize(decoded.len(), State::default());
                for &state in &decoded {
                    let key = state.number as usize & mask;
                    sorted[positions[key]] = state;
                    positions[key] += 1;
                }
                out.grouping_seconds += prepare.elapsed().as_secs_f64();
                let verify = Instant::now();
                decoded.clear();
                for (key, (&start, &count)) in starts.iter().zip(&counts).enumerate() {
                    if count == 0 {
                        continue;
                    }
                    out.buckets += 1;
                    if let Some(table) = table {
                        let row = &table[key];
                        out.table_lookups += 1;
                        for &state in &sorted[start..start + count] {
                            if mode == REGROUPED_TABLE {
                                if let Some(next) = advance(state, cutoff, width, row, &mut out) {
                                    decoded.push(next);
                                }
                            } else {
                                affine(state, cutoff, width, table, Some(row), &mut out);
                            }
                        }
                    } else {
                        for &state in &sorted[start..start + count] {
                            scalar(state, cutoff, &mut out);
                        }
                    }
                }
                out.verification_seconds += verify.elapsed().as_secs_f64();
            }
        } else {
            // Decoding/validation belongs to the measured processing cost in
            // every mode; grouped modes include it in their preparation cost.
            if let Some(table) = table {
                for state in decoded {
                    affine(state, cutoff, width, table, None, &mut out);
                }
            } else {
                for state in decoded {
                    scalar(state, cutoff, &mut out);
                }
            }
            out.verification_seconds += prepare.elapsed().as_secs_f64();
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn oracle(states: &[TailContinuation], cutoff: u32) -> (u64, u32, Vec<u128>) {
        let (mut iters, mut hash, mut flags) = (0, 0, Vec::new());
        for record in states {
            let (h, n, f) = step_until_below(record.base(), record.number(), 1u128 << cutoff);
            iters += n;
            hash ^= h;
            if f {
                flags.push(record.base());
            }
        }
        flags.sort_unstable();
        (iters, hash, flags)
    }

    #[test]
    fn all_modes_match_scalar_for_wide_states_cutoffs_and_duplicates() {
        let mut seed = 0xf148_d671_8223_a5cbu64;
        let mut random = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for cutoff in [97, 104, 112, 120, 123] {
            let mut states = Vec::new();
            for i in 0..4096 {
                let mut number = (((random() as u128) << 64) | random() as u128)
                    & ((1u128 << cutoff.min(122)) - 1);
                number = number.max(4);
                match i % 8 {
                    0 => number = (1u128 << cutoff.min(122)) - 2 - (i & 127),
                    1 => number = (number >> 32) << 32,
                    2 => number = ((number >> 32) << 32) + 1,
                    3 => number = ((number >> 32) << 32) + u32::MAX as u128,
                    _ => {}
                }
                let base = match (i / 8) % 5 {
                    0 => number - 1,
                    1 => number + 1,
                    2 => number >> 1,
                    3 => number >> 20,
                    _ => 1u128 << 72,
                }
                .max(3);
                states.push(TailContinuation::from_values(base, number));
            }
            // Explicitly preserve duplicate flagged origins and the mandatory
            // first iteration of initially descended states.
            states.extend([TailContinuation::from_values(27, (1u128 << 122) - 2); 3]);
            states.extend([
                TailContinuation::from_values(100, 2),
                TailContinuation::from_values(27, 27),
            ]);
            let expected = oracle(&states, cutoff);
            for width in [8, 10, 12, 14] {
                for mode in 0..=4 {
                    let mut result = verify_states(&states, cutoff, width, mode).unwrap();
                    result.flagged.sort_unstable();
                    assert_eq!(
                        (result.iters, result.hash, result.flagged),
                        expected,
                        "cutoff={cutoff} width={width} mode={mode}"
                    );
                    assert_eq!(result.cases, states.len() as u64);
                    assert_eq!(
                        result.iters,
                        result.scalar_iterations + result.jumped_iterations
                    );
                    if mode >= TABLE {
                        assert!(result.jumps > 0);
                    }
                }
            }
        }
    }

    #[test]
    fn affine_rows_and_integer_bounds_cover_every_residue() {
        for width in [8, 10, 12, 14] {
            let table = build_table(width);
            for (residue, row) in table.iter().enumerate() {
                for high in [1u128, 3, 1 << 40, 1 << 100] {
                    let number = (high << width) + residue as u128;
                    let mut current = number;
                    for prefix in &row.prefix[..row.count] {
                        for _ in 0..2 {
                            let a = ((current as u32) | (1 << 8)).trailing_zeros();
                            current = POW3[a as usize] * (current >> a);
                            assert!(
                                current <= high * row.max_peak_a as u128 + row.max_peak_b as u128
                            );
                            current -= 1;
                            let b = ((current as u32) | (1 << 31)).trailing_zeros();
                            current = (current >> b) + 1;
                        }
                        assert_eq!(
                            current,
                            (number >> prefix.shift) * prefix.multiplier as u128
                                + prefix.add as u128
                        );
                        assert!(
                            current
                                >= high * row.min_endpoint_a as u128 + row.min_endpoint_b as u128
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn rejects_invalid_inputs_and_handles_empty_batches() {
        assert!(verify_states(&[], 120, 12, SCALAR)
            .unwrap()
            .flagged
            .is_empty());
        assert!(verify_states(&[], 96, 12, SCALAR).is_err());
        assert!(verify_states(&[], 120, 11, TABLE).is_err());
        assert!(verify_states(&[], 120, 12, 5).is_err());
        for record in [
            TailContinuation::from_values(2, 4),
            TailContinuation::from_values(27, 0),
            TailContinuation::from_values(27, 1u128 << 123),
        ] {
            assert!(verify_states(&[record], 120, 12, SCALAR).is_err());
        }
    }

    #[test]
    fn chunk_boundaries_keep_duplicate_cases_and_shared_lookup_accounting() {
        let states = vec![TailContinuation::from_values(27, 28); 65_541];
        let expected = oracle(&states, 120);
        let result = verify_states(&states, 120, 12, GROUPED_TABLE).unwrap();
        assert_eq!((result.iters, result.hash, result.flagged), expected);
        assert_eq!(result.buckets, 2);
        assert_eq!(result.cases, states.len() as u64);
    }
}
