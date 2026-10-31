//! Exact trajectory metrics for captured candidate starts.
//!
//! These functions replay each supplied start all the way to 1. They neither
//! search for additional candidates nor establish that a candidate is a global
//! path record among every smaller positive start. Values are serialized as
//! decimal strings so consumers do not round integers above 2^53.
//!
//! Both maps are reported: the shortcut map uses `(3*n+1)/2` on odd inputs,
//! whereas the standard map includes the intermediate `3*n+1`. Each maximum's
//! step index and each total step count use that map's own transitions.
//! `first_descent_step` counts shortcut transitions.

use crate::arith::U256;
use anyhow::{bail, ensure, Context, Result};
use serde::Serialize;
use std::collections::BTreeMap;

/// Abort an unfinished replay after this many shortcut transitions. Reaching
/// this limit is an error, never evidence that the trajectory converged.
pub const MAX_SHORTCUT_STEPS: u64 = 10_000_000;

/// Complete trajectory metrics for one start, stopping on its first visit to 1.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct PathMetrics {
    pub start: String,
    pub shortcut_maximum: String,
    pub standard_maximum: String,
    /// First shortcut step attaining the shortcut maximum; the start is step 0.
    pub shortcut_peak_step: u64,
    /// First ordinary 3*n+1 or n/2 transition attaining the standard maximum;
    /// the start is step 0 and an odd shortcut transition takes two such steps.
    pub standard_peak_step: u64,
    pub total_shortcut_steps: u64,
    pub total_standard_steps: u64,
    /// First shortcut transition ending below the original start. This is None
    /// for start 1, whose trajectory is already complete at step 0.
    pub first_descent_step: Option<u64>,
}

/// One unique captured candidate, with the number of times it was supplied.
/// The metrics do not certify that the candidate sets a global path record.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct PathCandidate {
    pub occurrences: u64,
    #[serde(flatten)]
    pub metrics: PathMetrics,
}

fn checked_add(a: U256, b: U256) -> Option<U256> {
    let (lo, carry) = a.lo.overflowing_add(b.lo);
    let hi = a.hi.checked_add(b.hi)?.checked_add(u128::from(carry))?;
    Some(U256 { hi, lo })
}

fn checked_standard_odd(n: U256) -> Option<U256> {
    let twice = checked_add(n, n)?;
    let thrice = checked_add(twice, n)?;
    checked_add(thrice, U256::from(1))
}

fn half(n: U256) -> U256 {
    U256 {
        hi: n.hi >> 1,
        lo: (n.lo >> 1) | (n.hi << 127),
    }
}

fn is_larger(a: U256, b: U256) -> bool {
    (a.hi, a.lo) > (b.hi, b.lo)
}

/// Four-limb long division keeps each intermediate below 10 * 2^64.
fn decimal(n: U256) -> String {
    let mut limbs = [
        (n.hi >> 64) as u64,
        n.hi as u64,
        (n.lo >> 64) as u64,
        n.lo as u64,
    ];
    if limbs.iter().all(|&x| x == 0) {
        return "0".to_string();
    }
    let mut digits = Vec::with_capacity(78);
    while limbs.iter().any(|&x| x != 0) {
        let mut remainder = 0u128;
        for limb in &mut limbs {
            let dividend = (remainder << 64) | u128::from(*limb);
            *limb = (dividend / 10) as u64;
            remainder = dividend % 10;
        }
        digits.push(b'0' + remainder as u8);
    }
    digits.reverse();
    String::from_utf8(digits).expect("decimal digits are valid UTF-8")
}

/// Replay a positive u128 start to 1 with checked 256-bit arithmetic.
///
/// Returns an error if the standard odd intermediate `3*n+1` would exceed
/// 256 bits, even when its halved shortcut successor could fit, or if the
/// generous transition limit is reached. No peak or convergence is fabricated
/// after either condition.
pub fn trajectory_metrics(start: u128) -> Result<PathMetrics> {
    trajectory_with_limit(start, MAX_SHORTCUT_STEPS)
}

fn trajectory_with_limit(start: u128, limit: u64) -> Result<PathMetrics> {
    ensure!(start > 0, "trajectory start must be positive");
    let initial = U256::from(start);
    let mut n = initial;
    let mut shortcut_maximum = initial;
    let mut standard_maximum = initial;
    let mut shortcut_peak_step = 0;
    let mut standard_peak_step = 0;
    let mut total_shortcut_steps = 0u64;
    let mut total_standard_steps = 0u64;
    let mut first_descent_step = None;
    while n != U256::from(1) {
        if total_shortcut_steps == limit {
            bail!("trajectory from {start} has not reached 1 after {limit} shortcut steps");
        }
        total_shortcut_steps = total_shortcut_steps
            .checked_add(1)
            .context("shortcut step counter overflow")?;
        total_standard_steps = total_standard_steps
            .checked_add(1)
            .context("standard step counter overflow")?;
        if n.lo & 1 == 1 {
            let intermediate = checked_standard_odd(n).with_context(|| format!(
                "standard 3*n+1 intermediate exceeds 256 bits for start {start} at shortcut step {total_shortcut_steps}"))?;
            if is_larger(intermediate, standard_maximum) {
                standard_maximum = intermediate;
                standard_peak_step = total_standard_steps;
            }
            n = half(intermediate);
            total_standard_steps = total_standard_steps
                .checked_add(1)
                .context("standard step counter overflow")?;
        } else {
            n = half(n);
        }
        if is_larger(n, shortcut_maximum) {
            shortcut_maximum = n;
            shortcut_peak_step = total_shortcut_steps;
        }
        if is_larger(n, standard_maximum) {
            standard_maximum = n;
            standard_peak_step = total_standard_steps;
        }
        if first_descent_step.is_none() && (n.hi, n.lo) < (initial.hi, initial.lo) {
            first_descent_step = Some(total_shortcut_steps);
        }
    }
    Ok(PathMetrics {
        start: start.to_string(),
        shortcut_maximum: decimal(shortcut_maximum),
        standard_maximum: decimal(standard_maximum),
        shortcut_peak_step,
        standard_peak_step,
        total_shortcut_steps,
        total_standard_steps,
        first_descent_step,
    })
}

/// Replay unique captured starts in ascending numeric order. Duplicate starts
/// contribute to `occurrences`, but their complete trajectory is replayed once.
/// Every supplied start must be positive and finish within the replay limits.
pub fn candidate_metrics(starts: &[u128]) -> Result<Vec<PathCandidate>> {
    let mut occurrences = BTreeMap::<u128, u64>::new();
    for &start in starts {
        ensure!(start > 0, "candidate start must be positive");
        let count = occurrences.entry(start).or_default();
        *count = count
            .checked_add(1)
            .context("candidate occurrence counter overflow")?;
    }
    occurrences
        .into_iter()
        .map(|(start, occurrences)| {
            Ok(PathCandidate {
                occurrences,
                metrics: trajectory_metrics(start)?,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Independent oracle: explicitly take the ordinary odd step and its
    // following halving, using checked u128 rather than two-limb arithmetic.
    fn narrow_oracle(start: u128) -> PathMetrics {
        let mut n = start;
        let (mut shortcut_maximum, mut standard_maximum) = (n, n);
        let (mut shortcut_peak_step, mut standard_peak_step, mut steps) = (0, 0, 0);
        let mut standard_steps = 0;
        let mut first_descent_step = None;
        while n != 1 {
            steps += 1;
            standard_steps += 1;
            let odd = n % 2 != 0;
            n = if odd {
                n.checked_mul(3).unwrap().checked_add(1).unwrap()
            } else {
                n / 2
            };
            if n > standard_maximum {
                standard_maximum = n;
                standard_peak_step = standard_steps;
            }
            if odd {
                n /= 2;
                standard_steps += 1;
            }
            if n > shortcut_maximum {
                shortcut_maximum = n;
                shortcut_peak_step = steps;
            }
            if n < start && first_descent_step.is_none() {
                first_descent_step = Some(steps);
            }
        }
        PathMetrics {
            start: start.to_string(),
            shortcut_maximum: shortcut_maximum.to_string(),
            standard_maximum: standard_maximum.to_string(),
            shortcut_peak_step,
            standard_peak_step,
            total_shortcut_steps: steps,
            total_standard_steps: standard_steps,
            first_descent_step,
        }
    }

    #[test]
    fn complete_metrics_match_independent_narrow_oracle() {
        for n in 1..=5000 {
            assert_eq!(trajectory_metrics(n).unwrap(), narrow_oracle(n), "n={n}");
        }
    }

    #[test]
    fn twenty_seven_reports_both_peak_conventions_and_full_tail() {
        let metrics = trajectory_metrics(27).unwrap();
        assert_eq!(metrics.shortcut_maximum, "4616");
        assert_eq!(metrics.standard_maximum, "9232");
        assert_eq!(metrics.shortcut_peak_step, 45);
        assert_eq!(metrics.standard_peak_step, 77);
        assert_eq!(metrics.total_shortcut_steps, 70);
        assert_eq!(metrics.total_standard_steps, 111);
        assert_eq!(metrics.first_descent_step, Some(59));
    }

    #[test]
    fn published_record_has_exact_large_peaks() {
        let metrics = trajectory_metrics(10_709_980_568_908_647).unwrap();
        assert_eq!(
            metrics.shortcut_maximum,
            "175294593968539094415936960141122"
        );
        assert_eq!(
            metrics.standard_maximum,
            "350589187937078188831873920282244"
        );
        assert_eq!(metrics.shortcut_peak_step, 298);
        assert_eq!(metrics.standard_peak_step, 519);
        assert_eq!(metrics.total_standard_steps, 1165);
        assert_eq!(metrics.first_descent_step, Some(485));
        assert_eq!(metrics, narrow_oracle(10_709_980_568_908_647));
    }

    #[test]
    fn trajectory_above_u128_finishes_exactly() {
        // Independent Python integer replay of 2^127-1 yields these metrics;
        // the standard peak needs 203 bits and exercises the high limb.
        let metrics = trajectory_metrics((1u128 << 127) - 1).unwrap();
        assert_eq!(
            metrics.shortcut_maximum,
            "3930061525912861057173624287137506221892737197425280369698986"
        );
        assert_eq!(
            metrics.standard_maximum,
            "7860123051825722114347248574275012443785474394850560739397972"
        );
        assert_eq!(metrics.shortcut_peak_step, 127);
        assert_eq!(metrics.standard_peak_step, 253);
        assert_eq!(metrics.total_shortcut_steps, 1067);
        assert_eq!(metrics.total_standard_steps, 1660);
        assert_eq!(metrics.first_descent_step, Some(468));
    }

    #[test]
    fn decimal_conversion_covers_full_256_bit_range() {
        assert_eq!(decimal(U256::from(0)), "0");
        for n in [1, 9, 10, 99, 100, u64::MAX as u128, 1u128 << 64, u128::MAX] {
            assert_eq!(decimal(U256::from(n)), n.to_string());
        }
        assert_eq!(
            decimal(U256 { hi: 1, lo: 0 }),
            "340282366920938463463374607431768211456"
        );
        assert_eq!(
            decimal(U256 {
                hi: u128::MAX,
                lo: u128::MAX
            }),
            "115792089237316195423570985008687907853269984665640564039457584007913129639935"
        );
    }

    #[test]
    fn checked_arithmetic_rejects_overflow_in_skipped_standard_intermediate() {
        assert_eq!(
            checked_add(U256::from(u128::MAX), U256::from(1)),
            Some(U256 { hi: 1, lo: 0 })
        );
        assert!(checked_add(
            U256 {
                hi: u128::MAX,
                lo: u128::MAX
            },
            U256::from(1)
        )
        .is_none());
        // 3*(2^255+1)+1 exceeds 256 bits, though its shortcut half fits.
        assert!(checked_standard_odd(U256 {
            hi: 1u128 << 127,
            lo: 1
        })
        .is_none());
        assert_eq!(half(U256 { hi: 1, lo: 0 }), U256::from(1u128 << 127));
    }

    #[test]
    fn boundaries_and_limits_are_explicit() {
        assert_eq!(trajectory_metrics(1).unwrap(), narrow_oracle(1));
        assert!(trajectory_metrics(0).is_err());
        assert!(trajectory_with_limit(1, 0).is_ok());
        assert!(trajectory_with_limit(2, 0).is_err());
        assert!(trajectory_with_limit(2, 1).is_ok());
        assert!(trajectory_with_limit(27, 69).is_err());
        assert!(trajectory_with_limit(27, 70).is_ok());
    }

    #[test]
    fn candidate_output_is_sorted_unique_and_counts_occurrences() {
        let candidates = candidate_metrics(&[27, 2, 10, 2, 1, 27, 27]).unwrap();
        let summary: Vec<_> = candidates
            .iter()
            .map(|c| (c.metrics.start.as_str(), c.occurrences))
            .collect();
        assert_eq!(summary, vec![("1", 1), ("2", 2), ("10", 1), ("27", 3)]);
        assert!(candidate_metrics(&[]).unwrap().is_empty());
        assert!(candidate_metrics(&[1, 0]).is_err());
        let json = serde_json::to_value(&candidates[3]).unwrap();
        assert_eq!(json["start"], "27");
        assert_eq!(json["occurrences"], 3);
        assert_eq!(json["standard_maximum"], "9232");
    }
}
