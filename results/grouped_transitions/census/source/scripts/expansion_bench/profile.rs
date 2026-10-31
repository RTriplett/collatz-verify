//! Untimed diagnostics for the ternary screening experiment.
//!
//! Traversal always follows the production cull decisions. Counters measure
//! work at nodes that have already passed OEE, descent, and pm1. In particular,
//! they do not treat an impossible pm_inf certificate as permission to skip
//! a mixed-word or walkback rule.
use cv::arith::{slope_descends, weight_le_pow2k};
use cv::rules::{Cull, Rules};
use cv::tree::{Node, P16};
use serde_json::{json, Value};

const INV3: u128 = 0xAAAA_AAAA_AAAA_AAAA_AAAA_AAAA_AAAA_AAAB;
const THIRD: u128 = u128::MAX / 3;
const NEVER: usize = 81;

#[derive(Clone)]
struct Counters {
    oee_culls: u64,
    generated: u64,
    descent_culls: u64,
    pm1_culls: u64,
    after_baseline: u64,
    pm_calls: u64,
    pm_culls: u64,
    mix_culls: u64,
    walkback_culls: u64,
    survived: u64,
    endpoints: u64,
    m3_mod3_not2: u64,
    residue_fallback: u64,
    screened_out: u64,
    eligible: u64,
    eligible_without_cull: u64,
    full_width_loop_attempts: u64,
    full_width_successful_divisions: u64,
    avoided_loop_attempts: u64,
    avoided_successful_divisions: u64,
    repeated_divisions_on_eligible: u64,
    residue_loop_attempts: u64,
    residue_successful_divisions: u64,
    min_slope_t: [u64; 82],
    capped_full_width_v3: [u64; 81],
    residue_v3: [u64; 17],
    eligible_by_min_t: [u64; 82],
    pm_culls_by_min_t: [u64; 82],
    min_t_and_residue_v3: [[u64; 17]; 82],
}

impl Default for Counters {
    fn default() -> Self {
        Self {
            oee_culls: 0,
            generated: 0,
            descent_culls: 0,
            pm1_culls: 0,
            after_baseline: 0,
            pm_calls: 0,
            pm_culls: 0,
            mix_culls: 0,
            walkback_culls: 0,
            survived: 0,
            endpoints: 0,
            m3_mod3_not2: 0,
            residue_fallback: 0,
            screened_out: 0,
            eligible: 0,
            eligible_without_cull: 0,
            full_width_loop_attempts: 0,
            full_width_successful_divisions: 0,
            avoided_loop_attempts: 0,
            avoided_successful_divisions: 0,
            repeated_divisions_on_eligible: 0,
            residue_loop_attempts: 0,
            residue_successful_divisions: 0,
            min_slope_t: [0; 82],
            capped_full_width_v3: [0; 81],
            residue_v3: [0; 17],
            eligible_by_min_t: [0; 82],
            pm_culls_by_min_t: [0; 82],
            min_t_and_residue_v3: [[0; 17]; 82],
        }
    }
}

impl Counters {
    fn add(&mut self, other: &Self) {
        macro_rules! add_scalars {
            ($($field:ident),* $(,)?) => { $(self.$field += other.$field;)* };
        }
        add_scalars!(
            oee_culls,
            generated,
            descent_culls,
            pm1_culls,
            after_baseline,
            pm_calls,
            pm_culls,
            mix_culls,
            walkback_culls,
            survived,
            endpoints,
            m3_mod3_not2,
            residue_fallback,
            screened_out,
            eligible,
            eligible_without_cull,
            full_width_loop_attempts,
            full_width_successful_divisions,
            avoided_loop_attempts,
            avoided_successful_divisions,
            repeated_divisions_on_eligible,
            residue_loop_attempts,
            residue_successful_divisions,
        );
        for t in 0..82 {
            self.min_slope_t[t] += other.min_slope_t[t];
            self.eligible_by_min_t[t] += other.eligible_by_min_t[t];
            self.pm_culls_by_min_t[t] += other.pm_culls_by_min_t[t];
            for v in 0..17 {
                self.min_t_and_residue_v3[t][v] += other.min_t_and_residue_v3[t][v];
            }
        }
        for t in 0..81 {
            self.capped_full_width_v3[t] += other.capped_full_width_v3[t];
        }
        for t in 0..17 {
            self.residue_v3[t] += other.residue_v3[t];
        }
    }

    fn json(&self) -> Value {
        // Sparse histogram rows keep the per-depth report readable. A residue
        // valuation of 16 means at least 16; NEVER means no t <= f can pass.
        let histogram = |values: &[u64]| {
            values
                .iter()
                .enumerate()
                .filter(|(_, count)| **count != 0)
                .map(|(t, count)| json!({"t": t, "count": count}))
                .collect::<Vec<_>>()
        };
        let joint = self
            .min_t_and_residue_v3
            .iter()
            .enumerate()
            .flat_map(|(min_t, row)| {
                row.iter()
                    .enumerate()
                    .filter(|(_, count)| **count != 0)
                    .map(move |(v3, count)| {
                        json!({"min_t": min_t, "residue_v3": v3, "count": count})
                    })
            })
            .collect::<Vec<_>>();
        json!({
            "oee_culls": self.oee_culls,
            "generated": self.generated,
            "descent_culls": self.descent_culls,
            "pm1_culls": self.pm1_culls,
            "after_baseline": self.after_baseline,
            "pm_calls": self.pm_calls,
            "pm_culls": self.pm_culls,
            "mix_culls": self.mix_culls,
            "walkback_culls": self.walkback_culls,
            "survived": self.survived,
            "endpoints": self.endpoints,
            "m3_mod3_not2": self.m3_mod3_not2,
            "residue_fallback": self.residue_fallback,
            "screened_out": self.screened_out,
            "eligible": self.eligible,
            "eligible_without_cull": self.eligible_without_cull,
            "full_width_loop_attempts": self.full_width_loop_attempts,
            "full_width_successful_divisions": self.full_width_successful_divisions,
            "avoided_loop_attempts": self.avoided_loop_attempts,
            "avoided_successful_divisions": self.avoided_successful_divisions,
            "repeated_divisions_on_eligible": self.repeated_divisions_on_eligible,
            "residue_loop_attempts": self.residue_loop_attempts,
            "residue_successful_divisions": self.residue_successful_divisions,
            "min_slope_t": histogram(&self.min_slope_t),
            "capped_full_width_v3": histogram(&self.capped_full_width_v3),
            "residue_v3": histogram(&self.residue_v3),
            "eligible_by_min_t": histogram(&self.eligible_by_min_t),
            "pm_culls_by_min_t": histogram(&self.pm_culls_by_min_t),
            "min_t_and_residue_v3": joint,
        })
    }
}

struct Profile {
    by_k: Vec<Counters>,
    minimum_t: Vec<[u8; 81]>,
    sum_n: u128,
    sum_m: u128,
    state_xor: u64,
}

impl Profile {
    fn visit(&mut self, node: &Node, target: u32, rules: &Rules) {
        if node.k == target {
            self.by_k[target as usize].endpoints += 1;
            self.sum_n = self.sum_n.wrapping_add(node.n0);
            self.sum_m = self.sum_m.wrapping_add(node.m);
            self.state_xor ^= node.m3 as u64
                | ((node.f as u64) << 32)
                | ((node.k as u64) << 40)
                | ((node.oee as u64) << 48)
                | ((node.ne as u64) << 52);
            return;
        }
        for child in node.children(rules) {
            let row = &mut self.by_k[node.k as usize + 1];
            let Some(child) = child else {
                row.oee_culls += 1;
                continue;
            };
            row.generated += 1;
            if child.m < child.n0 || slope_descends(child.k, child.f) {
                row.descent_culls += 1;
                continue;
            }
            if rules.pm1 && child.ne == 2 && slope_descends(child.k - 1, child.f - 1) {
                row.pm1_culls += 1;
                continue;
            }
            row.after_baseline += 1;

            // This is the authoritative traversal decision, with the original
            // pm_inf, mixed-word, and walkback implementation and rule order.
            let decision = rules.extra_cull(child.n0, child.k, child.f, child.m, child.m3);
            if rules.pm_inf {
                row.pm_calls += 1;
                let min_t = self.minimum_t[child.k as usize][child.f as usize] as usize;
                row.min_slope_t[min_t] += 1;
                row.m3_mod3_not2 += (child.m3 % 3 != 2) as u64;

                let fallback = child.m3 == P16 as u32 - 1;
                let mut r = child.m3 + 1;
                let mut residue_t = 0u32;
                if fallback {
                    residue_t = 16;
                    row.residue_fallback += 1;
                } else {
                    loop {
                        row.residue_loop_attempts += 1;
                        if r % 3 != 0 {
                            break;
                        }
                        r /= 3;
                        residue_t += 1;
                        row.residue_successful_divisions += 1;
                    }
                }
                row.residue_v3[residue_t as usize] += 1;
                row.min_t_and_residue_v3[min_t][residue_t as usize] += 1;
                let capped_residue_t = residue_t.min(child.f);
                let eligible =
                    fallback || (capped_residue_t != 0 && capped_residue_t >= min_t as u32);

                // Count the actual reference loop's multiply-and-test attempts,
                // including its final failed test unless t reaches f.
                let mut q = child.m + 1;
                let mut full_t = 0u32;
                let mut attempts = 0u64;
                while full_t < child.f {
                    attempts += 1;
                    let d = q.wrapping_mul(INV3);
                    if d > THIRD {
                        break;
                    }
                    q = d;
                    full_t += 1;
                }
                row.capped_full_width_v3[full_t as usize] += 1;
                row.full_width_loop_attempts += attempts;
                row.full_width_successful_divisions += full_t as u64;
                if !fallback {
                    assert_eq!(full_t, capped_residue_t);
                }
                if eligible {
                    row.eligible += 1;
                    row.eligible_by_min_t[min_t] += 1;
                    // The existing screen computes the same successful
                    // divisions again on eligible non-fallback inputs.
                    if !fallback {
                        row.repeated_divisions_on_eligible += full_t as u64;
                    }
                } else {
                    row.screened_out += 1;
                    row.avoided_loop_attempts += attempts;
                    row.avoided_successful_divisions += full_t as u64;
                    assert_ne!(decision, Some(Cull::PmInf));
                }
                if decision == Some(Cull::PmInf) {
                    row.pm_culls_by_min_t[min_t] += 1;
                } else if eligible {
                    row.eligible_without_cull += 1;
                }
            }
            match decision {
                Some(Cull::PmInf) => row.pm_culls += 1,
                Some(Cull::Mix) => row.mix_culls += 1,
                Some(Cull::Walkback) => row.walkback_culls += 1,
                Some(_) => unreachable!("extra_cull returned a baseline rule"),
                None => {
                    row.survived += 1;
                    self.visit(&child, target, rules);
                }
            }
        }
    }
}

/// Profile original expansion decisions without modifying any timed DFS.
/// Roots must already have survived all requested rules. This routine is
/// intentionally serial and should run outside benchmark timing windows.
pub fn profile(roots: &[Node], target: u32, rules: &Rules) -> Value {
    assert!(target <= 120);
    assert!(roots.iter().all(|root| root.k <= target && root.f <= 80));
    let minimum_t = (0..=target)
        .map(|k| {
            let mut row = [NEVER as u8; 81];
            for f in 0..=80u32 {
                if let Some(t) = (0..=f).find(|&t| weight_le_pow2k(t, f - t, k)) {
                    row[f as usize] = t as u8;
                }
            }
            row
        })
        .collect();
    let mut profile = Profile {
        by_k: vec![Counters::default(); target as usize + 1],
        minimum_t,
        sum_n: 0,
        sum_m: 0,
        state_xor: 0,
    };
    for root in roots {
        profile.visit(root, target, rules);
    }
    let mut total = Counters::default();
    let mut by_k = Vec::new();
    for (k, row) in profile.by_k.iter().enumerate() {
        total.add(row);
        if row.generated != 0 || row.oee_culls != 0 || row.endpoints != 0 {
            by_k.push(json!({"k": k, "counts": row.json()}));
        }
    }
    assert_eq!(total.pm_calls, total.screened_out + total.eligible);
    assert_eq!(total.eligible, total.pm_culls + total.eligible_without_cull);
    json!({
        "kind": "ternary_profile",
        "roots": roots.len(),
        "target": target,
        "rules": rules.label(),
        "valuation_cap": 16,
        "no_slope_solution_sentinel": NEVER,
        "loop_attempts_include_final_failed_divisibility_test": true,
        "total": total.json(),
        "by_k": by_k,
        "digest": {
            "endpoints": total.endpoints,
            "sum_n": profile.sum_n.to_string(),
            "sum_m": profile.sum_m.to_string(),
            "state_xor": profile.state_xor,
        },
    })
}
