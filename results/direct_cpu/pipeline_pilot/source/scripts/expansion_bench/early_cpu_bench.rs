//! CPU-only replay of exact states exported immediately after GPU screening.
//!
//! Input is an unframed sequence of eight little-endian u32 limbs: original
//! start followed by T^M(start)+1, the TailContinuation ABI. All input states
//! are retained, including duplicate starts and their checksum multiplicity.

use cv::early_cpu::{verify_states_chunked, EarlyResult};
use cv::tail::{recheck_wide, step_until_below, TailContinuation};
use serde_json::json;
use std::collections::BTreeSet;
use std::path::PathBuf;
use std::time::Instant;

fn cpu_seconds() -> f64 {
    #[repr(C)]
    struct Timespec {
        seconds: std::os::raw::c_long,
        nanoseconds: std::os::raw::c_long,
    }
    unsafe extern "C" {
        fn clock_gettime(clock: i32, time: *mut Timespec) -> i32;
    }
    let mut t = Timespec {
        seconds: 0,
        nanoseconds: 0,
    };
    assert_eq!(unsafe { clock_gettime(2, &mut t) }, 0);
    t.seconds as f64 + t.nanoseconds as f64 * 1e-9
}

fn mode_name(mode: u32) -> &'static str {
    [
        "scalar",
        "bucket_scalar",
        "table",
        "initial_group_table",
        "regrouped_table",
    ][mode as usize]
}

fn signature(result: &mut EarlyResult) -> (u64, u64, u32, Vec<u128>) {
    result.flagged.sort_unstable();
    (
        result.cases,
        result.iters,
        result.hash,
        result.flagged.clone(),
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let mut input: Option<PathBuf> = None;
    let mut cutoff = 120u32;
    let mut repetitions = 5usize;
    let mut limit = usize::MAX;
    let mut chunks = vec![16_384usize, 65_536, 262_144];
    let mut widths = vec![8u32, 10, 12, 14];
    let mut modes = vec![0u32, 1, 2, 3, 4];
    while let Some(arg) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| format!("missing value after {arg}"))?;
        match arg.as_str() {
            "--input" => input = Some(value.into()),
            "--cutoff" => cutoff = value.parse()?,
            "--repetitions" => repetitions = value.parse()?,
            "--limit" => limit = value.parse()?,
            "--chunks" => chunks = value.split(',').map(str::parse).collect::<Result<_, _>>()?,
            "--widths" => widths = value.split(',').map(str::parse).collect::<Result<_, _>>()?,
            "--modes" => modes = value.split(',').map(str::parse).collect::<Result<_, _>>()?,
            _ => return Err(format!("unknown option {arg}").into()),
        }
    }
    let input = input.ok_or("use --input PATH [--cutoff 120] [--repetitions 5]")?;
    if repetitions < 3
        || chunks.is_empty()
        || chunks.iter().any(|&n| n == 0)
        || widths.is_empty()
        || modes.is_empty()
        || modes.iter().any(|&m| m > 4)
        || !(97..=123).contains(&cutoff)
        || widths.iter().any(|&w| !matches!(w, 8 | 10 | 12 | 14))
    {
        return Err("invalid benchmark options (at least three repetitions required)".into());
    }
    let bytes = std::fs::read(&input)?;
    if bytes.is_empty() || bytes.len() % 32 != 0 {
        return Err("fixture must contain a nonempty whole number of 32-byte records".into());
    }
    let fixture_fingerprint = bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| {
        (h ^ b as u64).wrapping_mul(0x100_0000_01b3)
    });
    let states: Vec<TailContinuation> = bytes
        .chunks_exact(32)
        .take(limit)
        .map(|record| {
            let mut limbs = [0u32; 8];
            for (slot, bytes) in limbs.iter_mut().zip(record.chunks_exact(4)) {
                *slot = u32::from_le_bytes(bytes.try_into().unwrap());
            }
            TailContinuation { limbs }
        })
        .collect();
    if states.is_empty() {
        return Err("fixture limit selected no states".into());
    }
    println!(
        "{}",
        json!({
            "type":"config", "experiment":"early_cpu_replay", "input":input,
            "input_bytes":bytes.len(), "input_fingerprint_fnv1a64":format!("{fixture_fingerprint:016x}"),
            "input_fingerprint_note":"FNV is an identity aid, not a cryptographic digest; capture metadata records provenance",
            "available_states":bytes.len()/32, "states":states.len(), "cutoff_bits":cutoff,
            "widths":widths, "chunks":chunks, "modes":modes, "repetitions":repetitions,
            "workers":1, "timed_scope":"decode and validate states, allocations, all sorting and exact verification; resident tables prebuilt",
            "excluded_scope":"fixture loading, table construction, correctness comparisons, wide rechecks, GPU screening and transfer",
            "order":"rotated by 13 positions per repetition, with alternate repetitions reversed",
            "grouping":"initial_group_table shares only the first row; regrouped_table buckets unfinished states after every transition"
        })
    );

    // Independent original scalar oracle. Do not let table implementation or
    // state ordering define correctness; preserve duplicate overflow origins.
    let mut expected = EarlyResult {
        cases: states.len() as u64,
        ..EarlyResult::default()
    };
    for state in &states {
        if state.base() < 3 || state.number() == 0 || state.number() >= 1u128 << 123 {
            return Err("fixture contains an invalid state".into());
        }
        let (hash, iters, flagged) =
            step_until_below(state.base(), state.number(), 1u128 << cutoff);
        expected.hash ^= hash;
        expected.iters += iters;
        if flagged {
            expected.flagged.push(state.base());
        }
    }
    let expected_signature = signature(&mut expected);
    let origins: BTreeSet<u128> = expected.flagged.iter().copied().collect();
    for &origin in &origins {
        if recheck_wide(origin).is_none() {
            return Err("unresolved wide recheck".into());
        }
    }
    println!(
        "{}",
        json!({"type":"oracle", "cases":expected.cases, "iters":expected.iters,
        "hash":expected.hash, "flags":expected.flagged.len(), "unique_wide_rechecks":origins.len(),
        "all_wide_rechecks_passed":true})
    );

    for &width in &widths {
        let start = Instant::now();
        let cpu = cpu_seconds();
        let prepared = verify_states_chunked(&[], cutoff, width, 2, 65_536)?;
        println!(
            "{}",
            json!({"type":"table_setup", "width":width,
            "wall_seconds":start.elapsed().as_secs_f64(), "cpu_seconds":cpu_seconds()-cpu,
            "table_setup_seconds":prepared.table_setup_seconds,
            "note":"empty-state call constructs or obtains the cached table without trajectory work"})
        );
    }
    let mut configs = Vec::new();
    for &chunk in &chunks {
        for &mode in &modes {
            if mode == 0 {
                configs.push((mode, 0u32, chunk));
            } else {
                for &width in &widths {
                    configs.push((mode, width, chunk));
                }
            }
        }
    }
    // A complete warm pass puts state and table pages through the same paths
    // that will be timed. It is independently checked, not counted as a repeat.
    for &(mode, width, chunk) in &configs {
        let mut result = verify_states_chunked(&states, cutoff, width, mode, chunk)?;
        if signature(&mut result) != expected_signature {
            return Err(format!("warmup mismatch mode={mode} width={width} chunk={chunk}").into());
        }
    }
    for repetition in 0..repetitions {
        let mut order: Vec<usize> = (0..configs.len())
            .map(|i| (i + 13 * repetition) % configs.len())
            .collect();
        if repetition % 2 == 1 {
            order.reverse();
        }
        for (position, index) in order.into_iter().enumerate() {
            let (mode, width, chunk) = configs[index];
            let cpu = cpu_seconds();
            let start = Instant::now();
            let mut result = verify_states_chunked(&states, cutoff, width, mode, chunk)?;
            let wall = start.elapsed().as_secs_f64();
            let cpu = cpu_seconds() - cpu;
            if signature(&mut result) != expected_signature {
                return Err(format!(
                    "trial mismatch rep={repetition} mode={mode} width={width} chunk={chunk}"
                )
                .into());
            }
            println!(
                "{}",
                json!({
                    "type":"trial", "repetition":repetition, "order_position":position,
                    "mode":mode, "mode_name":mode_name(mode), "width":width, "chunk_size":chunk,
                    "wall_seconds":wall, "cpu_seconds":cpu,
                    "cases_per_cpu_second":result.cases as f64/cpu,
                    "iterations_per_cpu_second":result.iters as f64/cpu,
                    "grouping_seconds":result.grouping_seconds,
                    "verification_seconds":result.verification_seconds,
                    "table_setup_seconds":result.table_setup_seconds,
                    "cases":result.cases, "iters":result.iters, "hash":result.hash,
                    "flags":result.flagged.len(), "buckets":result.buckets,
                    "jumps":result.jumps, "jumped_iterations":result.jumped_iterations,
                    "scalar_iterations":result.scalar_iterations, "table_lookups":result.table_lookups,
                    "exact_match":true
                })
            );
        }
    }
    println!(
        "{}",
        json!({"type":"complete", "configs":configs.len(),
        "trials":configs.len()*repetitions, "exact_match":true})
    );
    Ok(())
}
