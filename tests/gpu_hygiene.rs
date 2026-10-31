//! c3-hyg GPU regression tests.  They create CUDA contexts, so they are
//! ignored by default; run them under the GPU lock, one at a time:
//!
//!   flock gpu.lock cargo test --release --test gpu_hygiene -- --ignored --test-threads=1
//!
//! Every GPU result is compared with the CPU reference (cv::jump::for_each_case
//! with the code's TailConfig), as screenbench does.
#![cfg(feature = "cuda")]

use cv::gpu::{GpuBench, GpuTuning};
use cv::jump::{case_digest, for_each_case, xv_kind, TailConfig};
use cv::lookahead::{BvMode, Lookahead, WindowRules};
use cv::rules::{MixTables, Rules};
use cv::rowsieve::RowSieve;
use cv::runspec::TailCodes;
use cv::sample::Sampler;
use cv::tail::{Screen, TailParams};
use cv::tree::Node;
use cv::verify::VerifyConfig;
use rayon::prelude::*;
use std::sync::Arc;

extern "C" {
    fn cudaDeviceReset() -> i32;
}

const N: u32 = 72;
const A: u32 = 10;
const B: u32 = 24;

fn lookahead() -> Lookahead {
    let wr = WindowRules { pm_inf: true, mix: Some(Arc::new(MixTables::build(13))) };
    Lookahead::meta(N - A, A, B, 16, BvMode::Shared, &wr)
        .with_class_windows(&[B; 16])
        .with_rows(Some(RowSieve::standard(N - A, A)))
}

fn nodes(la: &Lookahead, class: usize, count: usize, seed: u64) -> Vec<Node> {
    let f = la.offset + class as u32;
    Sampler::new(&Rules::baseline(), la.m, Some((f, f))).draws(count, seed, None).into_iter().map(|d| d.node).collect()
}

/// (cases, iterations, hash, flags, sorted flagged starts, digest (sum, xor, count)).
type Tuple = (u64, u64, u32, u64, Vec<u128>, (u64, u64, u64));

fn cpu(la: &Lookahead, tp: &TailParams, nodes: &[Node], code: u32, class: usize) -> Tuple {
    let cfg = TailConfig::new(xv_kind(code).unwrap());
    let b = la.bs[class];
    let (c, it, h, fl, mut list, dg) = nodes.par_iter().map(|nd| {
        let (mut c, mut it, mut h, mut fl, mut list, mut dg) = (0u64, 0u64, 0u32, 0u64, Vec::new(), (0u64, 0u64, 0u64));
        let h0 = 485 * nd.f as i32 - 306 * la.m as i32;
        for_each_case(la, tp, nd, |base, x| {
            let (hash, iters, flagged) = cfg.run_h(base, x, tp.cutoff_bits, b, h0);
            c += 1;
            it += iters;
            h ^= hash;
            fl += flagged as u64;
            if flagged { list.push(base); }
            let d = case_digest(base, iters as u32, hash);
            dg = (dg.0.wrapping_add(d), dg.1 ^ d, dg.2 + 1);
        }, |_| panic!("wide start"));
        (c, it, h, fl, list, dg)
    }).reduce(|| (0, 0, 0, 0, Vec::new(), (0, 0, 0)), |mut a, b| {
        a.4.extend(b.4);
        (a.0 + b.0, a.1 + b.1, a.2 ^ b.2, a.3 + b.3, a.4, (a.5 .0.wrapping_add(b.5 .0), a.5 .1 ^ b.5 .1, a.5 .2 + b.5 .2))
    });
    list.sort_unstable();
    (c, it, h, fl, list, dg)
}

fn gpu(bench: &mut GpuBench, class: usize, nodes: &[Node], code: u32, wq: &str, screen: &Screen) -> anyhow::Result<Tuple> {
    let wq = cv::gpu::parse_wq(wq).unwrap();
    let g = bench.run_xs_wq(class, nodes, code, wq, screen, 0)?;
    let d = bench.last_dbg;
    Ok((g.cases, g.iters, bench.last_hash, bench.last_flags, bench.last_flag_list.clone(), (d[0], d[1], d[2])))
}

/// The production per-class kernels on a small sample, each checked against
/// the CPU: cv_kernel_jd<2,93> (class 0, mod 9), cv_wqj2_kernel<JdWqTail3<91>,1,43>
/// (class 1), cv_kernel_jds<111,1> (class 7, rows).
fn production_checks(bench: &mut GpuBench, la: &Lookahead, sets: &[(usize, u32, &str, Vec<Node>)]) -> Vec<Tuple> {
    let mut out = Vec::new();
    for (class, code, wq, ns) in sets {
        let screen = if *class >= 2 { Screen { rows: true, ..Screen::default() } } else { Screen::default() };
        let tp = TailParams { screen, ..TailParams::default() };
        let want = cpu(la, &tp, ns, *code, *class);
        let got = gpu(bench, *class, ns, *code, wq, &screen).expect("GPU launch");
        assert_eq!((got.0, got.1, got.2, got.3, &got.4), (want.0, want.1, want.2, want.3, &want.4),
            "class {class} code {code} {wq}: GPU != CPU");
        out.push(got);
    }
    out
}

fn sample_sets(la: &Lookahead) -> Vec<(usize, u32, &'static str, Vec<Node>)> {
    vec![
        (0, 93, "-", nodes(la, 0, 3000, 20261301)),
        (1, 91, "v43.k8", nodes(la, 1, 3000, 20261302)),
        (7, 111, "-", nodes(la, 7, 600, 20261303)),
    ]
}

#[test]
#[ignore = "needs the GPU (run under flock gpu.lock with --ignored)"]
fn per_context_tables_survive_teardown_reset_and_two_live_contexts() {
    let la = lookahead();
    let sets = sample_sets(&la);
    let strip = |v: &Vec<Tuple>| v.iter().map(|t| (t.0, t.1, t.2, t.3, t.4.clone())).collect::<Vec<_>>();
    // 1. First context.
    let mut a = GpuBench::new(&la, 120, true).unwrap();
    let ra = production_checks(&mut a, &la, &sets);
    drop(a);
    // 2. Teardown and re-creation: the new context uploads its own tables.
    let mut b = GpuBench::new(&la, 120, true).unwrap();
    assert_eq!(strip(&production_checks(&mut b, &la, &sets)), strip(&ra));
    drop(b);
    // 3. Device reset between contexts (round 2's process-global table cache
    //    would have kept device pointers of the destroyed context).
    assert_eq!(unsafe { cudaDeviceReset() }, 0);
    let mut c = GpuBench::new(&la, 120, true).unwrap();
    assert_eq!(strip(&production_checks(&mut c, &la, &sets)), strip(&ra));
    // 4. A context that outlives a reset is refused, not run on stale tables.
    assert!(c.context_ok());
    assert_eq!(unsafe { cudaDeviceReset() }, 0);
    assert!(!c.context_ok(), "the reset must be detected");
    // Every run on it is refused, before any endpoint is packed (the reset
    // also freed its pinned buffers).
    for (class, code, wq, ns) in &sets {
        let screen = Screen { rows: *class >= 2, ..Screen::default() };
        let e = format!("{:#}", gpu(&mut c, *class, ns, *code, wq, &screen).unwrap_err());
        assert!(e.contains("context is not current"), "{e}");
    }
    assert!(c.run_x(0, &sets[0].3, 5 | (93 << 24), 0, true).is_err());
    drop(c);
    // 5. Two live contexts in one process: independent tables; dropping one
    //    leaves the other intact.
    let mut d = GpuBench::new(&la, 120, true).unwrap();
    let mut e2 = GpuBench::new(&la, 120, true).unwrap();
    let rd = production_checks(&mut d, &la, &sets);
    let re = production_checks(&mut e2, &la, &sets);
    assert_eq!(strip(&rd), strip(&ra));
    assert_eq!(strip(&re), strip(&ra));
    drop(d);
    assert_eq!(strip(&production_checks(&mut e2, &la, &sets)), strip(&ra));
}

#[test]
#[ignore = "needs the GPU (run under flock gpu.lock with --ignored)"]
fn measurement_only_kernels_need_an_explicit_opt_in() {
    let la = lookahead();
    let ns = nodes(&la, 0, 500, 20261304);
    let mut bench = GpuBench::new(&la, 120, true).unwrap();
    // jd floor code 127 (counts cases, runs no tail) and the screen-only diagnostic.
    for _ in 0..2 {
        assert!(bench.run_x(0, &ns, 5 | (127 << 24), 0, true).is_err(), "floor code ran without opt-in");
        assert!(bench.run_xs_wq(0, &ns, 93, 0, &Screen::default(), cv::gpu::SCREEN_DIAG_SCAN).is_err(),
                "screen diagnostic ran without opt-in");
        let cap = (1 << 21) | cv::gpu::parse_wq("v43.k8").unwrap();
        assert!(bench.run_x(1, &nodes(&la, 1, 200, 20261305), 5 | (51 << 24) | cap, 0, true).is_err(),
                "capped test tail ran without opt-in");
    }
    // A verification launch on the same context still works.
    let want = cpu(&la, &TailParams::default(), &ns, 93, 0);
    let got = gpu(&mut bench, 0, &ns, 93, "-", &Screen::default()).unwrap();
    assert_eq!((got.0, got.1, got.2), (want.0, want.1, want.2));
    // With the opt-in the floor runs: same cases, no iterations.
    bench.allow_measurement_only().unwrap();
    let g = bench.run_x(0, &ns, 5 | (127 << 24), 0, true).unwrap();
    assert_eq!((g.cases, g.iters), (want.0, 0));
    assert_eq!(bench.run_xs_wq(0, &ns, 93, 0, &Screen::default(), cv::gpu::SCREEN_DIAG_SCAN).unwrap().cases, want.0);
}

#[test]
#[ignore = "needs the GPU (run under flock gpu.lock with --ignored)"]
fn session_refuses_measurement_and_mismatched_codes() {
    // Pipeline path (GpuSession / gpu_run): N32, A10, B22, 16 classes.
    let rules = Rules::baseline();
    let mut cfg = VerifyConfig {
        n: 32, a: 10, b: 22, num_bv: 16, bv_mode: BvMode::Split, window: WindowRules::default(), class_b: None,
        split: 12, rules, tail: TailParams::default(), units: None,
    };
    cfg.tail.codes = TailCodes::parse("93", 16).unwrap();
    let ok = GpuTuning { overflow_vote: false, oee_mask: true, xtail: cfg.tail.codes.gpu_xtail(), ..GpuTuning::default() };
    let rep = cv::gpu::verify_gpu_tuned(&cfg, ok).unwrap();
    let cpu = cv::verify::verify_cpu(&cfg);
    assert_eq!((rep.cases, rep.iters, &rep.hash), (cpu.cases, cpu.iters, &cpu.hash));
    assert_eq!(rep.config, cpu.config, "CPU and GPU reports carry the same configuration record");
    assert_eq!(rep.verification_scope, "full_range");
    // The GPU codes must be the run's codes (a class beyond a short list
    // used to run the original loop on the CPU).
    let mut short = ok;
    short.xtail[15] = 0;
    assert!(cv::gpu::verify_gpu_tuned(&cfg, short).is_err());
    let mut other = ok;
    other.xtail[3] = 90;
    assert!(cv::gpu::verify_gpu_tuned(&cfg, other).is_err());
    // Floor codes on a verification path.
    let mut floor = cfg.clone();
    floor.tail.codes = TailCodes::measurement(&[127], 16).unwrap();
    let ft = GpuTuning { xtail: floor.tail.codes.gpu_xtail(), ..ok };
    let e = format!("{:#}", cv::gpu::verify_gpu_tuned(&floor, ft).err().expect("floor code accepted"));
    assert!(e.contains("measurement-only"), "{e}");
    // Screen diagnostics on a verification path.
    let mut diag = cfg.clone();
    diag.tail.screen = Screen { rows: false, ..Screen::top(10) };
    let dt = GpuTuning { screen_diag: cv::gpu::SCREEN_DIAG_SETUP, ..ok };
    let e = format!("{:#}", cv::gpu::verify_gpu_tuned(&diag, dt).err().expect("screen diagnostic accepted"));
    assert!(e.contains("measurement-only"), "{e}");
    // A session whose CUDA context was reset is refused before its producers
    // write the (freed) pinned buffers.
    let mut session = cv::gpu::GpuSession::new(&cfg).unwrap();
    let again = session.run_transport_tuned(&cfg, &cv::expansion::DfsExpander, cv::gpu::GpuTransport::Replay16, &[], ok)
        .unwrap();
    assert_eq!((again.cases, again.iters, &again.hash), (cpu.cases, cpu.iters, &cpu.hash));
    assert_eq!(unsafe { cudaDeviceReset() }, 0);
    let e = format!("{:#}", session.run_transport_tuned(&cfg, &cv::expansion::DfsExpander,
        cv::gpu::GpuTransport::Replay16, &[], ok).err().expect("stale session ran"));
    assert!(e.contains("context is not current"), "{e}");
    drop(session);
}

#[test]
#[ignore = "needs the GPU (run under flock gpu.lock with --ignored)"]
fn warp_queue_digest_accumulates_across_launch_chunks() {
    // Layout 52 (per-case digest copy of the production layout 43), code 91.
    let la = lookahead();
    let ns = nodes(&la, 1, 3001, 20261306);
    let want = cpu(&la, &TailParams::default(), &ns, 91, 1);
    let mut bench = GpuBench::new(&la, 120, true).unwrap();
    let one = gpu(&mut bench, 1, &ns, 91, "v52.k8", &Screen::default()).unwrap();
    assert_eq!(one.5, want.5, "single-launch digest");
    bench.set_launch_chunk(700).unwrap();
    let many = gpu(&mut bench, 1, &ns, 91, "v52.k8", &Screen::default()).unwrap();
    assert_eq!((many.0, many.1, many.2, many.3), (want.0, want.1, want.2, want.3));
    assert_eq!(many.5, want.5, "digest over 5 launches: sum and count add, the XOR word XORs");
    // The pure accumulator: XOR on word 1 for digest layouts only.
    let mut acc = [1u64, 3, 5, 0, 0, 0];
    cv::gpu::accumulate_dbg(&mut acc, &[1, 3, 5, 0, 0, 0], cv::gpu::parse_wq("v52.k8").unwrap());
    assert_eq!(acc[..3], [2, 0, 10]);
    let mut acc = [1u64, 3, 5, 0, 0, 0];
    cv::gpu::accumulate_dbg(&mut acc, &[1, 3, 5, 0, 0, 0], cv::gpu::parse_wq("v24").unwrap());
    assert_eq!(acc[..3], [2, 6, 10], "instrumented layouts sum every counter");
}

#[test]
#[ignore = "needs the GPU (run under flock gpu.lock with --ignored)"]
fn w10_fd3_dips_use_the_field_width_on_cpu_and_gpu() {
    // Code 96: W10 FD3, KA0 (per-thread cv_kernel_jd<2,96>); code 110: W10 KA2 speculative.
    let la = lookahead();
    let mut bench = GpuBench::new(&la, 120, true).unwrap();
    let mut earlier = 0u64;
    for (class, code, count) in [(2usize, 96u32, 2000usize), (4, 96, 800), (5, 96, 600), (3, 110, 1000)] {
        let ns = nodes(&la, class, count, 20261307);
        let want = cpu(&la, &TailParams::default(), &ns, code, class);
        let g = bench.run_x(class, &ns, 5 | (code << 24), 0, true).unwrap();
        assert_eq!((g.cases, g.iters, bench.last_hash, bench.last_flags, &bench.last_flag_list),
                   (want.0, want.1, want.2, want.3, &want.4), "class {class} code {code}");
        // Round 2's u8 image clamped W10 dips at 255 quanta (2040): with that
        // table no case can stop earlier, and some stop later.
        let kind = xv_kind(code).unwrap();
        let mut old = TailConfig::new(kind);
        let t = old.dip.as_mut().unwrap();
        for d in t.dip.iter_mut() {
            *d = (*d).min(2040);
        }
        let b = la.bs[class];
        let tp = TailParams::default();
        let old_iters: u64 = ns.par_iter().map(|nd| {
            let h0 = 485 * nd.f as i32 - 306 * la.m as i32;
            let mut it = 0;
            for_each_case(&la, &tp, nd, |base, x| it += old.run_h(base, x, tp.cutoff_bits, b, h0).1, |_| panic!());
            it
        }).sum();
        assert!(want.1 <= old_iters, "a larger dip table can only stop cases earlier");
        earlier += old_iters - want.1;
    }
    assert!(earlier > 0, "fixture: the unclamped W10 field certifies some stops earlier");
}

#[test]
#[ignore = "needs the GPU (run under flock gpu.lock with --ignored)"]
fn flag_buffer_overflow_fails_loudly() {
    // Synthetic exact iterates far above 2^(cutoff - K): every completion is
    // flagged at its first boundary, far more than the 65536-slot buffer.
    let la = lookahead();
    let mut ns = nodes(&la, 0, 6000, 20261309);
    let mut r = 0x9E37_79B9_7F4A_7C15u64;
    for nd in &mut ns {
        r ^= r << 13;
        r ^= r >> 7;
        r ^= r << 17;
        nd.m = (1u128 << 115) | r as u128;
    }
    let mut bench = GpuBench::new(&la, 120, true).unwrap();
    let e = format!("{:#}", bench.run_xs_wq(0, &ns, 93, 0, &Screen::default(), 0).unwrap_err());
    assert!(e.contains("flag buffer overflow"), "{e}");
    // In buffer-sized pieces every flag is returned.
    bench.set_launch_chunk(400).unwrap();
    let g = bench.run_xs_wq(0, &ns, 93, 0, &Screen::default(), 0).unwrap();
    assert!(g.cases > 65_536 && bench.last_flags == g.cases && bench.last_flag_list.len() as u64 == g.cases);
}

/// c3-final: the two-phase kernel (c3-2ph, cuda/twophase.cuh) reads this
/// context's FD3 image and occupancy cache (c3-hyg's DevTables); at A = 11
/// the chunk layouts 29-31 screen one (top / bottom half) or two (full range)
/// 32-word chunks per endpoint; the A <= 10 layouts refuse A = 11 instead of
/// computing something else.
#[test]
#[ignore = "needs the GPU (run under flock gpu.lock with --ignored)"]
fn two_phase_kernels_use_per_context_tables_and_refuse_unsupported_ranges() {
    let la = lookahead();
    let rows = Screen { rows: true, ..Screen::default() };
    let sets = [
        (0usize, "t3", nodes(&la, 0, 3000, 20261341)),
        (4, "t5", nodes(&la, 4, 600, 20261342)),
        (1, "t12", nodes(&la, 1, 2000, 20261343)),   // per-case digest copy of t3
    ];
    let tp = TailParams { screen: rows, ..TailParams::default() };
    let want: Vec<Tuple> = sets.iter().map(|(c, _, ns)| cpu(&la, &tp, ns, 91, *c)).collect();
    let check = |bench: &mut GpuBench| {
        for ((class, wq, ns), w) in sets.iter().zip(&want) {
            let g = gpu(bench, *class, ns, 91, wq, &rows).expect("two-phase launch");
            assert_eq!((g.0, g.1, g.2, g.3, &g.4), (w.0, w.1, w.2, w.3, &w.4), "class {class} {wq}: GPU != CPU");
            if *wq == "t12" {
                assert_eq!(g.5, w.5, "class {class} {wq}: per-case digest");
            }
        }
    };
    let mut a = GpuBench::new(&la, 120, true).unwrap();
    check(&mut a);
    drop(a);
    assert_eq!(unsafe { cudaDeviceReset() }, 0);
    let mut b = GpuBench::new(&la, 120, true).unwrap();
    check(&mut b);
    // The A <= 10 batched layouts refuse a 64-word completion range (A = 11).
    drop(b);
    let wr = WindowRules { pm_inf: true, mix: Some(Arc::new(MixTables::build(13))) };
    let la11 = Lookahead::meta(N - 11, 11, B, 16, BvMode::Shared, &wr)
        .with_class_windows(&[B; 16])
        .with_rows(Some(RowSieve::standard(N - 11, 11)));
    let mut c = GpuBench::new(&la11, 120, true).unwrap();
    let top = Screen { rows: true, ..Screen::top(11) };
    let bot = Screen { rows: true, ..Screen::bottom(11) };
    let ns = nodes(&la11, 2, 1500, 20261344);
    for (screen, name) in [(top, "top"), (bot, "bottom"), (rows, "full")] {
        let w = cpu(&la11, &TailParams { screen, ..TailParams::default() }, &ns, 91, 2);
        for wq in ["t29", "t30", "t31"] {
            let g = gpu(&mut c, 2, &ns, 91, wq, &screen).expect("chunk-offset launch");
            assert_eq!((g.0, g.1, g.2, g.3, &g.4), (w.0, w.1, w.2, w.3, &w.4), "A = 11 {name} {wq}: GPU != CPU");
            if wq == "t31" {
                assert_eq!(g.5, w.5, "A = 11 {name} {wq}: per-case digest");
            }
        }
        assert!(gpu(&mut c, 2, &ns, 91, "t3", &screen).is_err(), "t3 at A = 11 must be refused");
    }
}
