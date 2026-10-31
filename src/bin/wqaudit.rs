//! Work-equality audit of the warp-queue jump kernels (cuda/wq_jump.cuh).
//!
//! wqaudit N A CLASSES DRAWS SEED MODE2 CUTOFF CLASS_B XVAR QVARS [CHUNK] [SMALL]
//!   XVAR   per-thread reference kernel, e.g. x51 (cv_kernel_x; jx semantics)
//!   QVARS  warp-queue variants, e.g. q51.v17,q51.v26 (DIG layouts 26-30 also
//!          return the per-case digest)
//!   CHUNK  endpoints per GPU launch (default 1048576; small values keep flags
//!          below the 65,536-entry buffer at low cutoffs)
//!   SMALL  1: also launch the first 1..=33 endpoints on their own
//!
//! For every class it checks, on identical draws: the CPU reference
//! (cv::jump::jump_until, coarse, the code's W) against the GPU x-kernel and
//! every q-kernel on cases, iterations, XOR hash and the sorted multiset of
//! flagged starts; for DIG layouts also the per-case digest sum, XOR and count
//! (cv::jump::case_digest).  Any mismatch exits nonzero.
use anyhow::{bail, ensure, Result};
use cv::gpu::GpuBench;
use cv::jump::{case_digest, for_each_case, xv_kind, TailConfig, TailKind};
use cv::lookahead::{BvMode, Lookahead, WindowRules};
use cv::rules::{MixTables, Rules};
use cv::sample::Sampler;
use cv::tail::TailParams;
use cv::tree::Node;
use rayon::prelude::*;
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq, Eq, Default)]
struct Work {
    cases: u64,
    iters: u64,
    hash: u32,
    flags: Vec<u128>,
    dsum: u64,
    dxor: u64,
    dcount: u64,
}

fn wq_pad(s: &str) -> (u32, u32) {
    // q<code>.v<layout>[.k][.c][.o]
    let mut parts = s.split('.');
    let code: u32 = parts.next().unwrap().strip_prefix('q').unwrap().parse().unwrap();
    let (mut v, mut k, mut c, mut o) = (0u32, 0u32, 1u32, 0u32);
    for part in parts {
        let (key, val) = part.split_at(1);
        let val: u32 = val.parse().unwrap();
        match key { "v" => v = val, "k" => k = val, "c" => c = val, "o" => o = val, _ => panic!("{part}") }
    }
    assert!(v < 64 && k < 32 && c < 8 && o < 8, "{s}: field out of range");
    (5 | (k << 3) | (c << 10) | (o << 13) | ((v & 31) << 16) | (1 << 22) | ((v >> 5) << 23) | (code << 24), code)
}

fn cpu(la: &Lookahead, tp: &TailParams, nodes: &[Node], cfg: &TailConfig, b: u32) -> Work {
    let parts: Vec<Work> = nodes.par_chunks(256).map(|chunk| {
        let mut w = Work::default();
        for nd in chunk {
            for_each_case(la, tp, nd, |base, x| {
                let (hash, iters, flagged) = cfg.run(base, x, tp.cutoff_bits, b);
                w.cases += 1;
                w.iters += iters;
                w.hash ^= hash;
                if flagged { w.flags.push(base); }
                let d = case_digest(base, iters as u32, hash);
                w.dsum = w.dsum.wrapping_add(d);
                w.dxor ^= d;
                w.dcount += 1;
            }, |_| panic!("wide start"));
        }
        w
    }).collect();
    let mut w = Work::default();
    for p in parts {
        w.cases += p.cases; w.iters += p.iters; w.hash ^= p.hash; w.flags.extend(p.flags);
        w.dsum = w.dsum.wrapping_add(p.dsum); w.dxor ^= p.dxor; w.dcount += p.dcount;
    }
    w.flags.sort_unstable();
    w
}

fn gpu(bench: &mut GpuBench, class: usize, nodes: &[Node], pad: u32, mode2: bool, chunk: usize) -> Result<Work> {
    let mut w = Work::default();
    for part in nodes.chunks(chunk) {
        let g = bench.run_x(class, part, pad, 0, mode2)?;
        w.cases += g.cases;
        w.iters += g.iters;
        w.hash ^= bench.last_hash;
        w.flags.extend(bench.last_flag_list.iter().copied());
        w.dsum = w.dsum.wrapping_add(bench.last_dbg[0]);
        w.dxor ^= bench.last_dbg[1];
        w.dcount += bench.last_dbg[2];
    }
    w.flags.sort_unstable();
    Ok(w)
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 11 {
        bail!("usage: wqaudit N A CLASSES DRAWS SEED MODE2 CUTOFF CLASS_B XVAR QVARS [CHUNK] [SMALL]");
    }
    let n: u32 = a[1].parse()?;
    let aa: u32 = a[2].parse()?;
    let classes: Vec<usize> = a[3].split(',').map(|x| x.parse().unwrap()).collect();
    let draws: usize = a[4].parse()?;
    let seed: u64 = a[5].parse()?;
    let mode2 = a[6] == "1";
    let cutoff: u32 = a[7].parse()?;
    let mut cb: Vec<u32> = a[8].split(',').map(|x| x.parse().unwrap()).collect();
    while cb.len() < 16 { cb.push(*cb.last().unwrap()); }
    let xvar = a[9].clone();
    let qvars: Vec<String> = a[10].split(',').map(String::from).collect();
    let chunk: usize = a.get(11).map_or(1 << 20, |s| s.parse().unwrap());
    let small = a.get(12).map_or(false, |s| s == "1");
    let m = n - aa;
    let wr = WindowRules { pm_inf: true, mix: Some(Arc::new(MixTables::build(13))) };
    let base_b = *cb.iter().min().unwrap();
    let la = Lookahead::meta(m, aa, base_b, 16, BvMode::Shared, &wr).with_class_windows(&cb);
    let tp = TailParams { cutoff_bits: cutoff, ..TailParams::default() };
    let mut bench = GpuBench::new(&la, tp.cutoff_bits, tp.tail_oee)?;
    let xcode: u32 = xvar.strip_prefix('x').expect("x<code>").parse()?;
    let xpad = 5 | (xcode << 24);
    let kind = xv_kind(xcode).expect("x code");
    ensure!(matches!(kind, TailKind::Jump { coarse: true, skip_window: false, .. }), "reference must be a coarse jump code");
    let cfg = TailConfig::new(kind);
    let mut failures = 0usize;
    for &class in &classes {
        let f = la.offset + class as u32;
        let b = la.bs[class];
        let sampler = Sampler::new(&Rules::baseline(), m, Some((f, f)));
        let nodes: Vec<Node> = sampler.draws(draws, seed ^ (class as u64 + 1).wrapping_mul(0x9E37_79B9), None)
            .iter().map(|d| d.node).collect();
        let want = cpu(&la, &tp, &nodes, &cfg, b);
        let xw = gpu(&mut bench, class, &nodes, xpad, mode2, chunk)?;
        let x_ok = xw.cases == want.cases && xw.iters == want.iters && xw.hash == want.hash && xw.flags == want.flags;
        failures += !x_ok as usize;
        println!("{{\"class\":{class},\"b\":{b},\"endpoints\":{},\"mode2\":{mode2},\"cutoff\":{cutoff},\"v\":\"cpu\",\"cases\":{},\"iters\":{},\"hash\":\"{:08x}\",\"flags\":{},\"dsum\":\"{:016x}\",\"dxor\":\"{:016x}\",\"dcount\":{}}}",
                 nodes.len(), want.cases, want.iters, want.hash, want.flags.len(), want.dsum, want.dxor, want.dcount);
        println!("{{\"class\":{class},\"v\":\"{xvar}\",\"ok\":{x_ok},\"cases\":{},\"iters\":{},\"hash\":\"{:08x}\",\"flags\":{}}}",
                 xw.cases, xw.iters, xw.hash, xw.flags.len());
        for qv in &qvars {
            let (pad, code) = wq_pad(qv);
            ensure!(xv_kind(code) == Some(kind), "{qv} must share {xvar}'s CPU semantics");
            let v = ((pad >> 16) & 31) | (((pad >> 23) & 1) << 5);
            let dig = (26..=30).contains(&v) || v == 36 || v == 42 || v == 49 || v == 52;
            let qw = gpu(&mut bench, class, &nodes, pad, mode2, chunk)?;
            let mut ok = qw.cases == want.cases && qw.iters == want.iters && qw.hash == want.hash && qw.flags == want.flags
                && qw.cases == xw.cases && qw.iters == xw.iters && qw.hash == xw.hash && qw.flags == xw.flags;
            if dig {
                ok &= qw.dsum == want.dsum && qw.dxor == want.dxor && qw.dcount == want.dcount;
            }
            failures += !ok as usize;
            println!("{{\"class\":{class},\"v\":\"{qv}\",\"digest\":{dig},\"ok\":{ok},\"cases\":{},\"iters\":{},\"hash\":\"{:08x}\",\"flags\":{},\"dsum\":\"{:016x}\",\"dxor\":\"{:016x}\",\"dcount\":{}}}",
                     qw.cases, qw.iters, qw.hash, qw.flags.len(), qw.dsum, qw.dxor, qw.dcount);
            if small {
                let mut bad = 0;
                let mut cases = 0;
                for len in 1..=33usize.min(nodes.len()) {
                    let sub = &nodes[..len];
                    let sw = cpu(&la, &tp, sub, &cfg, b);
                    let g = gpu(&mut bench, class, sub, pad, mode2, chunk)?;
                    let gx = gpu(&mut bench, class, sub, xpad, mode2, chunk)?;
                    let mut same = g.cases == sw.cases && g.iters == sw.iters && g.hash == sw.hash && g.flags == sw.flags
                        && gx.cases == sw.cases && gx.iters == sw.iters && gx.hash == sw.hash && gx.flags == sw.flags;
                    if dig { same &= g.dsum == sw.dsum && g.dxor == sw.dxor && g.dcount == sw.dcount; }
                    bad += !same as usize;
                    cases += sw.cases;
                }
                failures += bad;
                println!("{{\"class\":{class},\"v\":\"{qv}\",\"small_launches\":\"1..=33\",\"cases\":{cases},\"mismatches\":{bad}}}");
            }
        }
        eprintln!("class {class} done");
    }
    println!("{{\"failures\":{failures}}}");
    ensure!(failures == 0, "{failures} mismatches");
    Ok(())
}
