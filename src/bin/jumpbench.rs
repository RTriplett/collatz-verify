//! Paired GPU timing of experimental tail variants against the tuned kernel,
//! with exact CPU-reference checks of cases, iterations, hash and flags.
//!
//! jumpbench N A CLASSES GPU_N CPU_N REPS SEED MODE2 VARIANTS [CLASS_B]
//!   CLASSES  e.g. 0,1,2,3,4,5
//!   VARIANTS e.g. pad5,x1,x17   (pad5 = original tuned kernel; xK = cuda/xtail.cuh code K)
//!   CLASS_B  per-class windows (default 28 for all)

use anyhow::{ensure, Result};
use cv::gpu::GpuBench;
use cv::jump::{for_each_case, xv_kind, TailConfig, TailKind};
use cv::lookahead::{BvMode, Lookahead, WindowRules};
use cv::rules::{MixTables, Rules};
use cv::sample::Sampler;
use cv::tail::TailParams;
use cv::tree::Node;
use rayon::prelude::*;
use std::sync::Arc;

struct Variant {
    name: String,
    pad: u32,
    kind: TailKind,
}

/// Warp-queue variants: `q<code>[.v<layout>][.k<K>][.c<log2 fetch>][.o<blocks/SM>][.cap<J>]`,
/// e.g. `q51.v2.k16`.  The CPU semantics are those of xtail code `<code>`
/// (`.cap<J>`: the timing-only capped test tail, CPU `jump_until_cap`).
fn parse_wq(s: &str) -> (Variant, u32) {
    let mut parts = s.split('.');
    let code: u32 = parts.next().unwrap().strip_prefix('q').unwrap().parse().expect("q<code>");
    let (mut v, mut k, mut c, mut o, mut cap) = (0u32, 0u32, 1u32, 0u32, None);
    for part in parts {
        if let Some(x) = part.strip_prefix("cap") { cap = Some(x.parse::<u32>().unwrap()); }
        else if let Some(x) = part.strip_prefix('v') { v = x.parse().unwrap(); }
        else if let Some(x) = part.strip_prefix('k') { k = x.parse().unwrap(); }
        else if let Some(x) = part.strip_prefix('c') { c = x.parse().unwrap(); }
        else if let Some(x) = part.strip_prefix('o') { o = x.parse().unwrap(); }
        else { panic!("bad warp-queue option {part}"); }
    }
    assert!(v < 64 && k < 32 && c < 8 && o < 8);
    let kind = match (xv_kind(code).expect("unknown xv"), cap) {
        (TailKind::Jump { w, coarse: true, skip_window: false }, Some(cap)) => TailKind::JumpCap { w, cap },
        (kind, None) => kind,
        _ => panic!("cap needs a coarse jump code"),
    };
    let pad = 5 | (k << 3) | (c << 10) | (o << 13) | ((v & 31) << 16) | ((cap.is_some() as u32) << 21) | (1 << 22)
        | ((v >> 5) << 23) | (code << 24);
    (Variant { name: s.into(), pad, kind }, cap.unwrap_or(0))
}

fn parse_variant(s: &str) -> Variant {
    if s.starts_with('q') {
        return parse_wq(s).0;
    }
    if s == "pad5" {
        return Variant { name: s.into(), pad: 5, kind: TailKind::Base };
    }
    if s == "pad0" {
        return Variant { name: s.into(), pad: 0, kind: TailKind::Base };
    }
    let (s, persistent) = match s.strip_suffix('p') { Some(t) => (t, true), None => (s, false) };
    let (code, block) = match s.split_once('b') {
        Some((c, b)) => (c, b.parse::<u32>().unwrap()),
        None => (s, 256),
    };
    let xv: u32 = code.strip_prefix('x').expect("variant xK").parse().unwrap();
    let kind = xv_kind(xv).expect("unknown xv");
    let bsel = match block { 256 => 0, 128 => 1, 512 => 2, _ => panic!("block") };
    Variant { name: format!("{s}{}", if persistent { "p" } else { "" }), pad: 5 | (bsel << 8) | (xv << 24) | ((persistent as u32) << 23), kind }
}

/// KParams.tail_budget of a variant: skipped descent tests, or the cap of the
/// warp-queue test tail.
fn variant_budget(v: &Variant, cfg: &TailConfig, b: u32) -> u32 {
    match v.kind {
        TailKind::JumpCap { cap, .. } => cap,
        _ => cfg.skip(b),
    }
}

fn cpu_reference(la: &Lookahead, tp: &TailParams, nodes: &[Node], cfg: &TailConfig, b: u32) -> (u64, u64, u32, u64) {
    nodes
        .par_iter()
        .map(|nd| {
            let (mut c, mut it, mut h, mut fl) = (0u64, 0u64, 0u32, 0u64);
            for_each_case(la, tp, nd, |base, x| {
                let (hash, iters, flagged) = cfg.run(base, x, tp.cutoff_bits, b);
                c += 1;
                it += iters;
                h ^= hash;
                fl += flagged as u64;
            }, |_| panic!("wide start"));
            (c, it, h, fl)
        })
        .reduce(|| (0, 0, 0, 0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 ^ b.2, a.3 + b.3))
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let n: u32 = a[1].parse()?;
    let aa: u32 = a[2].parse()?;
    let classes: Vec<usize> = a[3].split(',').map(|x| x.parse().unwrap()).collect();
    let gpu_n: usize = a[4].parse()?;
    let cpu_n: usize = a[5].parse()?;
    let reps: usize = a[6].parse()?;
    let seed: u64 = a[7].parse()?;
    let mode2 = a[8] == "1";
    let variants: Vec<Variant> = a[9].split(',').map(parse_variant).collect();
    let mut cb: Vec<u32> = a.get(10).map_or(vec![28], |s| s.split(',').map(|x| x.parse().unwrap()).collect());
    while cb.len() < 16 {
        cb.push(*cb.last().unwrap());
    }
    let m = n - aa;
    let wr = WindowRules { pm_inf: true, mix: Some(Arc::new(MixTables::build(13))) };
    let base_b = *cb.iter().min().unwrap();
    let la = Lookahead::meta(m, aa, base_b, 16, BvMode::Shared, &wr).with_class_windows(&cb);
    let tp = TailParams::default();
    let mut bench = GpuBench::new(&la, tp.cutoff_bits, tp.tail_oee)?;
    // c3-hyg: jd floor codes and the capped test tail are measurement-only
    // kernels; the bench's context opts in explicitly when one is requested.
    if variants.iter().any(|v| (v.pad >> 21) & 1 == 1 || cv::runspec::is_floor_code(v.pad >> 24)) {
        bench.allow_measurement_only()?;
    }
    let baseline = Rules::baseline();
    println!("{{\"n\":{n},\"a\":{aa},\"gpu_n\":{gpu_n},\"cpu_n\":{cpu_n},\"reps\":{reps},\"seed\":{seed},\"mode2\":{mode2},\"class_b\":{cb:?},\"device\":\"{}\",\"classes\":[",
             cv::gpu::device_name());
    for (ci, &class) in classes.iter().enumerate() {
        let f = la.offset + class as u32;
        let b = la.bs[class];
        let sampler = Sampler::new(&baseline, m, Some((f, f)));
        let draws = sampler.draws(gpu_n, seed ^ (class as u64 + 1).wrapping_mul(0x9E37_79B9), None);
        let nodes: Vec<Node> = draws.iter().map(|d| d.node).collect();
        let sub = &nodes[..cpu_n.min(nodes.len())];
        // Exact checks on the subset.
        let mut checks = Vec::new();
        let mut cases0 = None;
        for v in &variants {
            let cfg = TailConfig::new(v.kind);
            let skip = variant_budget(v, &cfg, b);
            let g = bench.run_x(class, sub, v.pad, skip, mode2)?;
            let (gh, gf) = (bench.last_hash, bench.last_flags);
            let (cc, ci2, ch, cf) = cpu_reference(&la, &tp, sub, &cfg, b);
            ensure!(g.cases == cc && g.iters == ci2 && gh == ch && gf == cf,
                "class {class} {}: GPU cases {} iters {} hash {gh:08x} flags {gf} vs CPU {cc} {ci2} {ch:08x} {cf}",
                v.name, g.cases, g.iters);
            ensure!(*cases0.get_or_insert(cc) == cc, "case set changed for {}", v.name);
            checks.push(format!("{{\"v\":\"{}\",\"cases\":{cc},\"iters\":{ci2},\"hash\":\"{ch:08x}\",\"flags\":{cf}}}", v.name));
        }
        // Full-size flags/hash once per variant, then rotated paired timing.
        let mut full = Vec::new();
        for v in &variants {
            let skip = variant_budget(v, &TailConfig::new(v.kind), b);
            let g = bench.run_x(class, &nodes, v.pad, skip, mode2)?; // warm-up
            full.push(format!("{{\"v\":\"{}\",\"cases\":{},\"iters\":{},\"hash\":\"{:08x}\",\"flags\":{},\"dbg\":{:?}}}",
                              v.name, g.cases, g.iters, bench.last_hash, bench.last_flags, bench.last_dbg));
        }
        let mut times = vec![Vec::new(); variants.len()];
        for r in 0..reps {
            for k in 0..variants.len() {
                let i = (k + r) % variants.len();
                let v = &variants[i];
                let skip = variant_budget(v, &TailConfig::new(v.kind), b);
                let g = bench.run_x(class, &nodes, v.pad, skip, mode2)?;
                times[i].push(g.kernel_seconds);
            }
        }
        let med = |xs: &Vec<f64>| { let mut v = xs.clone(); v.sort_by(|a, b| a.total_cmp(b)); v[v.len() / 2] };
        let tstr: Vec<String> = variants.iter().zip(&times).map(|(v, t)| {
            let ratios: Vec<f64> = t.iter().zip(&times[0]).map(|(x, y)| x / y).collect();
            format!("{{\"v\":\"{}\",\"median_s\":{:.6},\"ns_per_ep\":{:.4},\"ratio_vs_first\":{:.4},\"ratio_min\":{:.4},\"ratio_max\":{:.4},\"times\":{:?}}}",
                v.name, med(t), med(t) * 1e9 / nodes.len() as f64, med(&ratios),
                ratios.iter().cloned().fold(f64::MAX, f64::min), ratios.iter().cloned().fold(0.0, f64::max),
                t.iter().map(|x| (x * 1e6).round() / 1e6).collect::<Vec<_>>())
        }).collect();
        println!("{{\"class\":{class},\"f\":{f},\"b\":{b},\"checks\":[{}],\"full\":[{}],\"timing\":[{}]}}{}",
                 checks.join(","), full.join(","), tstr.join(","), if ci + 1 < classes.len() { "," } else { "" });
        eprintln!("class {class} done");
    }
    println!("]}}");
    Ok(())
}
