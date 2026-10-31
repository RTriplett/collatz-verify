//! c-screen kernel-level benchmarks and audits (exact-iterate MODE 2 input, as
//! after parent expansion), with exact CPU-reference checks.
//!
//! screenbench kernel  N A CLASSES GPU_N CPU_N REPS SEED VARIANTS [CLASS_B] [ORDER]
//! screenbench wsweep  N A CLASSES GPU_N REPS SEED VARIANTS BLIST [CLASS_B] [ORDER]
//! screenbench partition N A B CODES [CPU.json]   complete-N CPU+GPU partition audit
//! screenbench partition-cpu N A B CODES ROWS OUT.json   its CPU half (no GPU)
//!
//! c3-hyg: the partition audit's CPU reference takes CODES itself
//! (TailParams::codes); CV_XTAIL_CPU is refused.  Diagnostic variants
//! (L/S/U) opt the bench's context in to measurement-only kernels.
//!
//! VARIANTS (comma separated):
//!   pad5           tuned original kernel (novote_oee) via GpuBench::run_x
//!   x<C>           jump agent's kernel cv_kernel_x code C (e.g. x51); xa = per-class production code
//!   s<C|a|d>[r][t|b][L|S|U][@<wq>]  screen kernel cv_kernel_xs (jump codes 49..53) or
//!                  cv_kernel_jds (jd codes, c-final): r rows, t top half, b bottom half,
//!                  u h >= 2^(A-2), l h < 2^(A-2), m 2^(A-2) <= h < 2^(A-1) (n73-port),
//!                  L load-only, S screen-only, U screen + case setup (diagnostics, no jumps),
//!                  P mod-9 word prefetch (no rows), Z lazy row-word fetch (rows), B 48-register
//!                  launch bound (jd rows kernels); P/Z/B are
//!                  work-preserving scan options and are checked like full variants.
//!                  a = production jx codes per class, d = per-class jd codes (CV_SB_JD);
//!                  @<wq> = warp-queue scheduler spec (v<layout>[.k<K>][.c<log2 fetch>][.o<n>]).
//!                  Without a screen flag the ordinary launch (cv_kernel_x / cv_kernel_jd /
//!                  warp queue) runs.
//! CLASS_B: per-class windows (default production 32,35,35,35,35,34,32,32,27x8).
//! ORDER: parent (default; K52 parents expanded to depth M in DFS order, the
//!        pipeline's batch order), uniform (independent depth-M draws per class), or
//!        bvsort (parent sample sorted by bitvector index: an untimed-sort upper
//!        bound for a device-side reorder before verification).
//! BLIST: windows to sweep, e.g. 27,28,29,30,31,32,33,34,35
//! CODES (partition): 16 per-class codes, e.g. 49,50,51,51,51,51,52,53,53,53,53,53,53,53,53,53
use anyhow::{ensure, Result};
use cv::expansion::DfsExpander;
use cv::gpu::{GpuBench, GpuSession, GpuTransport, GpuTuning};
use cv::jump::{for_each_case, xv_kind, TailConfig, TailKind};
use cv::lookahead::{BvMode, Lookahead, WindowRules};
use cv::rowsieve::RowSieve;
use cv::rules::{MixTables, Rules};
use cv::sample::Sampler;
use cv::tail::{Screen, TailParams};
use cv::tree::{dfs, Node, Sink, TreeStats};
use cv::verify::{verify_cpu, VerifyConfig};
use rayon::prelude::*;
use serde_json::json;
use std::sync::Arc;

const PROD_CB: &str = "32,35,35,35,35,34,32,32,27,27,27,27,27,27,27,27";
const PROD_CODES: [u32; 16] = [49, 50, 51, 51, 51, 51, 52, 53, 53, 53, 53, 53, 53, 53, 53, 53];

#[derive(Clone, Debug)]
enum Kind {
    Pad5,
    X(Option<u32>),
    /// code None = per-class codes `codes`; wq = warp-queue scheduler bits (0 = per thread).
    S { code: Option<u32>, codes: [u32; 16], screen: Screen, diag: u32, wq: u32 },
}

/// Per-class jd codes for `sd...` variants: CV_SB_JD (16 codes) or c-tail's
/// choice at L2-resident light-class windows.
fn jd_codes() -> [u32; 16] {
    let v = std::env::var("CV_SB_JD").unwrap_or_else(|_| "93,93,93,90,90,90,90,111,111,111,111,111,111,111,111,111".into());
    let v: Vec<u32> = v.split(',').map(|x| x.trim().parse().unwrap()).collect();
    assert_eq!(v.len(), 16, "CV_SB_JD: 16 codes");
    v.try_into().unwrap()
}

#[derive(Clone, Debug)]
struct Variant {
    name: String,
    kind: Kind,
}

fn parse_variant(v: &str, a: u32) -> Variant {
    let kind = if v == "pad5" {
        Kind::Pad5
    } else if let Some(c) = v.strip_prefix('x') {
        Kind::X(if c == "a" { None } else { Some(c.parse().expect("x<code>")) })
    } else if let Some(rest) = v.strip_prefix('s') {
        // s<C|a|d>[flags][@<warp-queue spec>]: a = production jx codes, d = per-class jd codes.
        let (rest, wq) = match rest.split_once('@') {
            Some((r, q)) => (r, cv::gpu::parse_wq(q).expect("warp-queue spec")),
            None => (rest, 0),
        };
        let digits: String = rest.chars().take_while(|c| c.is_ascii_digit() || *c == 'a' || *c == 'd').collect();
        let (code, codes) = match digits.as_str() {
            "a" => (None, PROD_CODES),
            "d" => (None, jd_codes()),
            d => (Some(d.parse().expect("s<code>")), PROD_CODES),
        };
        let mut screen = Screen::default();
        let mut diag = 0;
        for ch in rest[digits.len()..].chars() {
            match ch {
                'r' => screen.rows = true,
                't' => screen = Screen { rows: screen.rows, ..Screen::top(a) },
                'b' => screen = Screen { rows: screen.rows, ..Screen::bottom(a) },
                // n73-port: quarter ranges (A >= 8): u = h >= 2^(A-2) (n >= 2^(N-2); N73: [2^71, 2^73)),
                // l = h < 2^(A-2), m = 2^(A-2) <= h < 2^(A-1) (N73: [2^71, 2^72)).
                'u' => screen = Screen { rows: screen.rows, h_lo: 1 << (a - 2), ..Screen::default() },
                'l' => screen = Screen { rows: screen.rows, h_hi: 1 << (a - 2), ..Screen::default() },
                'm' => screen = Screen { rows: screen.rows, h_lo: 1 << (a - 2), h_hi: 1 << (a - 1), ..Screen::default() },
                'L' => diag = cv::gpu::SCREEN_DIAG_LOAD,
                'S' => diag = cv::gpu::SCREEN_DIAG_SCAN,
                'U' => diag = cv::gpu::SCREEN_DIAG_SETUP,
                'P' => diag = cv::gpu::SCREEN_OPT_M9PF,
                'Z' => diag = cv::gpu::SCREEN_OPT_LAZY,
                'B' => diag = cv::gpu::SCREEN_OPT_LB5,
                _ => panic!("unknown screen flag {ch} in {v}"),
            }
        }
        Kind::S { code, codes, screen, diag, wq }
    } else {
        panic!("unknown variant {v}")
    };
    Variant { name: v.to_string(), kind }
}

impl Variant {
    fn code(&self, class: usize) -> u32 {
        match &self.kind {
            Kind::Pad5 => 0,
            Kind::X(c) => c.unwrap_or(PROD_CODES[class.min(15)]),
            Kind::S { code: c, codes, .. } => c.unwrap_or(codes[class.min(15)]),
        }
    }
    fn screen(&self) -> Screen {
        match &self.kind {
            Kind::S { screen, .. } => *screen,
            _ => Screen::default(),
        }
    }
    fn diag(&self) -> u32 {
        match &self.kind {
            Kind::S { diag, .. } => *diag,
            _ => 0,
        }
    }
    fn tail_kind(&self, class: usize) -> TailKind {
        match self.kind {
            Kind::Pad5 => TailKind::Base,
            _ => xv_kind(self.code(class)).expect("code"),
        }
    }
    fn run(&self, bench: &mut GpuBench, class: usize, nodes: &[Node]) -> Result<cv::gpu::GpuClassStats> {
        match &self.kind {
            Kind::Pad5 => bench.run_x(class, nodes, 5, 0, true),
            Kind::X(_) => bench.run_x(class, nodes, 5 | (self.code(class) << 24), 0, true),
            Kind::S { screen, diag, wq, .. } => bench.run_xs_wq(class, nodes, self.code(class), *wq, screen, *diag),
        }
    }
}

struct Collect(Vec<Node>);
impl Sink for Collect {
    fn endpoint(&mut self, node: &Node) {
        self.0.push(*node);
    }
}

/// Endpoints per class in the pipeline's batch order: K(M-10) parents drawn
/// uniformly from the baseline population, kept if improved13 does not cull
/// them through depth M-12 (the pipeline's cutoff), expanded to depth M with
/// the baseline rules in DFS order.  (The K38 walkback is not applied.)
fn parent_endpoints(m: u32, la: &Lookahead, classes: &[usize], per_class: usize, seed: u64) -> Vec<Vec<Node>> {
    // c3-a: CV_SB_PARENT_W = device expansion width (default 10). The pipeline's K52 parents expand
    // M - 52 levels (9 at A = 11, 8 at A = 12), so set it to M - 52 to match the pipeline's batch order.
    let width: u32 = std::env::var("CV_SB_PARENT_W").map(|v| v.parse().expect("CV_SB_PARENT_W")).unwrap_or(10);
    assert!((2..=12).contains(&width) && width + 2 < m, "CV_SB_PARENT_W in 2..=12");
    let entry = m - width;
    let cutoff = m - width - 2;
    let base = Rules::baseline();
    let improved = Rules::improved(13);
    let sampler = Sampler::new(&base, entry, None);
    let mut out: Vec<Vec<Node>> = vec![Vec::new(); classes.len()];
    let mut round = 0u64;
    while out.iter().any(|v| v.len() < per_class) {
        let draws = sampler.draws(65536, seed ^ round.wrapping_mul(0x9E37_79B9_7F4A_7C15), Some(&improved));
        round += 1;
        let roots: Vec<Node> = draws.into_iter().filter(|d| d.culled.map_or(true, |(k, _)| k > cutoff)).map(|d| d.node).collect();
        let expanded: Vec<Vec<Node>> = roots.par_iter().map(|r| {
            let mut sink = Collect(Vec::new());
            let mut st = TreeStats::new(m);
            dfs::<_, false>(r, m, &base, &mut sink, &mut st);
            sink.0
        }).collect();
        for eps in expanded {
            for nd in eps {
                if let Some(c) = la.bigindex(nd.f) {
                    if let Some(i) = classes.iter().position(|&x| x == c) {
                        if out[i].len() < per_class {
                            out[i].push(nd);
                        }
                    }
                }
            }
        }
        assert!(round < 4096, "sampling did not fill every class");
    }
    out
}

fn uniform_endpoints(m: u32, la: &Lookahead, classes: &[usize], per_class: usize, seed: u64) -> Vec<Vec<Node>> {
    let base = Rules::baseline();
    classes.iter().map(|&c| {
        let f = la.offset + c as u32;
        let s = Sampler::new(&base, m, Some((f, f)));
        s.draws(per_class, seed ^ (c as u64 + 1).wrapping_mul(0x9E37_79B9), None).into_iter().map(|d| d.node).collect()
    }).collect()
}

/// c3-a: endpoint sets cached on disk (CV_SB_CACHE = directory), so that sampling, which is CPU-only, can run
/// outside the GPU lock (`screenbench sample ...` fills the cache; the timed run then loads it).  A file is keyed
/// by (M, parent width, seed, endpoints per class, classes, order) and stores (n0, k, f, oee, ne) per endpoint; the
/// iterate is recomputed exactly by Node::from_compact.
fn endpoint_sets(m: u32, la: &Lookahead, classes: &[usize], per_class: usize, seed: u64, order: &str) -> Vec<Vec<Node>> {
    let sample = || if order == "uniform" { uniform_endpoints(m, la, classes, per_class, seed) } else { parent_endpoints(m, la, classes, per_class, seed) };
    let Ok(dir) = std::env::var("CV_SB_CACHE") else { return sample() };
    let width: u32 = std::env::var("CV_SB_PARENT_W").map(|v| v.parse().unwrap()).unwrap_or(10);
    let cl: Vec<String> = classes.iter().map(|c| c.to_string()).collect();
    let kind = if order == "uniform" { "uniform" } else { "parent" };
    let path = format!("{dir}/eps_m{m}_w{width}_s{seed}_n{per_class}_c{}_{kind}.bin", cl.join("-"));
    if let Ok(bytes) = std::fs::read(&path) {
        let mut out = vec![Vec::new(); classes.len()];
        let mut i = 0usize;
        for ci in 0..classes.len() {
            let n = u64::from_le_bytes(bytes[i..i + 8].try_into().unwrap()) as usize;
            i += 8;
            for _ in 0..n {
                let n0 = u64::from_le_bytes(bytes[i..i + 8].try_into().unwrap());
                let (k, f, oee, ne) = (bytes[i + 8] as u32, bytes[i + 9] as u32, bytes[i + 10], bytes[i + 11]);
                i += 12;
                out[ci].push(Node::from_compact(n0, k, f, oee, ne));
            }
        }
        assert_eq!(i, bytes.len(), "endpoint cache {path}: trailing bytes");
        eprintln!("loaded endpoint cache {path}");
        return out;
    }
    let sets = sample();
    let mut bytes = Vec::new();
    for set in &sets {
        bytes.extend_from_slice(&(set.len() as u64).to_le_bytes());
        for nd in set {
            assert!(nd.n0 < 1u128 << 64 && nd.k <= 64);
            bytes.extend_from_slice(&(nd.n0 as u64).to_le_bytes());
            bytes.extend_from_slice(&[nd.k as u8, nd.f as u8, nd.oee, nd.ne]);
        }
    }
    std::fs::write(&path, &bytes).expect("write endpoint cache");
    eprintln!("wrote endpoint cache {path}");
    sets
}

/// Bitvector index of an endpoint's completion window (the kernel's `index`).
fn bv_index(la: &Lookahead, class: usize, nd: &Node) -> u64 {
    let lo1 = (nd.m as u64).wrapping_add(1);
    lo1.wrapping_mul(la.magic64[class]) & ((1u64 << la.bs[class]) - 1)
}

/// ORDER "bvsort": endpoints sorted by bitvector index (an upper bound on what a
/// device-side reorder before verification could gain; the sort is not timed).
fn maybe_sort(order: &str, la: &Lookahead, class: usize, nodes: &mut Vec<Node>) {
    if order == "bvsort" {
        nodes.sort_by_key(|nd| bv_index(la, class, nd));
    }
}

/// Exact CPU reference of a variant on `nodes`: (cases, iterations, hash,
/// flags, sorted flagged starts, per-case digest (sum, xor, count)).  jd codes
/// get H0 = 485 F - 306 M per endpoint.  The digest is cv::jump::case_digest
/// of (n, the case's jumps, the case's hash), as the warp-queue audit layouts
/// (26, 52) accumulate it on the GPU.
fn cpu_reference(la: &Lookahead, tp: &TailParams, nodes: &[Node], cfg: &TailConfig, b: u32)
    -> (u64, u64, u32, u64, Vec<u128>, (u64, u64, u64)) {
    let (c, it, h, fl, mut list, dg) = nodes.par_iter().map(|nd| {
        let (mut c, mut it, mut h, mut fl, mut list) = (0u64, 0u64, 0u32, 0u64, Vec::new());
        let mut dg = (0u64, 0u64, 0u64);
        let h0 = 485 * nd.f as i32 - 306 * la.m as i32;
        for_each_case(la, tp, nd, |base, x| {
            let (hash, iters, flagged) = cfg.run_h(base, x, tp.cutoff_bits, b, h0);
            c += 1;
            it += iters;
            h ^= hash;
            fl += flagged as u64;
            if flagged { list.push(base); }
            let d = cv::jump::case_digest(base, iters as u32, hash);
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

/// Warp-queue layouts that return the per-case digest in KOut.dbg[0..2].
fn digest_layout(wq: u32) -> bool {
    // c3-2ph: two-phase audit layouts (cuda/twophase.cuh, DIG: 2, 12, 14, 16..21, 24;
    // c3-final adds 31); c3-final: taken from the layout table itself.
    if cv::gpu::is_two_phase(wq) {
        return cv::gpu::tp_layout_kind((wq >> 16) & 31) & 2 != 0;
    }
    let v = ((wq >> 16) & 31) | (((wq >> 23) & 1) << 5);
    wq != 0 && matches!(v, 26 | 52)
}

fn lookahead(n: u32, a: u32, cb: &[u32]) -> Lookahead {
    let m = n - a;
    let wr = WindowRules { pm_inf: true, mix: Some(Arc::new(MixTables::build(13))) };
    let base_b = *cb.iter().min().unwrap();
    Lookahead::meta(m, a, base_b, 16, BvMode::Shared, &wr).with_class_windows(cb).with_rows(Some(RowSieve::standard(m, a)))
}

fn parse_list(s: &str) -> Vec<u32> {
    s.split(',').map(|x| x.trim().parse().unwrap()).collect()
}

fn class_windows(s: Option<&String>) -> Vec<u32> {
    let mut cb = parse_list(s.map_or(PROD_CB, |v| v.as_str()));
    while cb.len() < 16 {
        cb.push(*cb.last().unwrap());
    }
    cb
}

fn median(xs: &[f64]) -> f64 {
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

fn smi() -> String {
    std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=clocks.sm,power.draw,temperature.gpu", "--format=csv,noheader"])
        .output().map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()).unwrap_or_default()
}

/// Rotated paired timing of every variant on `nodes`; returns per-variant times.
fn time_variants(bench: &mut GpuBench, class: usize, nodes: &[Node], variants: &[Variant], reps: usize)
    -> Result<(Vec<Vec<f64>>, Vec<(u64, u64, u32, u64)>)> {
    let mut full = Vec::new();
    let mut dbg = Vec::new();
    for v in variants {
        let g = v.run(bench, class, nodes)?; // warm-up, full-size work tuple
        full.push((g.cases, g.iters, bench.last_hash, bench.last_flags));
        dbg.push(bench.last_dbg);
    }
    // c3-2ph: instrumented layouts (counters in KOut.dbg) report their counters.
    for (v, d) in variants.iter().zip(&dbg) {
        if d.iter().any(|&x| x != 0) {
            eprintln!("dbg class {class} {}: {:?}", v.name, d);
            println!("{}", json!({"type":"dbg","class":class,"v":v.name,"dbg":d}));
        }
    }
    let mut times = vec![Vec::new(); variants.len()];
    for r in 0..reps {
        for k in 0..variants.len() {
            let i = (k + r) % variants.len();
            let g = variants[i].run(bench, class, nodes)?;
            times[i].push(g.kernel_seconds);
        }
    }
    Ok((times, full))
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    match args[1].as_str() {
        "kernel" => kernel(&args),
        "wsweep" => wsweep(&args),
        "partition" => partition(&args),
        "words" => words(&args),
        "partition-cpu" => partition_cpu(&args),
        // c3-a: screenbench sample N A CLASSES GPU_N SEED [ORDER]: fill the CV_SB_CACHE endpoint cache (CPU only).
        "sample" => {
            let (n, aa): (u32, u32) = (args[2].parse()?, args[3].parse()?);
            let classes: Vec<usize> = args[4].split(',').map(|x| x.parse().unwrap()).collect();
            let (gpu_n, seed): (usize, u64) = (args[5].parse()?, args[6].parse()?);
            let order = args.get(7).map_or("parent", |s| s.as_str());
            ensure!(std::env::var("CV_SB_CACHE").is_ok(), "sample needs CV_SB_CACHE");
            let la = lookahead(n, aa, &class_windows(None));
            let t0 = std::time::Instant::now();
            let sets = endpoint_sets(n - aa, &la, &classes, gpu_n, seed, order);
            eprintln!("sampled {:?} endpoints in {:.1}s", sets.iter().map(|s| s.len()).collect::<Vec<_>>(), t0.elapsed().as_secs_f64());
            Ok(())
        }
        _ => panic!("usage: screenbench kernel|wsweep|partition|words ..."),
    }
}

fn kernel(a: &[String]) -> Result<()> {
    let n: u32 = a[2].parse()?;
    let aa: u32 = a[3].parse()?;
    let classes: Vec<usize> = a[4].split(',').map(|x| x.parse().unwrap()).collect();
    let gpu_n: usize = a[5].parse()?;
    let cpu_n: usize = a[6].parse()?;
    let reps: usize = a[7].parse()?;
    let seed: u64 = a[8].parse()?;
    let variants: Vec<Variant> = a[9].split(',').map(|v| parse_variant(v, aa)).collect();
    let cb = class_windows(a.get(10));
    let order = a.get(11).map_or("parent", |s| s.as_str());
    let m = n - aa;
    let la = lookahead(n, aa, &cb);
    let mut tp = TailParams::default();
    // c-final: CV_BENCH_CUTOFF forces a lower overflow cutoff (flag-path checks).
    if let Ok(c) = std::env::var("CV_BENCH_CUTOFF") {
        tp.cutoff_bits = c.parse()?;
    }
    let mut bench = GpuBench::new(&la, tp.cutoff_bits, tp.tail_oee)?;
    // c3-hyg: diagnostic flavours are measurement-only kernels (explicit opt-in).
    if variants.iter().any(|v| cv::gpu::measurement_diag(v.diag())) {
        bench.allow_measurement_only()?;
    }
    let t0 = std::time::Instant::now();
    let mut sets = endpoint_sets(m, &la, &classes, gpu_n, seed, order);
    for (ci, &class) in classes.iter().enumerate() {
        maybe_sort(order, &la, class, &mut sets[ci]);
    }
    eprintln!("sampled {} classes in {:.1}s", classes.len(), t0.elapsed().as_secs_f64());
    println!("{}", json!({"type":"config","n":n,"a":aa,"gpu_n":gpu_n,"cpu_n":cpu_n,"reps":reps,"seed":seed,"cutoff_bits":tp.cutoff_bits,
        "order":order,"class_b":cb,"device":cv::gpu::device_name(),"variants":variants.iter().map(|v| v.name.clone()).collect::<Vec<_>>(),
        "row_table_rows":la.rows.as_ref().map(|r| r.rows.len()),"smi_start":smi()}));
    for (ci, &class) in classes.iter().enumerate() {
        let nodes = &sets[ci];
        let b = la.bs[class];
        let sub = &nodes[..cpu_n.min(nodes.len())];
        // Exact CPU checks on the subset (diagnostics: case counts only).
        let mut checks = Vec::new();
        for v in &variants {
            let g = v.run(&mut bench, class, sub)?;
            let (gh, gf) = (bench.last_hash, bench.last_flags);
            tp.screen = v.screen();
            let cfg = TailConfig::new(v.tail_kind(class));
            let (cc, ci2, ch, cf, clist, cdg) = cpu_reference(&la, &tp, sub, &cfg, b);
            let mut digest_checked = false;
            match v.diag() {
                0 | cv::gpu::SCREEN_OPT_M9PF | cv::gpu::SCREEN_OPT_LAZY | cv::gpu::SCREEN_OPT_LB5 => {
                    ensure!(g.cases == cc && g.iters == ci2 && gh == ch && gf == cf,
                        "class {class} {}: GPU cases {} iters {} hash {gh:08x} flags {gf} vs CPU {cc} {ci2} {ch:08x} {cf}", v.name, g.cases, g.iters);
                    ensure!(bench.last_flag_list == clist, "class {class} {}: flagged-start multisets differ", v.name);
                    if let Kind::S { wq, .. } = &v.kind {
                        if digest_layout(*wq) {
                            ensure!(g.launches == 1, "digest checks need a single launch");
                            let gd = (bench.last_dbg[0], bench.last_dbg[1], bench.last_dbg[2]);
                            ensure!(gd == cdg, "class {class} {}: per-case digest GPU {gd:?} vs CPU {cdg:?}", v.name);
                            digest_checked = true;
                        }
                    }
                }
                cv::gpu::SCREEN_DIAG_LOAD => ensure!(g.cases == 0, "load-only diagnostic counted cases"),
                _ => ensure!(g.cases == cc, "class {class} {}: diagnostic cases {} vs CPU {cc}", v.name, g.cases),
            }
            checks.push(json!({"v":v.name,"cases":cc,"iters":ci2,"hash":format!("{ch:08x}"),"flags":cf,"gpu_cases":g.cases,"gpu_iters":g.iters,
                "digest_checked":digest_checked,"digest":format!("{:016x}/{:016x}/{}", cdg.0, cdg.1, cdg.2)}));
        }
        let (times, full) = time_variants(&mut bench, class, nodes, &variants, reps)?;
        let timing: Vec<_> = variants.iter().zip(&times).zip(&full).map(|((v, t), f)| {
            let ratios: Vec<f64> = t.iter().zip(&times[0]).map(|(x, y)| x / y).collect();
            json!({"v":v.name,"median_s":median(t),"ns_per_ep":median(t) * 1e9 / nodes.len() as f64,
                "ratio_vs_first":median(&ratios),
                "ratio_min":ratios.iter().cloned().fold(f64::MAX, f64::min),
                "ratio_max":ratios.iter().cloned().fold(0.0, f64::max),
                "cases":f.0,"iters":f.1,"hash":format!("{:08x}", f.2),"flags":f.3,
                "cases_per_ep":f.0 as f64 / nodes.len() as f64,"times":t})
        }).collect();
        println!("{}", json!({"type":"class","class":class,"f":la.offset + class as u32,"b":b,"endpoints":nodes.len(),
            "checks":checks,"timing":timing,"smi":smi()}));
        eprintln!("class {class} done ({:.1}s)", t0.elapsed().as_secs_f64());
    }
    Ok(())
}

fn wsweep(a: &[String]) -> Result<()> {
    let n: u32 = a[2].parse()?;
    let aa: u32 = a[3].parse()?;
    let classes: Vec<usize> = a[4].split(',').map(|x| x.parse().unwrap()).collect();
    let gpu_n: usize = a[5].parse()?;
    let reps: usize = a[6].parse()?;
    let seed: u64 = a[7].parse()?;
    let variants: Vec<Variant> = a[8].split(',').map(|v| parse_variant(v, aa)).collect();
    let blist = parse_list(&a[9]);
    let cb = class_windows(a.get(10));
    let order = a.get(11).map_or("parent", |s| s.as_str());
    let m = n - aa;
    let la = lookahead(n, aa, &cb);
    let tp = TailParams::default();
    let mut bench = GpuBench::new(&la, tp.cutoff_bits, tp.tail_oee)?;
    if variants.iter().any(|v| cv::gpu::measurement_diag(v.diag())) {
        bench.allow_measurement_only()?;   // c3-hyg: diagnostics are measurement-only
    }
    let t0 = std::time::Instant::now();
    let mut sets = endpoint_sets(m, &la, &classes, gpu_n, seed, order);
    println!("{}", json!({"type":"config","mode":"wsweep","n":n,"a":aa,"gpu_n":gpu_n,"reps":reps,"seed":seed,"order":order,
        "blist":blist,"device":cv::gpu::device_name(),"variants":variants.iter().map(|v| v.name.clone()).collect::<Vec<_>>(),
        "smi_start":smi(),"sampling_seconds":t0.elapsed().as_secs_f64()}));
    for (ci, &class) in classes.iter().enumerate() {
        for &b in &blist {
            bench.set_class_window(class, b)?;
            let mut lab = la.clone();
            lab.bs[class] = b;
            maybe_sort(order, &lab, class, &mut sets[ci]);
            let nodes = &sets[ci];
            let build0 = bench.build_seconds;
            let (times, full) = time_variants(&mut bench, class, nodes, &variants, reps)?;
            let timing: Vec<_> = variants.iter().zip(&times).zip(&full).map(|((v, t), f)| json!({
                "v":v.name,"median_s":median(t),"ns_per_ep":median(t) * 1e9 / nodes.len() as f64,
                "cases_per_ep":f.0 as f64 / nodes.len() as f64,"iters_per_ep":f.1 as f64 / nodes.len() as f64,
                "times":t})).collect();
            println!("{}", json!({"type":"window","class":class,"b":b,"endpoints":nodes.len(),
                "array_bytes":bench.lookahead().array_words(class) * 4,"build_seconds":bench.build_seconds - build0,
                "timing":timing,"smi":smi()}));
            eprintln!("class {class} B{b} done ({:.1}s)", t0.elapsed().as_secs_f64());
        }
    }
    Ok(())
}

/// screenbench words N A CLASSES COUNT SEED [CLASS_B]: the cooperative
/// per-lane-word routine (cuda/screen.cuh scr_mask_setup + scr_word, one warp
/// per endpoint) must produce exactly the completions the CPU enumerates
/// (jump::for_each_case) for mod 9 and rows, full range and top half.
fn words(a: &[String]) -> Result<()> {
    let n: u32 = a[2].parse()?;
    let aa: u32 = a[3].parse()?;
    let classes: Vec<usize> = a[4].split(',').map(|x| x.parse().unwrap()).collect();
    let count: usize = a[5].parse()?;
    let seed: u64 = a[6].parse()?;
    let cb = class_windows(a.get(7));
    let m = n - aa;
    let la = lookahead(n, aa, &cb);
    let mut tp = TailParams::default();
    let mut bench = GpuBench::new(&la, tp.cutoff_bits, tp.tail_oee)?;
    let sets = parent_endpoints(m, &la, &classes, count, seed);
    let words32 = (1usize << aa) / 32;
    for (ci, &class) in classes.iter().enumerate() {
        let nodes = &sets[ci];
        for screen in [Screen::default(), Screen::default().with_rows(true), Screen::top(aa),
                       Screen { rows: true, ..Screen::top(aa) }] {
            let gpu = bench.screen_words(class, nodes, &screen)?;
            tp.screen = screen;
            let mut mism = 0u64;
            let mut bits = 0u64;
            for (k, nd) in nodes.iter().enumerate() {
                let mut want = vec![0u32; words32];
                // for_each_case also applies the OEE parity tests; the words carry the same mask.
                for_each_case(&la, &tp, nd, |base, _x| {
                    let h = (base >> m) as usize;
                    want[h / 32] |= 1 << (h % 32);
                }, |_| panic!("wide start"));
                let got = &gpu[k * words32..(k + 1) * words32];
                mism += want.iter().zip(got).filter(|(x, y)| x != y).count() as u64;
                bits += got.iter().map(|w| w.count_ones() as u64).sum::<u64>();
            }
            println!("{}", json!({"type":"words","class":class,"b":la.bs[class],"screen":screen,"endpoints":nodes.len(),
                "set_bits":bits,"mismatched_words":mism}));
            ensure!(mism == 0, "cooperative words differ from the CPU enumeration (class {class}, {screen:?})");
        }
    }
    Ok(())
}

fn partition_config(n: u32, aa: u32, b: u32) -> VerifyConfig {
    let rules = Rules::improved(13);
    VerifyConfig {
        n, a: aa, b, num_bv: 16, bv_mode: BvMode::Shared,
        window: WindowRules { pm_inf: true, mix: rules.mix.clone() },
        class_b: Some(vec![b; 16]), split: 10, rules, tail: TailParams::default(), units: None,
    }
}

/// c3-hyg: the per-class codes (and optional CV_SB_WQ warp-queue specs) of a
/// partition run, as the run's own TailParams (CPU and GPU read the same value).
fn partition_tail(codes: &[u32]) -> Result<(cv::runspec::TailCodes, cv::runspec::WqSchedule)> {
    cv::runspec::reject_legacy_env()?;
    let tc = cv::runspec::TailCodes::from_codes(codes, 16)?;
    let ws = match std::env::var("CV_SB_WQ") {
        Ok(specs) => cv::runspec::WqSchedule::parse(&specs, 16)?,
        Err(_) => cv::runspec::WqSchedule::none(),
    };
    Ok((tc, ws))
}

/// Rows (when on) apply to bitvector classes >= CV_SB_ROWS_FROM (default 0) and the CPU-only classes.
fn partition_screens(aa: u32, rows: bool) -> [(&'static str, Screen); 3] {
    let rows_from: u32 = std::env::var("CV_SB_ROWS_FROM").ok().map_or(0, |v| v.parse().unwrap());
    let rows_from = if rows { rows_from } else { 0 };
    [("full", Screen { rows, rows_from, ..Screen::default() }), ("bottom", Screen { rows, rows_from, ..Screen::bottom(aa) }),
     ("top", Screen { rows, rows_from, ..Screen::top(aa) })]
}

/// screenbench partition-cpu N A B CODES ROWS(0|1) OUT.json: the CPU half of the
/// partition audit, run without the GPU (and without the GPU lock).
fn partition_cpu(a: &[String]) -> Result<()> {
    let n: u32 = a[2].parse()?;
    let aa: u32 = a[3].parse()?;
    let b: u32 = a[4].parse()?;
    let codes = parse_list(&a[5]);
    let rows = a[6] == "1";
    let (tc, ws) = partition_tail(&codes)?;
    let mut out = Vec::new();
    for (label, screen) in partition_screens(aa, rows) {
        let mut cfg = partition_config(n, aa, b);
        cfg.tail.screen = screen;
        cfg.tail.codes = tc;
        cfg.tail.schedule = ws;
        let t0 = std::time::Instant::now();
        let cpu = verify_cpu(&cfg);
        let mut fl = cpu.flagged.clone();
        fl.sort();
        out.push(json!({"range":label,"rows":rows,"screen":screen,"endpoints":cpu.tail.endpoints,"cases":cpu.tail.cases,
            "iters":cpu.tail.iters,"hash":cpu.hash,"flagged":fl,"generated":cpu.tree.generated,"cpu_seconds":t0.elapsed().as_secs_f64()}));
        eprintln!("{label} rows={rows}: cases {} ({:.1}s)", cpu.cases, t0.elapsed().as_secs_f64());
    }
    std::fs::write(&a[7], serde_json::to_string(&out)?)?;
    Ok(())
}

/// Complete verification of all n < 2^N three ways (full, bottom half h < 2^(A-1),
/// top half), each with and without the row table, on the CPU (c3-hyg: the
/// run's own TailParams::codes = CODES) and on the GPU (affine32
/// transport, screen kernels).  Checks CPU == GPU per screen and
/// bottom + top == full exactly (cases, iterations, XOR hash, flagged starts).
fn partition(a: &[String]) -> Result<()> {
    let n: u32 = a[2].parse()?;
    let aa: u32 = a[3].parse()?;
    let b: u32 = a[4].parse()?;
    let codes = parse_list(&a[5]);
    ensure!(codes.len() == 16 && codes.iter().all(|c| (49..=53).contains(c) || cv::jump::jd_kind(*c).is_some()),
        "16 codes: jump codes 49..53 or jd codes");
    let (tc, ws) = partition_tail(&codes)?;
    let mut base_cfg = partition_config(n, aa, b);
    base_cfg.tail.codes = tc;
    base_cfg.tail.schedule = ws;
    let mut session_cfg = base_cfg.clone();
    session_cfg.tail.screen.rows = true;
    let mut gpu = GpuSession::new_experimental(&session_cfg)?;
    // c3-hyg: the GPU codes and warp-queue bits are the run's TailParams values.
    let tuning = GpuTuning { overflow_vote: false, oee_mask: true, xtail: tc.gpu_xtail(), wq: ws.bits(),
                             ..GpuTuning::default() };
    gpu.prepare_jd_tables(&codes)?;
    // Optional 7th argument: CPU signatures from `partition-cpu` (then only the
    // GPU runs here, and the rows setting comes from that file).
    let cpu_file: Option<Vec<serde_json::Value>> = a.get(6).map(|p| serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap());
    let rows_list: Vec<bool> = match &cpu_file { Some(v) => vec![v[0]["rows"].as_bool().unwrap()], None => vec![false, true] };
    let mut results = Vec::new();
    for rows in rows_list {
        let mut sigs = Vec::new();
        for (k, (label, screen)) in partition_screens(aa, rows).into_iter().enumerate() {
            let mut cfg = base_cfg.clone();
            cfg.tail.screen = screen;
            let t0 = std::time::Instant::now();
            let (c_end, c_cases, c_iters, c_hash, fc, c_gen, cpu_s) = if let Some(v) = &cpu_file {
                let r = &v[k];
                ensure!(r["range"] == label && r["rows"] == rows, "CPU file order");
                let vec_u64 = |x: &serde_json::Value| x.as_array().unwrap().iter().map(|y| y.as_u64().unwrap()).collect::<Vec<u64>>();
                (vec_u64(&r["endpoints"]), vec_u64(&r["cases"]), vec_u64(&r["iters"]), r["hash"].as_str().unwrap().to_string(),
                 r["flagged"].as_array().unwrap().iter().map(|y| y.as_str().unwrap().to_string()).collect::<Vec<String>>(),
                 vec_u64(&r["generated"]), r["cpu_seconds"].as_f64().unwrap())
            } else {
                let cpu = verify_cpu(&cfg);
                let mut fc: Vec<String> = cpu.flagged.clone();
                fc.sort();
                (cpu.tail.endpoints.clone(), cpu.tail.cases.clone(), cpu.tail.iters.clone(), cpu.hash.clone(), fc,
                 cpu.tree.generated.clone(), t0.elapsed().as_secs_f64())
            };
            let g = gpu.run_transport_tuned(&cfg, &DfsExpander, GpuTransport::Affine32, &[], tuning)?;
            let mut fg: Vec<String> = g.flagged.clone();
            fg.sort();
            ensure!(c_end == g.tail.endpoints && c_cases == g.tail.cases && c_iters == g.tail.iters
                && c_hash == g.hash && fc == fg && c_gen == g.tree.generated,
                "CPU != GPU for {label} rows={rows}: cases {} vs {}, iters {} vs {}, hash {} vs {}",
                c_cases.iter().sum::<u64>(), g.cases, c_iters.iter().sum::<u64>(), g.iters, c_hash, g.hash);
            println!("{}", json!({"type":"partition_run","n":n,"a":aa,"b":b,"rows":rows,"range":label,"screen":cfg.tail.screen,
                "endpoints":c_end.iter().sum::<u64>(),"cases":c_cases.iter().sum::<u64>(),"iters":c_iters.iter().sum::<u64>(),
                "hash":c_hash,"flagged":fc.len(),
                "cpu_seconds":cpu_s,"gpu_seconds":g.seconds,"cpu_gpu_equal":true}));
            sigs.push((c_cases, c_iters, u32::from_str_radix(&c_hash, 16)?, fc, c_end));
        }
        let (full, bot, top) = (&sigs[0], &sigs[1], &sigs[2]);
        let add = |x: &Vec<u64>, y: &Vec<u64>| x.iter().zip(y).map(|(a, b)| a + b).collect::<Vec<u64>>();
        let mut flags: Vec<String> = bot.3.iter().chain(top.3.iter()).cloned().collect();
        flags.sort();
        let ok = add(&bot.0, &top.0) == full.0 && add(&bot.1, &top.1) == full.1 && (bot.2 ^ top.2) == full.2
            && flags == full.3 && bot.4 == full.4 && top.4 == full.4;
        println!("{}", json!({"type":"partition","n":n,"rows":rows,"bottom_plus_top_equals_full":ok,
            "full_cases":full.0.iter().sum::<u64>(),"bottom_cases":bot.0.iter().sum::<u64>(),"top_cases":top.0.iter().sum::<u64>(),
            "full_iters":full.1.iter().sum::<u64>(),"bottom_iters":bot.1.iter().sum::<u64>(),"top_iters":top.1.iter().sum::<u64>(),
            "full_hash":format!("{:08X}", full.2),"bottom_hash":format!("{:08X}", bot.2),"top_hash":format!("{:08X}", top.2)}));
        ensure!(ok, "partition failed (rows={rows})");
        results.push(ok);
    }
    Ok(())
}
