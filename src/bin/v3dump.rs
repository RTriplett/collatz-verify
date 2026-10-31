//! v3-sound: GPU dump tool for the independent checker (src/bin/v3check.rs),
//! adapted from v2-sound's v2dump for c3-final's tree (c3-hyg's per-context
//! tables, c3-2ph's two-phase kernel, c3-final's chunk layouts).
//!
//! Runs the REAL kernels of this build through cv::gpu::GpuBench::run_xs_wq,
//! the same cv_gpu_launch_scr -> cv_gpu_launch_impl -> tp_launch host path the
//! pipeline takes after parent expansion (MODE 2 exact iterates), on sampled or
//! complete endpoint sets, and writes for the checker:
//!   PREFIX.eps   binary endpoint records (n0, m = T^M(n0) or a synthetic start,
//!                f, class, oee, ne, family, row-table keep words of the uploaded
//!                (12, 9) table)
//!   PREFIX.jsonl config, variants, single-endpoint launch results ("g1") and
//!                per-class batch results ("gb", incl. KOut.dbg of digest layouts),
//!                optionally re-run with forced launch chunks ("chunk" field).
//! It contains no certificate logic: the checker recomputes everything from n0.
//!
//! v-n73-sound: item flags u / l / m add the quarter ranges h >= 2^(A-2), h < 2^(A-2) and
//! [2^(A-2), 2^(A-1)) (n in [2^(N-2), 2^N), n < 2^(N-2), n in [2^(N-2), 2^(N-1))).
//!
//! v3dump PREFIX N A CLASSES PER_CLASS SINGLE SEED CLASS_B VARIANTS [ORDER] [CUTOFF] [BATCHES]
//!   VARIANTS: NAME=ITEMS;NAME=ITEMS...  ITEMS = 16 comma-separated per-class items
//!             (or one item for every class): <code>[r][t|b][@<wq spec>]
//!             e.g. P=91r@t3,91r@t3,91r@t3,91r@t3,91r@t5 (padded with its last item)
//!   ORDER:    parent (default, the pipeline's batch order), uniform, complete (every
//!             depth-M survivor of improved(13)), mixsyn (parent order, every
//!             V2_SYN_EVERY-th endpoint (default 2) gets a synthetic start in
//!             [2^110, 2^111): each of its surviving completions is flagged at its
//!             first boundary and returned by identity in the flag buffer)
//!   BATCHES:  comma list of extra batch sizes (prefix launches), e.g. 1,2,3,17,33
//!   env V2_CHUNKS=c1,c2,...  also re-run every whole-class batch with the launch
//!             split into chunks of c endpoints (GpuBench::set_launch_chunk)
//!   env V2_BATCH_CHUNK=c  launch chunk of the whole-class and prefix batches
//!             (default 2^20: one launch per batch)
//!   env V2_CPU=1  also writes the project's CPU reference per batch ("cpu" lines).
//!   env V2_PHASE=gen  sample only (CPU, outside the GPU lock) into PREFIX.nodes;
//!   V2_PHASE=gpu  load PREFIX.nodes.  GPU work needs V2_LOCKED=1 (set by the
//!   locked wrapper only).
use anyhow::{ensure, Result};
use cv::gpu::GpuBench;
use cv::jump::{for_each_case, xv_kind, TailConfig};
use cv::lookahead::{BvMode, Lookahead, WindowRules};
use cv::rowsieve::RowSieve;
use cv::rules::{MixTables, Rules};
use cv::sample::Sampler;
use cv::tail::{Screen, TailParams};
use cv::tree::{dfs, Node, Sink, TreeStats};
use rayon::prelude::*;
use serde_json::json;
use std::io::Write;
use std::sync::Arc;

struct Collect(Vec<Node>);
impl Sink for Collect {
    fn endpoint(&mut self, node: &Node) {
        self.0.push(*node);
    }
}

#[derive(Clone, Debug)]
struct Item {
    code: u32,
    rows: bool,
    range: u8, // 0 full, 1 top, 2 bottom; v-n73-sound: 3 u2 (h >= 2^(A-2)), 4 l2 (h < 2^(A-2)), 5 m2 ([2^(A-2), 2^(A-1)))
    wq: u32,
    wq_spec: String,
}

fn parse_item(s: &str) -> Item {
    let (body, wq_spec) = match s.split_once('@') {
        Some((b, q)) => (b, q.to_string()),
        None => (s, "-".to_string()),
    };
    let digits: String = body.chars().take_while(|c| c.is_ascii_digit()).collect();
    let code: u32 = digits.parse().expect("item code");
    let mut it = Item { code, rows: false, range: 0, wq: cv::gpu::parse_wq(&wq_spec).expect("wq spec"), wq_spec };
    for ch in body[digits.len()..].chars() {
        match ch {
            'r' => it.rows = true,
            't' => it.range = 1,
            'b' => it.range = 2,
            // v-n73-sound: quarter ranges (fullrange_bench _u2 / _l2 / _m2, screenbench u / l / m)
            'u' => it.range = 3,
            'l' => it.range = 4,
            'm' => it.range = 5,
            _ => panic!("bad item flag {ch} in {s}"),
        }
    }
    it
}

/// Endpoints per class in the pipeline's batch order (screenbench's parent order:
/// K(M-10) baseline draws, DFS to depth M).
fn parent_endpoints(m: u32, la: &Lookahead, classes: &[usize], per_class: usize, seed: u64) -> Vec<Vec<Node>> {
    let entry = m - 10;
    let cutoff = m - 12;
    let base = Rules::baseline();
    let improved = Rules::improved(13);
    let sampler = Sampler::new(&base, entry, None);
    let mut out: Vec<Vec<Node>> = vec![Vec::new(); classes.len()];
    let mut round = 0u64;
    while out.iter().any(|v| v.len() < per_class) {
        // v3-sound: smaller rounds, and only endpoints of classes still being filled are
        // kept (bounded memory when rare classes need many rounds); parent order is kept.
        let wanted: Vec<usize> = classes.iter().zip(&out).filter(|(_, v)| v.len() < per_class).map(|(&c, _)| c).collect();
        let draws = sampler.draws(16384, seed ^ round.wrapping_mul(0x9E37_79B9_7F4A_7C15), Some(&improved));
        round += 1;
        let roots: Vec<Node> = draws.into_iter().filter(|d| d.culled.map_or(true, |(k, _)| k > cutoff)).map(|d| d.node).collect();
        let expanded: Vec<Vec<Node>> = roots
            .par_iter()
            .map(|r| {
                let mut sink = Collect(Vec::new());
                let mut st = TreeStats::new(m);
                dfs::<_, false>(r, m, &base, &mut sink, &mut st);
                let mut v = sink.0;
                v.retain(|nd| la.bigindex(nd.f).is_some_and(|c| wanted.contains(&c)));
                v
            })
            .collect();
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
        assert!(round < 65536, "sampling did not fill every class");
    }
    out
}

fn uniform_endpoints(m: u32, la: &Lookahead, classes: &[usize], per_class: usize, seed: u64) -> Vec<Vec<Node>> {
    let base = Rules::baseline();
    classes
        .iter()
        .map(|&c| {
            let f = la.offset + c as u32;
            let s = Sampler::new(&base, m, Some((f, f)));
            s.draws(per_class, seed ^ (c as u64 + 1).wrapping_mul(0x9E37_79B9), None).into_iter().map(|d| d.node).collect()
        })
        .collect()
}

/// Every depth-M survivor of improved(13) (root-armed OEE), split by class.
fn complete_endpoints(m: u32, la: &Lookahead, classes: &[usize]) -> Vec<Vec<Node>> {
    let rules = Rules::improved(13);
    let mut sink = Collect(Vec::new());
    let mut st = TreeStats::new(m);
    dfs::<_, false>(&Node::root(&rules), m, &rules, &mut sink, &mut st);
    let mut out: Vec<Vec<Node>> = vec![Vec::new(); classes.len()];
    for nd in sink.0 {
        if let Some(c) = la.bigindex(nd.f) {
            if let Some(i) = classes.iter().position(|&x| x == c) {
                out[i].push(nd);
            }
        }
    }
    out
}

fn u128s(v: &[u128]) -> Vec<String> {
    v.iter().map(|x| x.to_string()).collect()
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    if a.get(1).map(|s| s.as_str()) == Some("tables") {
        // CPU only: the host dip tables the GPU FD3 images are uploaded from
        // (cv::jdip::DipTable, production window rules), as stored (c3-hyg:
        // q * min(cap, floor(dip / q)), cap = the FD3 field), and exact (q = 1).
        let mut out = Vec::new();
        for (w, q) in [(10u32, 8u32), (11, 32), (12, 128)] {
            let t = cv::jdip::DipTable::build_seq(w, cv::jump::production_window_rules(), q, false);
            let exact = cv::jdip::DipTable::build_seq(w, cv::jump::production_window_rules(), 1, false);
            out.push(json!({"w": w, "q": q, "cap": cv::jdip::quanta_cap(w, q), "dip": t.dip, "exact": exact.dip}));
        }
        std::fs::write(&a[2], serde_json::to_vec(&out)?)?;
        return Ok(());
    }
    ensure!(a.len() >= 10, "usage: v3dump PREFIX N A CLASSES PER_CLASS SINGLE SEED CLASS_B VARIANTS [ORDER] [CUTOFF] [BATCHES]");
    let prefix = &a[1];
    let n: u32 = a[2].parse()?;
    let aa: u32 = a[3].parse()?;
    let classes: Vec<usize> = a[4].split(',').map(|x| x.parse().unwrap()).collect();
    let per_class: usize = a[5].parse()?;
    let single: usize = a[6].parse()?;
    let seed: u64 = a[7].parse()?;
    let mut cb: Vec<u32> = a[8].split(',').map(|x| x.parse().unwrap()).collect();
    while cb.len() < 16 {
        cb.push(*cb.last().unwrap());
    }
    let variants: Vec<(String, Vec<Item>)> = a[9]
        .split(';')
        .map(|v| {
            let (name, items) = v.split_once('=').expect("NAME=ITEMS");
            let mut its: Vec<Item> = items.split(',').map(parse_item).collect();
            while its.len() < 16 {
                its.push(its.last().unwrap().clone());
            }
            assert_eq!(its.len(), 16, "variant {name}: 16 items");
            (name.to_string(), its)
        })
        .collect();
    let order = a.get(10).map_or("parent", |s| s.as_str());
    let cutoff: u32 = a.get(11).map_or(Ok(120), |s| s.parse())?;
    let batches: Vec<usize> = a.get(12).map_or(Vec::new(), |s| if s.is_empty() || s == "-" { Vec::new() } else { s.split(',').map(|x| x.parse().unwrap()).collect() });
    let chunks: Vec<usize> = std::env::var("V2_CHUNKS").map_or(Vec::new(), |s| s.split(',').filter(|x| !x.is_empty()).map(|x| x.parse().unwrap()).collect());
    let syn_every: usize = std::env::var("V2_SYN_EVERY").map_or(2, |s| s.parse().unwrap());
    // Launch chunk of the whole-class and prefix batches (default one launch of up to 2^20).
    let batch_chunk: usize = std::env::var("V2_BATCH_CHUNK").map_or(1 << 20, |s| s.parse().unwrap());
    let m = n - aa;
    let wr = WindowRules { pm_inf: true, mix: Some(Arc::new(MixTables::build(13))) };
    let base_b = *cb.iter().min().unwrap();
    let la = Lookahead::meta(m, aa, base_b, 16, BvMode::Shared, &wr).with_class_windows(&cb).with_rows(Some(RowSieve::standard(m, aa)));
    let rs = la.rows.clone().unwrap();
    let mut tp = TailParams::default();
    tp.cutoff_bits = cutoff;
    let t0 = std::time::Instant::now();
    let phase = std::env::var("V2_PHASE").unwrap_or_default();
    let sets: Vec<Vec<Node>> = if phase == "gpu" {
        let raw = std::fs::read(format!("{prefix}.nodes"))?;
        let all: Vec<Vec<Node>> = serde_json::from_slice(&raw)?;
        ensure!(all.len() == classes.len(), "node file does not match CLASSES");
        all
    } else {
        match order {
            "uniform" => uniform_endpoints(m, &la, &classes, per_class, seed),
            "complete" => complete_endpoints(m, &la, &classes),
            "mixsyn" => {
                let mut sets = parent_endpoints(m, &la, &classes, per_class, seed);
                for set in sets.iter_mut() {
                    for (i, nd) in set.iter_mut().enumerate() {
                        if i % syn_every == syn_every - 1 {
                            let r = (nd.n0 as u128).wrapping_mul(0x9E37_79B9_7F4A_7C15_F39C_C060_5CED_C835) >> 30;
                            nd.m = (1u128 << 110) | (r & ((1u128 << 100) - 1));
                        }
                    }
                }
                sets
            }
            _ => parent_endpoints(m, &la, &classes, per_class, seed),
        }
    };
    if phase == "gen" {
        std::fs::write(format!("{prefix}.nodes"), serde_json::to_vec(&sets)?)?;
        eprintln!("gen: {:?} endpoints in {:.1}s", sets.iter().map(|s| s.len()).collect::<Vec<_>>(), t0.elapsed().as_secs_f64());
        return Ok(());
    }
    eprintln!("endpoints: {:?} in {:.1}s", sets.iter().map(|s| s.len()).collect::<Vec<_>>(), t0.elapsed().as_secs_f64());
    let words64 = (1usize << aa) / 64;
    let mut eps = std::io::BufWriter::new(std::fs::File::create(format!("{prefix}.eps"))?);
    for (ci, &c) in classes.iter().enumerate() {
        for nd in &sets[ci] {
            eps.write_all(&nd.n0.to_le_bytes())?;
            eps.write_all(&nd.m.to_le_bytes())?;
            eps.write_all(&nd.f.to_le_bytes())?;
            eps.write_all(&(c as u32).to_le_bytes())?;
            eps.write_all(&[nd.oee, nd.ne, la.family(nd.ne) as u8, 0])?;
            let (row, phase) = (rs.row_index(nd.n0), rs.phase(nd.n0));
            for w in 0..words64 {
                eps.write_all(&rs.word64(row, phase, w).to_le_bytes())?;
            }
        }
    }
    eps.flush()?;
    drop(eps);
    let mut out = std::io::BufWriter::new(std::fs::File::create(format!("{prefix}.jsonl"))?);
    let counts: Vec<usize> = sets.iter().map(|s| s.len()).collect();
    writeln!(out, "{}", json!({"t": "cfg", "n": n, "a": aa, "m": m, "offset": la.offset, "windows": cb, "cutoff": cutoff,
        "classes": classes, "counts": counts, "single": single, "seed": seed, "order": order, "words64": words64,
        "rec_bytes": 16 + 16 + 4 + 4 + 4 + 8 * words64, "batches": batches, "chunks": chunks, "syn_every": syn_every, "batch_chunk": batch_chunk,
        "rows_reachable": rs.rows.len()}))?;
    for (name, its) in &variants {
        writeln!(out, "{}", json!({"t": "var", "name": name,
            "codes": its.iter().map(|i| i.code).collect::<Vec<_>>(),
            "rows": its.iter().map(|i| i.rows).collect::<Vec<_>>(),
            "range": its.iter().map(|i| i.range).collect::<Vec<_>>(),
            "wq": its.iter().map(|i| i.wq_spec.clone()).collect::<Vec<_>>()}))?;
    }
    ensure!(std::env::var("V2_LOCKED").map_or(false, |v| v == "1"), "refusing GPU work outside the GPU lock (V2_LOCKED unset)");
    let mut bench = GpuBench::new(&la, cutoff, tp.tail_oee)?;
    let smi = || std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=clocks.sm,power.draw,temperature.gpu", "--format=csv,noheader"])
        .output().map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()).unwrap_or_default();
    writeln!(out, "{}", json!({"t": "smi", "at": "start", "v": smi()}))?;
    let cpu_ref = std::env::var("V2_CPU").map_or(false, |v| v == "1");
    for (ci, &c) in classes.iter().enumerate() {
        let nodes = &sets[ci];
        if nodes.is_empty() {
            continue;
        }
        for (name, its) in &variants {
            let it = &its[c.min(15)];
            // v-n73-sound: the quarter ranges are built exactly as n73-port's fullrange_bench range_screen.
            let a2 = 1u32 << (aa - 2);
            let mut screen = match it.range {
                1 => Screen::top(aa),
                2 => Screen::bottom(aa),
                3 => Screen { h_lo: a2, h_hi: 0, ..Screen::default() },
                4 => Screen { h_lo: 0, h_hi: a2, ..Screen::default() },
                5 => Screen { h_lo: a2, h_hi: 2 * a2, ..Screen::default() },
                _ => Screen::default(),
            };
            if it.rows {
                screen.rows = true;
                screen.rows_from = 0; // per-item rows flag decides for this class
            }
            let ts = std::time::Instant::now();
            bench.set_launch_chunk(1 << 20)?;
            for i in 0..single.min(nodes.len()) {
                let g = bench.run_xs_wq(c, &nodes[i..i + 1], it.code, it.wq, &screen, 0)?;
                writeln!(out, "{}", json!({"t": "g1", "v": name, "c": c, "i": i, "cases": g.cases, "iters": g.iters,
                    "hash": bench.last_hash, "flags": u128s(&bench.last_flag_list)}))?;
            }
            let mut sizes = vec![nodes.len()];
            sizes.extend(batches.iter().copied().filter(|&b| b <= nodes.len()));
            bench.set_launch_chunk(batch_chunk)?;
            for &k in &sizes {
                let g = bench.run_xs_wq(c, &nodes[..k], it.code, it.wq, &screen, 0)?;
                writeln!(out, "{}", json!({"t": "gb", "v": name, "c": c, "k": k, "chunk": batch_chunk, "cases": g.cases, "iters": g.iters,
                    "hash": bench.last_hash, "flags": u128s(&bench.last_flag_list), "launches": g.launches,
                    "dbg": bench.last_dbg.to_vec()}))?;
            }
            for &ch in &chunks {
                if ch >= nodes.len() {
                    continue;
                }
                bench.set_launch_chunk(ch)?;
                let g = bench.run_xs_wq(c, nodes, it.code, it.wq, &screen, 0)?;
                writeln!(out, "{}", json!({"t": "gb", "v": name, "c": c, "k": nodes.len(), "chunk": ch, "cases": g.cases, "iters": g.iters,
                    "hash": bench.last_hash, "flags": u128s(&bench.last_flag_list), "launches": g.launches,
                    "dbg": bench.last_dbg.to_vec()}))?;
            }
            bench.set_launch_chunk(1 << 20)?;
            eprintln!("class {c} variant {name}: {} single + {} batches + {} chunked in {:.1}s", single.min(nodes.len()), sizes.len(),
                chunks.iter().filter(|&&ch| ch < nodes.len()).count(), ts.elapsed().as_secs_f64());
            if cpu_ref {
                let cfg = TailConfig::new(xv_kind(it.code).expect("code"));
                let mut tpv = tp;
                tpv.screen = screen;
                let b = la.bs[c];
                let (cc, ii, hh, mut fl) = nodes.par_iter().map(|nd| {
                    let (mut cc, mut ii, mut hh, mut fl) = (0u64, 0u64, 0u32, Vec::new());
                    let h0 = 485 * nd.f as i32 - 306 * la.m as i32;
                    for_each_case(&la, &tpv, nd, |base, x| {
                        let (hash, iters, flagged) = cfg.run_h(base, x, tpv.cutoff_bits, b, h0);
                        cc += 1; ii += iters; hh ^= hash;
                        if flagged { fl.push(base); }
                    }, |_| panic!("wide start"));
                    (cc, ii, hh, fl)
                }).reduce(|| (0, 0, 0, Vec::new()), |mut x, y| { x.3.extend(y.3); (x.0 + y.0, x.1 + y.1, x.2 ^ y.2, x.3) });
                fl.sort_unstable();
                writeln!(out, "{}", json!({"t": "cpu", "v": name, "c": c, "k": nodes.len(), "cases": cc, "iters": ii, "hash": hh, "flags": u128s(&fl)}))?;
            }
            out.flush()?;
        }
    }
    writeln!(out, "{}", json!({"t": "smi", "at": "end", "v": smi()}))?;
    writeln!(out, "{}", json!({"t": "done", "seconds": t0.elapsed().as_secs_f64()}))?;
    out.flush()?;
    Ok(())
}
