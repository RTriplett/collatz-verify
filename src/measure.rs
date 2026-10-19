//! Measured workload at any N from exact counts times sampled unit work.
//!
//! For each bitvector class `i` (endpoints with `f = offset + i` at depth
//! `M = N - A`) the DP gives the exact count `S_i`.  Uniform draws from the
//! class give the mean steps-2+3 work per endpoint (cases, double
//! iterations, and — with a GPU — measured kernel seconds), plus the fraction
//! the value-dependent rules would have culled on the way down.  Totals are
//! `sum_i S_i * mean_i`; standard errors come from the draws.

use crate::dp::ForwardDp;
use crate::lookahead::{BvMode, Lookahead, WindowRules};
use crate::rules::{Rules, CULL_NAMES, NUM_CULLS};
use crate::sample::Sampler;
use crate::tail::{endpoint_work, TailParams};
use crate::tree::Node;
use anyhow::{ensure, Result};
use rayon::prelude::*;
use serde::Serialize;

/// Kernel timing callback: (class, endpoints) -> (seconds, cases, iters).
pub type GpuTimer<'a> = dyn FnMut(usize, &[Node]) -> Result<(f64, u64, u64)> + 'a;

#[derive(Clone, Debug, Serialize)]
pub struct MeanSe {
    pub mean: f64,
    pub se: f64,
}

impl MeanSe {
    fn of(xs: impl Iterator<Item = f64> + Clone) -> Self {
        let n = xs.clone().count() as f64;
        if n == 0.0 {
            return MeanSe { mean: 0.0, se: 0.0 };
        }
        let mean = xs.clone().sum::<f64>() / n;
        let var = xs.map(|x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1.0).max(1.0);
        MeanSe { mean, se: (var / n).sqrt() }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct ClassMeasure {
    /// Bitvector index (`num_bv` = beyond the last array: finished on CPU).
    pub class: usize,
    pub f_lo: u32,
    pub f_hi: u32,
    /// Exact baseline endpoint count.
    pub endpoints: f64,
    pub samples: usize,
    pub cases: MeanSe,
    pub iters: MeanSe,
    pub flagged: MeanSe,
    /// Fraction of baseline endpoints surviving the extra rules.
    pub survive: Option<MeanSe>,
    /// Per-endpoint work restricted to survivors (as a mean over all draws,
    /// i.e. already multiplied by the survival fraction).
    pub cases_kept: Option<MeanSe>,
    pub iters_kept: Option<MeanSe>,
    /// GPU seconds per endpoint (all draws / survivors only, per draw).
    pub gpu_s: Option<f64>,
    pub gpu_s_kept: Option<f64>,
    pub gpu_samples: usize,
    /// The GPU reproduced the CPU's exact cases and iterations on the
    /// CPU-evaluated draws.
    pub gpu_cpu_agree: Option<bool>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Totals {
    pub endpoints: f64,
    pub cases: f64,
    pub cases_se: f64,
    pub iters: f64,
    pub iters_se: f64,
    /// Expected trajectories reaching 2^cutoff (big-integer re-checks).
    pub flagged: f64,
    pub gpu_seconds: Option<f64>,
    /// Improved only: GPU seconds from timing the surviving draws directly
    /// (`gpu_seconds` instead scales each class's baseline time by the exact
    /// iteration ratio, which is far less noisy).
    pub gpu_seconds_direct: Option<f64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Measure {
    pub n: u32,
    pub a: u32,
    pub b: u32,
    pub num_bv: usize,
    pub bv_mode: BvMode,
    pub class_b: Option<Vec<u32>>,
    pub window: String,
    pub baseline: String,
    pub extra: Option<String>,
    /// Exact baseline step-1 node count (rule evaluations).
    pub tree_nodes: f64,
    pub classes: Vec<ClassMeasure>,
    pub base: Totals,
    pub improved: Option<Totals>,
    /// Among draws culled by the extra rules: (depth, rule) histogram.
    pub cull_depths: Vec<(u32, String, u64)>,
    /// Step-1 nodes under the extra rules: sum_k G(k) * psi(k-1), with the
    /// survival fraction psi sampled at a ladder of depths.
    pub tree_nodes_improved: Option<f64>,
    pub tree_survival: Vec<(u32, f64)>,
    pub seconds: f64,
}

pub struct MeasureConfig {
    pub n: u32,
    pub a: u32,
    pub b: u32,
    pub num_bv: usize,
    pub bv_mode: BvMode,
    pub window: WindowRules,
    pub class_b: Option<Vec<u32>>,
    pub per_class: usize,
    pub gpu_per_class: usize,
    pub seed: u64,
    pub tail: TailParams,
    /// Draws per depth for the step-1 survival ladder (0 = skip).
    pub tree_samples: usize,
    /// Evaluate windows directly instead of building arrays (any B <= 39;
    /// CPU work only — GPU timing then needs device-built arrays).
    pub direct: bool,
}

pub fn measure(cfg: &MeasureConfig, baseline: &Rules, extra: Option<&Rules>, mut gpu: Option<&mut GpuTimer>) -> Result<Measure> {
    let t0 = std::time::Instant::now();
    if let Some(x) = extra {
        ensure!(
            x.pm1 == baseline.pm1 && x.oee == baseline.oee && x.root_oee == baseline.root_oee,
            "extra rules must share the baseline parity rules"
        );
    }
    let m = cfg.n - cfg.a;
    ensure!(m <= crate::verify::MAX_M, "M = N - A = {m} exceeds {}; raise A", crate::verify::MAX_M);
    let la = if let Some(w) = &cfg.class_b {
        Lookahead::meta(m, cfg.a, cfg.b, cfg.num_bv, cfg.bv_mode, &cfg.window).with_class_windows(w)
    } else if cfg.direct || cfg.b > 30 {
        Lookahead::meta(m, cfg.a, cfg.b, cfg.num_bv, cfg.bv_mode, &cfg.window)
    } else {
        Lookahead::build_ext(m, cfg.a, cfg.b, cfg.num_bv, cfg.bv_mode, &cfg.window)
    };
    let fw = ForwardDp::from_root(baseline, m);
    let by_f = fw.layer(m).by_f();
    let fmax = by_f.len() as u32 - 1;
    let mut classes = Vec::new();
    let mut cull_hist = vec![[0u64; NUM_CULLS]; m as usize + 1];
    let ranges: Vec<(usize, u32, u32)> = (0..cfg.num_bv)
        .map(|i| (i, la.offset + i as u32, la.offset + i as u32))
        .chain(std::iter::once((cfg.num_bv, la.offset + cfg.num_bv as u32, fmax)))
        .collect();
    for (class, lo, hi) in ranges {
        let count: u128 = (lo..=hi.min(fmax)).map(|f| by_f[f as usize]).sum();
        if count == 0 {
            continue;
        }
        let want_gpu = gpu.is_some() && class < cfg.num_bv;
        let ndraw = cfg.per_class.max(if want_gpu { cfg.gpu_per_class } else { 0 });
        let sampler = Sampler::new(baseline, m, Some((lo, hi)));
        debug_assert_eq!(sampler.population(), count);
        let draws = sampler.draws(ndraw, cfg.seed ^ (class as u64 + 1).wrapping_mul(0x9E37_79B9), extra);
        let cpu: Vec<(f64, f64, bool, f64)> = draws[..cfg.per_class.min(ndraw)]
            .par_iter()
            .map(|d| {
                let w = endpoint_work(&la, &cfg.tail, &d.node, |_| {});
                (w.cases as f64, w.iters as f64, d.culled.is_none(), w.flagged as f64)
            })
            .collect();
        for d in &draws {
            if let Some((k, why)) = d.culled {
                cull_hist[k as usize][why as usize] += 1;
            }
        }
        let cases = MeanSe::of(cpu.iter().map(|x| x.0));
        let iters = MeanSe::of(cpu.iter().map(|x| x.1));
        let flagged = MeanSe::of(cpu.iter().map(|x| x.3));
        let (survive, cases_kept, iters_kept) = if extra.is_some() {
            (
                Some(MeanSe::of(cpu.iter().map(|x| x.2 as u8 as f64))),
                Some(MeanSe::of(cpu.iter().map(|x| if x.2 { x.0 } else { 0.0 }))),
                Some(MeanSe::of(cpu.iter().map(|x| if x.2 { x.1 } else { 0.0 }))),
            )
        } else {
            (None, None, None)
        };
        let (mut gpu_s, mut gpu_s_kept, mut gpu_cpu_agree) = (None, None, None);
        if want_gpu {
            let timer = gpu.as_mut().unwrap();
            let subset: Vec<Node> = draws[..cpu.len()].iter().map(|d| d.node).collect();
            let (_, gc, gi) = timer(class, &subset)?;
            let (cc, ci) = cpu.iter().fold((0u64, 0u64), |(a, b), x| (a + x.0 as u64, b + x.1 as u64));
            gpu_cpu_agree = Some(gc == cc && gi == ci);
            ensure!(gc == cc && gi == ci, "class {class}: GPU {gc} cases / {gi} iters vs CPU {cc} / {ci}");
            let nodes: Vec<Node> = draws.iter().map(|d| d.node).collect();
            let (secs, _, _) = timer(class, &nodes)?;
            gpu_s = Some(secs / nodes.len() as f64);
            if extra.is_some() {
                let kept: Vec<Node> = draws.iter().filter(|d| d.culled.is_none()).map(|d| d.node).collect();
                let (ks, _, _) = timer(class, &kept)?;
                gpu_s_kept = Some(ks / nodes.len() as f64);
            }
        }
        classes.push(ClassMeasure {
            class,
            f_lo: lo,
            f_hi: hi,
            endpoints: count as f64,
            samples: cpu.len(),
            cases,
            iters,
            flagged,
            survive,
            cases_kept,
            iters_kept,
            gpu_s,
            gpu_s_kept,
            gpu_samples: if want_gpu { draws.len() } else { 0 },
            gpu_cpu_agree,
        });
    }
    let tot = |sel: &dyn Fn(&ClassMeasure) -> (f64, Option<&MeanSe>, Option<&MeanSe>, Option<f64>)| {
        let mut t = Totals::default();
        let mut gpu_total = Some(0.0);
        let (mut cv, mut iv) = (0.0, 0.0);
        for c in &classes {
            let (eps, cs, is, g) = sel(c);
            t.endpoints += eps;
            t.flagged += c.endpoints * c.flagged.mean;
            if let Some(cs) = cs {
                t.cases += c.endpoints * cs.mean;
                cv += (c.endpoints * cs.se).powi(2);
            }
            if let Some(is) = is {
                t.iters += c.endpoints * is.mean;
                iv += (c.endpoints * is.se).powi(2);
            }
            gpu_total = match (gpu_total, g, c.class < cfg.num_bv) {
                (Some(x), Some(g), _) => Some(x + c.endpoints * g),
                (Some(x), None, false) => Some(x),
                _ => None,
            };
        }
        t.cases_se = cv.sqrt();
        t.iters_se = iv.sqrt();
        t.gpu_seconds = gpu_total;
        t
    };
    let base = tot(&|c| (c.endpoints, Some(&c.cases), Some(&c.iters), c.gpu_s));
    let improved = extra.map(|_| {
        let scaled = |c: &ClassMeasure| {
            let r = if c.iters.mean > 0.0 { c.iters_kept.as_ref().unwrap().mean / c.iters.mean } else { 1.0 };
            c.gpu_s.map(|g| g * r)
        };
        let mut t = tot(&|c| {
            (c.endpoints * c.survive.as_ref().unwrap().mean, c.cases_kept.as_ref(), c.iters_kept.as_ref(), scaled(c))
        });
        t.gpu_seconds_direct = tot(&|c| (0.0, None, None, c.gpu_s_kept)).gpu_seconds;
        t
    });
    let mut cull_depths = Vec::new();
    for (k, row) in cull_hist.iter().enumerate() {
        for (r, &c) in row.iter().enumerate() {
            if c > 0 {
                cull_depths.push((k as u32, CULL_NAMES[r].to_string(), c));
            }
        }
    }
    let (mut tree_nodes_improved, mut tree_survival) = (None, Vec::new());
    if let (Some(x), true) = (extra, cfg.tree_samples > 0) {
        // psi(k) at every 4th depth from 12, interpolated linearly in k.
        let mut ladder: Vec<u32> = (12..m).step_by(4).collect();
        ladder.push(m);
        for &k in &ladder {
            let s = Sampler::new(baseline, k, None);
            let d = s.draws(cfg.tree_samples, cfg.seed ^ 0xA5A5 ^ k as u64, Some(x));
            let psi = d.iter().filter(|d| d.culled.is_none()).count() as f64 / d.len() as f64;
            tree_survival.push((k, psi));
        }
        let psi_at = |k: u32| -> f64 {
            if k <= ladder[0] {
                return 1.0 - (1.0 - tree_survival[0].1) * k as f64 / ladder[0] as f64;
            }
            let i = tree_survival.iter().position(|&(kk, _)| kk >= k).unwrap();
            let (k1, p1) = tree_survival[i];
            if k1 == k {
                return p1;
            }
            let (k0, p0) = tree_survival[i - 1];
            p0 + (p1 - p0) * (k - k0) as f64 / (k1 - k0) as f64
        };
        let start = fw.start.k;
        let nodes: f64 = (start..=m)
            .map(|k| fw.generated_at(k) as f64 * if k > start { psi_at(k - 1) } else { 1.0 })
            .sum();
        tree_nodes_improved = Some(nodes);
    }
    Ok(Measure {
        tree_nodes_improved,
        tree_survival,
        n: cfg.n,
        a: cfg.a,
        b: cfg.b,
        num_bv: cfg.num_bv,
        bv_mode: cfg.bv_mode,
        class_b: cfg.class_b.clone(),
        window: cfg.window.label(),
        baseline: baseline.label(),
        extra: extra.map(|r| r.label()),
        tree_nodes: fw.generated_through(m) as f64,
        classes,
        base,
        improved,
        cull_depths,
        seconds: t0.elapsed().as_secs_f64(),
    })
}
