use anyhow::{bail, Result};
use clap::{Args, Parser, Subcommand};
use cv::dp::{predict_cases, ForwardDp};
use cv::lookahead::{BvMode, Lookahead, WindowRules};
use cv::rules::{Rules, CULL_NAMES, NUM_CULLS};
use cv::tail::TailParams;
use cv::verify::{census, verify_cpu, VerifyConfig};
use std::fs;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "cv", about = "LSB-tree Collatz verification, exact workload counts and cost model")]
struct Cli {
    /// Worker threads (default: all logical CPUs).
    #[arg(long, global = true)]
    threads: Option<usize>,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Args, Clone)]
struct RuleArgs {
    /// Rule preset: baseline (Angeltveit), improved (baseline + pm_inf + mix), descent.
    #[arg(long, default_value = "baseline")]
    rules: String,
    /// Mixed-word table modulus 3^J for `improved` (0 disables mix).
    #[arg(long, default_value_t = 12)]
    mix_j: u32,
    /// Disable maximal path merge in `improved`.
    #[arg(long)]
    no_pm_inf: bool,
    /// Start the OEE machine idle (Angeltveit's CPU program) instead of armed.
    #[arg(long)]
    no_root_oee: bool,
    #[arg(long)]
    no_oee: bool,
    #[arg(long)]
    no_pm1: bool,
    /// Apply the mixed-word rule only at depths >= this.
    #[arg(long, default_value_t = 0)]
    mix_from: u32,
    /// Exhaustive bounded walkback at this single checkpoint depth.
    #[arg(long)]
    walkback_depth: Option<u32>,
    #[arg(long, default_value_t = 65536)]
    walkback_states: u32,
}

impl RuleArgs {
    fn build(&self) -> Result<Rules> {
        let mut r = match self.rules.as_str() {
            "baseline" => Rules::baseline(),
            "improved" => Rules::improved(self.mix_j),
            "descent" => Rules::descent_only(),
            other => bail!("unknown rule preset {other}"),
        };
        if self.no_pm_inf {
            r.pm_inf = false;
        }
        if self.no_root_oee {
            r.root_oee = false;
        }
        if self.no_oee {
            r.oee = false;
        }
        if self.no_pm1 {
            r.pm1 = false;
        }
        r.mix_from = self.mix_from;
        r.walkback = self.walkback_depth.map(|depth| cv::rules::Walkback { depth, max_states: self.walkback_states });
        Ok(r)
    }
}

#[derive(Args, Clone)]
struct ShapeArgs {
    /// A: completion bits handled by the bitvector stage.
    #[arg(long, default_value_t = 10)]
    a: u32,
    /// B: bitvector lookahead window (2^B bits per array).
    #[arg(long, default_value_t = 24)]
    b: u32,
    /// Number of bitvector arrays (f classes).
    #[arg(long, default_value_t = 16)]
    nbv: usize,
    /// Use the reference single-family bitvectors (reproduces Angeltveit's
    /// counts; has unjustified exclusions, see tests/bv_audit.rs).
    #[arg(long)]
    bv_legacy: bool,
    /// One conservative array per class (no mod-3 credit before the window's
    /// first odd step): sound, half the bitvector memory.
    #[arg(long)]
    bv_shared: bool,
    /// Credit maximal inverse-odd chains inside the lookahead window.
    #[arg(long)]
    bv_pm_inf: bool,
    /// Credit mixed-word merges mod 3^J inside the window (0 = off).
    #[arg(long, default_value_t = 0)]
    bv_mix_j: u32,
    /// Larger window for the lowest-f classes (arrays built on the GPU).
    #[arg(long)]
    b_hi: Option<u32>,
    /// How many classes get --b-hi (classes 0..8 carry 99% of GPU time).
    #[arg(long, default_value_t = 8)]
    hi_classes: usize,
    /// Explicit per-class windows, e.g. 35,35,35,35,35,34,33,31 (overrides --b-hi).
    #[arg(long, value_delimiter = ',')]
    class_b: Option<Vec<u32>>,
}

impl ShapeArgs {
    fn mode(&self) -> BvMode {
        match (self.bv_legacy, self.bv_shared) {
            (true, _) => BvMode::Legacy,
            (false, true) => BvMode::Shared,
            _ => BvMode::Split,
        }
    }
    fn class_windows(&self) -> Result<Option<Vec<u32>>> {
        let w = match (&self.class_b, self.b_hi) {
            (Some(w), _) => Some(w.clone()),
            (None, Some(b)) => Some(vec![b; self.hi_classes.min(self.nbv)]),
            (None, None) => None,
        };
        if let Some(w) = &w {
            if w.len() > self.nbv || w.iter().any(|&x| x < self.b || x > cv::lookahead::MAX_B) {
                bail!("class windows must number at most --nbv and lie in [--b, {}]", cv::lookahead::MAX_B);
            }
        }
        Ok(w)
    }
    fn window(&self) -> WindowRules {
        WindowRules {
            pm_inf: self.bv_pm_inf,
            mix: (self.bv_mix_j > 0).then(|| std::sync::Arc::new(cv::rules::MixTables::build(self.bv_mix_j))),
        }
    }
}

#[derive(Subcommand)]
enum Cmd {
    /// Verify convergence for all n < 2^N.
    Verify {
        #[arg(long)]
        n: u32,
        #[command(flatten)]
        shape: ShapeArgs,
        #[command(flatten)]
        rules: RuleArgs,
        /// Breadth-first split depth for work units.
        #[arg(long, default_value_t = 24)]
        split: u32,
        /// Skip the OEE parity tests on completions.
        #[arg(long)]
        no_tail_oee: bool,
        #[arg(long, default_value_t = 120)]
        cutoff_bits: u32,
        /// Run steps 2+3 on the GPU.
        #[arg(long)]
        gpu: bool,
        /// Write the full JSON report here.
        #[arg(long)]
        json: Option<PathBuf>,
        /// Work plan from `cv plan`; with --package runs one package ("7") or a
        /// range ("7..12") in one process, so device arrays are built once.
        #[arg(long)]
        plan: Option<PathBuf>,
        #[arg(long)]
        package: Option<String>,
    },
    /// Split the search into work packages of balanced predicted GPU time.
    Plan {
        #[arg(long)]
        n: u32,
        #[command(flatten)]
        shape: ShapeArgs,
        #[command(flatten)]
        rules: RuleArgs,
        /// `cv measure` JSON with per-class GPU seconds per endpoint.
        #[arg(long)]
        costs: PathBuf,
        /// Target GPU hours per package.
        #[arg(long, default_value_t = 1.0)]
        target_hours: f64,
        #[arg(long)]
        out: PathBuf,
    },
    /// Build the bitvectors on the GPU and compare them with the host build.
    GpuCheck {
        #[arg(long)]
        n: u32,
        #[command(flatten)]
        shape: ShapeArgs,
    },
    /// Exact step-1 survivor counts per depth.
    Census {
        #[arg(long)]
        depth: u32,
        #[command(flatten)]
        rules: RuleArgs,
        #[arg(long, default_value_t = 22)]
        split: u32,
        #[arg(long)]
        json: Option<PathBuf>,
    },
    /// Exact baseline workload (DP) for a range of N.
    Predict {
        #[arg(long)]
        from: u32,
        #[arg(long)]
        to: u32,
        #[command(flatten)]
        shape: ShapeArgs,
        #[arg(long)]
        no_tail_oee: bool,
        #[command(flatten)]
        rules: RuleArgs,
        #[arg(long)]
        csv: Option<PathBuf>,
    },
    /// Exact counts x sampled per-endpoint work (and GPU time) at any N.
    Measure {
        #[arg(long)]
        n: u32,
        #[command(flatten)]
        shape: ShapeArgs,
        /// Rules evaluated on the sampled baseline endpoints (e.g. improved).
        #[command(flatten)]
        rules: RuleArgs,
        /// CPU-evaluated draws per bitvector class.
        #[arg(long, default_value_t = 100_000)]
        per_class: usize,
        /// Draws per class for GPU kernel timing (with --gpu).
        #[arg(long, default_value_t = 1 << 21)]
        gpu_per_class: usize,
        #[arg(long)]
        gpu: bool,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        #[arg(long)]
        no_tail_oee: bool,
        /// Draws per depth for the improved step-1 node estimate.
        #[arg(long, default_value_t = 50_000)]
        tree_samples: usize,
        /// Evaluate lookahead windows directly (no arrays; any B up to 39).
        #[arg(long)]
        direct: bool,
        #[arg(long)]
        json: Option<PathBuf>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if let Some(t) = cli.threads {
        rayon::ThreadPoolBuilder::new().num_threads(t).build_global()?;
    }
    match cli.cmd {
        Cmd::Verify { n, shape, rules, split, no_tail_oee, cutoff_bits, gpu, json, plan, package } => {
            if cutoff_bits > 123 || cutoff_bits < n {
                bail!("cutoff-bits must be in [N, 123]");
            }
            let units = match (&plan, &package) {
                (Some(p), Some(spec)) => {
                    let pl: cv::units::Plan = serde_json::from_str(&fs::read_to_string(p)?)?;
                    if pl.n != n || pl.a != shape.a {
                        bail!("plan was made for N={} A={}", pl.n, pl.a);
                    }
                    let (lo, hi) = match spec.split_once("..") {
                        Some((a, b)) => (a.parse::<usize>()?, b.parse::<usize>()?),
                        None => {
                            let i = spec.parse::<usize>()?;
                            (i, i + 1)
                        }
                    };
                    let nodes: Vec<cv::tree::Node> =
                        pl.packages.into_iter().filter(|x| x.id >= lo && x.id < hi).flat_map(|x| x.nodes).collect();
                    if nodes.is_empty() {
                        bail!("no packages in {spec}");
                    }
                    Some(nodes)
                }
                (None, None) => None,
                _ => bail!("--plan and --package go together"),
            };
            let cfg = VerifyConfig {
                n,
                a: shape.a,
                b: shape.b,
                num_bv: shape.nbv,
                bv_mode: shape.mode(),
                window: shape.window(),
                class_b: shape.class_windows()?,
                split,
                rules: rules.build()?,
                tail: TailParams { cutoff_bits, tail_oee: !no_tail_oee },
                units,
            };
            let rep = if gpu { run_gpu(&cfg)? } else { verify_cpu(&cfg) };
            println!(
                "N={} A={} B={} rules={} backend={} units={} tree_nodes={} endpoints={} cases={} iters={} hash={} flagged={} rechecked={} seconds={:.2}",
                rep.n, rep.a, rep.b, rep.rules, rep.backend, rep.units, rep.tree_nodes, rep.endpoints, rep.cases,
                rep.iters, rep.hash, rep.flagged.len(), rep.flagged_rechecked, rep.seconds
            );
            if rep.flagged_rechecked != rep.flagged.len() {
                bail!("a flagged trajectory could not be finished in 256 bits");
            }
            for x in rep.flagged.iter().take(20) {
                println!("flagged {x}");
            }
            if rep.flagged.len() > 20 {
                println!("flagged ... ({} more; full list in --json)", rep.flagged.len() - 20);
            }
            if let Some(p) = json {
                fs::write(p, serde_json::to_string_pretty(&rep)?)?;
            }
        }
        Cmd::Plan { n, shape, rules, costs, target_hours, out } => {
            let r = rules.build()?;
            let base = Rules { pm_inf: false, mix: None, walkback: None, ..r.clone() };
            // Planning needs only the class layout, not the arrays.
            let mut la = Lookahead::meta(n - shape.a, shape.a, shape.b, shape.nbv, shape.mode(), &shape.window());
            if let Some(w) = shape.class_windows()? {
                la = la.with_class_windows(&w);
            }
            let ms: serde_json::Value = serde_json::from_str(&fs::read_to_string(costs)?)?;
            let classes = ms["classes"].as_array().ok_or_else(|| anyhow::anyhow!("bad measure json"))?;
            let mut sec = vec![0f64; shape.nbv + 1];
            let mut last = 0f64;
            for c in classes {
                let i = c["class"].as_u64().unwrap() as usize;
                // Without GPU timings, cost by iterations at ~6.7 ps per
                // double iteration (RTX 4090, N = 72 measurement).
                let g = c["gpu_s"].as_f64().unwrap_or_else(|| c["iters"]["mean"].as_f64().unwrap_or(0.0) * 6.7e-12);
                sec[i] = g;
                last = last.max(g);
            }
            sec[shape.nbv] = last;
            let oracle = cv::units::CostOracle::new(&base, &la, &sec);
            let p = cv::units::plan(&base, &la, n, &oracle, target_hours * 3600.0);
            println!(
                "N={} packages={} endpoints={:.4e} predicted_gpu_hours={:.1} max_package_hours={:.3} nodes={}",
                n,
                p.packages.len(),
                p.total_endpoints,
                p.total_gpu_seconds / 3600.0,
                p.packages.iter().map(|x| x.gpu_seconds).fold(0.0, f64::max) / 3600.0,
                p.packages.iter().map(|x| x.nodes.len()).sum::<usize>()
            );
            fs::write(out, serde_json::to_string(&p)?)?;
        }
        Cmd::GpuCheck { n, shape } => {
            let la = Lookahead::build_ext(n - shape.a, shape.a, shape.b, shape.nbv, shape.mode(), &shape.window());
            let (bad, words, secs) = gpu_check(&la)?;
            println!(
                "N={n} A={} B={} window={}: device build {:.2} s, {} of {} u32 words differ",
                shape.a,
                shape.b,
                la.window.label(),
                secs,
                bad,
                words
            );
            if bad != 0 {
                bail!("device-built bitvectors differ from the host build");
            }
        }
        Cmd::Census { depth, rules, split, json } => {
            let r = rules.build()?;
            let c = census(&r, depth, split);
            println!("# census rules={} depth={} seconds={:.2}", c.rules, depth, c.seconds);
            print!("k,generated,survived");
            for name in CULL_NAMES {
                print!(",{name}");
            }
            println!();
            for k in 2..=depth as usize {
                print!("{},{},{}", k, c.stats.generated[k], c.stats.survived[k]);
                for i in 0..NUM_CULLS {
                    print!(",{}", c.stats.culls[k][i]);
                }
                println!();
            }
            if let Some(p) = json {
                fs::write(p, serde_json::to_string_pretty(&c)?)?;
            }
        }
        Cmd::Predict { from, to, shape, no_tail_oee, rules, csv } => {
            let r = rules.build()?;
            if !r.is_parity_only() {
                bail!("predict counts parity-state rules only; use `sample` for value-dependent rules");
            }
            let fw = ForwardDp::from_root(&r, to - shape.a);
            let window = shape.window();
            let mut out = String::from("N,M,tree_nodes,endpoints,endpoints_no_bv,cases,fraction\n");
            for n in from..=to {
                let la = Lookahead::build_ext(n - shape.a, shape.a, shape.b, shape.nbv, shape.mode(), &window);
                let p = predict_cases(&fw, &la, n, !no_tail_oee);
                let line = format!(
                    "{},{},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e}\n",
                    n, n - shape.a, p.tree_nodes, p.endpoints, p.endpoints_no_bv, p.cases, p.fraction
                );
                print!("{line}");
                out.push_str(&line);
            }
            if let Some(p) = csv {
                fs::write(p, out)?;
            }
        }
        Cmd::Measure { n, shape, rules, per_class, gpu_per_class, gpu, seed, no_tail_oee, tree_samples, direct, json } => {
            let extra = rules.build()?;
            let baseline = Rules { pm_inf: false, mix: None, walkback: None, ..extra.clone() };
            let extra = (!extra.is_parity_only()).then_some(extra);
            let tail = TailParams { cutoff_bits: 120, tail_oee: !no_tail_oee };
            let cfg = cv::measure::MeasureConfig {
                n,
                a: shape.a,
                b: shape.b,
                num_bv: shape.nbv,
                bv_mode: shape.mode(),
                window: shape.window(),
                class_b: shape.class_windows()?,
                per_class,
                gpu_per_class,
                seed,
                tail,
                tree_samples,
                direct,
            };
            let rep = run_measure(&cfg, &baseline, extra.as_ref(), gpu)?;
            print_measure(&rep);
            if let Some(p) = json {
                fs::write(p, serde_json::to_string_pretty(&rep)?)?;
            }
        }
    }
    Ok(())
}

fn print_measure(r: &cv::measure::Measure) {
    println!(
        "# measure N={} A={} B={} window={} baseline={} extra={} tree_nodes={:.4e} seconds={:.1}",
        r.n,
        r.a,
        r.b,
        r.window,
        r.baseline,
        r.extra.as_deref().unwrap_or("-"),
        r.tree_nodes,
        r.seconds
    );
    println!("class,f,endpoints,samples,cases/ep,iters/ep,survive,gpu_ns/ep,gpu_ns/ep_kept");
    for c in &r.classes {
        println!(
            "{},{}{},{:.4e},{},{:.2},{:.2},{},{},{}",
            c.class,
            c.f_lo,
            if c.f_hi > c.f_lo { format!("-{}", c.f_hi) } else { String::new() },
            c.endpoints,
            c.samples,
            c.cases.mean,
            c.iters.mean,
            c.survive.as_ref().map_or("-".into(), |s| format!("{:.5}+-{:.5}", s.mean, s.se)),
            c.gpu_s.map_or("-".into(), |g| format!("{:.3}", g * 1e9)),
            c.gpu_s_kept.map_or("-".into(), |g| format!("{:.3}", g * 1e9)),
        );
    }
    let show = |name: &str, t: &cv::measure::Totals| {
        println!(
            "{name}: endpoints={:.4e} cases={:.4e}(+-{:.1e}) iters={:.4e}(+-{:.1e}) flagged={:.3e} gpu_seconds={}",
            t.endpoints,
            t.cases,
            t.cases_se,
            t.iters,
            t.iters_se,
            t.flagged,
            t.gpu_seconds.map_or("-".into(), |g| format!("{:.4e} ({:.3} GPU-years)", g, g / 3.15576e7))
        );
    };
    show("baseline", &r.base);
    if let Some(t) = &r.improved {
        show("improved", t);
        if let Some(tn) = r.tree_nodes_improved {
            println!("tree nodes: baseline {:.4e} improved {:.4e} ({:.2}%)", r.tree_nodes, tn, 100.0 * (1.0 - tn / r.tree_nodes));
        }
        println!(
            "reduction: endpoints {:.2}%  cases {:.2}%  iters {:.2}%{}",
            100.0 * (1.0 - t.endpoints / r.base.endpoints),
            100.0 * (1.0 - t.cases / r.base.cases),
            100.0 * (1.0 - t.iters / r.base.iters),
            match (t.gpu_seconds, r.base.gpu_seconds) {
                (Some(a), Some(b)) => format!("  gpu {:.2}%", 100.0 * (1.0 - a / b)),
                _ => String::new(),
            }
        );
    }
}

#[cfg(feature = "cuda")]
fn run_measure(cfg: &cv::measure::MeasureConfig, base: &Rules, extra: Option<&Rules>, gpu: bool) -> Result<cv::measure::Measure> {
    if !gpu {
        return cv::measure::measure(cfg, base, extra, None);
    }
    let m = cfg.n - cfg.a;
    let la = if let Some(w) = &cfg.class_b {
        Lookahead::meta(m, cfg.a, cfg.b, cfg.num_bv, cfg.bv_mode, &cfg.window).with_class_windows(w)
    } else if cfg.direct || cfg.b > 30 {
        Lookahead::meta(m, cfg.a, cfg.b, cfg.num_bv, cfg.bv_mode, &cfg.window)
    } else {
        Lookahead::build_ext(m, cfg.a, cfg.b, cfg.num_bv, cfg.bv_mode, &cfg.window)
    };
    let mut bench = cv::gpu::GpuBench::new(&la, cfg.tail.cutoff_bits, cfg.tail.tail_oee)?;
    let mut timer = |idx: usize, nodes: &[cv::tree::Node]| -> Result<(f64, u64, u64)> {
        let s = bench.run_median(idx, nodes, 3)?;
        Ok((s.kernel_seconds, s.cases, s.iters))
    };
    cv::measure::measure(cfg, base, extra, Some(&mut timer))
}

#[cfg(not(feature = "cuda"))]
fn run_measure(cfg: &cv::measure::MeasureConfig, base: &Rules, extra: Option<&Rules>, gpu: bool) -> Result<cv::measure::Measure> {
    if gpu {
        bail!("built without the `cuda` feature");
    }
    cv::measure::measure(cfg, base, extra, None)
}

#[cfg(feature = "cuda")]
fn gpu_check(la: &Lookahead) -> Result<(u64, u64, f64)> {
    cv::gpu::check_device_build(la)
}

#[cfg(not(feature = "cuda"))]
fn gpu_check(_la: &Lookahead) -> Result<(u64, u64, f64)> {
    bail!("built without the `cuda` feature")
}

#[cfg(feature = "cuda")]
fn run_gpu(cfg: &VerifyConfig) -> Result<cv::verify::VerifyReport> {
    cv::gpu::verify_gpu(cfg)
}

#[cfg(not(feature = "cuda"))]
fn run_gpu(_cfg: &VerifyConfig) -> Result<cv::verify::VerifyReport> {
    bail!("built without the `cuda` feature")
}
