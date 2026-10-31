//! Divert compact terminal groups to CPU workers before GPU packing.
//! GPU and CPU may use different M/A and lookahead windows. The source-depth
//! partition is audited separately from CPU-native descendant/tail counters.
#[allow(dead_code)] mod affine;
#[allow(dead_code)] mod terminal;
#[allow(dead_code)] mod ternary;
#[allow(dead_code)] mod fast_tail;
#[cfg(test)] #[allow(dead_code)] mod combined;
#[cfg(test)] #[allow(dead_code)] mod tail_policy;
#[cfg(test)] #[allow(dead_code)] mod terminal_walk;

use affine::Affine;
use cv::arith::{mod_pow3, POW3};
use cv::direct_cpu::{DirectWork, Plan};
use cv::expansion::Expander;
use cv::gpu::{GpuSession, GpuTransport, GpuTuning};
use cv::lookahead::{BvMode, WindowRules};
use cv::rules::{walkback_certificate, Rules};
use cv::sample::{Rng, Sampler};
use cv::tail::{recheck_wide, step_until_below, TailParams};
use cv::tree::{dfs, Node, Sink, TerminalWord, TreeStats};
use cv::verify::{verify_cpu_with_expander, VerifyConfig, VerifyReport};
use rayon::prelude::*;
use serde::Serialize;
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashSet};
use std::sync::{Arc, Mutex, atomic::{AtomicU64, AtomicUsize, Ordering}, mpsc::{sync_channel, SyncSender, TrySendError}};
use std::thread;
use std::time::Instant;

const F_SLOTS: usize = 81;

fn cpu_seconds(clock: i32) -> f64 {
    #[repr(C)] struct Timespec { seconds: std::os::raw::c_long, nanos: std::os::raw::c_long }
    unsafe extern "C" { fn clock_gettime(clock: i32, time: *mut Timespec) -> i32; }
    let mut time = Timespec { seconds: 0, nanos: 0 };
    assert_eq!(unsafe { clock_gettime(clock, &mut time) }, 0);
    time.seconds as f64 + time.nanos as f64 * 1e-9
}

fn env_number(name: &str, default: usize) -> usize {
    std::env::var(name).map(|s| s.parse().unwrap()).unwrap_or(default)
}

fn mix(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// Commutative descriptor identifier, not a cryptographic proof of coverage.
#[derive(Clone, Copy, Default, Serialize, Debug, PartialEq, Eq)]
struct Digest { count: u64, xor: u64, sum: u64 }
impl Digest {
    fn add(&mut self, key: u64) { self.count += 1; self.xor ^= key; self.sum = self.sum.wrapping_add(key); }
    fn merge(&mut self, other: Self) { self.count += other.count; self.xor ^= other.xor; self.sum = self.sum.wrapping_add(other.sum); }
}

fn group_key(parent: &Node, entry: u32, count: usize, target: u32) -> u64 {
    // Canonical parent residue/depth plus a range in this run's immutable
    // terminal table identifies a descriptor; the remaining Node fields are
    // determined by the prefix. Keep selection cheap enough for every group.
    debug_assert!(parent.k <= 64 && parent.n0 < 1u128 << parent.k);
    mix(parent.n0 as u64 ^ ((entry as u64) << 32)
        ^ (parent.k as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ (count as u64).wrapping_mul(0xd1b5_4a32_d192_ed03) ^ target as u64)
}

#[derive(Clone, Debug, Serialize)]
struct Variant { name: String, cpu_a: u32, cpu_b: Option<u32>, stride: u64, workers: usize, wrapper: bool }
impl Variant {
    fn parse(name: &str, gpu_a: u32) -> Self {
        if name == "gpu" || name == "route_control" {
            return Self { name: name.into(), cpu_a: gpu_a, cpu_b: None, stride: 0,
                workers: 0, wrapper: name == "route_control" };
        }
        let fields: Vec<_> = name.split('_').collect();
        assert_eq!(fields.len(), 5, "use direct_a8_b20_s1024_w4");
        assert_eq!(fields[0], "direct");
        let cpu_a = fields[1].strip_prefix('a').unwrap().parse().unwrap();
        let b = fields[2].strip_prefix('b').unwrap();
        let cpu_b = (b != "none").then(|| b.parse().unwrap());
        let stride = fields[3].strip_prefix('s').unwrap().parse().unwrap();
        let workers = fields[4].strip_prefix('w').unwrap().parse().unwrap();
        assert!(cpu_a <= gpu_a && stride > 0 && (1..=8).contains(&workers));
        Self { name: name.into(), cpu_a, cpu_b, stride, workers, wrapper: true }
    }
}

#[derive(Clone, Serialize)]
struct RoutingStats {
    original_by_f: Vec<u64>, gpu_by_f: Vec<u64>, cpu_by_f: Vec<u64>,
    groups: Digest, gpu_groups: Digest, cpu_groups: Digest,
    eligible_groups: u64, selected_groups: u64, queue_full_groups: u64, startup_blocked_groups: u64,
    copied_descriptor_bytes: u64, successful_descriptor_bytes: u64, rejected_descriptor_bytes: u64,
}
impl Default for RoutingStats {
    fn default() -> Self {
        Self { original_by_f: vec![0; F_SLOTS], gpu_by_f: vec![0; F_SLOTS], cpu_by_f: vec![0; F_SLOTS],
            groups: Digest::default(), gpu_groups: Digest::default(), cpu_groups: Digest::default(),
            eligible_groups: 0, selected_groups: 0, queue_full_groups: 0,
            startup_blocked_groups: 0, copied_descriptor_bytes: 0,
            successful_descriptor_bytes: 0, rejected_descriptor_bytes: 0 }
    }
}
impl RoutingStats {
    fn merge(&mut self, other: &Self) {
        for i in 0..F_SLOTS { self.original_by_f[i] += other.original_by_f[i]; self.gpu_by_f[i] += other.gpu_by_f[i]; self.cpu_by_f[i] += other.cpu_by_f[i]; }
        self.groups.merge(other.groups); self.gpu_groups.merge(other.gpu_groups); self.cpu_groups.merge(other.cpu_groups);
        self.eligible_groups += other.eligible_groups; self.selected_groups += other.selected_groups;
        self.queue_full_groups += other.queue_full_groups; self.startup_blocked_groups += other.startup_blocked_groups;
        self.copied_descriptor_bytes += other.copied_descriptor_bytes;
        self.successful_descriptor_bytes += other.successful_descriptor_bytes;
        self.rejected_descriptor_bytes += other.rejected_descriptor_bytes;
    }
}

struct Job { parent: Node, entries: Vec<TerminalWord>, target: u32, key: u64 }
struct Router {
    sender: Option<SyncSender<Job>>, stats: Mutex<RoutingStats>, stride: u64,
    classes: Vec<u32>, offset: u32, num_bv: usize,
    startup_gpu_endpoints: u64, gpu_forwarded_during_startup: AtomicU64,
    credits: Arc<AtomicUsize>, peak_credits: AtomicUsize, capacity: usize,
}
struct Routed<'a> { affine: &'a Affine, router: &'a Router }
struct Intercept<'a, S> { sink: &'a mut S, router: &'a Router, local: RoutingStats }

impl<S: Sink> Intercept<'_, S> {
    fn reserve(&self) -> bool {
        let credits = &self.router.credits;
        let mut old = credits.load(Ordering::Relaxed);
        loop {
            if old >= self.router.capacity { return false; }
            match credits.compare_exchange_weak(old, old + 1, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => {
                    self.router.peak_credits.fetch_max(old + 1, Ordering::Relaxed);
                    return true;
                }
                Err(actual) => old = actual,
            }
        }
    }
    fn forward_startup(&self, count: usize) {
        // Once the threshold is reached, this counter stops changing. It is
        // an initial feed guard, not a live GPU-queue occupancy measurement.
        if self.router.gpu_forwarded_during_startup.load(Ordering::Relaxed) < self.router.startup_gpu_endpoints {
            self.router.gpu_forwarded_during_startup.fetch_add(count as u64, Ordering::Relaxed);
        }
    }
}
impl<S: Sink> Sink for Intercept<'_, S> {
    fn endpoint(&mut self, node: &Node) { self.sink.endpoint(node); }
    fn compact_endpoint(&mut self, n0: u64, k: u32, f: u32, oee: u8, ne: u8) {
        self.sink.compact_endpoint(n0, k, f, oee, ne);
    }
    fn terminal_group(&mut self, parent: &Node, entry_start: u32, entries: &[TerminalWord], target: u32) {
        if entries.is_empty() { return; }
        let f = entries[0].f;
        debug_assert!(entries.iter().all(|entry| entry.f == f));
        let key = group_key(parent, entry_start, entries.len(), target);
        self.local.original_by_f[f as usize] += entries.len() as u64;
        self.local.groups.add(key);
        let gpu_class = f >= self.router.offset && (f - self.router.offset) < self.router.num_bv as u32;
        let eligible = gpu_class && self.router.classes.contains(&f);
        if eligible {
            self.local.eligible_groups += 1;
            if self.router.stride > 0 && key % self.router.stride == 0 {
                self.local.selected_groups += 1;
                if self.router.gpu_forwarded_during_startup.load(Ordering::Relaxed) >= self.router.startup_gpu_endpoints {
                    // Reserve bounded in-flight capacity before allocating or
                    // copying. A busy CPU pool costs only this failed try.
                    if self.reserve() {
                        let job = Job { parent: *parent, entries: entries.to_vec(), target, key };
                        let bytes = (std::mem::size_of::<Node>() + std::mem::size_of_val(entries)) as u64;
                        self.local.copied_descriptor_bytes += bytes;
                        match self.router.sender.as_ref().unwrap().try_send(job) {
                            Ok(()) => {
                                self.local.cpu_by_f[f as usize] += entries.len() as u64;
                                self.local.cpu_groups.add(key);
                                self.local.successful_descriptor_bytes += bytes;
                                return;
                            }
                            Err(TrySendError::Full(_)) => {
                                self.router.credits.fetch_sub(1, Ordering::Relaxed);
                                self.local.queue_full_groups += 1;
                                self.local.rejected_descriptor_bytes += bytes;
                            }
                            Err(TrySendError::Disconnected(_)) => {
                                self.router.credits.fetch_sub(1, Ordering::Relaxed);
                                panic!("direct CPU workers disconnected");
                            }
                        }
                    } else { self.local.queue_full_groups += 1; }
                } else { self.local.startup_blocked_groups += 1; }
            }
        }
        self.local.gpu_by_f[f as usize] += entries.len() as u64;
        self.local.gpu_groups.add(key);
        if gpu_class { self.forward_startup(entries.len()); }
        self.sink.terminal_group(parent, entry_start, entries, target);
    }
}
impl Expander for Routed<'_> {
    fn expand<S: Sink, const STATS: bool>(&self, root: &Node, target: u32, rules: &Rules, sink: &mut S, stats: &mut TreeStats) {
        let mut intercepted = Intercept { sink, router: self.router, local: RoutingStats::default() };
        self.affine.expand::<_, STATS>(root, target, rules, &mut intercepted, stats);
        self.router.stats.lock().unwrap().merge(&intercepted.local);
    }
}

struct WorkerDone {
    work: DirectWork, original_by_f: Vec<u64>, groups: Digest, sources: Vec<Node>,
    cpu_seconds: f64, reconstruct_seconds: f64, verification_seconds: f64,
}
impl Default for WorkerDone {
    fn default() -> Self { Self { work: DirectWork::default(), original_by_f: vec![0; F_SLOTS], groups: Digest::default(), sources: Vec::new(), cpu_seconds: 0.0, reconstruct_seconds: 0.0, verification_seconds: 0.0 } }
}

fn work_json(work: &DirectWork) -> Value {
    let mut flags = work.flagged.clone(); flags.sort_unstable();
    json!({"source_endpoints":work.source_endpoints,"endpoints":work.endpoints,
        "cases":work.cases,"iters":work.iters,"hash":format!("{:08X}",work.hash),
        "flagged":flags.iter().map(u128::to_string).collect::<Vec<_>>()})
}

struct Outcome { report: VerifyReport, routing: Option<RoutingStats>, cpu: WorkerDone,
    wall_seconds: f64, process_cpu_seconds: f64, drain_seconds: f64, wide_recheck_seconds: f64,
    startup_forwarded: u64, capture: bool, peak_credits: usize, capacity: usize }
impl Outcome {
    fn json(&self) -> Value {
        json!({"report":self.report,"routing":self.routing,"cpu_native":work_json(&self.cpu.work),
            "report_scope":"Existing GPU pipeline after diversion; CPU-native work is recorded separately.",
            "cpu_original_by_f":self.cpu.original_by_f,"cpu_groups":self.cpu.groups,
            "wall_seconds":self.wall_seconds,"process_cpu_seconds":self.process_cpu_seconds,
            "average_busy_cpu_cores":self.process_cpu_seconds/self.wall_seconds,
            "cpu_worker_seconds":self.cpu.cpu_seconds,"cpu_reconstruction_seconds":self.cpu.reconstruct_seconds,
            "cpu_verification_seconds":self.cpu.verification_seconds,"cpu_drain_seconds":self.drain_seconds,
            "cpu_wide_recheck_seconds":self.wide_recheck_seconds,
            "startup_gpu_endpoints_forwarded":self.startup_forwarded,
            "cpu_source_capture_enabled":self.capture,"captured_cpu_sources":self.cpu.sources.len(),
            "cpu_max_inflight_groups":self.peak_credits,"cpu_inflight_capacity_groups":self.capacity,
            "cpu_flagged_rechecked":self.cpu.work.flagged.len(),"partition_equal":true})
    }
}

fn tuning() -> GpuTuning { GpuTuning { overflow_vote: false, oee_mask: true, ..GpuTuning::default() } }

fn run(gpu: &mut GpuSession, cfg: &VerifyConfig, affine: &Affine, variant: &Variant,
       plan: Option<Arc<Plan>>, classes: &[u32], guard: u64, queue: usize, capture: bool) -> Outcome {
    let cpu_start = cpu_seconds(2);
    let started = Instant::now();
    let mut handles = Vec::new();
    let (sender, receiver) = sync_channel::<Job>(queue);
    let receiver = Arc::new(Mutex::new(receiver));
    let credits = Arc::new(AtomicUsize::new(0));
    for _ in 0..variant.workers {
        let receiver = Arc::clone(&receiver);
        let plan = Arc::clone(plan.as_ref().unwrap());
        let credits = Arc::clone(&credits);
        handles.push(thread::spawn(move || {
            let mut done = WorkerDone::default();
            loop {
                let job = match receiver.lock().unwrap().recv() { Ok(job) => job, Err(_) => break };
                let cpu = cpu_seconds(3);
                done.groups.add(job.key);
                let t = Instant::now();
                // Bounded parent-sized batches, reconstructed only on workers.
                let sources: Vec<_> = job.entries.iter().map(|word| word.endpoint(&job.parent, job.target)).collect();
                done.reconstruct_seconds += t.elapsed().as_secs_f64();
                let t = Instant::now();
                for source in &sources {
                    done.original_by_f[source.f as usize] += 1;
                    plan.process_into(source, &mut done.work);
                }
                done.verification_seconds += t.elapsed().as_secs_f64();
                if capture { done.sources.extend(sources); }
                done.cpu_seconds += cpu_seconds(3) - cpu;
                assert!(credits.fetch_sub(1, Ordering::Relaxed) > 0);
            }
            done
        }));
    }
    drop(receiver);
    let router = Router { sender: Some(sender), stats: Mutex::new(RoutingStats::default()), stride: variant.stride,
        classes: classes.to_vec(), offset: cfg.m() * 306 / 485 + 1, num_bv: cfg.num_bv,
        startup_gpu_endpoints: guard, gpu_forwarded_during_startup: AtomicU64::new(0),
        credits: Arc::clone(&credits), peak_credits: AtomicUsize::new(0), capacity: queue };
    let report = if variant.wrapper {
        gpu.run_transport_tuned(cfg, &Routed { affine, router: &router }, GpuTransport::Parents, affine.table_words(), tuning())
    } else {
        gpu.run_transport_tuned(cfg, affine, GpuTransport::Parents, affine.table_words(), tuning())
    }.expect("direct pipeline GPU run");
    let Router { sender, stats, gpu_forwarded_during_startup, peak_credits, .. } = router;
    drop(sender);
    let drain = Instant::now();
    let mut done = WorkerDone::default();
    for handle in handles {
        let part = handle.join().expect("direct CPU worker panicked");
        done.work.merge(&part.work); done.groups.merge(part.groups);
        for (a, b) in done.original_by_f.iter_mut().zip(part.original_by_f) { *a += b; }
        done.sources.extend(part.sources);
        done.cpu_seconds += part.cpu_seconds;
        done.reconstruct_seconds += part.reconstruct_seconds;
        done.verification_seconds += part.verification_seconds;
    }
    let drain_seconds = drain.elapsed().as_secs_f64();
    assert_eq!(credits.load(Ordering::Relaxed), 0);
    // Both ordinary and diverted work must finish their wide rechecks before
    // the pipeline timer stops; duplicate flagged originals are retained.
    assert_eq!(report.flagged.len(), report.flagged_rechecked);
    let wide = Instant::now();
    assert!(done.work.flagged.iter().all(|&n| recheck_wide(n).is_some()));
    let wide_recheck_seconds = wide.elapsed().as_secs_f64();
    let wall_seconds = started.elapsed().as_secs_f64();
    let process_cpu_seconds = cpu_seconds(2) - cpu_start;
    let routing = variant.wrapper.then(|| stats.into_inner().unwrap());
    if let Some(routing) = &routing {
        assert_eq!(routing.cpu_by_f, done.original_by_f);
        assert_eq!(routing.cpu_groups, done.groups);
        assert_eq!(routing.selected_groups, routing.cpu_groups.count + routing.queue_full_groups + routing.startup_blocked_groups);
        assert_eq!(routing.copied_descriptor_bytes, routing.successful_descriptor_bytes + routing.rejected_descriptor_bytes);
        let mut all = routing.gpu_groups; all.merge(routing.cpu_groups); assert_eq!(all, routing.groups);
        for f in 0..F_SLOTS { assert_eq!(routing.original_by_f[f], routing.gpu_by_f[f] + routing.cpu_by_f[f]); }
        assert_eq!(routing.original_by_f.iter().sum::<u64>(), report.tree.survived[cfg.m() as usize]);
        assert_eq!(routing.gpu_by_f.iter().sum::<u64>(), report.endpoints);
        assert_eq!(routing.cpu_by_f.iter().sum::<u64>(), done.work.source_endpoints);
        let gpu_classes = report.gpu.as_ref().unwrap()["classes"].as_array().unwrap();
        let offset = cfg.m() * 306 / 485 + 1;
        for (i, class) in gpu_classes.iter().enumerate() {
            assert_eq!(class["endpoints"].as_u64().unwrap(), routing.gpu_by_f[offset as usize + i]);
        }
        let fallback: u64 = routing.gpu_by_f.iter().enumerate()
            .filter(|(f, _)| *f < offset as usize || *f >= offset as usize + cfg.num_bv).map(|(_, n)| *n).sum();
        assert_eq!(fallback, report.tail.endpoints[cfg.num_bv]);
    }
    Outcome { report, routing, cpu: done, wall_seconds, process_cpu_seconds, drain_seconds, wide_recheck_seconds,
        startup_forwarded: gpu_forwarded_during_startup.into_inner(), capture,
        peak_credits: peak_credits.into_inner(), capacity: queue }
}

fn config(n: u32, a: u32, b: u32, num_bv: usize) -> VerifyConfig {
    let rules = Rules::improved(13);
    VerifyConfig { n, a, b, num_bv, bv_mode: BvMode::Shared,
        window: WindowRules { pm_inf: true, mix: rules.mix.clone() }, class_b: None,
        split: 10, rules, tail: TailParams::default(), units: None }
}

/// Independent structural oracle: normal Node children/culls, direct window
/// evaluation, and the original scalar verifier. No compiled CPU suffix table.
fn reference_sources(plan: &Plan, sources: &[Node], cutoff_bits: u32) -> DirectWork {
    struct Oracle<'a> { plan: &'a Plan, work: DirectWork, cutoff_bits: u32 }
    impl Sink for Oracle<'_> {
        fn endpoint(&mut self, node: &Node) {
            self.work.endpoints += 1;
            let la = self.plan.lookahead();
            let r9 = mod_pow3(node.n0, 2) as usize;
            for h in 0..(1u128 << self.plan.a()) {
                if la.completion_word(None, 0, r9, h as usize / 64) & (1u64 << (h as usize % 64)) == 0 { continue; }
                let base = node.n0 + (h << self.plan.m());
                let Some(x) = POW3[node.f as usize].checked_mul(h).and_then(|x| x.checked_add(node.m)).filter(|&x| x < 1u128 << 123) else {
                    self.work.cases += 1; self.work.flagged.push(base); continue;
                };
                if self.plan.b().is_some() && !la.keeps(node.f, node.ne, (x + 1) as u64) { continue; }
                if (node.oee == 2 && x & 1 == 0) || (node.oee == 1 && x & 3 == 0) { continue; }
                let (hash, iterations, flagged) = step_until_below(base, x + 1, 1u128 << self.cutoff_bits);
                self.work.cases += 1; self.work.iters += iterations; self.work.hash ^= hash;
                if flagged { self.work.flagged.push(base); }
            }
        }
    }
    let mut oracle = Oracle { plan, work: DirectWork::default(), cutoff_bits };
    let baseline = Rules::baseline();
    for source in sources {
        dfs::<_, false>(source, plan.m(), &baseline, &mut oracle, &mut TreeStats::default());
        oracle.work.source_endpoints += 1;
    }
    oracle.work
}

fn complete_signature(report: &VerifyReport, extra: &DirectWork) -> Value {
    let mut flags: Vec<u128> = report.flagged.iter().map(|s| s.parse().unwrap()).collect();
    flags.extend(extra.flagged.iter().copied()); flags.sort_unstable();
    json!({"endpoints":report.endpoints+extra.endpoints,"cases":report.cases+extra.cases,
        "iters":report.iters+extra.iters,"hash":format!("{:08X}",report.tail.hash^extra.hash),
        "flags":flags.iter().map(u128::to_string).collect::<Vec<_>>(),"tree":report.tree})
}

fn validate(profile: &str, threads: usize) {
    let wide = profile != "validate";
    let overflow = profile == "validate-device";
    let mut cfg = config(if wide { 72 } else { 32 }, 10, 12, if overflow { 24 } else { 2 });
    if wide {
        let baseline = Rules::baseline();
        let sampler = Sampler::new(&baseline, 52, None);
        let mut rng = Rng(31_419);
        let mut roots = Vec::new();
        if overflow {
            cfg.tail.cutoff_bits = 110;
            let mut high = Node::root(&baseline);
            while high.k < 52 { high = high.children(&baseline)[1].unwrap(); }
            assert_eq!(high.n0, (1u128 << 52) - 1);
            roots.push(high);
        }
        while roots.len() < 16 + usize::from(overflow) {
            let draw = sampler.draw(&mut rng, Some(&cfg.rules));
            if draw.culled.is_none() { roots.push(draw.node); }
        }
        cfg.units = Some(roots);
    } else { cfg.rules.walkback = Some(cv::rules::Walkback { depth: 12, max_states: 256 }); }
    let affine = Affine::new(cfg.m(), if wide { 50 } else { 16 }, if wide { 52 } else { 18 }, &cfg.rules);
    let cpu = verify_cpu_with_expander(&cfg, &affine);
    let expected = complete_signature(&cpu, &DirectWork::default());
    let mut gpu = GpuSession::new_experimental(&cfg).expect("validation GPU setup");
    gpu.set_terminal_table(affine.table_words()).unwrap();
    println!("{}", json!({"type":"config","experiment":"direct_pipeline","profile":profile,"threads":threads,
        "scope":if wide { "N72 packages, exact same-A oracle and mixed-A source replay" } else { "complete N32, exact same-A oracle and mixed-A source replay" },
        "cutoff_bits":cfg.tail.cutoff_bits,"roots":cfg.units}));
    let names = if overflow {
        ["gpu", "route_control", "direct_a10_b12_s1_w2", "direct_a8_b12_s1_w2", "direct_a6_bnone_s1_w2", "direct_a0_b12_s1_w2"]
    } else {
        ["gpu", "route_control", "direct_a10_b12_s1_w2", "direct_a8_b12_s1_w2", "direct_a6_bnone_s2_w2", "direct_a0_b12_s2_w2"]
    };
    // The forced-overflow input has one all-odd M62 endpoint. Selecting only
    // its F62 class guarantees CPU arithmetic flags without relying on queue
    // timing among unrelated classes.
    let classes = if overflow { vec![cfg.m()] } else { vec![cfg.m() * 306 / 485 + 1, cfg.m() * 306 / 485 + 2] };
    for name in names {
        let variant = Variant::parse(name, cfg.a);
        let plan = (variant.workers > 0).then(|| Arc::new(Plan::new(cfg.n, cfg.m(), variant.cpu_a, variant.cpu_b, cfg.num_bv, &cfg.window).with_cutoff_bits(cfg.tail.cutoff_bits)));
        let outcome = run(&mut gpu, &cfg, &affine, &variant, plan.clone(), &classes, 0, 8, true);
        if variant.workers > 0 {
            assert!(outcome.cpu.work.source_endpoints > 0, "validation did not route CPU work");
            assert_eq!(outcome.cpu.sources.len() as u64, outcome.cpu.work.source_endpoints);
            let reference = reference_sources(plan.as_ref().unwrap(), &outcome.cpu.sources, cfg.tail.cutoff_bits);
            assert_eq!(work_json(&reference), work_json(&outcome.cpu.work), "CPU policy oracle {name}");
            if wide {
                assert!(outcome.cpu.sources.iter().any(|node| node.m > u64::MAX as u128), "wide CPU iterate fixture");
            }
            if overflow { assert!(!outcome.cpu.work.flagged.is_empty(), "forced cutoff must flag diverted CPU work"); }
        }
        let same_policy = variant.workers == 0 || (variant.cpu_a == cfg.a && variant.cpu_b == Some(cfg.b));
        if same_policy { assert_eq!(complete_signature(&outcome.report, &outcome.cpu.work), expected, "same-policy {profile} {name}"); }
        println!("{}", json!({"type":"validation","variant":variant,"source_partition_equal":true,
            "cpu_policy_oracle_equal":true,"same_policy_complete_signature_equal":same_policy,"result":outcome.json()}));
    }
    println!("{}", json!({"type":"complete","validation_count":names.len()}));
}

fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let profile = args.first().map(String::as_str).unwrap_or("validate");
    let number = |index: usize, default: usize| args.get(index).map(|s| s.parse().unwrap()).unwrap_or(default);
    let threads = number(3, 2);
    rayon::ThreadPoolBuilder::new().num_threads(threads).build_global().unwrap();
    if matches!(profile, "validate" | "validate-wide" | "validate-device") { validate(profile, threads); return; }
    assert_eq!(profile, "target", "supported profiles are validate, validate-wide, validate-device and target");
    let (drawn, repetitions, seed) = (number(1, 128), number(2, 2), number(4, 20261103) as u64);
    let a = number(7, 10) as u32;
    assert!((8..=10).contains(&a));
    let variants: Vec<_> = args.get(5).map(String::as_str).unwrap_or("gpu,route_control,direct_a8_b20_s1024_w4")
        .split(',').map(|name| Variant::parse(name, a)).collect();
    assert!(!variants.is_empty() && variants.iter().map(|v| &v.name).collect::<HashSet<_>>().len() == variants.len());
    let count = env_number("CV_FULLRANGE_COHORTS", 4);
    let cutoff = env_number("CV_FULLRANGE_CUTOFF", 50) as u32;
    let entry = env_number("CV_FULLRANGE_ENTRY", 52) as u32;
    let aggregates = env_number("CV_FULLRANGE_AGGREGATE", 2);
    let guard = env_number("CV_DIRECT_STARTUP_ENDPOINTS", 1 << 22) as u64;
    let queue = env_number("CV_DIRECT_QUEUE", 64);
    assert!(drawn > 0 && repetitions > 0 && count > 0 && queue > 0 && cutoff >= 38);
    let mut cfg = config(72, a, 27, 16);
    cfg.class_b = Some(args.get(6).map(String::as_str).unwrap_or("32,35,35,35,35,34,32,32,27,27,27,27,27,27,27,27")
        .split(',').map(|s| s.parse().unwrap()).collect());
    assert_eq!(cfg.class_b.as_ref().unwrap().len(), cfg.num_bv);
    let offset = cfg.m() * 306 / 485 + 1;
    let classes: Vec<u32> = std::env::var("CV_DIRECT_CLASSES").unwrap_or_else(|_| offset.to_string()).split(',').map(|s| s.parse().unwrap()).collect();
    assert!(!classes.is_empty() && classes.iter().all(|&f| f >= offset && f < offset + cfg.num_bv as u32));
    let started = Instant::now();
    let affine = Affine::new(cfg.m(), cutoff, entry, &cfg.rules);
    let mut resident_plans = BTreeMap::new();
    let plans: Vec<_> = variants.iter().map(|variant| (variant.workers > 0).then(|| {
        Arc::clone(resident_plans.entry((variant.cpu_a, variant.cpu_b)).or_insert_with(||
            Arc::new(Plan::new(cfg.n, cfg.m(), variant.cpu_a, variant.cpu_b, cfg.num_bv, &cfg.window).with_cutoff_bits(cfg.tail.cutoff_bits))))
    })).collect();
    println!("{}", json!({"type":"config","experiment":"direct_pipeline","profile":profile,
        "n":cfg.n,"gpu_a":cfg.a,"gpu_m":cfg.m(),"gpu_class_b":cfg.class_b,"num_bv":cfg.num_bv,
        "rules":cfg.rules.label(),"window":cfg.window.label(),"cutoff_bits":cfg.tail.cutoff_bits,
        "threads":threads,"draws_per_cohort":drawn,"cohort_count":count,"repetitions":repetitions,
        "aggregate_repetitions":aggregates,"seed":seed,"checkpoint":38,"walkback_cap":65536,
        "cutoff":cutoff,"terminal_entry":entry,"variants":variants,"selected_original_f":classes,
        "queue_capacity_groups":queue,"startup_gpu_endpoint_guard":guard,
        "setup_seconds":started.elapsed().as_secs_f64(),
        "cpu_plan_bytes":plans.iter().map(|p| p.as_ref().map(|p| json!({"table":p.table_bytes(),"lookahead":p.lookahead_bytes()}))).collect::<Vec<_>>(),
        "routing":"Deterministic descriptor-hash candidates; nonblocking admission; full CPU queue falls back to unchanged GPU group. Startup guard is not a live queue watermark.",
        "timing_scope":"Warm pipeline including worker creation, dispatch, CPU worker drain and wide rechecks; resident tables and input sampling excluded.",
        "coverage":"Original-depth descriptor partition conserved. CPU-native descendants and tail signatures use their own M/A/B; cross-policy hashes may differ.",
        "fallback":"Existing out-of-range-F CPU fallback remains unchanged on the producer.",
        "gpu_by_f_note":"Routing gpu_by_f counts groups forwarded to the existing GPU sink, including that sink's ordinary CPU fallback; subtract out-of-range F for actual GPU endpoints.",
        "descriptor_bytes_scope":"Node struct plus copied terminal words; excludes Vec and channel bookkeeping. Credits are reserved before copying, so a full CPU pool does not copy rejected groups.",
        "kernel_seconds_meaning":"Overlapping CUDA event sums, not additive GPU wall seconds"}));
    let sampler = Sampler::new(&Rules::baseline(), 38, None);
    let mut seeder = Rng(seed);
    let mut cohorts = Vec::new();
    for index in 0..count {
        let cohort_seed = seeder.next_u64();
        let draws = sampler.draws(drawn, cohort_seed, Some(&cfg.rules));
        let classified: Vec<_> = draws.into_par_iter().map(|draw| {
            let reason = if draw.culled.is_some() { 1 } else if walkback_certificate(draw.node.n0, draw.node.k, draw.node.f, draw.node.m, 65536).is_some() { 2 } else { 0 };
            (draw.node, reason)
        }).collect();
        let cheap = classified.iter().filter(|(_, why)| *why == 1).count();
        let walk = classified.iter().filter(|(_, why)| *why == 2).count();
        let roots: Vec<_> = classified.into_iter().filter_map(|(node, why)| (why == 0).then_some(node)).collect();
        assert_eq!(roots.len()+cheap+walk, drawn);
        println!("{}", json!({"type":"cohort","cohort":index,"seed":cohort_seed,"drawn":drawn,
            "cheap_rejected":cheap,"walk_rejected":walk,"accepted":roots.len(),
            "duplicates_retained":roots.len()-roots.iter().map(|n| n.n0).collect::<HashSet<_>>().len(),"roots":roots}));
        cohorts.push(roots);
    }
    let started = Instant::now();
    let mut gpu = GpuSession::new_experimental(&cfg).expect("GPU setup");
    gpu.set_terminal_table(affine.table_words()).unwrap();
    println!("{}", json!({"type":"gpu_setup","wall_seconds":started.elapsed().as_secs_f64()}));
    cfg.units = Some(cohorts.iter().flatten().take(32).copied().collect());
    for (variant, plan) in variants.iter().zip(&plans) {
        let result = run(&mut gpu, &cfg, &affine, variant, plan.clone(), &classes, guard, queue, false);
        println!("{}", json!({"type":"warmup","variant":variant.name,"result":result.json()}));
    }
    let mut expected_trees: Vec<Option<Value>> = vec![None; count];
    let mut expected_controls: Vec<Option<Value>> = vec![None; count];
    for repetition in 0..repetitions {
        for position in 0..count {
            let cohort = (position + repetition) % count;
            cfg.units = Some(cohorts[cohort].clone());
            for order in 0..variants.len() {
                let index = (order + cohort + repetition) % variants.len();
                let result = run(&mut gpu, &cfg, &affine, &variants[index], plans[index].clone(), &classes, guard, queue, false);
                let tree = serde_json::to_value(&result.report.tree).unwrap();
                if let Some(expected) = &expected_trees[cohort] { assert_eq!(*expected, tree); } else { expected_trees[cohort] = Some(tree); }
                if variants[index].workers == 0 {
                    let signature = complete_signature(&result.report, &result.cpu.work);
                    if let Some(expected) = &expected_controls[cohort] { assert_eq!(*expected, signature); }
                    else { expected_controls[cohort] = Some(signature); }
                }
                println!("{}", json!({"type":"trial","cohort":cohort,"repetition":repetition,"order_position":order,
                    "variant":variants[index].name,"drawn":drawn,"result":result.json()}));
            }
        }
    }
    let roots: Vec<_> = cohorts.iter().flatten().copied().collect();
    println!("{}", json!({"type":"aggregate_config","drawn":drawn*count,"accepted":roots.len(),"roots":roots}));
    cfg.units = Some(roots);
    let mut aggregate_tree = None;
    let mut aggregate_control = None;
    for repetition in 0..aggregates {
        for order in 0..variants.len() {
            let index = (order + repetition) % variants.len();
            let result = run(&mut gpu, &cfg, &affine, &variants[index], plans[index].clone(), &classes, guard, queue, false);
            let tree = serde_json::to_value(&result.report.tree).unwrap();
            if let Some(expected) = &aggregate_tree { assert_eq!(*expected, tree); } else { aggregate_tree = Some(tree); }
            if variants[index].workers == 0 {
                let signature = complete_signature(&result.report, &result.cpu.work);
                if let Some(expected) = &aggregate_control { assert_eq!(*expected, signature); }
                else { aggregate_control = Some(signature); }
            }
            println!("{}", json!({"type":"aggregate_trial","repetition":repetition,"order_position":order,
                "variant":variants[index].name,"drawn":drawn*count,"result":result.json()}));
        }
    }
    println!("{}", json!({"type":"complete","cohorts":count,"variants":variants.len(),"repetitions":repetitions,"aggregate_repetitions":aggregates}));
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn full_cpu_pool_forwards_group_without_copying() {
        struct Forwarded(usize);
        impl Sink for Forwarded {
            fn endpoint(&mut self, _: &Node) { unreachable!(); }
            fn terminal_group(&mut self, _: &Node, _: u32, entries: &[TerminalWord], _: u32) { self.0 += entries.len(); }
        }
        let (sender, receiver) = sync_channel(1);
        let credits = Arc::new(AtomicUsize::new(0));
        let router = Router { sender: Some(sender), stats: Mutex::new(RoutingStats::default()), stride: 1,
            classes: vec![14], offset: 14, num_bv: 2, startup_gpu_endpoints: 0,
            gpu_forwarded_during_startup: AtomicU64::new(0), credits: Arc::clone(&credits),
            peak_credits: AtomicUsize::new(0), capacity: 1 };
        let mut forwarded = Forwarded(0);
        let mut sink = Intercept { sink: &mut forwarded, router: &router, local: RoutingStats::default() };
        let parent = Node::root(&Rules::baseline());
        let entries = [TerminalWord { r: 0, q_coeff: 0, iterate: 0, pow3: 1, f: 14, oee: 0, ne: 0, pad: 0 }];
        sink.terminal_group(&parent, 0, &entries, 22);
        let copied = sink.local.copied_descriptor_bytes;
        assert!(copied > 0);
        sink.terminal_group(&parent, 0, &entries, 22);
        assert_eq!(sink.local.copied_descriptor_bytes, copied);
        assert_eq!(sink.local.rejected_descriptor_bytes, 0);
        assert_eq!(sink.local.queue_full_groups, 1);
        assert_eq!(sink.local.cpu_by_f[14], 1);
        assert_eq!(sink.local.gpu_by_f[14], 1);
        let mut partition = sink.local.cpu_groups; partition.merge(sink.local.gpu_groups);
        assert_eq!(partition, sink.local.groups);
        assert_eq!(receiver.try_recv().unwrap().entries, entries);
        credits.fetch_sub(1, Ordering::Relaxed);
        sink.terminal_group(&parent, 0, &entries, 22);
        assert_eq!(sink.local.cpu_by_f[14], 2);
        drop(sink);
        assert_eq!(forwarded.0, 1);
    }
    #[test]
    fn cpu_policy_oracle_checks_deeper_extension() {
        let rules = Rules::baseline();
        let sampler = Sampler::new(&rules, 18, None);
        let sources: Vec<_> = sampler.draws(8, 42, None).into_iter().map(|d| d.node).collect();
        let window = WindowRules::default();
        for a in [10, 8, 6, 0] { for b in [None, Some(12)] {
            let plan = Plan::new(28, 18, a, b, 4, &window);
            let expected = reference_sources(&plan, &sources, 120);
            let mut actual = DirectWork::default(); for source in &sources { plan.process_into(source, &mut actual); }
            assert_eq!(work_json(&expected), work_json(&actual));
        } }
    }
}
