//! Bounded CPU work after GPU candidate screening, before trajectory stepping.
use super::GpuStats;
use crate::early_cpu::verify_states;
use crate::tail::TailContinuation;
use anyhow::{anyhow, ensure, Context, Result};
use std::io::{BufWriter, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{sync_channel, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Instant;

const ENDPOINTS_PER_CHUNK: usize = 1024;

struct Job {
    class: usize,
    m: u32,
    power3: u128,
    words32: usize,
    affine: Vec<u32>,
    masks: Vec<u32>,
}

fn decode(job: &Job, limit: usize) -> Result<Vec<TailContinuation>> {
    decode_range(job, 0, job.affine.len() / 8, limit)
}

fn decode_range(job: &Job, start: usize, end: usize, limit: usize) -> Result<Vec<TailContinuation>> {
    ensure!(job.m < 128 && job.words32 > 0 && job.affine.len() % 8 == 0,
        "invalid early endpoint shape");
    let endpoints = job.affine.len() / 8;
    ensure!(endpoints <= super::BATCH && job.masks.len() == endpoints * job.words32
        && start <= end && end <= endpoints,
        "invalid early mask shape");
    let masks = &job.masks[start * job.words32..end * job.words32];
    let affine = &job.affine[start * 8..end * 8];
    let count = masks.iter().map(|w| w.count_ones() as usize).sum::<usize>().min(limit);
    let mut states = Vec::with_capacity(count);
    for (affine, masks) in affine.chunks_exact(8).zip(masks.chunks_exact(job.words32)) {
        let n0 = affine[0] as u128 | (affine[1] as u128) << 32
            | (affine[2] as u128) << 64 | ((affine[3] & 0x1fff_ffff) as u128) << 96;
        let iterate = affine[4] as u128 | (affine[5] as u128) << 32
            | (affine[6] as u128) << 64 | (affine[7] as u128) << 96;
        for (word, &mask) in masks.iter().enumerate() {
            let mut mask = mask;
            while mask != 0 {
                if states.len() == count { return Ok(states); }
                let aval = (32 * word + mask.trailing_zeros() as usize) as u128;
                mask &= mask - 1;
                ensure!(aval <= (u128::MAX >> job.m), "early candidate shift overflow");
                let base = n0.checked_add(aval << job.m).context("early candidate base overflow")?;
                let number = aval.checked_mul(job.power3).and_then(|x| x.checked_add(iterate))
                    .and_then(|x| x.checked_add(1)).context("early candidate iterate overflow")?;
                states.push(TailContinuation::from_values(base, number));
            }
        }
    }
    Ok(states)
}

#[derive(Default)]
struct Completed {
    cases: Vec<u64>,
    iterations: Vec<u64>,
    hash: u32,
    flagged: Vec<u128>,
    cpu_seconds: f64,
    decode_seconds: f64,
    grouping_seconds: f64,
    verification_seconds: f64,
    table_setup_seconds: f64,
    buckets: u64,
    jumps: u64,
    jumped_iterations: u64,
    scalar_iterations: u64,
    table_lookups: u64,
}

fn thread_cpu_seconds() -> f64 {
    #[repr(C)]
    struct Timespec { seconds: std::os::raw::c_long, nanoseconds: std::os::raw::c_long }
    unsafe extern "C" { fn clock_gettime(clock: i32, time: *mut Timespec) -> i32; }
    let mut t = Timespec { seconds: 0, nanoseconds: 0 };
    assert_eq!(unsafe { clock_gettime(3, &mut t) }, 0);
    t.seconds as f64 + t.nanoseconds as f64 * 1e-9
}

struct Capture {
    writer: BufWriter<std::fs::File>,
    limit: usize,
    cases: usize,
}

pub(super) struct EarlyWorkers {
    sender: Option<SyncSender<Job>>,
    workers: Vec<JoinHandle<Result<Completed>>>,
    send_seconds: f64,
    copy_seconds: f64,
    submitted_cases: u64,
    capture: Option<Capture>,
}

impl EarlyWorkers {
    pub fn new(classes: usize, workers: usize, cutoff_bits: u32, width: u32, mode: u32) -> Result<Self> {
        ensure!(classes > 0 && (1..=8).contains(&workers), "invalid early worker count");
        ensure!(matches!(width, 8 | 10 | 12 | 14) && mode <= 4, "invalid early CPU mode");
        // The optional diagnostic capture is claimed by the first early run in
        // this process. Benchmark runners claim it during warmup and audit
        // that subsequent timed trials report zero captured states.
        static CAPTURE_CLAIMED: AtomicBool = AtomicBool::new(false);
        let capture = if let Some(path) = std::env::var_os("CV_EARLY_CAPTURE") {
            if !CAPTURE_CLAIMED.swap(true, Ordering::SeqCst) {
                let limit = std::env::var("CV_EARLY_CAPTURE_LIMIT").unwrap_or_else(|_| "262144".into())
                    .parse::<usize>().context("invalid CV_EARLY_CAPTURE_LIMIT")?;
                ensure!(limit > 0, "CV_EARLY_CAPTURE_LIMIT must be positive");
                Some(Capture { writer: BufWriter::new(std::fs::OpenOptions::new().write(true)
                    .create_new(true).open(path).context("create early CPU fixture")?), limit, cases: 0 })
            } else { None }
        } else { None };
        // Each launch contributes at most one compact shard per worker. This
        // lets GPU dispatch continue without waiting for per-chunk CPU work.
        // At most two launches are queued and one launch is being processed;
        // state vectors remain bounded to ENDPOINTS_PER_CHUNK inside workers.
        let (sender, receiver) = sync_channel::<Job>(2 * workers);
        let receiver = Arc::new(Mutex::new(receiver));
        let workers = (0..workers).map(|_| {
            let receiver = Arc::clone(&receiver);
            std::thread::spawn(move || {
                let mut done = Completed { cases: vec![0; classes], iterations: vec![0; classes],
                    ..Default::default() };
                let mut error = None;
                loop {
                    let Ok(job) = receiver.lock().unwrap().recv() else { break; };
                    let cpu = thread_cpu_seconds();
                    let result = (|| -> Result<()> {
                        ensure!(job.class < classes, "invalid early CPU class");
                        for start in (0..job.affine.len() / 8).step_by(ENDPOINTS_PER_CHUNK) {
                            let end = (start + ENDPOINTS_PER_CHUNK).min(job.affine.len() / 8);
                            let decoding = Instant::now();
                            let states = decode_range(&job, start, end, usize::MAX)?;
                            done.decode_seconds += decoding.elapsed().as_secs_f64();
                            let result = verify_states(&states, cutoff_bits, width, mode)?;
                            ensure!(result.cases == states.len() as u64, "early CPU case mismatch");
                            done.cases[job.class] += result.cases;
                            done.iterations[job.class] += result.iters;
                            done.hash ^= result.hash;
                            done.flagged.extend(result.flagged);
                            done.grouping_seconds += result.grouping_seconds;
                            done.verification_seconds += result.verification_seconds;
                            done.table_setup_seconds += result.table_setup_seconds;
                            done.buckets += result.buckets;
                            done.jumps += result.jumps;
                            done.jumped_iterations += result.jumped_iterations;
                            done.scalar_iterations += result.scalar_iterations;
                            done.table_lookups += result.table_lookups;
                        }
                        Ok(())
                    })();
                    if let Err(e) = result { error.get_or_insert(e); }
                    done.cpu_seconds += thread_cpu_seconds() - cpu;
                    // Continue consuming after errors so blocked producers drain.
                }
                if let Some(error) = error { return Err(error); }
                Ok(done)
            })
        }).collect();
        Ok(Self { sender: Some(sender), workers, send_seconds: 0.0, copy_seconds: 0.0,
            submitted_cases: 0, capture })
    }

    pub fn submit(&mut self, class: usize, m: u32, power3: u128, words32: usize,
                  affine: &[u32], masks: &[u32]) -> Result<()> {
        ensure!(affine.len() % 8 == 0 && affine.len() / 8 <= super::BATCH
            && masks.len() == affine.len() / 8 * words32 && words32 > 0,
            "invalid early CPU batch shape");
        let copy_start = thread_cpu_seconds();
        let cases = masks.iter().map(|w| w.count_ones() as u64).sum::<u64>();
        if cases == 0 {
            self.copy_seconds += thread_cpu_seconds() - copy_start;
            return Ok(());
        }
        let job = Job { class, m, power3, words32, affine: affine.to_vec(), masks: masks.to_vec() };
        self.copy_seconds += thread_cpu_seconds() - copy_start;
        if let Some(capture) = self.capture.as_mut() {
            if capture.cases < capture.limit {
                let states = decode(&job, capture.limit - capture.cases)?;
                for state in &states {
                    for limb in state.limbs { capture.writer.write_all(&limb.to_le_bytes())?; }
                }
                capture.cases += states.len();
            }
        }
        let start = Instant::now();
        let result = self.sender.as_ref().unwrap().send(job);
        self.send_seconds += start.elapsed().as_secs_f64();
        result.map_err(|_| anyhow!("early CPU workers stopped"))?;
        self.submitted_cases += cases;
        Ok(())
    }

    pub fn finish(mut self, stats: &mut GpuStats) -> Result<()> {
        let drain = Instant::now();
        drop(self.sender.take());
        stats.early_class_endpoints.resize(stats.classes.len(), 0);
        stats.early_class_cases.resize(stats.classes.len(), 0);
        stats.early_class_iterations.resize(stats.classes.len(), 0);
        let mut cases = 0;
        let mut error = None;
        for worker in self.workers.drain(..) {
            match worker.join().map_err(|_| anyhow!("early CPU worker panicked")) {
                Ok(Ok(done)) => {
                    cases += done.cases.iter().sum::<u64>();
                    for (idx, ((class, count), iterations)) in stats.classes.iter_mut()
                        .zip(done.cases).zip(done.iterations).enumerate() {
                        class.cases += count;
                        class.iters += iterations;
                        stats.early_class_cases[idx] += count;
                        stats.early_class_iterations[idx] += iterations;
                        stats.early_iterations += iterations;
                    }
                    stats.early_hash ^= done.hash;
                    stats.early_flags += done.flagged.len() as u64;
                    stats.flagged.extend(done.flagged);
                    stats.early_cpu_seconds += done.cpu_seconds;
                    stats.early_decode_seconds += done.decode_seconds;
                    stats.early_group_seconds += done.grouping_seconds;
                    stats.early_verification_seconds += done.verification_seconds;
                    stats.early_table_setup_seconds += done.table_setup_seconds;
                    stats.early_buckets += done.buckets;
                    stats.early_jumps += done.jumps;
                    stats.early_jumped_iterations += done.jumped_iterations;
                    stats.early_scalar_iterations += done.scalar_iterations;
                    stats.early_table_lookups += done.table_lookups;
                }
                Ok(Err(e)) | Err(e) => { error.get_or_insert(e); }
            }
        }
        if let Some(error) = error { return Err(error); }
        ensure!(cases == self.submitted_cases, "not every screened CPU case completed");
        stats.early_cases = cases;
        stats.hash ^= stats.early_hash;
        stats.flags += stats.early_flags;
        stats.early_queue_wait_seconds = self.send_seconds;
        stats.early_host_copy_seconds = self.copy_seconds;
        stats.early_drain_seconds = drain.elapsed().as_secs_f64();
        if let Some(capture) = self.capture.as_mut() {
            capture.writer.flush()?;
            stats.early_capture_cases = capture.cases as u64;
        }
        Ok(())
    }
}

impl Drop for EarlyWorkers {
    fn drop(&mut self) {
        drop(self.sender.take());
        for worker in self.workers.drain(..) { let _ = worker.join(); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn endpoint(n0: u128, iterate: u128) -> Vec<u32> {
        let mut words = TailContinuation::from_values(n0, iterate).limbs.to_vec();
        words[3] |= 3 << 29; // Metadata must not become bits of the start value.
        words
    }

    #[test]
    fn masks_reconstruct_high_bits_and_word_boundaries() {
        let job = Job { class: 0, m: 64, power3: 81, words32: 2,
            affine: endpoint(27, (1u128 << 100) + 3), masks: vec![0x8000_0001, 3] };
        let states = decode(&job, usize::MAX).unwrap();
        assert_eq!(states.len(), 4);
        for (state, aval) in states.iter().zip([0u128, 31, 32, 33]) {
            assert_eq!(state.base(), 27 + (aval << 64));
            assert_eq!(state.number(), (1u128 << 100) + 4 + aval * 81);
        }
        assert_eq!(decode(&job, 2).unwrap(), states[..2]);
    }

    #[test]
    fn bounded_workers_preserve_counts_hashes_classes_and_flags() {
        for workers in [1, 2] {
            let affine = endpoint(27, (1u128 << 115) + 6);
            let masks = [7u32];
            let job = Job { class: 0, m: 6, power3: 243, words32: 1,
                affine: affine.clone(), masks: masks.to_vec() };
            let states = decode(&job, usize::MAX).unwrap();
            let oracle = verify_states(&states, 110, 10, 0).unwrap();
            let mut pool = EarlyWorkers::new(2, workers, 110, 10, 0).unwrap();
            for class in (0..16).map(|j| j % 2) {
                pool.submit(class, 6, 243, 1, &affine, &masks).unwrap();
            }
            let mut stats = GpuStats { classes: vec![Default::default(); 2], ..Default::default() };
            pool.finish(&mut stats).unwrap();
            assert_eq!(stats.early_cases, 16 * oracle.cases);
            assert_eq!(stats.early_iterations, 16 * oracle.iters);
            assert_eq!(stats.hash, 0);
            assert_eq!(stats.flags, 16 * oracle.flagged.len() as u64);
            assert!(stats.flags > 0);
            assert_eq!(stats.classes[0].cases, 8 * oracle.cases);
            assert_eq!(stats.classes[1].iters, 8 * oracle.iters);
            assert_eq!(stats.early_class_cases, vec![8 * oracle.cases; 2]);
            assert_eq!(stats.early_class_iterations, vec![8 * oracle.iters; 2]);
            assert!(stats.early_cpu_seconds > 0.0);
        }
    }

    #[test]
    fn worker_errors_drain_the_bounded_queue_before_returning() {
        let mut pool = EarlyWorkers::new(1, 1, 120, 10, 0).unwrap();
        for _ in 0..12 { pool.submit(99, 6, 243, 1, &endpoint(27, 30), &[1]).unwrap(); }
        let mut stats = GpuStats { classes: vec![Default::default()], ..Default::default() };
        assert!(pool.finish(&mut stats).is_err());
    }

    #[test]
    fn compact_shards_are_reconstructed_across_multiple_bounded_chunks() {
        let endpoints = ENDPOINTS_PER_CHUNK + 7;
        let affine = endpoint(27, (1u128 << 115) + 6).repeat(endpoints);
        let masks = vec![7u32; endpoints];
        let oracle = verify_states(&decode(&Job { class: 0, m: 6, power3: 243, words32: 1,
            affine: endpoint(27, (1u128 << 115) + 6), masks: vec![7] }, usize::MAX).unwrap(),
            110, 10, 0).unwrap();
        let mut pool = EarlyWorkers::new(1, 1, 110, 10, 3).unwrap();
        pool.submit(0, 6, 243, 1, &affine, &masks).unwrap();
        let mut stats = GpuStats { classes: vec![Default::default()], ..Default::default() };
        pool.finish(&mut stats).unwrap();
        assert_eq!(stats.early_cases, endpoints as u64 * oracle.cases);
        assert_eq!(stats.early_iterations, endpoints as u64 * oracle.iters);
        assert_eq!(stats.flags, endpoints as u64 * oracle.flagged.len() as u64);
        assert_eq!(stats.hash, oracle.hash); // An odd number of duplicate endpoints.
    }
}
