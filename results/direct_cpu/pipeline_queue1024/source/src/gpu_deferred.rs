//! Bounded CPU continuations for the experimental GPU tail budget.
use super::GpuStats;
use crate::tail::{finish_continuation, TailContinuation};
use anyhow::{anyhow, ensure, Result};
use std::sync::mpsc::{sync_channel, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Instant;

struct Job {
    class: usize,
    records: Vec<TailContinuation>,
}

struct Completed {
    cases: u64,
    iterations: Vec<u64>,
    hash: u32,
    flagged: Vec<u128>,
    cpu_seconds: f64,
}

fn thread_cpu_seconds() -> f64 {
    #[repr(C)]
    struct Timespec {
        seconds: std::os::raw::c_long,
        nanoseconds: std::os::raw::c_long,
    }
    unsafe extern "C" {
        fn clock_gettime(clock: i32, time: *mut Timespec) -> i32;
    }
    let mut t = Timespec { seconds: 0, nanoseconds: 0 };
    assert_eq!(unsafe { clock_gettime(3, &mut t) }, 0);
    t.seconds as f64 + t.nanoseconds as f64 * 1e-9
}

pub(super) struct DeferredWorkers {
    sender: Option<SyncSender<Job>>,
    workers: Vec<JoinHandle<Result<Completed>>>,
    send_seconds: f64,
}

impl DeferredWorkers {
    pub fn new(classes: usize, workers: usize, cutoff_bits: u32) -> Self {
        assert!(classes > 0 && (1..=8).contains(&workers));
        // At most two queued batches per worker, plus batches being processed.
        // GPU buffers impose a second independent per-batch bound.
        let (sender, receiver) = sync_channel::<Job>(2 * workers);
        let receiver = Arc::new(Mutex::new(receiver));
        let workers = (0..workers).map(|_| {
            let receiver = Arc::clone(&receiver);
            std::thread::spawn(move || {
                let mut done = Completed {
                    cases: 0, iterations: vec![0; classes], hash: 0,
                    flagged: Vec::new(), cpu_seconds: 0.0,
                };
                let mut error = None;
                loop {
                    let job = receiver.lock().unwrap().recv();
                    let Ok(job) = job else { break; };
                    let cpu = thread_cpu_seconds();
                    for record in job.records {
                        let result = finish_continuation(record, cutoff_bits);
                        match result {
                            Ok((hash, iterations, flagged)) if job.class < classes => {
                                done.cases += 1;
                                done.iterations[job.class] += iterations;
                                done.hash ^= hash;
                                if flagged { done.flagged.push(record.base()); }
                            }
                            _ => {
                                // Keep draining so an invalid record cannot deadlock
                                // a GPU worker trying to send to the bounded queue.
                                error.get_or_insert("invalid CPU continuation record or class");
                            }
                        }
                    }
                    done.cpu_seconds += thread_cpu_seconds() - cpu;
                }
                if let Some(error) = error { return Err(anyhow!(error)); }
                Ok(done)
            })
        }).collect();
        Self { sender: Some(sender), workers, send_seconds: 0.0 }
    }

    pub fn submit(&mut self, class: usize, records: Vec<TailContinuation>) -> Result<()> {
        let start = Instant::now();
        let result = self.sender.as_ref().unwrap().send(Job { class, records });
        self.send_seconds += start.elapsed().as_secs_f64();
        result.map_err(|_| anyhow!("CPU continuation workers stopped"))
    }

    pub fn finish(mut self, stats: &mut GpuStats) -> Result<()> {
        let drain = Instant::now();
        drop(self.sender.take());
        let mut cases = 0u64;
        let mut error = None;
        // Join every worker before returning, even if one reports an error.
        for worker in self.workers.drain(..) {
            match worker.join().map_err(|_| anyhow!("CPU continuation worker panicked")) {
                Ok(Ok(done)) => {
                    cases += done.cases;
                    stats.deferred_cpu_seconds += done.cpu_seconds;
                    stats.deferred_hash ^= done.hash;
                    stats.deferred_flags += done.flagged.len() as u64;
                    for (class, iterations) in stats.classes.iter_mut().zip(done.iterations) {
                        class.iters += iterations;
                        stats.deferred_iterations += iterations;
                    }
                    stats.flagged.extend(done.flagged);
                }
                Ok(Err(e)) | Err(e) => { error.get_or_insert(e); }
            }
        }
        if let Some(error) = error { return Err(error); }
        ensure!(cases == stats.deferred_cases, "not every GPU continuation completed");
        stats.hash ^= stats.deferred_hash;
        stats.flags += stats.deferred_flags;
        stats.deferred_queue_wait_seconds = self.send_seconds;
        stats.deferred_drain_seconds = drain.elapsed().as_secs_f64();
        Ok(())
    }
}

impl Drop for DeferredWorkers {
    fn drop(&mut self) {
        drop(self.sender.take());
        // Error paths still finish in-flight work and cannot leave CPU tasks
        // running beyond the measured pipeline call.
        for worker in self.workers.drain(..) { let _ = worker.join(); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_workers_preserve_duplicate_cases_classes_and_cutoff_flags() {
        for count in [1, 2] {
            let records = [
                TailContinuation::from_values(27, 28),
                TailContinuation::from_values((1u128 << 72) + 3, (1u128 << 120) + 7),
            ];
            let mut workers = DeferredWorkers::new(2, count, 110);
            let mut expected = GpuStats { classes: vec![Default::default(); 2], ..Default::default() };
            for job in 0..17 {
                let class = job % 2;
                workers.submit(class, records.to_vec()).unwrap();
                for record in records {
                    let (hash, iterations, flagged) = finish_continuation(record, 110).unwrap();
                    expected.hash ^= hash;
                    expected.classes[class].iters += iterations;
                    if flagged { expected.flagged.push(record.base()); }
                }
            }
            let mut actual = GpuStats {
                classes: vec![Default::default(); 2], deferred_cases: 34, ..Default::default()
            };
            workers.finish(&mut actual).unwrap();
            actual.flagged.sort_unstable();
            expected.flagged.sort_unstable();
            assert_eq!(actual.flagged, expected.flagged);
            assert!(!actual.flagged.is_empty());
            assert_eq!(actual.hash, expected.hash);
            assert_eq!(actual.flags, expected.flagged.len() as u64);
            assert_eq!(actual.classes.iter().map(|c| c.iters).collect::<Vec<_>>(),
                expected.classes.iter().map(|c| c.iters).collect::<Vec<_>>());
            assert_eq!(actual.deferred_iterations, actual.classes.iter().map(|c| c.iters).sum::<u64>());
            assert!(actual.deferred_cpu_seconds > 0.0);
        }
    }

    #[test]
    fn invalid_record_is_reported_after_bounded_queue_drains() {
        let mut workers = DeferredWorkers::new(1, 1, 120);
        for _ in 0..12 {
            workers.submit(0, vec![TailContinuation::from_values(27, 26)]).unwrap();
        }
        let mut stats = GpuStats { classes: vec![Default::default()], deferred_cases: 12, ..Default::default() };
        assert!(workers.finish(&mut stats).is_err());
    }
}
