//! CPU-only diagnostic of the current early-screen export preparation.
//! CUDA is used only for cudaHostAllocDefault pinned source buffers; no kernels
//! or transfers run. Keep default x86-64 code generation to match the pipeline.
use std::ffi::{c_char, c_int, c_long, c_void};
use std::hint::black_box;
use std::time::Instant;

const ENDPOINTS: usize = 1 << 20;
const SHARDS: usize = 4;
const AFFINE_WORDS: usize = ENDPOINTS * 8;
const MASK_WORDS: usize = ENDPOINTS * 32;
const REPS: usize = 5;

#[link(name = "cudart")]
unsafe extern "C" {
    fn cudaHostAlloc(ptr: *mut *mut c_void, size: usize, flags: u32) -> c_int;
    fn cudaFreeHost(ptr: *mut c_void) -> c_int;
    fn cudaGetErrorString(error: c_int) -> *const c_char;
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Timeval { seconds: c_long, micros: c_long }
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Usage { user: Timeval, system: Timeval, counters: [c_long; 14] }
#[repr(C)]
#[derive(Default)]
struct Timespec { seconds: c_long, nanos: c_long }
unsafe extern "C" {
    fn getrusage(who: c_int, usage: *mut Usage) -> c_int;
    fn clock_gettime(clock: c_int, ts: *mut Timespec) -> c_int;
}
fn usage() -> Usage {
    let mut result = Usage::default();
    assert_eq!(unsafe { getrusage(1, &mut result) }, 0); // RUSAGE_THREAD
    result
}
fn cpu_seconds() -> f64 {
    let mut ts = Timespec::default();
    assert_eq!(unsafe { clock_gettime(3, &mut ts) }, 0); // CLOCK_THREAD_CPUTIME_ID
    ts.seconds as f64 + ts.nanos as f64 * 1e-9
}
fn seconds(t: Timeval) -> f64 { t.seconds as f64 + t.micros as f64 * 1e-6 }
struct Pinned { ptr: *mut u32, words: usize }
impl Pinned {
    fn new(words: usize) -> Self {
        let mut ptr = std::ptr::null_mut();
        let error = unsafe { cudaHostAlloc(&mut ptr, words * 4, 0) };
        if error != 0 {
            let message = unsafe { std::ffi::CStr::from_ptr(cudaGetErrorString(error)) };
            panic!("pinned allocation failed: {}", message.to_string_lossy());
        }
        Self { ptr: ptr.cast(), words }
    }
    fn slice(&self) -> &[u32] { unsafe { std::slice::from_raw_parts(self.ptr, self.words) } }
    fn slice_mut(&mut self) -> &mut [u32] { unsafe { std::slice::from_raw_parts_mut(self.ptr, self.words) } }
}
impl Drop for Pinned {
    fn drop(&mut self) { assert_eq!(unsafe { cudaFreeHost(self.ptr.cast()) }, 0); }
}

#[no_mangle]
#[inline(never)]
pub fn software_popcount(masks: &[u32]) -> u64 {
    masks.iter().map(|w| w.count_ones() as u64).sum::<u64>()
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "popcnt")]
#[no_mangle]
#[inline(never)]
unsafe fn hardware_popcount(masks: &[u32]) -> u64 {
    // Reading two little-endian u32 words as one u64 preserves total popcount.
    // Four independent sums avoid a single loop-carried accumulator chain.
    use std::arch::x86_64::_popcnt64;
    let (mut a, mut b, mut c, mut d) = (0u64, 0u64, 0u64, 0u64);
    let mut i = 0;
    while i + 8 <= masks.len() {
        a += _popcnt64(std::ptr::read_unaligned(masks.as_ptr().add(i).cast::<i64>())) as u64;
        b += _popcnt64(std::ptr::read_unaligned(masks.as_ptr().add(i + 2).cast::<i64>())) as u64;
        c += _popcnt64(std::ptr::read_unaligned(masks.as_ptr().add(i + 4).cast::<i64>())) as u64;
        d += _popcnt64(std::ptr::read_unaligned(masks.as_ptr().add(i + 6).cast::<i64>())) as u64;
        i += 8;
    }
    while i < masks.len() { a += _popcnt64(masks[i] as i64) as u64; i += 1; }
    a + b + c + d
}
#[derive(Clone, Copy)]
enum Variant { SoftwareCount, HardwareCount, AllocateOnly, AllocateCopy,
               FirstTouchCopy, ReusedCopy, SoftwareAllocateCopy, SoftwareReusedCopy }
impl Variant {
    fn name(self) -> &'static str {
        match self {
            Self::SoftwareCount => "software_popcount",
            Self::HardwareCount => "hardware_popcnt64",
            Self::AllocateOnly => "allocate_without_touch",
            Self::AllocateCopy => "allocate_and_copy",
            Self::FirstTouchCopy => "first_touch_copy_after_allocation",
            Self::ReusedCopy => "resident_reused_copy",
            Self::SoftwareAllocateCopy => "software_scan_allocate_copy",
            Self::SoftwareReusedCopy => "software_scan_reused_copy",
        }
    }
    fn scans(self) -> bool { matches!(self,Self::SoftwareCount|Self::HardwareCount|Self::SoftwareAllocateCopy|Self::SoftwareReusedCopy) }
    fn copies(self) -> bool { matches!(self,Self::AllocateCopy|Self::FirstTouchCopy|Self::ReusedCopy|Self::SoftwareAllocateCopy|Self::SoftwareReusedCopy) }
}
struct Buffers { affine: Vec<u32>, masks: Vec<u32> }
fn reserved() -> Vec<Buffers> {
    (0..SHARDS).map(|_| Buffers {
        affine: Vec::with_capacity(AFFINE_WORDS / SHARDS),
        masks: Vec::with_capacity(MASK_WORDS / SHARDS),
    }).collect()
}
fn verify_copies(copies: &[Buffers], affine: &[u32], masks: &[u32]) {
    assert_eq!(copies.len(),SHARDS);
    for (i, copy) in copies.iter().enumerate() {
        let asrc = &affine[i * AFFINE_WORDS / SHARDS..(i + 1) * AFFINE_WORDS / SHARDS];
        let msrc = &masks[i * MASK_WORDS / SHARDS..(i + 1) * MASK_WORDS / SHARDS];
        assert_eq!(copy.affine.len(),asrc.len());assert_eq!(copy.masks.len(),msrc.len());
        for k in [0,1,asrc.len()/2,asrc.len()-1] { assert_eq!(copy.affine[k],asrc[k]); }
        for k in [0,1,msrc.len()/2,msrc.len()-1] { assert_eq!(copy.masks[k],msrc[k]); }
    }
}
fn trial(variant: Variant, rep: usize, position: usize, warmup: bool,
         affine: &[u32], masks: &[u32], reused: &mut [Buffers], expected: u64) {
    // In this diagnostic case only, reserve virtual destination storage before
    // timing; its first write remains inside the measured copy. Page-fault
    // counts reveal whether the allocator actually returned resident pages.
    let mut jobs = if matches!(variant,Variant::FirstTouchCopy) { reserved() }
                   else { Vec::with_capacity(SHARDS) };
    let before = usage();
    let cpu = cpu_seconds();
    let start = Instant::now();
    let mut count = 0;
    for shard in 0..SHARDS {
        let a = &affine[shard*AFFINE_WORDS/SHARDS..(shard+1)*AFFINE_WORDS/SHARDS];
        let m = &masks[shard*MASK_WORDS/SHARDS..(shard+1)*MASK_WORDS/SHARDS];
        match variant {
            Variant::SoftwareCount => count += software_popcount(black_box(m)),
            Variant::HardwareCount => count += unsafe { hardware_popcount(black_box(m)) },
            Variant::AllocateOnly => jobs.push(Buffers { affine: Vec::with_capacity(a.len()), masks: Vec::with_capacity(m.len()) }),
            Variant::AllocateCopy => jobs.push(Buffers { affine: black_box(a).to_vec(), masks: black_box(m).to_vec() }),
            Variant::FirstTouchCopy => { jobs[shard].affine.extend_from_slice(black_box(a)); jobs[shard].masks.extend_from_slice(black_box(m)); },
            Variant::ReusedCopy => { reused[shard].affine.copy_from_slice(black_box(a)); reused[shard].masks.copy_from_slice(black_box(m)); },
            Variant::SoftwareAllocateCopy => {
                count += software_popcount(black_box(m));
                jobs.push(Buffers { affine: black_box(a).to_vec(), masks: black_box(m).to_vec() });
            },
            Variant::SoftwareReusedCopy => {
                count += software_popcount(black_box(m));
                reused[shard].affine.copy_from_slice(black_box(a)); reused[shard].masks.copy_from_slice(black_box(m));
            },
        }
    }
    black_box(&jobs);black_box(&reused);black_box(count);
    let wall = start.elapsed().as_secs_f64();
    let cpu = cpu_seconds()-cpu;
    let after = usage();
    if variant.scans() { assert_eq!(count,expected); }
    if variant.copies() {
        if matches!(variant,Variant::ReusedCopy|Variant::SoftwareReusedCopy) { verify_copies(reused,affine,masks); }
        else { verify_copies(&jobs,affine,masks); }
    }
    println!("{{\"type\":\"trial\",\"variant\":\"{}\",\"repetition\":{},\"order_position\":{},\"warmup\":{},\"wall_seconds\":{:.9},\"cpu_seconds\":{:.9},\"user_seconds\":{:.9},\"system_seconds\":{:.9},\"minor_faults\":{},\"major_faults\":{},\"voluntary_switches\":{},\"involuntary_switches\":{},\"count\":{},\"checks_passed\":true}}",
        variant.name(),rep,position,warmup,wall,cpu,seconds(after.user)-seconds(before.user),seconds(after.system)-seconds(before.system),
        after.counters[4]-before.counters[4],after.counters[5]-before.counters[5],after.counters[12]-before.counters[12],after.counters[13]-before.counters[13],count);
    // Job destruction is outside the host copy timer, as in the pipeline where
    // workers drop jobs after verification. No worker threads run in this test.
}
fn main() {
    let hardware = std::is_x86_feature_detected!("popcnt");
    let mut affine = Pinned::new(AFFINE_WORDS);
    let mut masks = Pinned::new(MASK_WORDS);
    let mut random = 0x0c58_897f_29f7_054bu64;
    let mut next = || { random ^= random << 13; random ^= random >> 7; random ^= random << 17; random as u32 };
    for word in affine.slice_mut() { *word = next(); }
    let mut expected = 0;
    for word in masks.slice_mut() { *word = next(); expected += word.count_ones() as u64; }
    let mut reused = reserved();
    for job in &mut reused {
        job.affine.resize(AFFINE_WORDS/SHARDS,0);job.affine.fill(0xa5a5_a5a5);
        job.masks.resize(MASK_WORDS/SHARDS,0);job.masks.fill(0x5a5a_5a5a);
    }
    let mut variants = vec![Variant::SoftwareCount,Variant::AllocateOnly,Variant::AllocateCopy,
        Variant::FirstTouchCopy,Variant::ReusedCopy,Variant::SoftwareAllocateCopy,Variant::SoftwareReusedCopy];
    if hardware { variants.insert(1,Variant::HardwareCount); }
    println!("{{\"type\":\"config\",\"endpoints\":{},\"shards\":{},\"affine_bytes\":{},\"mask_bytes\":{},\"repetitions\":{},\"pinned_flags\":0,\"popcnt_supported\":{},\"expected_count\":{},\"source\":\"deterministic CPU-filled cudaHostAllocDefault, no DMA\",\"order\":\"rotate by 3 positions per repetition and reverse odd repetitions\",\"scope\":\"CPU count/allocation/copy only; validation and destination frees excluded\"}}",ENDPOINTS,SHARDS,AFFINE_WORDS*4,MASK_WORDS*4,REPS,hardware,expected);
    for (i,&variant) in variants.iter().enumerate() { trial(variant,0,i,true,affine.slice(),masks.slice(),&mut reused,expected); }
    for rep in 0..REPS {
        let mut order = variants.clone();let length=order.len();order.rotate_left((3*rep)%length);
        if rep%2==1 { order.reverse(); }
        for (i,variant) in order.into_iter().enumerate() { trial(variant,rep,i,false,affine.slice(),masks.slice(),&mut reused,expected); }
    }
}
