// Warp-queue scheduler with a pluggable per-lane tail (c-sched, round 2).
//
// Same work as the per-thread kernels (cv_kernel_x for the jump tails of
// cuda/xtail.cuh): the same screening (bitvector window, mod-9 mask, carried
// OEE mask applied before extraction) and, per case, exactly the tail's
// stepping.  Only the scheduling differs.  A persistent warp
//
//  * fetches chunks of 2^c endpoints dynamically (atomic counter KOut.next),
//  * screens ONE endpoint's 32-word completion chunk at a time cooperatively:
//    lane i forms completion word i, a warp prefix sum of popcounts writes the
//    surviving completion indices with ranks [qbase, qbase + QCAP) to a
//    per-warp shared queue (u16; a chunk with more survivors is screened
//    again with qbase += QCAP, so QCAP may be below the 570 maximum),
//  * materializes case records 32-wide (lane j builds record j with
//    Tail::make_record) into a per-warp shared ring,
//  * steps every lane U times per loop trip (Tail::step) and hands ring
//    records to idle lanes (Tail::start / pop_load).
//
// Schedulers (selected by the layout, WqCfg<V>):
//   cv_wqj_kernel   round-1 style: pop rounds when >= K lanes are idle,
//                   materialize when the ring is empty.
//   cv_wqj2_kernel  "trip-pop": idle lanes take records by rank on every trip
//                   (K = minimum idle lanes); the ring is topped up whenever it
//                   holds fewer records than there are idle lanes.  Best.
//   cv_wqj3_kernel  trip-pop with NL independent case slots per lane (ILP).
//   cv_wqpt_kernel  per-thread reference scheduler (one endpoint per thread)
//                   driving the same tail functor, for apples-to-apples tests.
//
// Idle lanes keep stepping stale values (one instruction stream for the
// warp) but every tail masks them out of the hash, the iteration count and
// the flag buffer (amask == 0).  Cases, iterations and the XOR hash are sums
// and XORs, so they are order independent; flags are raised by the same
// trajectories (only the buffer order differs).  Audit layouts (DIG) also
// return a per-case digest of (n, case jumps, case hash), see wqaudit.
//
// Tail functors: JumpTail<TV, CAP, FOUR, DIG> (4-limb steps, deferred flags)
// and LeanJumpTail<TV, CAP, DIG> (3-limb steps, rare 4-limb completion); the
// interface is described in c-sched/README.md.
//
// KParams.pad when bit 22 is set (validated by wq_pad_valid):
//   bits 0-2   must be 5 (no vote, constant powers, early OEE)
//   bits 3-7   K (pop rounds: refill when >= K lanes idle, default 8;
//              trip-pop: minimum idle lanes for a pop, default 1)
//   bits 10-12 log2 of the endpoints per dynamic fetch (0..7)
//   bits 13-15 blocks per SM cap (0 = occupancy limit)
//   bits 16-20 layout V, low 5 bits (WqCfg<V>: warps/block, ring, queue
//              capacity, launch bounds, steps per trip U, step mode,
//              scheduler, ILP slots, audit flags)
//   bit  21    TIMING-ONLY test tail: stop every case after at most
//              KParams.tail_budget jumps (unsound; refused by the pipeline)
//   bit  22    warp-queue scheduler
//   bit  23    layout V, bit 5 (V = 0..63)
//   bits 24-31 tail code: xtail.cuh jump codes 49..53 (W, table layout;
//              same per-case semantics as cv_kernel_x with that code)
#pragma once

#include <map>
#include <type_traits>
#include <cstdlib>

#define WQJ_FULL 0xFFFFFFFFu

// ---------------------------------------------------------------------------
// Layout variants.  QCAP >= 570 never rescreens (5 of 9 residues pass mod 9,
// so a 1024-index chunk has at most 570 survivors).
template<int V> struct WqCfg { static constexpr bool valid = false; };
// V: warps/block, ring records, queue capacity, min blocks (launch bounds),
// steps per refill check, step mode (0: the code's own 3-limb vote, 1: 4 limbs).
#define WQ_CFGL(V, WARPS_, RING_, QCAP_, MINB_, U_, FOUR_, INSTR_, POP2_, DIG_, NL_, PT_, POPX_, LEAN_) template<> struct WqCfg<V> { \
    static constexpr bool valid = true; static constexpr int WARPS = WARPS_, RING = RING_, QCAP = QCAP_, MINB = MINB_; \
    static constexpr int U = U_; static constexpr bool FOUR = FOUR_, INSTR = INSTR_, POP2 = POP2_, DIG = DIG_; \
    static constexpr int NL = NL_; static constexpr bool PT = PT_, POPX = POPX_, LEAN = LEAN_; };
#define WQ_CFGT(V, WARPS_, RING_, QCAP_, MINB_, U_, FOUR_, INSTR_, POP2_, DIG_, NL_, PT_) WQ_CFGL(V, WARPS_, RING_, QCAP_, MINB_, U_, FOUR_, INSTR_, POP2_, DIG_, NL_, PT_, false, false)
#define WQ_CFGN(V, WARPS_, RING_, QCAP_, MINB_, U_, FOUR_, INSTR_, POP2_, DIG_, NL_) WQ_CFGT(V, WARPS_, RING_, QCAP_, MINB_, U_, FOUR_, INSTR_, POP2_, DIG_, NL_, false)
#define WQ_CFGD(V, WARPS_, RING_, QCAP_, MINB_, U_, FOUR_, INSTR_, POP2_, DIG_) WQ_CFGN(V, WARPS_, RING_, QCAP_, MINB_, U_, FOUR_, INSTR_, POP2_, DIG_, 1)
#define WQ_CFGP(V, WARPS_, RING_, QCAP_, MINB_, U_, FOUR_, INSTR_, POP2_) WQ_CFGD(V, WARPS_, RING_, QCAP_, MINB_, U_, FOUR_, INSTR_, POP2_, false)
#define WQ_CFGI(V, WARPS_, RING_, QCAP_, MINB_, U_, FOUR_, INSTR_) WQ_CFGP(V, WARPS_, RING_, QCAP_, MINB_, U_, FOUR_, INSTR_, false)
#define WQ_CFG(V, WARPS_, RING_, QCAP_, MINB_, U_, FOUR_) WQ_CFGI(V, WARPS_, RING_, QCAP_, MINB_, U_, FOUR_, false)
WQ_CFG(0, 8, 32, 576, 1, 1, false)   // round-1 layout (occupancy-limited with large tables)
WQ_CFG(1, 20, 32, 288, 1, 1, true)   // 2 blocks/SM = 40 warps with a 16 KB table
WQ_CFG(2, 20, 32, 288, 1, 2, true)
WQ_CFG(3, 20, 32, 288, 1, 2, false)
WQ_CFG(4, 16, 32, 288, 1, 2, true)   // 32 warps/SM with a 16 KB table
WQ_CFG(5, 10, 16, 96, 1, 2, true)    // 4 blocks/SM = 40 warps with a 16 KB table
WQ_CFG(6, 20, 32, 288, 1, 3, true)
WQ_CFG(7, 24, 16, 128, 2, 2, true)   // 48 warps/SM if registers fit 40
WQ_CFG(8, 32, 32, 160, 1, 2, true)   // one 1024-thread block: 32 warps/SM
WQ_CFG(9, 20, 16, 160, 1, 2, true)
WQ_CFG(10, 8, 32, 384, 1, 2, true)   // 5 blocks/SM = 40 warps with a 4 KB table (W10)
WQ_CFG(11, 8, 32, 144, 1, 2, true)   // 5 blocks/SM = 40 warps with an 8 KB table (W11)
WQ_CFG(12, 20, 32, 288, 2, 2, true)  // as 2, launch bounds force <= 48 registers
WQ_CFG(13, 8, 32, 384, 5, 2, true)   // as 10, launch bounds force <= 48 registers
// Instrumented copies (KOut.dbg counters; counts only, timings not used).
WQ_CFGI(14, 20, 32, 288, 1, 2, true, true)    // = 2
WQ_CFGI(15, 20, 32, 288, 1, 1, true, true)    // = 1
WQ_CFGI(16, 8, 32, 384, 1, 2, true, true)     // = 10
// Scheduler 2 (trip-pop, append refill).
WQ_CFGP(17, 20, 32, 288, 2, 2, true, false, true)   // 40 warps/SM with a 16 KB table (<= 48 registers)
WQ_CFGP(18, 20, 32, 288, 2, 1, true, false, true)
WQ_CFGP(19, 20, 32, 288, 2, 3, true, false, true)
WQ_CFGP(20, 20, 32, 288, 1, 2, true, false, true)   // as 17 without register bound
WQ_CFGP(21, 8, 32, 384, 5, 2, true, false, true)    // 40 warps/SM with a 4 KB table (W10)
WQ_CFGP(22, 8, 32, 144, 5, 2, true, false, true)    // 40 warps/SM with an 8 KB table (W11)
WQ_CFGP(23, 20, 32, 288, 2, 2, false, false, true)  // as 17 with the code's 3-limb vote
WQ_CFGP(24, 20, 32, 288, 1, 2, true, true, true)    // instrumented 17
WQ_CFGP(25, 20, 32, 288, 1, 1, true, true, true)    // instrumented 18
// Audit copies: per-case digest sum/xor/count in KOut.dbg[0..2] (validation only).
WQ_CFGD(26, 20, 32, 288, 1, 2, true, false, true, true)    // = 17 (trip-pop)
WQ_CFGD(27, 20, 32, 288, 1, 2, true, false, false, true)   // = 2 (pop rounds)
WQ_CFGD(28, 8, 32, 384, 1, 2, true, false, true, true)     // = 21 (8 warps, small table)
WQ_CFGD(29, 20, 32, 288, 1, 2, false, false, true, true)   // = 23 (3-limb vote)
WQ_CFGD(30, 8, 32, 576, 1, 1, false, false, false, true)   // = 0 (round-1 layout)
// Two independent case slots per lane (ILP2), trip-pop (cv_wqj3_kernel).
WQ_CFGN(31, 16, 32, 288, 2, 1, true, false, true, false, 2)   // 32 warps/SM, <= 64 registers
WQ_CFGN(32, 16, 32, 288, 2, 2, true, false, true, false, 2)
WQ_CFGN(33, 18, 32, 288, 2, 1, true, false, true, false, 2)   // 36 warps/SM, <= 56 registers
WQ_CFGN(34, 18, 32, 288, 2, 2, true, false, true, false, 2)
WQ_CFGN(35, 8, 32, 384, 4, 1, true, false, true, false, 2)    // W10 table: 32 warps/SM
WQ_CFGN(36, 16, 32, 288, 1, 1, true, false, true, true, 2)    // audit copy of 31 (digest)
WQ_CFGN(37, 16, 32, 288, 2, 1, false, false, true, false, 2)  // 31 with the code's 3-limb vote
// Trip-pop layouts sized for other tables / more warps.
WQ_CFGP(38, 20, 16, 64, 2, 2, true, false, true)    // 32 KB u64 table: 2 blocks = 40 warps/SM
WQ_CFGP(39, 24, 16, 128, 2, 2, true, false, true)   // 16 KB table: 2 blocks = 48 warps/SM if <= 40 registers
WQ_CFGN(40, 16, 16, 96, 2, 1, true, false, true, false, 2)    // ILP2 with a 32 KB u64 table: 32 warps/SM
// Per-thread reference scheduler (cv_wqpt_kernel): one endpoint per thread as
// in cv_kernel_x, driving the same tail functor (4-limb steps).  Not a warp
// queue; for apples-to-apples comparisons of tails (e.g. the capped test tail).
WQ_CFGT(41, 8, 32, 32, 1, 1, true, false, false, false, 1, true)
WQ_CFGT(42, 8, 32, 32, 1, 1, true, false, false, true, 1, true)     // audit copy of 41 (digest)
// More capped-tail characterization layouts (trip-pop).
WQ_CFGP(43, 20, 32, 288, 2, 4, true, false, true)
// Trip-pop with predicated pops (POPX) and/or the lean 3-limb tail (LEAN).
WQ_CFGL(44, 20, 32, 288, 2, 2, true, false, true, false, 1, false, true, false)    // 17 + POPX
WQ_CFGL(45, 20, 32, 288, 2, 2, true, false, true, false, 1, false, false, true)    // 17 + LEAN
WQ_CFGL(46, 20, 32, 288, 2, 2, true, false, true, false, 1, false, true, true)     // 17 + POPX + LEAN
WQ_CFGL(47, 20, 32, 288, 2, 1, true, false, true, false, 1, false, true, true)     // 46 with U = 1
WQ_CFGL(48, 20, 32, 288, 2, 3, true, false, true, false, 1, false, true, true)     // 46 with U = 3
WQ_CFGL(49, 20, 32, 288, 1, 2, true, false, true, true, 1, false, true, true)      // audit copy of 46 (digest)
WQ_CFGL(50, 20, 32, 288, 1, 2, true, true, true, false, 1, false, true, true)      // instrumented 46
WQ_CFGL(51, 8, 32, 384, 5, 2, true, false, true, false, 1, false, true, true)      // 46 for the 4 KB W10 table: 40 warps
WQ_CFGD(52, 20, 32, 288, 1, 4, true, false, true, true)    // audit copy of 43 (digest)
#undef WQ_CFG
#undef WQ_CFGI
#undef WQ_CFGP
#undef WQ_CFGD
#undef WQ_CFGN
#undef WQ_CFGT
#undef WQ_CFGL

// ---------------------------------------------------------------------------
// Launch parameters, precomputed on the host so that the hot loop reads them
// from the constant bank instead of holding them in registers.
struct WqParams {
    u64 b_mask;      // 2^B - 1
    u64 fam_stride;  // u32 words from a family-0 array to its family-1 twin
    u64 magic;       // 3^-f mod 2^64 (bitvector index multiplier)
    u32 tp_0, tp_1, tp_2, tp_3;   // 3^f limbs
    u32 n;           // endpoints
    u32 words32;     // completion words per endpoint (2^A / 32)
    u32 nchunks;     // 32-word screening chunks per endpoint
    u32 m_shift;     // M % 32
    u32 guard_hi;    // flag when limb 3 >= guard_hi before a jump (cutoff_hi >> K)
    u32 tail_oee;
    u32 flag_cap;
    u32 kmin;        // refill when at least kmin lanes are idle (1..32)
    u32 fetch;       // endpoints per dynamic fetch
    u32 cap;         // capped test tail only: jumps per case
};

// Where flagged starts go.  Kept in shared memory and read with volatile
// loads inside the rare flag branch, so that nvcc does not hoist the
// flag-buffer address arithmetic (or call arguments) into the hot loop.
struct WqSink {
    KOut* out;
    u32* flags;
    u32 cap;
};

__device__ __forceinline__ void wq_flag_out(u32 b0, u32 b1, u32 b2, u32 b3, const WqSink* sink) {
    const volatile WqSink* vs = sink;
    KOut* out = vs->out;
    u32* flags = vs->flags;
    const u32 cap = vs->cap;
    const u32 slot = atomicAdd(&out->flags, 1u);
    if (slot < cap) {
        flags[4 * slot] = b0;
        flags[4 * slot + 1] = b1;
        flags[4 * slot + 2] = b2;
        flags[4 * slot + 3] = b3;
    }
}

// Per-case audit digest (DIG layouts): a strong 64-bit mix of
// (n, jumps of the case, XOR hash of the case), summed and XORed over cases.
// The CPU side is cv::jump::case_digest.
__device__ __forceinline__ u64 wq_mix64(u64 a) {
    a ^= a >> 30; a *= 0xBF58476D1CE4E5B9ull;
    a ^= a >> 27; a *= 0x94D049BB133111EBull;
    a ^= a >> 31;
    return a;
}
__device__ __forceinline__ u64 wq_case_digest(u32 b0, u32 b1, u32 b2, u32 b3, u32 jumps, u32 hash) {
    const u64 lo = (u64)b0 | ((u64)b1 << 32), hi = (u64)b2 | ((u64)b3 << 32);
    return wq_mix64(lo ^ wq_mix64(hi ^ wq_mix64(((u64)jumps << 32) | hash)));
}

// Predicated record load straight into lane registers (POPX pops): no branch,
// no register moves.  saddr is a shared-memory address of a 32-byte record.
__device__ __forceinline__ void wq_pred_load_rec(bool p, u32 saddr, u32& a0, u32& a1, u32& a2, u32& a3,
                                                 u32& b0, u32& b1, u32& b2, u32& b3) {
    asm volatile("{\n\t.reg .pred q;\n\tsetp.ne.b32 q, %8, 0;\n\t"
                 "@q ld.shared.v4.u32 {%0, %1, %2, %3}, [%9];\n\t"
                 "@q ld.shared.v4.u32 {%4, %5, %6, %7}, [%9+16];\n\t}"
                 : "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3), "+r"(b0), "+r"(b1), "+r"(b2), "+r"(b3)
                 : "r"((u32)p), "r"(saddr) : "memory");
}

// Endpoint record of the current screening chunk.
struct WqEp {
    uint4 n0;   // n0 limbs, flag bits cleared
    uint4 it;   // T^M(n0)
};

// Per-warp screening state (shared memory; touched only on screening passes).
struct WqWarp {
    WqEp e;
    const u32* bvw;     // window words of the current endpoint
    const u32* m9;      // its mod-9 mask row
    u32 offset, oee_mask;
    u32 ep, ep_end;     // current endpoint, end of the fetched chunk
    u32 chunk, qbase;   // 32-word chunk and first rank of the current pass
    u32 have_ep, pad_;
};

// ---------------------------------------------------------------------------
// Jump tail (xtail.cuh semantics, CPU reference jump::jump_until(coarse)):
//   start at x = T^M(n); before every jump x >= 2^(cutoff-K) flags and stops
//   the case; after every jump hash ^= x mod 2^32, iters += 1, and the case
//   stops when D(x, n) holds: (x >> 32) < (n >> 32) for 2^32 <= n < 2^96,
//   exact x < n otherwise.
//   Step modes: FOUR = 4-limb jumps; otherwise (codes with X::three) a warp
//   vote picks the 3-limb jump when no active lane is at or above 2^(96-K),
//   where it is exact.  Cases with n outside [2^32, 2^96) (rare: only small
//   n0 at h = 0 in production) run to completion at pop with the exact test,
//   as cv_kernel_x's small_base path does, so the warp loop never needs it.
//   CAP (timing only, unsound): a case also stops after w.cap jumps.
template<int TV, bool CAP, bool FOUR, bool DIG = false>
struct JumpTail {
    using X = XTraits<TV>;
    static_assert(X::valid && X::jump && X::opt && X::rep == 1 && X::prefix == 0 && !X::skip,
                  "warp-queue jump tails use the optimized unreplicated codes");
    static constexpr int W = X::w;
    static constexpr int K = x_growth_bits(W);
    static constexpr bool THREE = X::three && !FOUR;
    static constexpr bool DEFERS = true;   // step() defers flags through pend
    static constexpr u32 TABLE_BYTES = x_table_bytes<TV>();
    static constexpr u32 LIM3 = 1u << (32 - K);   // x < 2^(96-K) <=> x3 == 0 && x2 < LIM3

    struct Lane {
        u32 x0, x1, x2, x3;
        u32 b0, b1, b2, b3;
        u32 pend;    // the case stopped at the guard; its flag is recorded at the next refill
        u32 left;    // CAP only
        u32 chash, cjumps;   // DIG only: the current case's own hash and jumps
        u64 dsum, dxor;      // DIG only
        u32 dcount;          // DIG only
    };

    __device__ __forceinline__ static void idle(Lane& L) {
        L.x0 = L.x1 = L.x2 = L.x3 = 0;
        L.b0 = L.b1 = L.b2 = L.b3 = 0;
        L.pend = 0;
        L.left = 0;
        L.chash = L.cjumps = 0;
        L.dsum = L.dxor = 0;
        L.dcount = 0;
    }

    __device__ __forceinline__ static void digest(Lane& L) {
        const u64 d = wq_case_digest(L.b0, L.b1, L.b2, L.b3, L.cjumps, L.chash);
        L.dsum += d;
        L.dxor ^= d;
        L.dcount += 1;
    }

    // Record a pending flag (the lane is idle and still holds n).
    __device__ __forceinline__ static void settle(Lane& L, const WqSink* sink) {
        if (L.pend) {
            flag(L, sink);
            L.pend = 0;
        }
    }

    // rec0 = T^M(n) = T^M(n0) + aval 3^f, rec1 = n = n0 + aval 2^M.
    template<int MLIMB>
    __device__ __forceinline__ static void make_record(uint4& rec0, uint4& rec1, const WqEp& ep, u32 aval,
                                                       const WqParams& w) {
        const u64 p0 = (u64)aval * w.tp_0 + ep.it.x;
        const u64 p1 = (u64)aval * w.tp_1 + ep.it.y;
        const u64 p2 = (u64)aval * w.tp_2 + ep.it.z;
        const u32 p3 = aval * w.tp_3 + ep.it.w;
        rec0.x = (u32)p0; u32 carry = (u32)(p0 >> 32);
        u64 t = p1 + carry; rec0.y = (u32)t; carry = (u32)(t >> 32);
        t = p2 + carry; rec0.z = (u32)t; carry = (u32)(t >> 32);
        rec0.w = p3 + carry;
        const u64 sh = (u64)aval << w.m_shift;   // n0 < 2^M
        rec1 = ep.n0;
        if constexpr (MLIMB == 0) { rec1.x |= (u32)sh; rec1.y |= (u32)(sh >> 32); }
        if constexpr (MLIMB == 1) { rec1.y |= (u32)sh; rec1.z |= (u32)(sh >> 32); }
        if constexpr (MLIMB == 2) { rec1.z |= (u32)sh; rec1.w |= (u32)(sh >> 32); }
    }

    __device__ __forceinline__ static void flag(const Lane& L, const WqSink* sink) {
        wq_flag_out(L.b0, L.b1, L.b2, L.b3, sink);
    }

    __device__ __forceinline__ static bool coarse(const Lane& L) {
        return (L.x3 == 0) & ((((u64)L.x2 << 32) | L.x1) < (((u64)L.b2 << 32) | L.b1));
    }

    // Pop.  Returns the lane's new amask (zero if the case already finished).
    __device__ __forceinline__ static u32 start(Lane& L, const uint4& r0, const uint4& r1, u32& hash, u32& iters,
                                                const u32* s_tab, const WqParams& w, const WqSink* sink) {
        settle(L, sink);   // a finished flagged case still owns L
        L.x0 = r0.x; L.x1 = r0.y; L.x2 = r0.z; L.x3 = r0.w;
        L.b0 = r1.x; L.b1 = r1.y; L.b2 = r1.z; L.b3 = r1.w;
        if constexpr (CAP) L.left = w.cap;
        if constexpr (DIG) { L.chash = 0; L.cjumps = 0; }
        const bool small = ((L.b1 | L.b2) == 0) | (L.b3 != 0);
        if ((L.x3 >= w.guard_hi) | small) {
            // Rare.  The guard before the first jump, or a whole exact-test case.
            if constexpr (CAP) {
                if (w.cap == 0 && L.x3 < w.guard_hi) {
                    if constexpr (DIG) digest(L);
                    return 0u;
                }
            }
            while (true) {
                if (L.x3 >= w.guard_hi) {
                    flag(L, sink);
                    break;
                }
                x_jump4o<TV>(s_tab, 0, L.x0, L.x1, L.x2, L.x3);
                hash ^= L.x0;
                iters += 1;
                if constexpr (DIG) { L.chash ^= L.x0; L.cjumps += 1; }
                if (x_below4(L.x0, L.x1, L.x2, L.x3, L.b0, L.b1, L.b2, L.b3)) break;
                if constexpr (CAP) { if (--L.left == 0) break; }
            }
            if constexpr (DIG) digest(L);
            L.x0 = L.x1 = L.x2 = L.x3 = 0;
            return 0u;
        }
        if constexpr (CAP) {
            if (w.cap == 0) {
                if constexpr (DIG) digest(L);
                return 0u;
            }
        }
        return WQJ_FULL;
    }

    // One jump on every lane (inactive lanes compute but are masked out).
    // POPX pop interface (the pend flag must be settled before pop_load).
    __device__ __forceinline__ static void pop_load(Lane& L, bool take, u32 saddr, const WqParams& w) {
        wq_pred_load_rec(take, saddr, L.x0, L.x1, L.x2, L.x3, L.b0, L.b1, L.b2, L.b3);
        if constexpr (CAP) { if (take) L.left = w.cap; }
        if constexpr (DIG) { if (take) { L.chash = 0; L.cjumps = 0; } }
    }
    __device__ __forceinline__ static bool pop_rare(const Lane& L, const WqParams& w) {
        const bool small = ((L.b1 | L.b2) == 0) | (L.b3 != 0);
        bool rare = (L.x3 >= w.guard_hi) | small;
        if constexpr (CAP) rare |= w.cap == 0;
        return rare;
    }
    __device__ __forceinline__ static u32 start_rare(Lane& L, u32& hash, u32& iters, const u32* s_tab,
                                                     const WqParams& w, const WqSink* sink) {
        // Same as the rare branch of start(): L already holds the record.
        const uint4 r0 = make_uint4(L.x0, L.x1, L.x2, L.x3), r1 = make_uint4(L.b0, L.b1, L.b2, L.b3);
        return start(L, r0, r1, hash, iters, s_tab, w, sink);
    }

    __device__ __forceinline__ static void step(Lane& L, u32& amask, u32& hash, u32& iters, const u32* s_tab,
                                                const WqParams& w, const WqSink* = nullptr) {
        bool four = true;
        if constexpr (THREE) {
            const bool need4 = (amask != 0) & ((L.x3 != 0) | (L.x2 >= LIM3));
            four = __any_sync(WQJ_FULL, need4);
        }
        bool stop;
        if (!four) {
            // Every active lane is below 2^(96-K): the 3-limb jump is exact and
            // ends below 2^96 <= 2^(cutoff-K), so no guard is needed.
            x_jump3o<TV>(s_tab, 0, L.x0, L.x1, L.x2);
            hash ^= L.x0 & amask;
            iters -= amask;
            if constexpr (DIG) { L.chash ^= L.x0 & amask; L.cjumps -= amask; }
            stop = (((u64)L.x2 << 32) | L.x1) < (((u64)L.b2 << 32) | L.b1);
        } else {
            x_jump4o<TV>(s_tab, 0, L.x0, L.x1, L.x2, L.x3);
            hash ^= L.x0 & amask;
            iters -= amask;
            if constexpr (DIG) { L.chash ^= L.x0 & amask; L.cjumps -= amask; }
            const bool desc = coarse(L);
            // desc implies x3 == 0 < guard_hi, so desc and over exclude each other.
            // A flagged lane goes idle with n intact; settle() records it.
            const bool over = L.x3 >= w.guard_hi;
            L.pend |= (u32)(over & (amask != 0));
            stop = desc | over;
        }
        if constexpr (CAP) {
            L.left -= amask & 1u;
            stop |= L.left == 0;
        }
        if constexpr (DIG) { if (stop && amask) digest(L); }
        amask = stop ? 0u : amask;
    }
};

// ---------------------------------------------------------------------------
// Lean jump tail: identical per-case semantics to JumpTail (jump::jump_until,
// coarse test), different stepping.  Invariant for every active lane at a step:
// 2^32 <= n < 2^96 and x < 2^(96-K).  Then the 3-limb jump is exact, ends below
// 2^96 <= 2^(cutoff-K) (so the guard cannot fire before the next jump), and the
// coarse test is (x2:x1) < (n2:n1).  A lane whose new x reaches 2^(96-K) (the
// next jump needs 4 limbs and the guard) finishes its case at once in the exact
// 4-limb loop behind a warp vote (rare: P(x >= 2^88) ~ 1e-4 per lane-jump for
// classes 0-4 at N72), so no stale 3-limb step ever touches it.  Cases that do
// not satisfy the invariant at pop (small n, x >= 2^(96-K), guard) also run the
// 4-limb loop at pop.  Flags are recorded immediately on those rare paths, so
// this tail never defers a flag.
template<int TV, bool CAP, bool DIG = false>
struct LeanJumpTail {
    using X = XTraits<TV>;
    static_assert(X::valid && X::jump && X::opt && X::rep == 1 && X::prefix == 0 && !X::skip,
                  "warp-queue jump tails use the optimized unreplicated codes");
    static constexpr int W = X::w;
    static constexpr int K = x_growth_bits(W);
    static constexpr bool THREE = false;   // no warp-level operation inside a lane's own step decision
    static constexpr bool DEFERS = false;  // never sets pend
    static constexpr u32 TABLE_BYTES = x_table_bytes<TV>();
    static constexpr u32 LIM3 = 1u << (32 - K);

    struct Lane {
        u32 x0, x1, x2, x3;
        u32 b0, b1, b2, b3;
        u32 pend;            // unused (kept for the scheduler interface)
        u32 left;            // CAP only
        u32 chash, cjumps;   // DIG only
        u64 dsum, dxor;
        u32 dcount;
    };

    __device__ __forceinline__ static void idle(Lane& L) {
        L.x0 = L.x1 = L.x2 = L.x3 = 0;
        L.b0 = L.b1 = L.b2 = L.b3 = 0;
        L.pend = 0; L.left = 0;
        L.chash = L.cjumps = 0; L.dsum = L.dxor = 0; L.dcount = 0;
    }
    __device__ __forceinline__ static void digest(Lane& L) {
        const u64 d = wq_case_digest(L.b0, L.b1, L.b2, L.b3, L.cjumps, L.chash);
        L.dsum += d; L.dxor ^= d; L.dcount += 1;
    }
    __device__ __forceinline__ static void settle(Lane&, const WqSink*) {}

    template<int MLIMB>
    __device__ __forceinline__ static void make_record(uint4& rec0, uint4& rec1, const WqEp& ep, u32 aval,
                                                       const WqParams& w) {
        JumpTail<TV, CAP, true, DIG>::template make_record<MLIMB>(rec0, rec1, ep, aval, w);
    }

    // Exact 4-limb loop from the current state to the end of the case
    // (jump_until: guard before every jump, coarse or exact test after it).
    __device__ __forceinline__ static void finish4(Lane& L, u32& hash, u32& iters, const u32* s_tab,
                                                   const WqParams& w, const WqSink* sink) {
        const bool small = ((L.b1 | L.b2) == 0) | (L.b3 != 0);
        while (true) {
            if (L.x3 >= w.guard_hi) {
                wq_flag_out(L.b0, L.b1, L.b2, L.b3, sink);
                break;
            }
            if constexpr (CAP) { if (L.left == 0) break; }
            x_jump4o<TV>(s_tab, 0, L.x0, L.x1, L.x2, L.x3);
            hash ^= L.x0;
            iters += 1;
            if constexpr (DIG) { L.chash ^= L.x0; L.cjumps += 1; }
            if constexpr (CAP) { L.left -= 1; }
            const bool desc = small ? x_below4(L.x0, L.x1, L.x2, L.x3, L.b0, L.b1, L.b2, L.b3)
                                    : ((L.x3 == 0) & ((((u64)L.x2 << 32) | L.x1) < (((u64)L.b2 << 32) | L.b1)));
            if (desc) break;
            if constexpr (CAP) {
                if (L.x3 >= w.guard_hi) {   // CPU jump_until_cap: guard before the cap
                    wq_flag_out(L.b0, L.b1, L.b2, L.b3, sink);
                    break;
                }
                if (L.left == 0) break;
            }
        }
        if constexpr (DIG) digest(L);
    }

    // POPX pop: predicated load into the lane; returns whether the case must
    // take the rare start path.
    __device__ __forceinline__ static void pop_load(Lane& L, bool take, u32 saddr, const WqParams& w) {
        wq_pred_load_rec(take, saddr, L.x0, L.x1, L.x2, L.x3, L.b0, L.b1, L.b2, L.b3);
        if constexpr (CAP) { if (take) L.left = w.cap; }
        if constexpr (DIG) { if (take) { L.chash = 0; L.cjumps = 0; } }
    }
    __device__ __forceinline__ static bool pop_rare(const Lane& L, const WqParams& w) {
        const bool small = ((L.b1 | L.b2) == 0) | (L.b3 != 0);
        bool rare = (L.x3 != 0) | (L.x2 >= LIM3) | small;
        if constexpr (CAP) rare |= w.cap == 0;
        return rare;
    }
    __device__ __forceinline__ static u32 start_rare(Lane& L, u32& hash, u32& iters, const u32* s_tab,
                                                     const WqParams& w, const WqSink* sink) {
        finish4(L, hash, iters, s_tab, w, sink);
        L.x0 = L.x1 = L.x2 = L.x3 = 0;
        return 0u;
    }
    // Ring-pop interface of the other schedulers (start = load + rare path).
    __device__ __forceinline__ static u32 start(Lane& L, const uint4& r0, const uint4& r1, u32& hash, u32& iters,
                                                const u32* s_tab, const WqParams& w, const WqSink* sink) {
        L.x0 = r0.x; L.x1 = r0.y; L.x2 = r0.z; L.x3 = r0.w;
        L.b0 = r1.x; L.b1 = r1.y; L.b2 = r1.z; L.b3 = r1.w;
        if constexpr (CAP) L.left = w.cap;
        if constexpr (DIG) { L.chash = 0; L.cjumps = 0; }
        if (pop_rare(L, w)) return start_rare(L, hash, iters, s_tab, w, sink);
        return WQJ_FULL;
    }

    // One 3-limb jump on every lane; rare 4-limb completion behind a vote.
    __device__ __forceinline__ static void step(Lane& L, u32& amask, u32& hash, u32& iters, const u32* s_tab,
                                                const WqParams& w, const WqSink* sink) {
        x_jump3o<TV>(s_tab, 0, L.x0, L.x1, L.x2);
        hash ^= L.x0 & amask;
        iters -= amask;
        if constexpr (DIG) { L.chash ^= L.x0 & amask; L.cjumps -= amask; }
        if constexpr (CAP) L.left -= amask & 1u;
        const bool desc = (((u64)L.x2 << 32) | L.x1) < (((u64)L.b2 << 32) | L.b1);
        // desc implies x < n < 2^96 - K bits headroom, so desc and big exclude each other.
        bool done = desc;
        if constexpr (CAP) done |= L.left == 0;
        const bool big = (amask != 0) & !done & (L.x2 >= LIM3);
        if constexpr (DIG) { if (done && amask) digest(L); }
        amask = done ? 0u : amask;
        if (__any_sync(WQJ_FULL, big)) {   // rare
            if (big) {
                L.x3 = 0;   // x < 2^96 after a 3-limb jump from below 2^(96-K)
                if constexpr (CAP) {
                    // jump_until_cap: guard (cannot fire here) then the cap.
                    if (L.left == 0) { if constexpr (DIG) digest(L); amask = 0u; return; }
                }
                finish4(L, hash, iters, s_tab, w, sink);
                L.x0 = L.x1 = L.x2 = 0;
                amask = 0u;
            }
        }
    }
};

// ---------------------------------------------------------------------------
// The scheduler.  MLIMB = M / 32 (limb receiving aval * 2^M).
template<class Tail, int MLIMB, int V>
__global__ void __launch_bounds__(32 * WqCfg<V>::WARPS, WqCfg<V>::MINB)
cv_wqj_kernel(const u32* __restrict__ bv, const u32* __restrict__ mod9, const u32* __restrict__ affine,
              const WqParams w, KOut* out, u32* flags, const u32* __restrict__ xtab) {
    using C = WqCfg<V>;
    constexpr int WARPS = C::WARPS, RING = C::RING, QCAP = C::QCAP;
    static_assert((RING & (RING - 1)) == 0 && RING <= 32, "ring size");
    extern __shared__ __align__(16) u32 s_tab[];
    __shared__ uint4 s_rec[WARPS][RING][2];
    __shared__ WqWarp s_w[WARPS];
    __shared__ u16 s_queue[WARPS][QCAP];
    __shared__ WqSink s_sink;
    if (threadIdx.x == 0) {
        s_sink.out = out;
        s_sink.flags = flags;
        s_sink.cap = w.flag_cap;
    }

    if constexpr (Tail::TABLE_BYTES > 0) {
        const uint4* src = reinterpret_cast<const uint4*>(xtab);
        uint4* dst = reinterpret_cast<uint4*>(s_tab);
        for (u32 i = threadIdx.x; i < Tail::TABLE_BYTES / 16; i += blockDim.x) dst[i] = __ldg(src + i);
    }
    const u32 lane = threadIdx.x & 31u;
    const u32 wib = threadIdx.x >> 5;
    if (lane == 0) {
        s_w[wib].ep = 0;
        s_w[wib].ep_end = 0;
        s_w[wib].have_ep = 0;
    }
    __syncthreads();

    typename Tail::Lane L;
    Tail::idle(L);
    u32 amask = 0;
    u32 hash = 0, iters32 = 0, cases = 0;
    u64 iters64 = 0;
    const u32 lt_mask = (1u << lane) - 1u;
    u64 d_trips = 0, d_active = 0, d_pops = 0, d_mats = 0, d_screens = 0;   // INSTR only
    // warp-uniform scheduler state
    u32 qhead = 0, qcount = 0;   // queued completion indices of the current pass
    u32 rhead = 0, rcount = 0;   // materialized records
    bool more = true;

    while (true) {
        u32 idle = __ballot_sync(WQJ_FULL, amask == 0);
        if constexpr (C::INSTR) { d_trips += 1; }
        if ((u32)__popc(idle) >= w.kmin) {
            if (more) {
                while (true) {
                    if (rcount) {
                        if constexpr (C::INSTR) { d_pops += 1; }
                        const u32 rank = __popc(idle & lt_mask);
                        if (amask == 0 && rank < rcount) {
                            const u32 slot = (rhead + rank) & (RING - 1);
                            const uint4 r0 = s_rec[wib][slot][0];
                            const uint4 r1 = s_rec[wib][slot][1];
                            iters64 += iters32;   // iters32 holds at most one finished case
                            iters32 = 0;
                            cases += 1;
                            amask = Tail::start(L, r0, r1, hash, iters32, s_tab, w, &s_sink);
                        }
                        const u32 want = __popc(idle);
                        const u32 took = want < rcount ? want : rcount;
                        rhead += took;
                        rcount -= took;
                        if (took == want) break;
                        idle = __ballot_sync(WQJ_FULL, amask == 0);
                        continue;
                    }
                    WqWarp& sw = s_w[wib];
                    if (qcount == 0) {
                        // Next screening pass: same chunk (rescreen), next chunk or next endpoint.
                        if (!sw.have_ep) {
                            u32 ep = sw.ep, ep_end = sw.ep_end;
                            if (ep >= ep_end) {
                                u32 start = 0;
                                if (lane == 0) start = atomicAdd(&out->next, w.fetch);
                                start = __shfl_sync(WQJ_FULL, start, 0);
                                if (start >= w.n) { more = false; break; }
                                ep = start;
                                ep_end = start + w.fetch < w.n ? start + w.fetch : w.n;
                            }
                            const uint4 rb = reinterpret_cast<const uint4*>(affine)[(size_t)ep * 2];
                            const uint4 ri = reinterpret_cast<const uint4*>(affine)[(size_t)ep * 2 + 1];
                            const u32 oee = rb.w >> 30;
                            const u32 family = (rb.w >> 29) & 1u;
                            const u32 b3 = rb.w & 0x1FFFFFFFu;
                            const u64 lo1 = ((u64)ri.x | ((u64)ri.y << 32)) + 1;   // low bits of T^M(n0) + 1
                            u32 redmod9 = (rb.x & 0x3FFFF) + (rb.x >> 18)
                                        + 4 * (rb.y & 0x3FFFF) + 4 * (rb.y >> 18)
                                        + 7 * (rb.z & 0x3FFFF) + 7 * (rb.z >> 18)
                                        + (b3 & 0x3FFFF) + (b3 >> 18);
                            redmod9 %= 9;
                            const u64 index = (w.magic * lo1) & w.b_mask;
                            u32 oee_mask = 0xFFFFFFFFu;
                            if (w.tail_oee) {
                                if (oee == 2) {
                                    oee_mask = ((u32)lo1 & 1) ? 0xAAAAAAAAu : 0x55555555u;
                                } else if (oee == 1) {
                                    const u32 forbidden = ((1u - (u32)lo1) * w.tp_0) & 3u;
                                    oee_mask = ~(0x11111111u << forbidden);
                                }
                            }
                            __syncwarp();   // earlier reads of sw by other lanes
                            if (lane == 0) {
                                sw.e.n0 = make_uint4(rb.x, rb.y, rb.z, b3);
                                sw.e.it = ri;
                                sw.bvw = bv + family * w.fam_stride + (index >> 5);
                                sw.m9 = mod9 + w.words32 * redmod9;
                                sw.offset = (u32)(index & 31);
                                sw.oee_mask = oee_mask;
                                sw.ep = ep;
                                sw.ep_end = ep_end;
                                sw.chunk = 0;
                                sw.qbase = 0;
                                sw.have_ep = 1;
                            }
                            __syncwarp();
                        }
                        if constexpr (C::INSTR) { d_screens += 1; }
                        const u32 chunk = sw.chunk, qbase = sw.qbase;
                        const u32 wi = chunk * 32u + lane;
                        u32 word = 0;
                        if (wi < w.words32) {
                            const u32* bvw = sw.bvw;
                            word = sw.m9[wi] & __funnelshift_r(bvw[wi], bvw[wi + 1], sw.offset) & sw.oee_mask;
                        }
                        const u32 cnt = __popc(word);
                        u32 incl = cnt;
#pragma unroll
                        for (int o = 1; o < 32; o <<= 1) {
                            const u32 up = __shfl_up_sync(WQJ_FULL, incl, o);
                            if (lane >= (u32)o) incl += up;
                        }
                        const u32 total = __shfl_sync(WQJ_FULL, incl, 31);
                        u16* q = s_queue[wib];
                        if constexpr (QCAP >= 576) {
                            u32 pos = incl - cnt;
                            while (word) {
                                q[pos++] = (u16)(32u * wi + (__ffs(word) - 1));
                                word &= word - 1;
                            }
                            qcount = total;
                        } else {
                            u32 r = incl - cnt;
                            if (r + cnt > qbase && r < qbase + QCAP) {
                                while (word) {
                                    if (r >= qbase && r < qbase + QCAP) q[r - qbase] = (u16)(32u * wi + (__ffs(word) - 1));
                                    r++;
                                    word &= word - 1;
                                }
                            }
                            const u32 left = total > qbase ? total - qbase : 0u;
                            qcount = left < (u32)QCAP ? left : (u32)QCAP;
                        }
                        qhead = 0;
                        __syncwarp();   // queue writes before reads; sw reads before the update below
                        if (lane == 0) {
                            if (QCAP < 576 && total > qbase + QCAP) {
                                sw.qbase = qbase + QCAP;   // same chunk again for the remaining ranks
                            } else {
                                sw.qbase = 0;
                                if (chunk + 1 == w.nchunks) {
                                    sw.chunk = 0;
                                    sw.ep = sw.ep + 1;
                                    sw.have_ep = 0;
                                } else {
                                    sw.chunk = chunk + 1;
                                }
                            }
                        }
                        __syncwarp();
                        if (qcount == 0) continue;
                    }
                    // Materialize m records (the ring is empty here).  The barrier
                    // orders earlier pops (ring reads) before these ring writes.
                    const u32 m = qcount < (u32)RING ? qcount : (u32)RING;
                    if constexpr (C::INSTR) { d_mats += 1; }
                    __syncwarp();
                    if (lane < m) {
                        const u32 aval = s_queue[wib][qhead + lane];
                        const WqEp e = sw.e;
                        uint4 r0, r1;
                        Tail::template make_record<MLIMB>(r0, r1, e, aval, w);
                        const u32 slot = (rhead + lane) & (RING - 1);
                        s_rec[wib][slot][0] = r0;
                        s_rec[wib][slot][1] = r1;
                    }
                    __syncwarp();
                    qhead += m;
                    qcount -= m;
                    rcount = m;
                }
            }
            if (!more) {
                idle = __ballot_sync(WQJ_FULL, amask == 0);
                if (idle == WQJ_FULL) {
                    Tail::settle(L, &s_sink);
                    break;
                }
            }
        }
#pragma unroll
        for (int u = 0; u < C::U; u++) {
            if constexpr (C::INSTR) { d_active += __popc(__ballot_sync(WQJ_FULL, amask != 0)); }
            Tail::step(L, amask, hash, iters32, s_tab, w, &s_sink);
        }
    }

    u64 c = cases;
    u64 iters = iters64 + iters32;
    u64 ds = 0, dx = 0, dn = 0;
    if constexpr (C::DIG) { ds = L.dsum; dx = L.dxor; dn = L.dcount; }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        hash ^= __shfl_xor_sync(WQJ_FULL, hash, o);
        c += __shfl_xor_sync(WQJ_FULL, c, o);
        iters += __shfl_xor_sync(WQJ_FULL, iters, o);
        if constexpr (C::DIG) {
            ds += __shfl_xor_sync(WQJ_FULL, ds, o);
            dx ^= __shfl_xor_sync(WQJ_FULL, dx, o);
            dn += __shfl_xor_sync(WQJ_FULL, dn, o);
        }
    }
    if (lane == 0) {
        atomicXor(&out->hash, hash);
        atomicAdd(&out->cases, (unsigned long long)c);
        atomicAdd(&out->iters, (unsigned long long)iters);
        if constexpr (C::DIG) {
            atomicAdd(&out->dbg[0], (unsigned long long)ds);
            atomicXor(&out->dbg[1], (unsigned long long)dx);
            atomicAdd(&out->dbg[2], (unsigned long long)dn);
        }
        if constexpr (C::INSTR) {
            atomicAdd(&out->dbg[0], (unsigned long long)d_trips);
            atomicAdd(&out->dbg[1], (unsigned long long)d_active);
            atomicAdd(&out->dbg[2], (unsigned long long)d_pops);
            atomicAdd(&out->dbg[3], (unsigned long long)d_mats);
            atomicAdd(&out->dbg[4], (unsigned long long)d_screens);
            atomicAdd(&out->dbg[5], (unsigned long long)C::U);
        }
    }
}

// ---------------------------------------------------------------------------
// Scheduler 2 ("trip-pop", layouts with POP2): every loop trip, idle lanes
// take records from the ring by rank (no separate pop rounds, no K wait);
// whenever the ring holds fewer records than there are idle lanes, it is
// topped up by appending min(queue, RING - rcount) materialized records.
// KParams K (bits 3-7) here is the minimum number of idle lanes for a pop
// (1 = pop whenever a lane is idle).  Pending flags (cases stopped at the
// guard) are recorded before the lane takes a new case.
template<class Tail, int MLIMB, int V>
__global__ void __launch_bounds__(32 * WqCfg<V>::WARPS, WqCfg<V>::MINB)
cv_wqj2_kernel(const u32* __restrict__ bv, const u32* __restrict__ mod9, const u32* __restrict__ affine,
               const WqParams w, KOut* out, u32* flags, const u32* __restrict__ xtab) {
    using C = WqCfg<V>;
    constexpr int WARPS = C::WARPS, RING = C::RING, QCAP = C::QCAP;
    static_assert((RING & (RING - 1)) == 0 && RING <= 32, "ring size");
    extern __shared__ __align__(16) u32 s_tab[];
    __shared__ uint4 s_rec[WARPS][RING][2];
    __shared__ WqWarp s_w[WARPS];
    __shared__ u16 s_queue[WARPS][QCAP];
    __shared__ WqSink s_sink;
    if (threadIdx.x == 0) {
        s_sink.out = out;
        s_sink.flags = flags;
        s_sink.cap = w.flag_cap;
    }

    if constexpr (Tail::TABLE_BYTES > 0) {
        const uint4* src = reinterpret_cast<const uint4*>(xtab);
        uint4* dst = reinterpret_cast<uint4*>(s_tab);
        for (u32 i = threadIdx.x; i < Tail::TABLE_BYTES / 16; i += blockDim.x) dst[i] = __ldg(src + i);
    }
    const u32 lane = threadIdx.x & 31u;
    const u32 wib = threadIdx.x >> 5;
    if (lane == 0) {
        s_w[wib].ep = 0;
        s_w[wib].ep_end = 0;
        s_w[wib].have_ep = 0;
    }
    __syncthreads();

    typename Tail::Lane L;
    Tail::idle(L);
    u32 amask = 0;
    u32 hash = 0, iters32 = 0, cases = 0;
    u64 iters64 = 0;
    const u32 lt_mask = (1u << lane) - 1u;
    u64 d_trips = 0, d_active = 0, d_pops = 0, d_mats = 0, d_screens = 0;   // INSTR only
    u32 qhead = 0, qcount = 0;   // queued completion indices of the current pass
    u32 rhead = 0, rcount = 0;   // ring: records [rhead, rhead + rcount)
    bool more = true;

    while (true) {
        u32 idle = __ballot_sync(WQJ_FULL, amask == 0);
        if constexpr (C::INSTR) { d_trips += 1; }
        u32 nidle = __popc(idle);
        if (nidle >= w.kmin) {
            // Top up the ring when it cannot serve every idle lane (at most a
            // full ring: with NL > 1 or RING < 32 there can be more idle slots
            // than ring entries, and an uncapped target would never be met).
            const u32 want_records = nidle < (u32)RING ? nidle : (u32)RING;
            while (rcount < want_records && more) {
                WqWarp& sw = s_w[wib];
                if (qcount == 0) {
                    if (!sw.have_ep) {
                        u32 ep = sw.ep, ep_end = sw.ep_end;
                        if (ep >= ep_end) {
                            u32 start = 0;
                            if (lane == 0) start = atomicAdd(&out->next, w.fetch);
                            start = __shfl_sync(WQJ_FULL, start, 0);
                            if (start >= w.n) { more = false; break; }
                            ep = start;
                            ep_end = start + w.fetch < w.n ? start + w.fetch : w.n;
                        }
                        const uint4 rb = reinterpret_cast<const uint4*>(affine)[(size_t)ep * 2];
                        const uint4 ri = reinterpret_cast<const uint4*>(affine)[(size_t)ep * 2 + 1];
                        const u32 oee = rb.w >> 30;
                        const u32 family = (rb.w >> 29) & 1u;
                        const u32 b3 = rb.w & 0x1FFFFFFFu;
                        const u64 lo1 = ((u64)ri.x | ((u64)ri.y << 32)) + 1;   // low bits of T^M(n0) + 1
                        u32 redmod9 = (rb.x & 0x3FFFF) + (rb.x >> 18)
                                    + 4 * (rb.y & 0x3FFFF) + 4 * (rb.y >> 18)
                                    + 7 * (rb.z & 0x3FFFF) + 7 * (rb.z >> 18)
                                    + (b3 & 0x3FFFF) + (b3 >> 18);
                        redmod9 %= 9;
                        const u64 index = (w.magic * lo1) & w.b_mask;
                        u32 oee_mask = 0xFFFFFFFFu;
                        if (w.tail_oee) {
                            if (oee == 2) {
                                oee_mask = ((u32)lo1 & 1) ? 0xAAAAAAAAu : 0x55555555u;
                            } else if (oee == 1) {
                                const u32 forbidden = ((1u - (u32)lo1) * w.tp_0) & 3u;
                                oee_mask = ~(0x11111111u << forbidden);
                            }
                        }
                        __syncwarp();   // earlier reads of sw by other lanes
                        if (lane == 0) {
                            sw.e.n0 = make_uint4(rb.x, rb.y, rb.z, b3);
                            sw.e.it = ri;
                            sw.bvw = bv + family * w.fam_stride + (index >> 5);
                            sw.m9 = mod9 + w.words32 * redmod9;
                            sw.offset = (u32)(index & 31);
                            sw.oee_mask = oee_mask;
                            sw.ep = ep;
                            sw.ep_end = ep_end;
                            sw.chunk = 0;
                            sw.qbase = 0;
                            sw.have_ep = 1;
                        }
                        __syncwarp();
                    }
                    if constexpr (C::INSTR) { d_screens += 1; }
                    const u32 chunk = sw.chunk, qbase = sw.qbase;
                    const u32 wi = chunk * 32u + lane;
                    u32 word = 0;
                    if (wi < w.words32) {
                        const u32* bvw = sw.bvw;
                        word = sw.m9[wi] & __funnelshift_r(bvw[wi], bvw[wi + 1], sw.offset) & sw.oee_mask;
                    }
                    const u32 cnt = __popc(word);
                    u32 incl = cnt;
#pragma unroll
                    for (int o = 1; o < 32; o <<= 1) {
                        const u32 up = __shfl_up_sync(WQJ_FULL, incl, o);
                        if (lane >= (u32)o) incl += up;
                    }
                    const u32 total = __shfl_sync(WQJ_FULL, incl, 31);
                    u16* q = s_queue[wib];
                    if constexpr (QCAP >= 576) {
                        u32 pos = incl - cnt;
                        while (word) {
                            q[pos++] = (u16)(32u * wi + (__ffs(word) - 1));
                            word &= word - 1;
                        }
                        qcount = total;
                    } else {
                        u32 r = incl - cnt;
                        if (r + cnt > qbase && r < qbase + QCAP) {
                            while (word) {
                                if (r >= qbase && r < qbase + QCAP) q[r - qbase] = (u16)(32u * wi + (__ffs(word) - 1));
                                r++;
                                word &= word - 1;
                            }
                        }
                        const u32 left = total > qbase ? total - qbase : 0u;
                        qcount = left < (u32)QCAP ? left : (u32)QCAP;
                    }
                    qhead = 0;
                    __syncwarp();   // queue writes before reads; sw reads before the update below
                    if (lane == 0) {
                        if (QCAP < 576 && total > qbase + QCAP) {
                            sw.qbase = qbase + QCAP;   // same chunk again for the remaining ranks
                        } else {
                            sw.qbase = 0;
                            if (chunk + 1 == w.nchunks) {
                                sw.chunk = 0;
                                sw.ep = sw.ep + 1;
                                sw.have_ep = 0;
                            } else {
                                sw.chunk = chunk + 1;
                            }
                        }
                    }
                    __syncwarp();
                    if (qcount == 0) continue;
                }
                // Append m records after the ring's current contents.  The
                // barrier orders earlier pops (ring reads) before these writes.
                const u32 space = (u32)RING - rcount;
                const u32 m = qcount < space ? qcount : space;
                if constexpr (C::INSTR) { d_mats += 1; }
                __syncwarp();
                if (lane < m) {
                    const u32 aval = s_queue[wib][qhead + lane];
                    const WqEp e = sw.e;
                    uint4 r0, r1;
                    Tail::template make_record<MLIMB>(r0, r1, e, aval, w);
                    const u32 slot = (rhead + rcount + lane) & (RING - 1);
                    s_rec[wib][slot][0] = r0;
                    s_rec[wib][slot][1] = r1;
                }
                __syncwarp();
                qhead += m;
                qcount -= m;
                rcount += m;
            }
            if (rcount) {
                if constexpr (C::INSTR) { d_pops += 1; }
                const u32 rank = __popc(idle & lt_mask);
                if constexpr (C::POPX) {
                    // Predicated pop: loads land directly in the lane registers.
                    const bool take = (amask == 0) & (rank < rcount);
                    if constexpr (Tail::DEFERS) {
                        if (__any_sync(WQJ_FULL, take & (L.pend != 0))) {   // rare: flag before L is replaced
                            if (take) Tail::settle(L, &s_sink);
                        }
                    }
                    const u32 saddr = (u32)__cvta_generic_to_shared(&s_rec[wib][0][0])
                                    + (((rhead + rank) & (RING - 1)) << 5);
                    Tail::pop_load(L, take, saddr, w);
                    if (take) {
                        iters64 += iters32;
                        iters32 = 0;
                        cases += 1;
                    }
                    amask = take ? WQJ_FULL : amask;
                    const bool rare = take && Tail::pop_rare(L, w);
                    if (__any_sync(WQJ_FULL, rare)) {   // guard at start, small n, or (lean) x >= 2^(96-K)
                        if (rare) amask = Tail::start_rare(L, hash, iters32, s_tab, w, &s_sink);
                    }
                } else if (amask == 0 && rank < rcount) {
                    const u32 slot = (rhead + rank) & (RING - 1);
                    const uint4 r0 = s_rec[wib][slot][0];
                    const uint4 r1 = s_rec[wib][slot][1];
                    iters64 += iters32;   // iters32 never holds more than the lane's current case
                    iters32 = 0;
                    cases += 1;
                    amask = Tail::start(L, r0, r1, hash, iters32, s_tab, w, &s_sink);
                }
                const u32 took = nidle < rcount ? nidle : rcount;
                rhead += took;
                rcount -= took;
            } else if (!more && nidle == 32) {
                Tail::settle(L, &s_sink);
                break;   // no work left anywhere in this warp
            }
        }
#pragma unroll
        for (int u = 0; u < C::U; u++) {
            if constexpr (C::INSTR) { d_active += __popc(__ballot_sync(WQJ_FULL, amask != 0)); }
            Tail::step(L, amask, hash, iters32, s_tab, w, &s_sink);
        }
    }

    u64 c = cases;
    u64 iters = iters64 + iters32;
    u64 ds = 0, dx = 0, dn = 0;
    if constexpr (C::DIG) { ds = L.dsum; dx = L.dxor; dn = L.dcount; }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        hash ^= __shfl_xor_sync(WQJ_FULL, hash, o);
        c += __shfl_xor_sync(WQJ_FULL, c, o);
        iters += __shfl_xor_sync(WQJ_FULL, iters, o);
        if constexpr (C::DIG) {
            ds += __shfl_xor_sync(WQJ_FULL, ds, o);
            dx ^= __shfl_xor_sync(WQJ_FULL, dx, o);
            dn += __shfl_xor_sync(WQJ_FULL, dn, o);
        }
    }
    if (lane == 0) {
        atomicXor(&out->hash, hash);
        atomicAdd(&out->cases, (unsigned long long)c);
        atomicAdd(&out->iters, (unsigned long long)iters);
        if constexpr (C::DIG) {
            atomicAdd(&out->dbg[0], (unsigned long long)ds);
            atomicXor(&out->dbg[1], (unsigned long long)dx);
            atomicAdd(&out->dbg[2], (unsigned long long)dn);
        }
        if constexpr (C::INSTR) {
            atomicAdd(&out->dbg[0], (unsigned long long)d_trips);
            atomicAdd(&out->dbg[1], (unsigned long long)d_active);
            atomicAdd(&out->dbg[2], (unsigned long long)d_pops);
            atomicAdd(&out->dbg[3], (unsigned long long)d_mats);
            atomicAdd(&out->dbg[4], (unsigned long long)d_screens);
            atomicAdd(&out->dbg[5], (unsigned long long)C::U);
        }
    }
}

template<class Tail, int MLIMB, int V>
__global__ void __launch_bounds__(32 * WqCfg<V>::WARPS, WqCfg<V>::MINB)
cv_wqj3_kernel(const u32* __restrict__ bv, const u32* __restrict__ mod9, const u32* __restrict__ affine,
               const WqParams w, KOut* out, u32* flags, const u32* __restrict__ xtab) {
    using C = WqCfg<V>;
    constexpr int WARPS = C::WARPS, RING = C::RING, QCAP = C::QCAP;
    static_assert((RING & (RING - 1)) == 0 && RING <= 32, "ring size");
    extern __shared__ __align__(16) u32 s_tab[];
    __shared__ uint4 s_rec[WARPS][RING][2];
    __shared__ WqWarp s_w[WARPS];
    __shared__ u16 s_queue[WARPS][QCAP];
    __shared__ WqSink s_sink;
    if (threadIdx.x == 0) {
        s_sink.out = out;
        s_sink.flags = flags;
        s_sink.cap = w.flag_cap;
    }

    if constexpr (Tail::TABLE_BYTES > 0) {
        const uint4* src = reinterpret_cast<const uint4*>(xtab);
        uint4* dst = reinterpret_cast<uint4*>(s_tab);
        for (u32 i = threadIdx.x; i < Tail::TABLE_BYTES / 16; i += blockDim.x) dst[i] = __ldg(src + i);
    }
    const u32 lane = threadIdx.x & 31u;
    const u32 wib = threadIdx.x >> 5;
    if (lane == 0) {
        s_w[wib].ep = 0;
        s_w[wib].ep_end = 0;
        s_w[wib].have_ep = 0;
    }
    __syncthreads();

    constexpr int NL = C::NL;   // independent case slots per lane (ILP)
    typename Tail::Lane L[NL];
    u32 amask[NL];
#pragma unroll
    for (int j = 0; j < NL; j++) { Tail::idle(L[j]); amask[j] = 0; }
    u32 hash = 0, iters32 = 0, cases = 0;
    u64 iters64 = 0;
    const u32 lt_mask = (1u << lane) - 1u;
    u64 d_trips = 0, d_active = 0, d_pops = 0, d_mats = 0, d_screens = 0;   // INSTR only
    u32 qhead = 0, qcount = 0;   // queued completion indices of the current pass
    u32 rhead = 0, rcount = 0;   // ring: records [rhead, rhead + rcount)
    bool more = true;

    while (true) {
        u32 idle[NL];
        u32 nidle = 0;
#pragma unroll
        for (int j = 0; j < NL; j++) { idle[j] = __ballot_sync(WQJ_FULL, amask[j] == 0); nidle += __popc(idle[j]); }
        if constexpr (C::INSTR) { d_trips += 1; }
        if (nidle >= w.kmin) {
            // Top up the ring when it cannot serve every idle lane (at most a
            // full ring: with NL > 1 or RING < 32 there can be more idle slots
            // than ring entries, and an uncapped target would never be met).
            const u32 want_records = nidle < (u32)RING ? nidle : (u32)RING;
            while (rcount < want_records && more) {
                WqWarp& sw = s_w[wib];
                if (qcount == 0) {
                    if (!sw.have_ep) {
                        u32 ep = sw.ep, ep_end = sw.ep_end;
                        if (ep >= ep_end) {
                            u32 start = 0;
                            if (lane == 0) start = atomicAdd(&out->next, w.fetch);
                            start = __shfl_sync(WQJ_FULL, start, 0);
                            if (start >= w.n) { more = false; break; }
                            ep = start;
                            ep_end = start + w.fetch < w.n ? start + w.fetch : w.n;
                        }
                        const uint4 rb = reinterpret_cast<const uint4*>(affine)[(size_t)ep * 2];
                        const uint4 ri = reinterpret_cast<const uint4*>(affine)[(size_t)ep * 2 + 1];
                        const u32 oee = rb.w >> 30;
                        const u32 family = (rb.w >> 29) & 1u;
                        const u32 b3 = rb.w & 0x1FFFFFFFu;
                        const u64 lo1 = ((u64)ri.x | ((u64)ri.y << 32)) + 1;   // low bits of T^M(n0) + 1
                        u32 redmod9 = (rb.x & 0x3FFFF) + (rb.x >> 18)
                                    + 4 * (rb.y & 0x3FFFF) + 4 * (rb.y >> 18)
                                    + 7 * (rb.z & 0x3FFFF) + 7 * (rb.z >> 18)
                                    + (b3 & 0x3FFFF) + (b3 >> 18);
                        redmod9 %= 9;
                        const u64 index = (w.magic * lo1) & w.b_mask;
                        u32 oee_mask = 0xFFFFFFFFu;
                        if (w.tail_oee) {
                            if (oee == 2) {
                                oee_mask = ((u32)lo1 & 1) ? 0xAAAAAAAAu : 0x55555555u;
                            } else if (oee == 1) {
                                const u32 forbidden = ((1u - (u32)lo1) * w.tp_0) & 3u;
                                oee_mask = ~(0x11111111u << forbidden);
                            }
                        }
                        __syncwarp();   // earlier reads of sw by other lanes
                        if (lane == 0) {
                            sw.e.n0 = make_uint4(rb.x, rb.y, rb.z, b3);
                            sw.e.it = ri;
                            sw.bvw = bv + family * w.fam_stride + (index >> 5);
                            sw.m9 = mod9 + w.words32 * redmod9;
                            sw.offset = (u32)(index & 31);
                            sw.oee_mask = oee_mask;
                            sw.ep = ep;
                            sw.ep_end = ep_end;
                            sw.chunk = 0;
                            sw.qbase = 0;
                            sw.have_ep = 1;
                        }
                        __syncwarp();
                    }
                    if constexpr (C::INSTR) { d_screens += 1; }
                    const u32 chunk = sw.chunk, qbase = sw.qbase;
                    const u32 wi = chunk * 32u + lane;
                    u32 word = 0;
                    if (wi < w.words32) {
                        const u32* bvw = sw.bvw;
                        word = sw.m9[wi] & __funnelshift_r(bvw[wi], bvw[wi + 1], sw.offset) & sw.oee_mask;
                    }
                    const u32 cnt = __popc(word);
                    u32 incl = cnt;
#pragma unroll
                    for (int o = 1; o < 32; o <<= 1) {
                        const u32 up = __shfl_up_sync(WQJ_FULL, incl, o);
                        if (lane >= (u32)o) incl += up;
                    }
                    const u32 total = __shfl_sync(WQJ_FULL, incl, 31);
                    u16* q = s_queue[wib];
                    if constexpr (QCAP >= 576) {
                        u32 pos = incl - cnt;
                        while (word) {
                            q[pos++] = (u16)(32u * wi + (__ffs(word) - 1));
                            word &= word - 1;
                        }
                        qcount = total;
                    } else {
                        u32 r = incl - cnt;
                        if (r + cnt > qbase && r < qbase + QCAP) {
                            while (word) {
                                if (r >= qbase && r < qbase + QCAP) q[r - qbase] = (u16)(32u * wi + (__ffs(word) - 1));
                                r++;
                                word &= word - 1;
                            }
                        }
                        const u32 left = total > qbase ? total - qbase : 0u;
                        qcount = left < (u32)QCAP ? left : (u32)QCAP;
                    }
                    qhead = 0;
                    __syncwarp();   // queue writes before reads; sw reads before the update below
                    if (lane == 0) {
                        if (QCAP < 576 && total > qbase + QCAP) {
                            sw.qbase = qbase + QCAP;   // same chunk again for the remaining ranks
                        } else {
                            sw.qbase = 0;
                            if (chunk + 1 == w.nchunks) {
                                sw.chunk = 0;
                                sw.ep = sw.ep + 1;
                                sw.have_ep = 0;
                            } else {
                                sw.chunk = chunk + 1;
                            }
                        }
                    }
                    __syncwarp();
                    if (qcount == 0) continue;
                }
                // Append m records after the ring's current contents.  The
                // barrier orders earlier pops (ring reads) before these writes.
                const u32 space = (u32)RING - rcount;
                const u32 m = qcount < space ? qcount : space;
                if constexpr (C::INSTR) { d_mats += 1; }
                __syncwarp();
                if (lane < m) {
                    const u32 aval = s_queue[wib][qhead + lane];
                    const WqEp e = sw.e;
                    uint4 r0, r1;
                    Tail::template make_record<MLIMB>(r0, r1, e, aval, w);
                    const u32 slot = (rhead + rcount + lane) & (RING - 1);
                    s_rec[wib][slot][0] = r0;
                    s_rec[wib][slot][1] = r1;
                }
                __syncwarp();
                qhead += m;
                qcount -= m;
                rcount += m;
            }
            if (rcount) {
                if constexpr (C::INSTR) { d_pops += 1; }
                u32 before = 0;   // idle slots of earlier slot planes
#pragma unroll
                for (int j = 0; j < NL; j++) {
                    const u32 rank = before + __popc(idle[j] & lt_mask);
                    if (amask[j] == 0 && rank < rcount) {
                        const u32 slot = (rhead + rank) & (RING - 1);
                        const uint4 r0 = s_rec[wib][slot][0];
                        const uint4 r1 = s_rec[wib][slot][1];
                        iters64 += iters32;   // iters32: at most one case per slot
                        iters32 = 0;
                        cases += 1;
                        amask[j] = Tail::start(L[j], r0, r1, hash, iters32, s_tab, w, &s_sink);
                    }
                    before += __popc(idle[j]);
                }
                const u32 took = nidle < rcount ? nidle : rcount;
                rhead += took;
                rcount -= took;
            } else if (!more && nidle == 32 * NL) {
#pragma unroll
                for (int j = 0; j < NL; j++) Tail::settle(L[j], &s_sink);
                break;   // no work left anywhere in this warp
            }
        }
#pragma unroll
        for (int u = 0; u < C::U; u++) {
#pragma unroll
            for (int j = 0; j < NL; j++) {
                if constexpr (C::INSTR) { d_active += __popc(__ballot_sync(WQJ_FULL, amask[j] != 0)); }
                Tail::step(L[j], amask[j], hash, iters32, s_tab, w, &s_sink);
            }
        }
    }

    u64 c = cases;
    u64 iters = iters64 + iters32;
    u64 ds = 0, dx = 0, dn = 0;
    if constexpr (C::DIG) {
#pragma unroll
        for (int j = 0; j < NL; j++) { ds += L[j].dsum; dx ^= L[j].dxor; dn += L[j].dcount; }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        hash ^= __shfl_xor_sync(WQJ_FULL, hash, o);
        c += __shfl_xor_sync(WQJ_FULL, c, o);
        iters += __shfl_xor_sync(WQJ_FULL, iters, o);
        if constexpr (C::DIG) {
            ds += __shfl_xor_sync(WQJ_FULL, ds, o);
            dx ^= __shfl_xor_sync(WQJ_FULL, dx, o);
            dn += __shfl_xor_sync(WQJ_FULL, dn, o);
        }
    }
    if (lane == 0) {
        atomicXor(&out->hash, hash);
        atomicAdd(&out->cases, (unsigned long long)c);
        atomicAdd(&out->iters, (unsigned long long)iters);
        if constexpr (C::DIG) {
            atomicAdd(&out->dbg[0], (unsigned long long)ds);
            atomicXor(&out->dbg[1], (unsigned long long)dx);
            atomicAdd(&out->dbg[2], (unsigned long long)dn);
        }
        if constexpr (C::INSTR) {
            atomicAdd(&out->dbg[0], (unsigned long long)d_trips);
            atomicAdd(&out->dbg[1], (unsigned long long)d_active);
            atomicAdd(&out->dbg[2], (unsigned long long)d_pops);
            atomicAdd(&out->dbg[3], (unsigned long long)d_mats);
            atomicAdd(&out->dbg[4], (unsigned long long)d_screens);
            atomicAdd(&out->dbg[5], (unsigned long long)C::U);
        }
    }
}

// ---------------------------------------------------------------------------
// Per-thread reference scheduler: thread = endpoint (grid-stride), per-lane
// completion-word scan with prefetch, each case run to completion with the
// tail functor (Tail must not use warp-level operations: FOUR step mode).
template<class Tail, int MLIMB, int V>
__global__ void __launch_bounds__(32 * WqCfg<V>::WARPS, WqCfg<V>::MINB)
cv_wqpt_kernel(const u32* __restrict__ bv, const u32* __restrict__ mod9, const u32* __restrict__ affine,
               const WqParams w, KOut* out, u32* flags, const u32* __restrict__ xtab) {
    using C = WqCfg<V>;
    static_assert((C::FOUR || !Tail::THREE) && !C::LEAN, "per-thread tails cannot vote");
    extern __shared__ __align__(16) u32 s_tab[];
    __shared__ WqSink s_sink;
    if (threadIdx.x == 0) {
        s_sink.out = out;
        s_sink.flags = flags;
        s_sink.cap = w.flag_cap;
    }
    if constexpr (Tail::TABLE_BYTES > 0) {
        const uint4* src = reinterpret_cast<const uint4*>(xtab);
        uint4* dst = reinterpret_cast<uint4*>(s_tab);
        for (u32 i = threadIdx.x; i < Tail::TABLE_BYTES / 16; i += blockDim.x) dst[i] = __ldg(src + i);
    }
    __syncthreads();
    typename Tail::Lane L;
    Tail::idle(L);
    u32 hash = 0, iters32 = 0, cases = 0;
    u64 iters64 = 0;
    for (u32 idx = blockIdx.x * blockDim.x + threadIdx.x; idx < w.n; idx += gridDim.x * blockDim.x) {
        const uint4 rb = reinterpret_cast<const uint4*>(affine)[(size_t)idx * 2];
        const uint4 ri = reinterpret_cast<const uint4*>(affine)[(size_t)idx * 2 + 1];
        const u32 oee = rb.w >> 30;
        const u32 family = (rb.w >> 29) & 1u;
        const u32 b3 = rb.w & 0x1FFFFFFFu;
        const u64 lo1 = ((u64)ri.x | ((u64)ri.y << 32)) + 1;
        u32 redmod9 = (rb.x & 0x3FFFF) + (rb.x >> 18)
                    + 4 * (rb.y & 0x3FFFF) + 4 * (rb.y >> 18)
                    + 7 * (rb.z & 0x3FFFF) + 7 * (rb.z >> 18)
                    + (b3 & 0x3FFFF) + (b3 >> 18);
        redmod9 %= 9;
        const u64 index = (w.magic * lo1) & w.b_mask;
        const u32 offset = (u32)(index & 31);
        const u32* bvw = bv + family * w.fam_stride + (index >> 5);
        const u32* m9 = mod9 + w.words32 * redmod9;
        u32 oee_mask = 0xFFFFFFFFu;
        if (w.tail_oee) {
            if (oee == 2) {
                oee_mask = ((u32)lo1 & 1) ? 0xAAAAAAAAu : 0x55555555u;
            } else if (oee == 1) {
                const u32 forbidden = ((1u - (u32)lo1) * w.tp_0) & 3u;
                oee_mask = ~(0x11111111u << forbidden);
            }
        }
        WqEp e;
        e.n0 = make_uint4(rb.x, rb.y, rb.z, b3);
        e.it = ri;
        u32 wi = 0;
        u32 bv_lo = bvw[0], bv_hi = bvw[1];
        u32 bv_next = w.words32 > 1 ? bvw[2] : 0u;
        u32 word = m9[0] & __funnelshift_r(bv_lo, bv_hi, offset) & oee_mask;
        while (true) {
            while (word == 0) {
                if (++wi == w.words32) goto next_endpoint;
                bv_lo = bv_hi;
                bv_hi = bv_next;
                if (wi + 2 <= w.words32) bv_next = bvw[wi + 2];
                word = m9[wi] & __funnelshift_r(bv_lo, bv_hi, offset) & oee_mask;
            }
            {
                const u32 aval = 32 * wi + (__ffs(word) - 1);
                word &= word - 1;
                uint4 r0, r1;
                Tail::template make_record<MLIMB>(r0, r1, e, aval, w);
                iters64 += iters32;
                iters32 = 0;
                cases += 1;
                u32 amask = Tail::start(L, r0, r1, hash, iters32, s_tab, w, &s_sink);
                while (amask) Tail::step(L, amask, hash, iters32, s_tab, w, &s_sink);
            }
        }
    next_endpoint:;
    }
    Tail::settle(L, &s_sink);
    u64 c = cases;
    u64 iters = iters64 + iters32;
    u64 ds = 0, dx = 0, dn = 0;
    if constexpr (C::DIG) { ds = L.dsum; dx = L.dxor; dn = L.dcount; }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        hash ^= __shfl_xor_sync(WQJ_FULL, hash, o);
        c += __shfl_xor_sync(WQJ_FULL, c, o);
        iters += __shfl_xor_sync(WQJ_FULL, iters, o);
        if constexpr (C::DIG) {
            ds += __shfl_xor_sync(WQJ_FULL, ds, o);
            dx ^= __shfl_xor_sync(WQJ_FULL, dx, o);
            dn += __shfl_xor_sync(WQJ_FULL, dn, o);
        }
    }
    if ((threadIdx.x & 31) == 0) {
        atomicXor(&out->hash, hash);
        atomicAdd(&out->cases, (unsigned long long)c);
        atomicAdd(&out->iters, (unsigned long long)iters);
        if constexpr (C::DIG) {
            atomicAdd(&out->dbg[0], (unsigned long long)ds);
            atomicXor(&out->dbg[1], (unsigned long long)dx);
            atomicAdd(&out->dbg[2], (unsigned long long)dn);
        }
    }
}

// (cv_wqj3_kernel above: the trip-pop scheduler with NL independent case
// slots per lane, interleaved for instruction-level parallelism.)

// ---------------------------------------------------------------------------
// Host side.  Per-context caches live in WqState (one per Ctx).
struct WqState {
    int sms = 0;
    std::map<const void*, int> per_sm;   // kernel -> resident blocks per SM
    std::map<int, u32*> tables;          // tail code -> device table image
    DevTables* dt = nullptr;             // c3-hyg: the owning context's jd tables (cuda/devtables.cuh)
};

// Layout V: pad bits 16-20 plus bit 23 as its sixth bit (0..63).
static inline u32 wq_layout_of(u32 pad) {
    return ((pad >> 16) & 31u) | (((pad >> 23) & 1u) << 5);
}

static bool wq_layout_valid(u32 v) {
    switch (v) {
#define WV(V) case V: return WqCfg<V>::valid;
        WV(0) WV(1) WV(2) WV(3) WV(4) WV(5) WV(6) WV(7) WV(8) WV(9) WV(10) WV(11) WV(12) WV(13) WV(14) WV(15) WV(16) \
        WV(17) WV(18) WV(19) WV(20) WV(21) WV(22) WV(23) WV(24) WV(25) WV(26) WV(27) WV(28) WV(29) WV(30) \
        WV(31) WV(32) WV(33) WV(34) WV(35) WV(36) WV(37) WV(38) WV(39) WV(40) WV(41) WV(42) WV(43) \
        WV(44) WV(45) WV(46) WV(47) WV(48) WV(49) WV(50) WV(51) WV(52)
#undef WV
        default: return false;
    }
}

// c-final: jd FD3 codes 91/92/94/95 (c-tail's JdWqTail3, cuda/wq_jd.cuh) as well.
static bool wq_jd_code(u32 tv);
static bool wq_tail_valid(u32 tv) {
    return (tv >= 49 && tv <= 53) || wq_jd_code(tv);
}

// KParams.tail_budget (WqParams.cap) belongs to the tail family: the jump
// tails use it only for the timing-only capped test tail (pad bit 21).  A tail
// family that needs a per-class constant (e.g. a lookahead budget H0) declares
// it here.
static bool wq_tail_budget_ok(u32 pad, u32 tail_budget) {
    const u32 tv = pad >> 24;
    if (tv >= 49 && tv <= 53) return ((pad >> 21) & 1u) || tail_budget == 0;
    if (wq_jd_code(tv)) return !((pad >> 21) & 1u);   // c-final: jd tails carry H0 (any value); no capped tail
    return false;
}

// The caller has checked bit 22.  Jump guards need cutoff >= 96 + K.
static bool wq_pad_valid(u32 pad, u32 cutoff_hi, u32 words32, u32 m_limb) {
    if ((pad & 7u) != 5u || (pad & 0x300u)) return false;
    const u32 tv = pad >> 24, v = wq_layout_of(pad);
    if (!wq_tail_valid(tv) || !wq_layout_valid(v)) return false;
    if (tv >= 64 ? (cutoff_hi >> x_growth_bits(jd_variant_width(tv))) == 0 : !x_cutoff_ok(tv, cutoff_hi)) return false;
    if (((pad >> 3) & 31u) > 32u) return false;
    if (words32 == 0 || words32 > 2048u || m_limb > 2) return false;   // u16 completion indices, M <= 80
    return true;
}

static const u32* wq_table(WqState& st, int tv) {
    auto it = st.tables.find(tv);
    if (it != st.tables.end()) return it->second;
    int w = x_variant_width(tv);
    int layout = 0, rep = 1;
    bool shpow = false;
    switch (tv) {
#define WT(CODE) case CODE: layout = XTraits<CODE>::layout; rep = XTraits<CODE>::rep; shpow = XTraits<CODE>::shpow; break;
        WT(49) WT(50) WT(51) WT(52) WT(53)
#undef WT
        default: return nullptr;
    }
    const auto host = x_build_table(tv, w, layout, shpow, rep);
    u32* device = nullptr;
    if (cudaMalloc(&device, host.size() * 4) != cudaSuccess ||
        cudaMemcpy(device, host.data(), host.size() * 4, cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "wq table upload failed\n");
        return nullptr;
    }
    st.tables[tv] = device;
    return device;
}

template<class Tail, int MLIMB, int V>
static int wq_launch_kernel(WqState& st, const u32* bv, const u32* mod9, const u32* affine, u32 n, u64 magic,
                            const u32* tp, const KParams& p, KOut* out, u32* flags, const u32* table,
                            cudaStream_t stream) {
    auto kernel = [] {
        if constexpr (WqCfg<V>::PT) return cv_wqpt_kernel<Tail, MLIMB, V>;
        else if constexpr (WqCfg<V>::NL > 1) return cv_wqj3_kernel<Tail, MLIMB, V>;
        else if constexpr (WqCfg<V>::POP2) return cv_wqj2_kernel<Tail, MLIMB, V>;
        else return cv_wqj_kernel<Tail, MLIMB, V>;
    }();
    constexpr u32 threads = 32 * WqCfg<V>::WARPS;
    constexpr u32 dyn = Tail::TABLE_BYTES;
    auto it = st.per_sm.find((const void*)kernel);
    int per_sm;
    if (it == st.per_sm.end()) {
        // Static plus dynamic shared memory may exceed the 48 KB default.
        if (dyn > 0 && cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn) != cudaSuccess)
            return -1;
        per_sm = 0;
        if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, threads, dyn) != cudaSuccess || per_sm <= 0)
            return -1;
        st.per_sm[(const void*)kernel] = per_sm;
        if (getenv("CV_WQ_VERBOSE")) {
            cudaFuncAttributes fa;
            cudaFuncGetAttributes(&fa, kernel);
            fprintf(stderr, "wq kernel layout %d tail %d limb %d: %d regs, %zu B static + %u B dynamic smem, %d blocks/SM (%d warps)\n",
                    V, (int)(p.pad >> 24), MLIMB, fa.numRegs, fa.sharedSizeBytes, dyn, per_sm, per_sm * WqCfg<V>::WARPS);
        }
    } else {
        per_sm = it->second;
    }
    const u32 cap_sm = (p.pad >> 13) & 7u;
    if (cap_sm && (int)cap_sm < per_sm) per_sm = (int)cap_sm;
    WqParams w;
    w.b_mask = p.b_mask;
    w.fam_stride = p.fam_stride;
    w.magic = magic;
    w.tp_0 = tp[0]; w.tp_1 = tp[1]; w.tp_2 = tp[2]; w.tp_3 = tp[3];
    w.n = n;
    w.words32 = p.words32;
    w.nchunks = (p.words32 + 31u) >> 5;
    w.m_shift = p.m_shift;
    w.guard_hi = p.cutoff_hi >> Tail::K;
    w.tail_oee = p.tail_oee;
    w.flag_cap = p.flag_cap;
    w.kmin = (p.pad >> 3) & 31u;
    if (w.kmin == 0) w.kmin = WqCfg<V>::POP2 ? 1u : 8u;
    w.fetch = 1u << ((p.pad >> 10) & 7u);
    w.cap = p.tail_budget;
    const u64 grid = (u64)st.sms * (u64)per_sm;
    u64 need = ((u64)n + (u64)w.fetch * WqCfg<V>::WARPS - 1) / ((u64)w.fetch * WqCfg<V>::WARPS);
    if constexpr (WqCfg<V>::PT) need = ((u64)n + threads - 1) / threads;   // one endpoint per thread
    const u32 blocks = (u32)(need < grid || WqCfg<V>::PT ? need : grid);
    kernel<<<blocks, threads, dyn, stream>>>(bv, mod9, affine, w, out, flags, table);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

// Every layout is instantiated for M in [32, 64) (production M = 62, A = 10..12);
// limbs 0 and 2 (validation shapes such as N32/N40 and M = 64) only for the
// layouts in WQ_ALL_LIMBS, to bound compile time.
template<int V> struct WqAllLimbs { static constexpr bool value = V == 0 || V == 2 || V == 10 || V == 11 || V == 17 || V == 21 || V == 22 || (V >= 26 && V <= 31) || V == 35 || V == 36 || V == 41 || V == 42 || V == 43 || V == 44 || V == 46 || V == 49 || V == 51 || V == 52; };

template<class Tail, int V>
static int wq_launch_m(WqState& st, const u32* bv, const u32* mod9, const u32* affine, u32 n, u64 magic,
                       const u32* tp, const KParams& p, KOut* out, u32* flags, const u32* table,
                       cudaStream_t stream) {
    if (p.m_limb == 1)
        return wq_launch_kernel<Tail, 1, V>(st, bv, mod9, affine, n, magic, tp, p, out, flags, table, stream);
    if constexpr (WqAllLimbs<V>::value) {
        if (p.m_limb == 0)
            return wq_launch_kernel<Tail, 0, V>(st, bv, mod9, affine, n, magic, tp, p, out, flags, table, stream);
        if (p.m_limb == 2)
            return wq_launch_kernel<Tail, 2, V>(st, bv, mod9, affine, n, magic, tp, p, out, flags, table, stream);
    }
    return -1;
}

// Layouts instantiated for the capped test tail (timing-only characterization).
template<int V> struct WqCapLayout { static constexpr bool value = V == 2 || V == 10 || V == 11 || V == 17 || V == 18 || V == 19 || V == 21 || V == 26 || V == 31 || V == 32 || V == 35 || V == 39 || V == 41 || V == 42 || V == 43 || V == 46 || V == 47 || V == 48 || V == 49; };

template<int TV, bool CAP>
static int wq_launch_v(WqState& st, const u32* bv, const u32* mod9, const u32* affine, u32 n, u64 magic,
                       const u32* tp, const KParams& p, KOut* out, u32* flags, const u32* table,
                       cudaStream_t stream) {
    switch (wq_layout_of(p.pad)) {
#define WL(V) case V: \
        if constexpr (!CAP || WqCapLayout<V>::value) { \
            if (CAP && p.m_limb != 1) return -1; \
            using T_ = typename std::conditional<WqCfg<V>::LEAN, LeanJumpTail<TV, CAP, WqCfg<V>::DIG>, \
                                                  JumpTail<TV, CAP, WqCfg<V>::FOUR, WqCfg<V>::DIG>>::type; \
            return wq_launch_m<T_, V>(st, bv, mod9, affine, n, magic, tp, p, out, flags, table, stream); \
        } else return -1;
        WL(0) WL(1) WL(2) WL(3) WL(4) WL(5) WL(6) WL(7) WL(8) WL(9) WL(10) WL(11) WL(12) WL(13) WL(14) WL(15) WL(16) \
        WL(17) WL(18) WL(19) WL(20) WL(21) WL(22) WL(23) WL(24) WL(25) WL(26) WL(27) WL(28) WL(29) WL(30) \
        WL(31) WL(32) WL(33) WL(34) WL(35) WL(36) WL(37) WL(38) WL(39) WL(40) WL(41) WL(42) WL(43) \
        WL(44) WL(45) WL(46) WL(47) WL(48) WL(49) WL(50) WL(51) WL(52)
#undef WL
        default: return -1;
    }
}

// Input: exact-iterate affine32 records (modes 2-4 after expansion; mode 0
// converted by cv_replay_affine first).
static int wq_launch_jd(WqState& st, const u32* bv, const u32* mod9, const u32* affine, u32 n, u64 magic,
                        const u32* tp, const KParams& p, KOut* out, u32* flags, cudaStream_t stream);
static int wq_launch(WqState& st, const u32* bv, const u32* mod9, const u32* affine, u32 n, u64 magic,
                     const u32* tp, const KParams& p, KOut* out, u32* flags, cudaStream_t stream) {
    const int tv = (int)(p.pad >> 24);
    if (tv >= 64) return wq_launch_jd(st, bv, mod9, affine, n, magic, tp, p, out, flags, stream);   // c-final
    const bool cap = (p.pad >> 21) & 1u;
    const u32* table = wq_table(st, tv);
    if (!table) return -1;
    switch (tv) {
#define WJ(CODE) case CODE: return cap ? wq_launch_v<CODE, true>(st, bv, mod9, affine, n, magic, tp, p, out, flags, table, stream) \
                                       : wq_launch_v<CODE, false>(st, bv, mod9, affine, n, magic, tp, p, out, flags, table, stream);
        WJ(49) WJ(50) WJ(51) WJ(52) WJ(53)
#undef WJ
        default: return -1;
    }
}

// Replay16 -> affine32: n0 record unchanged, second uint4 = T^M(n0).
__global__ void cv_wq_replay_affine(const u32* __restrict__ pairs, u32 n, u32 m, u32* __restrict__ affine) {
    const u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const uint4 in = reinterpret_cast<const uint4*>(pairs)[idx];
    u64 lo = (u64)in.x | ((u64)in.y << 32);
    u64 hi = (u64)in.z | ((u64)(in.w & 0x1FFFFFFFu) << 32);
    for (u32 j = 0; j < m; j++) {
        if (lo & 1) {  // n + n/2 + 1
            u64 h_lo = (lo >> 1) | (hi << 63), h_hi = hi >> 1;
            u64 s_lo = lo + h_lo;
            u64 c1 = s_lo < lo;
            u64 s2 = s_lo + 1;
            u64 c2 = s2 < s_lo;
            hi = hi + h_hi + c1 + c2;
            lo = s2;
        } else {
            lo = (lo >> 1) | (hi << 63);
            hi >>= 1;
        }
    }
    uint4* o = reinterpret_cast<uint4*>(affine) + (size_t)idx * 2;
    o[0] = in;
    o[1] = make_uint4((u32)lo, (u32)(lo >> 32), (u32)hi, (u32)(hi >> 32));
}
