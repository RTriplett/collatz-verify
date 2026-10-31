// c3-2ph: two-phase verification kernels.
//
// Phase 1 (warp-cooperative screening): a warp screens ONE endpoint's
// 32-word completion chunk at a time, lane i forming completion word
// chunk * 32 + i with c-screen's cooperative routine (scr_mask_setup /
// scr_word: bitvector window word with the funnel shift, mod-9 or (12, 9)
// row-table mask word, carried OEE mask, completion-word range).  A warp
// prefix sum of popcounts writes the surviving completion indices (aval) to a
// per-warp shared case list (u16 entries: endpoint slot << 13 | aval); the
// endpoint record (n0, T^M(n0)) goes to one of EPN per-warp shared slots.  A
// chunk with more survivors than list space is screened again for the
// remaining ranks (qbase), as in c-sched's queue.
//
// Phase 2 (per-lane jd tails): lanes take cases from the list and run
// c-tail's per-lane jd tail (cuda/jdip.cuh JdLane: exact W-step jumps, FD3
// window-dip certificate, 3-limb steps while x < 2^(96-K), then 4-limb steps
// with the overflow guard).  Two schedulers (TpCfg<V>::POP):
//   POP 0  static rounds: each lane takes the case of rank `lane` and runs it
//          to completion with exactly cv_kernel_jds's loops; the warp then
//          takes the next 32 cases (refilling the list below 32).
//   POP 1  dynamic trip-pop: every trip runs U boundaries on every active
//          lane; when at least K lanes are idle, the list is topped up to the
//          number of idle lanes and idle lanes take cases by rank.
//
// Work is exactly the per-thread kernels' (cv_kernel_jds / cv_kernel_jd with
// the same code and screen): the screen forms the same completion words (same
// routine as the warp queue, audited bit-exactly by `screenbench words`), the
// list holds every surviving completion exactly once (ranks partition each
// chunk's survivors; the list is FIFO and never overwritten before it is
// consumed), and each case runs the same per-case code (JdLane steps with the
// same 3-limb/4-limb switch, guard, small-base exact-descent loop and flag).
// Cases, iterations and the XOR hash are order independent; flags are the
// same multiset.  DIG layouts also return a per-case digest of
// (n, case jumps, case hash) in KOut.dbg[0..2] (cv::jump::case_digest).
//
// KParams.pad (bit 21 set, bit 22 clear; validated by tp_pad_valid):
//   bits 0-2   5 (no vote, constant powers, early OEE)
//   bits 3-7   K: POP 1 tops up / pops only when >= K lanes are idle (0 = 1)
//   bits 10-12 log2 endpoints per dynamic fetch (KOut.next)
//   bits 13-15 resident blocks per SM cap (0 = occupancy limit)
//   bits 16-20 layout V (TpCfg<V>)
//   bit  21    two-phase kernel
//   bits 24-31 jd FD3 code (TP_CODES)
// tail_budget carries H0 = 485 F - 306 M (as every jd launch).
#pragma once

template<int V> struct TpCfg { static constexpr bool valid = false; };
// V: scheduler POP (0 static rounds, 1 dynamic masked steps, 2 dynamic per-lane capped loops,
// 3 lean dynamic masked 4-limb steps: warp control state and case bases in shared memory),
// boundaries per trip U (POP 1: exactly U; POP 2: at most U per live lane), warps/block,
// min blocks/SM (launch bounds), list capacity (u16 entries, power of 2), endpoint slots
// (power of 2, <= 8), DIG (per-case digest in KOut.dbg[0..2]), FOUR (POP 1: 4-limb masked
// steps always, guard folded in; otherwise 3-limb steps with a per-boundary warp vote),
// INSTR (counters in KOut.dbg[3..5]: trips, pop phases, screening passes; timing not used),
// ALLM (also instantiated for M-limbs 0 and 2, i.e. M < 32 and 64 <= M < 96; every layout
// has the production M = 62 limb 1).
// BS (batched setup, A <= 10 only): endpoints are fetched 32 at a time (KOut.next += 32,
// the pad's fetch field is ignored) and lane j computes endpoint j's screening setup
// (bitvector word index and family, mask-word offset, funnel offsets, OEE mask) into a
// 16-byte per-warp shared table; a screening pass then reads one entry (broadcast).
// EPB (endpoints per block; 0 = persistent): with EPB > 0 the grid has ceil(n / EPB) blocks and
// block b owns endpoints [b EPB, min((b + 1) EPB, n)), handed to its warps through a shared
// counter (non-persistent: blocks are short-lived, so kernels of other streams interleave).
#define TP_CFGE(V_, POP_, U_, WARPS_, MINB_, LCAP_, EPN_, DIG_, FOUR_, INSTR_, ALLM_, BS_, EPB_) template<> struct TpCfg<V_> { \
    static constexpr bool valid = true; static constexpr int POP = POP_, U = U_, WARPS = WARPS_, MINB = MINB_; \
    static constexpr int LCAP = LCAP_, EPN = EPN_; static constexpr bool DIG = DIG_, FOUR = FOUR_, INSTR = INSTR_, ALLM = ALLM_; \
    static constexpr bool BS = BS_; static constexpr u32 EPB = EPB_; };
#define TP_CFGB(V_, POP_, U_, WARPS_, MINB_, LCAP_, EPN_, DIG_, FOUR_, INSTR_, ALLM_, BS_) TP_CFGE(V_, POP_, U_, WARPS_, MINB_, LCAP_, EPN_, DIG_, FOUR_, INSTR_, ALLM_, BS_, 0)
#define TP_CFGX(V_, POP_, U_, WARPS_, MINB_, LCAP_, EPN_, DIG_, FOUR_, INSTR_, ALLM_) TP_CFGB(V_, POP_, U_, WARPS_, MINB_, LCAP_, EPN_, DIG_, FOUR_, INSTR_, ALLM_, false)
#define TP_CFG(V_, POP_, U_, WARPS_, MINB_, LCAP_, EPN_, DIG_, FOUR_) TP_CFGX(V_, POP_, U_, WARPS_, MINB_, LCAP_, EPN_, DIG_, FOUR_, false, false)
// Production-candidate and timing layouts (8 warps x 4 blocks or 16 warps x 2 blocks = 32 warps/SM, <= 64 registers).
TP_CFGE(1, 1, 6, 8, 4, 256, 8, false, true, false, true, false, 512)    // POP1 FOUR U6, block ranges of 512 endpoints
TP_CFGE(3, 1, 4, 16, 2, 512, 8, false, true, false, true, true, 1024)   // BS, 16 x 2, list 512, EPB 1024, U = 4 (production, classes 1-3)
TP_CFGE(5, 1, 5, 16, 2, 512, 8, false, true, false, true, true, 1024)   // as 3, U = 5 (production, classes 4-15)
TP_CFGE(6, 1, 8, 16, 2, 512, 8, false, true, false, false, true, 1024)  // as 3, U = 8
TP_CFGE(7, 1, 6, 16, 2, 512, 8, false, true, false, false, true, 512)   // as 3, U = 6, EPB 512
TP_CFGX(4, 0, 1, 8, 4, 256, 8, false, false, false, true)               // static rounds (cv_kernel_jds's loops), persistent
TP_CFGX(8, 1, 6, 8, 4, 256, 8, false, true, false, true)                // POP1 FOUR U6, persistent
TP_CFGE(9, 1, 6, 16, 2, 256, 8, false, true, false, true, true, 1024)   // BS, 16 x 2, list 256, EPB 1024
TP_CFGE(10, 1, 6, 16, 2, 256, 8, false, true, false, true, true, 512)   // BS, 16 x 2, list 256, EPB 512
TP_CFGE(11, 1, 6, 16, 2, 512, 4, false, true, false, false, true, 1024) // as 23 with 4 endpoint slots
TP_CFGE(22, 1, 6, 16, 2, 512, 8, false, true, false, false, true, 2048) // as 23, EPB 2048
TP_CFGE(23, 1, 6, 16, 2, 512, 8, false, true, false, true, true, 1024)  // BS, 16 x 2, list 512, EPB 1024, U = 6
// Audit copies (per-case digest in KOut.dbg[0..2]; every M-limb).
TP_CFGE(2, 1, 6, 16, 2, 512, 8, true, true, false, true, true, 1024)    // = 23
TP_CFGE(12, 1, 4, 16, 2, 512, 8, true, true, false, true, true, 1024)   // = 3
TP_CFGE(14, 1, 2, 8, 4, 64, 2, true, true, false, true, true, 64)       // BS, U = 2, list 64, 2 slots, EPB 64 (rescreens, slot stalls)
TP_CFGX(16, 1, 4, 8, 4, 256, 8, true, false, false, true)               // POP1 U4 3-limb + vote, persistent
TP_CFGX(17, 0, 1, 8, 4, 256, 8, true, false, false, true)               // = 4
TP_CFGX(18, 1, 2, 8, 4, 64, 2, true, false, false, true)                // POP1 U2 vote, list 64, 2 slots
TP_CFGX(19, 0, 1, 8, 4, 64, 2, true, false, false, true)                // static, list 64, 2 slots
TP_CFGX(20, 1, 4, 8, 4, 256, 8, true, true, false, true)                // POP1 FOUR U4, persistent
TP_CFGX(21, 1, 2, 8, 4, 64, 2, true, true, false, true)                 // as 18, 4-limb
TP_CFGE(24, 1, 5, 16, 2, 512, 8, true, true, false, true, true, 1024)   // = 5
// Instrumented copies (counters in KOut.dbg[3..5]).
TP_CFGE(13, 1, 6, 16, 2, 512, 8, false, true, true, false, true, 1024)  // = 23
TP_CFGE(15, 1, 6, 8, 4, 256, 8, false, true, true, false, false, 512)   // = 1
TP_CFGE(25, 1, 6, 16, 2, 256, 8, false, true, true, false, true, 512)   // = 10
TP_CFGX(26, 1, 6, 8, 4, 256, 8, false, true, true, false)               // = 8
TP_CFGX(27, 0, 1, 8, 4, 256, 8, false, false, true, false)              // = 4
TP_CFGE(28, 1, 6, 16, 2, 256, 8, false, true, true, false, true, 1024)  // = 9
// c3-final: chunk copies of the production BS layouts for A >= 11.  A batched screening pass
// forms the 32 words of one completion chunk c of the endpoint (lane i forms word 32 c + i); an
// endpoint's chunks chunk_lo..chunk_hi-1 of the screened range [w_first, w_end) are screened in
// order into the same endpoint slot, as refill_pass does (A = 11: the top half is chunk 1, the
// bottom half chunk 0, the full range chunks 0 and 1).  Layouts without the TpChunk flag are
// unchanged (one chunk, lane i forms word i; A <= 10).
TP_CFGE(29, 1, 4, 16, 2, 512, 8, false, true, false, true, true, 1024)  // = 3 + chunk offset (U = 4)
TP_CFGE(30, 1, 5, 16, 2, 512, 8, false, true, false, true, true, 1024)  // = 5 + chunk offset (U = 5)
TP_CFGE(31, 1, 4, 16, 2, 512, 8, true, true, false, true, true, 1024)   // = 29 with the per-case digest (audit copy)
#undef TP_CFG
#undef TP_CFGX
template<int V> struct TpChunk { static constexpr bool on = false; };
template<> struct TpChunk<29> { static constexpr bool on = true; };
template<> struct TpChunk<30> { static constexpr bool on = true; };
template<> struct TpChunk<31> { static constexpr bool on = true; };

#define TP_LAYOUTS(X) X(1) X(2) X(3) X(4) X(5) X(6) X(7) X(8) X(9) X(10) X(11) X(12) X(13) X(14) X(15) \
                      X(16) X(17) X(18) X(19) X(20) X(21) X(22) X(23) X(24) X(25) X(26) X(27) X(28) \
                      X(29) X(30) X(31)
#define TP_CODES(X) X(90) X(91) X(92) X(93) X(95) X(111)

struct TpEp { uint4 n0; uint4 it; };     // endpoint record: n0 (flag bits cleared), T^M(n0)
struct TpCtl {                           // POP 3: warp-uniform control state, kept in shared memory
    u32 lhead, ltail, seq, ep, ep_end, chunk, qbase, cur_slot, flags, pad_[3];   // flags: have | slot_set << 1 | more << 2
};
struct TpWarp {                          // per-warp screening state of the current endpoint
    uint4 n0, it;
    const u32* bvw;
    const u32* mw;
    u32 offset, oee_mask, sg, pad_;      // sg = row funnel offset | h = 0 guard << 8
};

// Masked per-lane jd step for the dynamic scheduler.  One boundary of
// JdLane<TV>::step_emb (same test, same exact jump, same order) computed on
// every lane without a branch; `live` (0 or ~0) masks the accounting, and a
// firing test clears it.  A lane whose case has stopped, or that holds no
// case, still executes the arithmetic on stale values (table indices stay in
// range: entries are masked to W bits, pair offsets are 8 f <= 120 bytes), but
// its hash, jump count and state no longer matter.  LIMBS = 3 is exact while
// x < 2^(96 - K) before the step (the caller guarantees it for live lanes).
template<int TV> struct TpLane {
    u32 x0, x1, x2, x3;
    int h;               // H << 13
    u32 m0, d0; int g0;  // entry of the next jump
    u32 m1, d1; int g1;  // KA 2: entry of the jump after
    int dip0;            // KA 0: dip field of the current window
    u32 jumps, hash;     // per case (DIG) or per lane (otherwise)
    u32 live;

    __device__ __forceinline__ void init(const JdTables<TV>& t, u32 a0, u32 a1, u32 a2, u32 a3, int h0) {
        using J = JDTraits<TV>;
        x0 = a0; x1 = a1; x2 = a2; x3 = a3;
        h = h0 * 8192;
        const u32 e = t.entry(x0);
        t.split(e, m0, d0, g0);
        dip0 = JdTables<TV>::dipfield(e);
        if constexpr (J::ka == 2) {
            const u32 y1 = m0 * __funnelshift_r(x0, x1, J::w) + d0;
            t.split(t.entry(y1), m1, d1, g1);
        }
    }
    __device__ __forceinline__ bool fits3() const {
        constexpr int K = x_growth_bits(JDTraits<TV>::w);
        return x3 == 0 && x2 < (1u << (32 - K));
    }
    template<int LIMBS>
    __device__ __forceinline__ void finish(u64 p0) {
        constexpr int W = JDTraits<TV>::w;
        if constexpr (LIMBS == 3) {
            const u32 t1 = __funnelshift_r(x1, x2, W);
            const u32 t2 = x2 >> W;
            const u64 p1 = (u64)t1 * m0;
            const u32 q2 = t2 * m0;
            asm("add.cc.u32 %0, %3, %4;\n\t"
                "addc.cc.u32 %1, %5, %6;\n\t"
                "addc.u32 %2, %7, %8;"
                : "=r"(x0), "=r"(x1), "=r"(x2)
                : "r"((u32)p0), "r"(d0), "r"((u32)p1), "r"((u32)(p0 >> 32)), "r"(q2), "r"((u32)(p1 >> 32)));
        } else {
            const u32 t1 = __funnelshift_r(x1, x2, W);
            const u32 t2 = __funnelshift_r(x2, x3, W);
            const u32 t3 = x3 >> W;
            const u64 p1 = (u64)t1 * m0;
            const u64 p2 = (u64)t2 * m0;
            const u32 q3 = t3 * m0;
            asm("add.cc.u32 %0, %4, %5;\n\t"
                "addc.cc.u32 %1, %6, %7;\n\t"
                "addc.cc.u32 %2, %8, %9;\n\t"
                "addc.u32 %3, %10, %11;"
                : "=r"(x0), "=r"(x1), "=r"(x2), "=r"(x3)
                : "r"((u32)p0), "r"(d0), "r"((u32)p1), "r"((u32)(p0 >> 32)), "r"((u32)p2), "r"((u32)(p1 >> 32)),
                  "r"(q3), "r"((u32)(p2 >> 32)));
        }
    }
    template<int LIMBS>
    __device__ __forceinline__ void step(const JdTables<TV>& t) {
        using J = JDTraits<TV>;
        constexpr int W = J::w;
        const u32 tt0 = __funnelshift_r(x0, x1, W);
        const u64 p0 = (u64)tt0 * m0;
        const u32 y1 = (u32)p0 + d0;                 // x_{j+1} mod 2^32
        bool fire;
        if constexpr (J::ka == 0) {
            fire = dip0 > h;
            const u32 e1 = t.entry(y1);
            finish<LIMBS>(p0);
            h += g0;
            t.split(e1, m0, d0, g0);
            dip0 = JdTables<TV>::dipfield(e1);
        } else if constexpr (J::ka == 1) {
            // spec and non-spec orders differ only in when the jump is computed; masked, both commit the same state.
            const u32 e1 = t.entry(y1);
            const int h1 = h + g0;
            fire = JdTables<TV>::dipfield(e1) > h1;
            finish<LIMBS>(p0);
            h = h1;
            t.split(e1, m0, d0, g0);
        } else {
            static_assert(J::wd <= 32 - W, "two-ahead index needs WD <= 32 - W");
            const u32 y2 = m1 * (y1 >> W) + d1;      // x_{j+2} mod 2^(32-W)
            const u32 e2 = t.entry(y2);
            const int h1 = h + g0;
            fire = JdTables<TV>::dipfield(e2) > h1 + g1;
            finish<LIMBS>(p0);
            h = h1;
            m0 = m1; d0 = d1; g0 = g1;
            t.split(e2, m1, d1, g1);
        }
        const u32 lv = fire ? 0u : live;
        hash ^= x0 & lv;
        jumps -= lv;                                  // lv = ~0: +1
        live = lv;
    }
};

// 128-bit shared-memory store/load that the compiler can neither forward nor keep
// in registers (POP 3 keeps case bases out of the register file between pops).
__device__ __forceinline__ void tp_sts128(uint4* ptr, uint4 v) {
    asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};"
                 :: "r"((u32)__cvta_generic_to_shared(ptr)), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
__device__ __forceinline__ uint4 tp_lds128(const uint4* ptr) {
    uint4 v;
    asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"((u32)__cvta_generic_to_shared(ptr)) : "memory");
    return v;
}

// Per-case digest (as wq_case_digest / cv::jump::case_digest).
__device__ __forceinline__ void tp_digest(u64& ds, u64& dx, u32& dn, u32 b0, u32 b1, u32 b2, u32 b3, u32 jumps, u32 hash) {
    const u64 d = wq_case_digest(b0, b1, b2, b3, jumps, hash);
    ds += d; dx ^= d; dn += 1;
}

template<int TV, int SCR, int V, int MLIMB>
__global__ void __launch_bounds__(32 * TpCfg<V>::WARPS, TpCfg<V>::MINB)
cv_tp_kernel(const u32* __restrict__ bv, const u32* __restrict__ mod9, const u32* __restrict__ affine,
             u32 n, u64 magic, u32 tp_0, u32 tp_1, u32 tp_2, u32 tp_3,
             KParams p, KOut* out, u32* flags, const u32* __restrict__ jtab_g, ScreenParams sp) {
    using C = TpCfg<V>;
    using J = JDTraits<TV>;
    static_assert(C::valid, "unknown two-phase layout");
    static_assert(J::valid && J::lay == 1 && J::floor == 0 && J::pref, "two-phase kernels take the FD3 jd codes");
    static_assert((C::LCAP & (C::LCAP - 1)) == 0 && C::LCAP >= 64 && C::LCAP <= 8192, "list capacity");
    static_assert((C::EPN & (C::EPN - 1)) == 0 && C::EPN >= 1 && C::EPN <= 8, "endpoint slots");
    constexpr bool ROWS = (SCR & SCR_ROWS) != 0;
    constexpr bool RANGE = (SCR & SCR_RANGE) != 0;
    static_assert((SCR & ~(SCR_ROWS | SCR_RANGE)) == 0, "two-phase screens: rows and range");
    static_assert(MLIMB >= 0 && MLIMB <= 2, "M-limb (M / 32) 0..2");
    constexpr int WARPS = C::WARPS, LCAP = C::LCAP, EPN = C::EPN;
    constexpr u32 LMASK = LCAP - 1;
    extern __shared__ __align__(16) u32 s_jd[];
    __shared__ u16 s_list[WARPS][LCAP];
    __shared__ TpEp s_ep[WARPS][EPN];
    __shared__ TpWarp s_w[C::BS ? 1 : WARPS];
    __shared__ uint4 s_bat[C::BS ? WARPS : 1][C::BS ? 32 : 1];   // BS: per-endpoint screening setup of the batch
    __shared__ u32 s_next;                                        // EPB > 0: next endpoint of the block's range
    static_assert(C::EPB % 32 == 0, "block ranges are whole 32-endpoint batches");
    static_assert(!C::BS || C::POP == 1, "batched setup is implemented for the dynamic masked scheduler");
    constexpr int NB3 = C::POP == 3 ? WARPS : 1;
    __shared__ uint4 s_base[NB3][C::POP == 3 ? 32 : 1];   // POP 3: case bases n (read at settle only)
    __shared__ TpCtl s_ctl[NB3];                          // POP 3: warp control state between pop phases
    {
        constexpr u32 JB = jd_jtab_bytes<TV>();
        const uint4* src = reinterpret_cast<const uint4*>(jtab_g);
        uint4* dst = reinterpret_cast<uint4*>(s_jd);
        for (u32 i = threadIdx.x; i < JB / 16; i += blockDim.x) dst[i] = __ldg(src + i);
        if (threadIdx.x == 0) s_next = 0;
        __syncthreads();
    }
    JdTables<TV> tabs;
    tabs.jtab = s_jd;
    tabs.dip = nullptr;

    const u32 lane = threadIdx.x & 31u;
    const u32 wib = threadIdx.x >> 5;
    const u32 lt_mask = (1u << lane) - 1u;
    const int h0 = (int)p.tail_budget;
    constexpr int K = x_growth_bits(J::w);
    const u32 guard_hi = p.cutoff_hi >> K;
    u32 kmin = (p.pad >> 3) & 31u;
    if (kmin == 0) kmin = 1;
    const u32 fetch = 1u << ((p.pad >> 10) & 7u);
    const u32 w_lo = RANGE ? sp.w_first : 0u;
    const u32 w_hi = RANGE ? sp.w_end : p.words32;
    const u32 chunk_lo = w_lo >> 5;
    const u32 chunk_hi = (w_hi + 31u) >> 5;
    u16* const list = s_list[wib];
    TpEp* const eps = s_ep[wib];
    TpWarp& sw = s_w[C::BS ? 0 : wib];

    // Warp-uniform control state (every lane holds the same values).
    u32 lhead = 0, ltail = 0;      // list ring [lhead, ltail)
    u32 seq = 0;                   // endpoint slot sequence (slot = seq mod EPN)
    u32 ep = 0, ep_end = 0;        // fetched endpoints [ep, ep_end)
    u32 chunk = chunk_lo, qbase = 0, cur_slot = 0;
    bool have = false, slot_set = false, more = true;

    // Per-lane accumulators.
    u64 n_trips = 0, n_pops = 0, n_passes = 0;   // INSTR (warp-level, counted on every lane alike)
    u32 rhash = 0, cases = 0;
    u64 iters = 0;
    u64 dsum = 0, dxor = 0;
    u32 dcount = 0;

    // Next F endpoints for this warp into [ep, ep_end): from the launch-wide counter
    // KOut.next (EPB == 0) or from the block's range (EPB > 0).  False when none are left.
    auto fetch_eps = [&](u32 F) -> bool {
        u32 start = 0;
        if constexpr (C::EPB == 0) {
            if (lane == 0) start = atomicAdd(&out->next, F);
            start = __shfl_sync(WQJ_FULL, start, 0);
            if (start >= n) return false;
            ep = start;
            ep_end = start + F < n ? start + F : n;
        } else {
            const u32 b_lo = blockIdx.x * C::EPB;
            const u32 b_hi = b_lo + C::EPB < n ? b_lo + C::EPB : n;
            if (lane == 0) start = b_lo + atomicAdd(&s_next, F);
            start = __shfl_sync(WQJ_FULL, start, 0);
            if (start >= b_hi) return false;
            ep = start;
            ep_end = start + F < b_hi ? start + F : b_hi;
        }
        return true;
    };

    // One screening pass: queue survivors of the current chunk with ranks
    // [qbase, qbase + take).  Returns false when no endpoint could be started
    // (all fetched, or every slot still referenced by queued cases).
    auto refill_pass = [&]() -> bool {
        if (!have) {
            if (ltail != lhead) {
                // Live endpoints span seq - seq(head entry) <= EPN (by construction).
                const u32 hslot = (u32)list[lhead & LMASK] >> 13;
                const u32 span = ((seq - 1u - hslot) & (u32)(EPN - 1)) + 1u;
                if (span >= (u32)EPN) return false;
            }
            if (ep >= ep_end) {
                if (!fetch_eps(fetch)) { more = false; return false; }
            }
            const uint4 rb = reinterpret_cast<const uint4*>(affine)[(size_t)ep * 2];
            const uint4 ri = reinterpret_cast<const uint4*>(affine)[(size_t)ep * 2 + 1];
            const u32 oee = rb.w >> 30;
            const u32 family = (rb.w >> 29) & 1u;
            const u32 b3 = rb.w & 0x1FFFFFFFu;
            const u64 lo1 = ((u64)ri.x | ((u64)ri.y << 32)) + 1;   // low 64 bits of T^M(n0) + 1
            const u32* mw;
            u32 soff, guard0;
            scr_mask_setup<ROWS>(rb.x, rb.y, rb.z, b3, mod9, p.words32, sp, mw, soff, guard0);
            const u64 index = (magic * lo1) & p.b_mask;
            u32 oee_mask = 0xFFFFFFFFu;
            if (p.tail_oee) {
                if (oee == 2) {
                    oee_mask = ((u32)lo1 & 1) ? 0xAAAAAAAAu : 0x55555555u;
                } else if (oee == 1) {
                    const u32 forbidden = ((1u - (u32)lo1) * tp_0) & 3u;
                    oee_mask = ~(0x11111111u << forbidden);
                }
            }
            __syncwarp();
            if (lane == 0) {
                sw.n0 = make_uint4(rb.x, rb.y, rb.z, b3);
                sw.it = ri;
                sw.bvw = bv + family * p.fam_stride + (index >> 5);
                sw.mw = mw;
                sw.offset = (u32)(index & 31);
                sw.oee_mask = oee_mask;
                sw.sg = soff | (guard0 << 8);
            }
            __syncwarp();
            have = true;
            chunk = chunk_lo;
            qbase = 0;
            slot_set = false;
        }
        if constexpr (C::INSTR) n_passes += 1;
        const u32 wi = chunk * 32u + lane;
        u32 word = 0;
        if (wi >= w_lo && wi < w_hi)
            word = scr_word<ROWS>(sw.bvw, sw.offset, sw.mw, sw.sg & 31u, sw.sg >> 8, sw.oee_mask, wi);
        const u32 cnt = __popc(word);
        u32 incl = cnt;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const u32 up = __shfl_up_sync(WQJ_FULL, incl, o);
            if (lane >= (u32)o) incl += up;
        }
        const u32 total = __shfl_sync(WQJ_FULL, incl, 31);
        const u32 space = (u32)LCAP - (ltail - lhead);
        const u32 left = total - qbase;            // total >= qbase
        const u32 take = left < space ? left : space;
        if (take) {
            if (!slot_set) {
                cur_slot = seq & (u32)(EPN - 1);
                seq += 1;
                slot_set = true;
                if (lane < 2) reinterpret_cast<uint4*>(&eps[cur_slot])[lane] = lane == 0 ? sw.n0 : sw.it;
            }
            u32 r = incl - cnt;
            if (r + cnt > qbase && r < qbase + take) {
                const u32 tag = cur_slot << 13;
                while (word) {
                    if (r >= qbase && r < qbase + take)
                        list[(ltail + r - qbase) & LMASK] = (u16)(tag | (32u * wi + (__ffs(word) - 1)));
                    r++;
                    word &= word - 1;
                }
            }
            ltail += take;
        }
        if (take < left) {
            qbase += take;                          // same chunk again for the remaining ranks
        } else {
            qbase = 0;
            if (chunk + 1 >= chunk_hi) { have = false; ep += 1; }
            else chunk += 1;
        }
        __syncwarp();
        return true;
    };

    // BS screening pass (single 32-word chunk: lane = completion word).
    auto refill_bs = [&]() -> bool {
        uint4* const bat = s_bat[C::BS ? wib : 0];
        if (!have) {
            if (ltail != lhead) {
                const u32 hslot = (u32)list[lhead & LMASK] >> 13;
                const u32 span = ((seq - 1u - hslot) & (u32)(EPN - 1)) + 1u;
                if (span >= (u32)EPN) return false;
            }
            if (ep >= ep_end) {
                // New batch of 32 endpoints (starts are multiples of 32): lane j sets up endpoint ep + j.
                if (!fetch_eps(32u)) { more = false; return false; }
                const u32 e = ep + lane;
                if (e < ep_end) {
                    const uint4 rb = reinterpret_cast<const uint4*>(affine)[(size_t)e * 2];
                    const uint4 ri = reinterpret_cast<const uint4*>(affine)[(size_t)e * 2 + 1];
                    const u32 oee = rb.w >> 30;
                    const u32 family = (rb.w >> 29) & 1u;
                    const u32 b3 = rb.w & 0x1FFFFFFFu;
                    const u64 lo1 = ((u64)ri.x | ((u64)ri.y << 32)) + 1;
                    const u32* mw;
                    u32 soff, guard0;
                    scr_mask_setup<ROWS>(rb.x, rb.y, rb.z, b3, mod9, p.words32, sp, mw, soff, guard0);
                    const u64 index = (magic * lo1) & p.b_mask;
                    u32 oee_mask = 0xFFFFFFFFu;
                    if (p.tail_oee) {
                        if (oee == 2) {
                            oee_mask = ((u32)lo1 & 1) ? 0xAAAAAAAAu : 0x55555555u;
                        } else if (oee == 1) {
                            const u32 forbidden = ((1u - (u32)lo1) * tp_0) & 3u;
                            oee_mask = ~(0x11111111u << forbidden);
                        }
                    }
                    const u32 mwo = (u32)(mw - (ROWS ? sp.rows : mod9));
                    bat[lane] = make_uint4((u32)(index >> 5), mwo,
                                           (u32)(index & 31) | (soff << 5) | (family << 10) | (guard0 << 11), oee_mask);
                }
                __syncwarp();
            }
            have = true;
            qbase = 0;
            slot_set = false;
            if constexpr (TpChunk<V>::on) chunk = chunk_lo;
        }
        if constexpr (C::INSTR) n_passes += 1;
        const uint4 be = bat[ep & 31u];
        const u32* const bvw = bv + (size_t)((be.z >> 10) & 1u) * p.fam_stride + be.x;
        const u32* const mw = (ROWS ? sp.rows : mod9) + be.y;
        // c3-final: completion word of this lane (chunk layouts: word 32 chunk + lane of the
        // endpoint's current chunk, chunk_lo <= chunk < chunk_hi).
        u32 wl = lane;
        if constexpr (TpChunk<V>::on) wl = (chunk << 5) + lane;
        u32 word = 0;
        if (wl >= w_lo && wl < w_hi)
            word = scr_word<ROWS>(bvw, be.z & 31u, mw, (be.z >> 5) & 31u, (be.z >> 11) & 1u, be.w, wl);
        const u32 cnt = __popc(word);
        u32 incl = cnt;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const u32 up = __shfl_up_sync(WQJ_FULL, incl, o);
            if (lane >= (u32)o) incl += up;
        }
        const u32 total = __shfl_sync(WQJ_FULL, incl, 31);
        const u32 space = (u32)LCAP - (ltail - lhead);
        const u32 left = total - qbase;
        const u32 take = left < space ? left : space;
        if (take) {
            if (!slot_set) {
                cur_slot = seq & (u32)(EPN - 1);
                seq += 1;
                slot_set = true;
                if (lane < 2) {
                    uint4 r = reinterpret_cast<const uint4*>(affine)[(size_t)ep * 2 + lane];
                    if (lane == 0) r.w &= 0x1FFFFFFFu;   // n0 without the OEE / family bits
                    reinterpret_cast<uint4*>(&eps[cur_slot])[lane] = r;
                }
            }
            u32 r = incl - cnt;
            if (r + cnt > qbase && r < qbase + take) {
                const u32 tag = cur_slot << 13;
                while (word) {
                    if (r >= qbase && r < qbase + take)
                        list[(ltail + r - qbase) & LMASK] = (u16)(tag | (32u * wl + (__ffs(word) - 1)));
                    r++;
                    word &= word - 1;
                }
            }
            ltail += take;
        }
        if (take < left) {
            qbase += take;
        } else {
            qbase = 0;
            if constexpr (TpChunk<V>::on) {
                // c3-final: the next chunk of the same endpoint (same slot), else the next endpoint.
                if (chunk + 1u < chunk_hi) { chunk += 1u; }
                else { have = false; ep += 1; }
            } else {
                have = false;
                ep += 1;
            }
        }
        __syncwarp();
        return true;
    };
    auto refill = [&]() -> bool {
        if constexpr (C::BS) return refill_bs();
        else return refill_pass();
    };

    // Case start from a list entry: x = T^M(n0) + aval 3^f, n = n0 + aval 2^M.
    auto load_case = [&](u32 e, u32& x0, u32& x1, u32& x2, u32& x3, u32& b0, u32& b1, u32& b2, u32& b3) {
        const u32 slot = e >> 13, aval = e & 0x1FFFu;
        const uint4 rn = eps[slot].n0;
        const uint4 ri = eps[slot].it;
        const u64 q0 = (u64)aval * tp_0 + ri.x;
        const u64 q1 = (u64)aval * tp_1 + ri.y;
        const u64 q2 = (u64)aval * tp_2 + ri.z;
        const u32 q3 = aval * tp_3 + ri.w;
        x0 = (u32)q0; u32 carry = (u32)(q0 >> 32);
        u64 t = q1 + carry; x1 = (u32)t; carry = (u32)(t >> 32);
        t = q2 + carry; x2 = (u32)t; carry = (u32)(t >> 32);
        x3 = q3 + carry;
        // n = n0 + aval 2^M (add_aval with M / 32 = MLIMB fixed at compile time).
        const u64 sh = (u64)aval << p.m_shift;
        b0 = rn.x; b1 = rn.y; b2 = rn.z; b3 = rn.w;
        if constexpr (MLIMB == 0) { b0 |= (u32)sh; b1 |= (u32)(sh >> 32); }
        if constexpr (MLIMB == 1) { b1 |= (u32)sh; b2 |= (u32)(sh >> 32); }
        if constexpr (MLIMB == 2) { b2 |= (u32)sh; b3 |= (u32)(sh >> 32); }
    };

    // Exact descent loop for n < 2^32 or n >= 2^96 (no dip certificate), as cv_kernel_jds.
    u32 small_j = 0, small_h = 0;   // POP 3 (non-DIG): set just before small_case, folded right after
    auto small_case = [&](u32 x0, u32 x1, u32 x2, u32 x3, u32 b0, u32 b1, u32 b2, u32 b3) {
        u32 jumps = 0, chash = 0;
        bool finished = false;
        while (x3 < guard_hi) {
            u32 m, d; int g;
            tabs.decode(x0, m, d, g);
            constexpr int W = J::w;
            const u32 t0 = __funnelshift_r(x0, x1, W);
            const u32 t1 = __funnelshift_r(x1, x2, W);
            const u32 t2 = __funnelshift_r(x2, x3, W);
            const u32 t3 = x3 >> W;
            const u64 p0 = (u64)t0 * m;
            const u64 p1 = (u64)t1 * m;
            const u64 p2 = (u64)t2 * m;
            const u32 q3b = t3 * m;
            asm("add.cc.u32 %0, %4, %5;\n\t"
                "addc.cc.u32 %1, %6, %7;\n\t"
                "addc.cc.u32 %2, %8, %9;\n\t"
                "addc.u32 %3, %10, %11;"
                : "=r"(x0), "=r"(x1), "=r"(x2), "=r"(x3)
                : "r"((u32)p0), "r"(d), "r"((u32)p1), "r"((u32)(p0 >> 32)), "r"((u32)p2), "r"((u32)(p1 >> 32)),
                  "r"(q3b), "r"((u32)(p2 >> 32)));
            chash ^= x0;
            jumps++;
            if (x_below4(x0, x1, x2, x3, b0, b1, b2, b3)) { finished = true; break; }
        }
        if (!finished) x_flag(out, flags, p, b0, b1, b2, b3);
        if constexpr (C::POP == 3 && !C::DIG) {
            small_j += jumps;      // folded into the lane's running totals (TpLane counters)
            small_h ^= chash;
        } else {
            iters += jumps;
            rhash ^= chash;
        }
        if constexpr (C::DIG) tp_digest(dsum, dxor, dcount, b0, b1, b2, b3, jumps, chash);
    };

    if constexpr (C::POP == 0) {
        // Static rounds: rank = lane, cases run to completion (cv_kernel_jds's loops).
        while (true) {
            while (more && ltail - lhead < 32u) {
                if (!refill_pass()) break;
            }
            const u32 avail = ltail - lhead;
            if (avail == 0) break;
            if constexpr (C::INSTR) { n_trips += 1; n_pops += 1; }
            const u32 take = avail < 32u ? avail : 32u;
            if (lane < take) {
                const u32 e = list[(lhead + lane) & LMASK];
                u32 x0, x1, x2, x3, b0, b1, b2, b3;
                load_case(e, x0, x1, x2, x3, b0, b1, b2, b3);
                cases += 1;
                const bool small_base = ((b1 | b2) == 0) | (b3 != 0);
                if (!small_base) {
                    JdLane<TV> s;
                    s.init(tabs, x0, x1, x2, x3, h0);
                    bool finished = false;
                    if constexpr (J::three) {
                        while (s.fits3()) {
                            if (s.template step<3>(tabs)) { finished = true; break; }
                        }
                    }
                    if (!finished) {
                        while (!s.flag_due(guard_hi)) {
                            if (s.template step<4>(tabs)) { finished = true; break; }
                        }
                    }
                    if (!finished) x_flag(out, flags, p, b0, b1, b2, b3);
                    iters += s.jumps;
                    rhash ^= s.hash;
                    if constexpr (C::DIG) tp_digest(dsum, dxor, dcount, b0, b1, b2, b3, s.jumps, s.hash);
                } else {
                    small_case(x0, x1, x2, x3, b0, b1, b2, b3);
                }
            }
            __syncwarp();   // list reads before later overwrites
            lhead += take;
        }
    } else if constexpr (C::POP == 1) {
        // Dynamic trip-pop, masked steps.
        TpLane<TV> s;
        s.x0 = s.x1 = s.x2 = s.x3 = 0;
        s.h = 0; s.m0 = s.d0 = 0; s.g0 = 0; s.m1 = s.d1 = 0; s.g1 = 0; s.dip0 = 0;
        s.jumps = 0; s.hash = 0; s.live = 0;
        u32 b0 = 0, b1 = 0, b2 = 0, b3 = 0;
        u32 pend = 0, owns = 0;     // flag due at settle; lane holds an unsettled case (DIG)
        while (true) {
            const u32 idle = __ballot_sync(WQJ_FULL, s.live == 0);
            const u32 nidle = __popc(idle);
            if constexpr (C::INSTR) n_trips += 1;
            if (nidle >= kmin) {
                if constexpr (C::INSTR) n_pops += 1;
                if (s.live == 0 && (pend | owns)) {
                    // Settle the lane's finished case: its flag, and (DIG) its digest.
                    if (pend) x_flag(out, flags, p, b0, b1, b2, b3);
                    if constexpr (C::DIG) {
                        iters += s.jumps;
                        rhash ^= s.hash;
                        tp_digest(dsum, dxor, dcount, b0, b1, b2, b3, s.jumps, s.hash);
                        s.jumps = 0;
                        s.hash = 0;
                    }
                    pend = 0;
                    owns = 0;
                }
                while (more && ltail - lhead < nidle) {
                    if (!refill()) break;
                }
                const u32 avail = ltail - lhead;
                if (avail == 0 && nidle == 32u) break;   // nothing queued, nothing running, nothing left to fetch
                const u32 rank = __popc(idle & lt_mask);
                if (s.live == 0 && rank < avail) {
                    const u32 e = list[(lhead + rank) & LMASK];
                    u32 x0, x1, x2, x3;
                    load_case(e, x0, x1, x2, x3, b0, b1, b2, b3);
                    cases += 1;
                    const bool small_base = ((b1 | b2) == 0) | (b3 != 0);
                    if (!small_base) {
                        s.init(tabs, x0, x1, x2, x3, h0);
                        s.live = 0xFFFFFFFFu;
                        owns = C::DIG ? 1u : 0u;
                    } else {
                        small_case(x0, x1, x2, x3, b0, b1, b2, b3);
                    }
                }
                __syncwarp();   // list reads before later overwrites
                lhead += nidle < avail ? nidle : avail;
            }
#pragma unroll
            for (int u = 0; u < C::U; u++) {
                if constexpr (J::three && !C::FOUR) {
                    // 3-limb steps unless some live lane has x >= 2^(96-K): then this boundary
                    // runs the 4-limb step (exact for every x) with the overflow guard first.
                    const bool big = (s.live != 0) & !s.fits3();
                    if (__any_sync(WQJ_FULL, big)) {
                        if ((s.live != 0) & (s.x3 >= guard_hi)) { pend = 1; s.live = 0; }
                        s.template step<4>(tabs);
                    } else {
                        s.template step<3>(tabs);
                    }
                } else {
                    if ((s.live != 0) & (s.x3 >= guard_hi)) { pend = 1; s.live = 0; }
                    s.template step<4>(tabs);
                }
            }
        }
        if (pend) x_flag(out, flags, p, b0, b1, b2, b3);
        if constexpr (C::DIG) {
            if (owns) {
                iters += s.jumps;
                rhash ^= s.hash;
                tp_digest(dsum, dxor, dcount, b0, b1, b2, b3, s.jumps, s.hash);
            }
        } else {
            iters += s.jumps;
            rhash ^= s.hash;
        }
    } else if constexpr (C::POP == 3) {
        // Lean dynamic trip-pop: POP 1's masked 4-limb steps, with the warp control
        // state and the case bases n kept in shared memory between pop phases so that
        // only the TpLane state is live across the steps (fits 48 registers).
        TpLane<TV> s;
        s.x0 = s.x1 = s.x2 = s.x3 = 0;
        s.h = 0; s.m0 = s.d0 = 0; s.g0 = 0; s.m1 = s.d1 = 0; s.g1 = 0; s.dip0 = 0;
        s.jumps = 0; s.hash = 0; s.live = 0;
        u32 pend = 0, owns = 0;     // flag due at settle; (DIG) unsettled case
        volatile TpCtl* const vc = &s_ctl[wib];
        uint4* const vb = &s_base[wib][lane];
        if (lane == 0) {
            vc->lhead = 0; vc->ltail = 0; vc->seq = 0; vc->ep = 0; vc->ep_end = 0;
            vc->chunk = chunk_lo; vc->qbase = 0; vc->cur_slot = 0; vc->flags = 4u;   // more
        }
        __syncwarp();
        while (true) {
            const u32 idle = __ballot_sync(WQJ_FULL, s.live == 0);
            const u32 nidle = __popc(idle);
            if constexpr (C::INSTR) n_trips += 1;
            if (nidle >= kmin) {
                if constexpr (C::INSTR) n_pops += 1;
                lhead = vc->lhead; ltail = vc->ltail; seq = vc->seq; ep = vc->ep; ep_end = vc->ep_end;
                chunk = vc->chunk; qbase = vc->qbase; cur_slot = vc->cur_slot;
                {
                    const u32 fl = vc->flags;
                    have = (fl & 1u) != 0; slot_set = (fl & 2u) != 0; more = (fl & 4u) != 0;
                }
                if (s.live == 0 && (pend | owns)) {
                    const uint4 bb = tp_lds128(vb);
                    if (pend) x_flag(out, flags, p, bb.x, bb.y, bb.z, bb.w);
                    if constexpr (C::DIG) {
                        iters += s.jumps;
                        rhash ^= s.hash;
                        tp_digest(dsum, dxor, dcount, bb.x, bb.y, bb.z, bb.w, s.jumps, s.hash);
                        s.jumps = 0;
                        s.hash = 0;
                    }
                    pend = 0;
                    owns = 0;
                }
                while (more && ltail - lhead < nidle) {
                    if (!refill_pass()) break;
                }
                const u32 avail = ltail - lhead;
                if (avail == 0 && nidle == 32u) break;   // nothing queued, nothing running, nothing left to fetch
                const u32 rank = __popc(idle & lt_mask);
                if (s.live == 0 && rank < avail) {
                    const u32 e = list[(lhead + rank) & LMASK];
                    u32 x0, x1, x2, x3, b0, b1, b2, b3;
                    load_case(e, x0, x1, x2, x3, b0, b1, b2, b3);
                    cases += 1;
                    const bool small_base = ((b1 | b2) == 0) | (b3 != 0);
                    if (!small_base) {
                        tp_sts128(vb, make_uint4(b0, b1, b2, b3));
                        s.init(tabs, x0, x1, x2, x3, h0);
                        s.live = 0xFFFFFFFFu;
                        owns = C::DIG ? 1u : 0u;
                    } else {
                        small_j = 0; small_h = 0;
                        small_case(x0, x1, x2, x3, b0, b1, b2, b3);
                        if constexpr (!C::DIG) { s.jumps += small_j; s.hash ^= small_h; }
                    }
                }
                lhead += nidle < avail ? nidle : avail;
                __syncwarp();   // list reads before later overwrites; control state below
                if (lane == 0) {
                    vc->lhead = lhead; vc->ltail = ltail; vc->seq = seq; vc->ep = ep; vc->ep_end = ep_end;
                    vc->chunk = chunk; vc->qbase = qbase; vc->cur_slot = cur_slot;
                    vc->flags = (have ? 1u : 0u) | (slot_set ? 2u : 0u) | (more ? 4u : 0u);
                }
                __syncwarp();
            }
#pragma unroll
            for (int u = 0; u < C::U; u++) {
                if ((s.live != 0) & (s.x3 >= guard_hi)) { pend = 1; s.live = 0; }
                s.template step<4>(tabs);
            }
        }
        if (s.live == 0 && (pend | owns)) {
            const uint4 bb = tp_lds128(vb);
            if (pend) x_flag(out, flags, p, bb.x, bb.y, bb.z, bb.w);
            if constexpr (C::DIG) {
                iters += s.jumps;
                rhash ^= s.hash;
                tp_digest(dsum, dxor, dcount, bb.x, bb.y, bb.z, bb.w, s.jumps, s.hash);
            }
        }
        if constexpr (!C::DIG) {
            iters += s.jumps;
            rhash ^= s.hash;
        }
    } else {
        // Dynamic trip-pop, per-lane capped loops: every live lane runs at most U
        // boundaries of cv_kernel_jds's own loops (JdLane 3-limb steps while x fits,
        // then 4-limb steps with the overflow guard before each), exiting on its stop.
        JdLane<TV> s;
        s.x0 = s.x1 = s.x2 = s.x3 = 0;
        s.h = 0; s.m0 = s.d0 = 0; s.g0 = 0; s.m1 = s.d1 = 0; s.g1 = 0; s.dip0 = 0;
        s.jumps = 0; s.hash = 0;
        u32 b0 = 0, b1 = 0, b2 = 0, b3 = 0;
        u32 live = 0, pend = 0, owns = 0;   // case running; flag due at settle; unsettled case
        while (true) {
            const u32 idle = __ballot_sync(WQJ_FULL, live == 0);
            const u32 nidle = __popc(idle);
            if constexpr (C::INSTR) n_trips += 1;
            if (nidle >= kmin) {
                if constexpr (C::INSTR) n_pops += 1;
                if (live == 0 && owns) {
                    // Settle the lane's finished case (JdLane counts per case).
                    if (pend) x_flag(out, flags, p, b0, b1, b2, b3);
                    iters += s.jumps;
                    rhash ^= s.hash;
                    if constexpr (C::DIG) tp_digest(dsum, dxor, dcount, b0, b1, b2, b3, s.jumps, s.hash);
                    pend = 0;
                    owns = 0;
                }
                while (more && ltail - lhead < nidle) {
                    if (!refill_pass()) break;
                }
                const u32 avail = ltail - lhead;
                if (avail == 0 && nidle == 32u) break;   // nothing queued, nothing running, nothing left to fetch
                const u32 rank = __popc(idle & lt_mask);
                if (live == 0 && rank < avail) {
                    const u32 e = list[(lhead + rank) & LMASK];
                    u32 x0, x1, x2, x3;
                    load_case(e, x0, x1, x2, x3, b0, b1, b2, b3);
                    cases += 1;
                    const bool small_base = ((b1 | b2) == 0) | (b3 != 0);
                    if (!small_base) {
                        s.init(tabs, x0, x1, x2, x3, h0);
                        live = 1;
                        owns = 1;
                    } else {
                        small_case(x0, x1, x2, x3, b0, b1, b2, b3);
                    }
                }
                __syncwarp();   // list reads before later overwrites
                lhead += nidle < avail ? nidle : avail;
            }
            if (live) {
                bool done = false;
                int u = 0;
                if constexpr (J::three) {
                    // Hot loop: cv_kernel_jds's 3-limb loop, capped at U boundaries.
                    for (; u < C::U; u++) {
                        if (!s.fits3()) break;
                        if (s.template step<3>(tabs)) { done = true; break; }
                    }
                }
                if (!done) {
                    // 4-limb boundaries (x >= 2^(96-K), or every boundary for 4-limb codes):
                    // the overflow guard, then the step, as cv_kernel_jds's 4-limb loop.
                    for (; u < C::U; u++) {
                        if (s.flag_due(guard_hi)) { pend = 1; done = true; break; }
                        if (s.template step<4>(tabs)) { done = true; break; }
                    }
                }
                if (done) live = 0;
            }
        }
        if (owns) {
            if (pend) x_flag(out, flags, p, b0, b1, b2, b3);
            iters += s.jumps;
            rhash ^= s.hash;
            if constexpr (C::DIG) tp_digest(dsum, dxor, dcount, b0, b1, b2, b3, s.jumps, s.hash);
        }
    }

    u64 c = cases;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        rhash ^= __shfl_xor_sync(WQJ_FULL, rhash, o);
        c += __shfl_xor_sync(WQJ_FULL, c, o);
        iters += __shfl_xor_sync(WQJ_FULL, iters, o);
        if constexpr (C::DIG) {
            dsum += __shfl_xor_sync(WQJ_FULL, dsum, o);
            dxor ^= __shfl_xor_sync(WQJ_FULL, dxor, o);
            dcount += __shfl_xor_sync(WQJ_FULL, dcount, o);
        }
    }
    if (lane == 0) {
        atomicXor(&out->hash, rhash);
        atomicAdd(&out->cases, (unsigned long long)c);
        atomicAdd(&out->iters, (unsigned long long)iters);
        if constexpr (C::DIG) {
            atomicAdd(&out->dbg[0], (unsigned long long)dsum);
            atomicXor(&out->dbg[1], (unsigned long long)dxor);
            atomicAdd(&out->dbg[2], (unsigned long long)dcount);
        }
        if constexpr (C::INSTR) {
            atomicAdd(&out->dbg[3], (unsigned long long)n_trips);
            atomicAdd(&out->dbg[4], (unsigned long long)n_pops);
            atomicAdd(&out->dbg[5], (unsigned long long)n_passes);
        }
    }
}

// ---------------------------------------------------------------------------
// Host side.

static bool tp_code_valid(u32 tv) {
#define TPCV(C) if (tv == C) return true;
    TP_CODES(TPCV)
#undef TPCV
    return false;
}
static bool tp_layout_valid(u32 v) {
#define TPLV(V) if (v == V) return true;
    TP_LAYOUTS(TPLV)
#undef TPLV
    return false;
}

// Two-phase pad fields (bit 21 with layout, K, fetch, blocks/SM cap).
static constexpr u32 TP_FIELD_MASK = (1u << 21) | (31u << 16) | (7u << 13) | (7u << 10) | (31u << 3);

static bool tp_pad_valid(u32 pad) {
    if (!((pad >> 21) & 1u) || ((pad >> 22) & 1u)) return false;
    if ((pad & ~(TP_FIELD_MASK | 0xFF000007u)) != 0) return false;   // block-size bits 8-9 and bit 23 must be 0
    if ((pad & 7u) != 5u) return false;
    return tp_code_valid(pad >> 24) && tp_layout_valid((pad >> 16) & 31u);
}

// c3-final: the dip/FD3 image and the occupancy query come from the launching
// context's DevTables (c3-hyg, cuda/devtables.cuh), not from process-global
// caches; device code is unchanged.
template<int TV, int SCR, int V, int MLIMB>
static bool tp_launch_one(DevTables& tabs, const u32* bv, const u32* mod9, const u32* input, u32 n, u64 magic, const u32* tp,
                          const KParams& p, KOut* out, u32* flags, cudaStream_t stream, const ScreenParams& sp) {
    using J = JDTraits<TV>;
    constexpr u32 bytes = jd_jtab_bytes<TV>();
    const u32* jt = jd_device_emb(tabs, J::w, J::dq, J::desc);   // this context's FD3 image
    if (!jt) {
        fprintf(stderr, "two-phase variant %d: dip table (w %d, q %d, desc %d) not uploaded to this context\n",
                TV, J::w, J::dq, (int)J::desc);
        return false;
    }
    auto kernel = cv_tp_kernel<TV, SCR, V, MLIMB>;
    constexpr u32 threads = 32 * TpCfg<V>::WARPS;
    int per_sm = devtables_resident(tabs, (const void*)kernel, threads, bytes);
    if (per_sm <= 0) return false;
    if (getenv("CV_TP_VERBOSE")) {
        cudaFuncAttributes fa;
        cudaFuncGetAttributes(&fa, kernel);
        fprintf(stderr, "tp kernel layout %d code %d scr %d mlimb %d: %d regs, %zu B static + %u B dynamic smem, %zu B local, %d blocks/SM (%d warps)\n",
                V, TV, SCR, MLIMB, fa.numRegs, fa.sharedSizeBytes, bytes, fa.localSizeBytes, per_sm, per_sm * TpCfg<V>::WARPS);
    }
    const u32 cap_sm = (p.pad >> 13) & 7u;
    if (cap_sm && (int)cap_sm < per_sm) per_sm = (int)cap_sm;
    // BS layouts fetch 32 endpoints per warp, one 32-word chunk (A <= 10), u32 word indices (B <= 37).
    // c3-final: chunk-offset layouts (TpChunk) screen one chunk of a larger completion range
    // (A >= 11 half ranges): the screened word range must lie inside a single 32-word chunk.
    if constexpr (TpChunk<V>::on) {
        constexpr bool RANGE = (SCR & SCR_RANGE) != 0;
        const u32 lo = RANGE ? sp.w_first : 0u, hi = RANGE ? sp.w_end : p.words32;
        if (hi <= lo || hi > p.words32 || p.words32 > 256 || (p.b_mask >> 37) != 0) {
            fprintf(stderr, "two-phase layout %d (chunks) needs a non-empty screened range, A <= 13 and B <= 37\n", V);
            return false;
        }
    } else if (TpCfg<V>::BS && (p.words32 > 32 || (p.b_mask >> 37) != 0)) {
        fprintf(stderr, "two-phase layout %d (batched setup) needs A <= 10 and B <= 37\n", V);
        return false;
    }
    const u32 fetch = TpCfg<V>::BS ? 32u : 1u << ((p.pad >> 10) & 7u);
    const u64 grid = (u64)(tabs.sms > 0 ? tabs.sms : 1) * (u64)per_sm;
    const u64 need = ((u64)n + (u64)fetch * TpCfg<V>::WARPS - 1) / ((u64)fetch * TpCfg<V>::WARPS);
    // Persistent layouts: at most one wave of resident blocks; block-range layouts: ceil(n / EPB) blocks.
    const u32 blocks = TpCfg<V>::EPB ? (u32)(((u64)n + TpCfg<V>::EPB - 1) / TpCfg<V>::EPB) : (u32)(need < grid ? need : grid);
    kernel<<<blocks, threads, bytes, stream>>>(bv, mod9, input, n, magic, tp[0], tp[1], tp[2], tp[3], p, out, flags, jt, sp);
    return cudaGetLastError() == cudaSuccess;
}

template<int TV, int V>
static bool tp_launch_v(DevTables& tabs, u32 scr, const u32* bv, const u32* mod9, const u32* input, u32 n, u64 magic, const u32* tp,
                        const KParams& p, KOut* out, u32* flags, cudaStream_t stream, const ScreenParams& sp) {
#define TPL_S(S_, M_) if (scr == S_) return tp_launch_one<TV, S_, V, M_>(tabs, bv, mod9, input, n, magic, tp, p, out, flags, stream, sp);
    if (p.m_limb == 1) { TPL_S(0, 1) TPL_S(1, 1) TPL_S(2, 1) TPL_S(3, 1) return false; }
    if constexpr (TpCfg<V>::ALLM) {
        if (p.m_limb == 0) { TPL_S(0, 0) TPL_S(1, 0) TPL_S(2, 0) TPL_S(3, 0) return false; }
        if (p.m_limb == 2) { TPL_S(0, 2) TPL_S(1, 2) TPL_S(2, 2) TPL_S(3, 2) return false; }
    }
#undef TPL_S
    fprintf(stderr, "two-phase layout %d is not instantiated for M-limb %u\n", V, p.m_limb);
    return false;
}

// Returns false (nothing launched) for an unsupported code, layout, screen or M.
static bool tp_launch(DevTables& tabs, u32 scr, const u32* bv, const u32* mod9, const u32* input, u32 n, u64 magic, const u32* tp,
                      const KParams& p, KOut* out, u32* flags, cudaStream_t stream, const ScreenParams& sp) {
    if (!tp_pad_valid(p.pad) || scr > 3) return false;
    const u32 tv = p.pad >> 24, v = (p.pad >> 16) & 31u;
#define TPL_V(TV_, V_) if (tv == TV_ && v == V_) return tp_launch_v<TV_, V_>(tabs, scr, bv, mod9, input, n, magic, tp, p, out, flags, stream, sp);
#define TPL_T(TV_) TPL_V(TV_, 1) TPL_V(TV_, 2) TPL_V(TV_, 3) TPL_V(TV_, 4) TPL_V(TV_, 5) TPL_V(TV_, 6) TPL_V(TV_, 7) \
    TPL_V(TV_, 8) TPL_V(TV_, 9) TPL_V(TV_, 10) TPL_V(TV_, 11) TPL_V(TV_, 12) TPL_V(TV_, 13) TPL_V(TV_, 14) TPL_V(TV_, 15) \
    TPL_V(TV_, 16) TPL_V(TV_, 17) TPL_V(TV_, 18) TPL_V(TV_, 19) TPL_V(TV_, 20) TPL_V(TV_, 21) TPL_V(TV_, 22) TPL_V(TV_, 23) \
    TPL_V(TV_, 24) TPL_V(TV_, 25) TPL_V(TV_, 26) TPL_V(TV_, 27) TPL_V(TV_, 28) TPL_V(TV_, 29) TPL_V(TV_, 30) TPL_V(TV_, 31)
    TP_CODES(TPL_T)
#undef TPL_T
#undef TPL_V
    return false;
}
