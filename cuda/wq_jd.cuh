// c-final: c-tail's jd FD3 tails in c-sched's warp-queue scheduler, and the
// trip-pop scheduler with cooperative completion screens.
//
// 1. JdWqTail3<TV, FOUR, DIG>: c-tail's JdWqTail2 (cuda/jd_wq2.cuh in the
//    c-tail tree, written for an intermediate c-sched interface) adapted to
//    c-sched's final interface: step() takes the WqSink pointer, DEFERS is
//    declared.  Per-case semantics are unchanged: cv_kernel_jd's, CPU
//    reference cv::jdip::jump_dip_until_k (KA 1 or 2).  H0 = 485 F - 306 M
//    reaches WqParams.cap through KParams.tail_budget.
// 2. cv_wqs_kernel<Tail, MLIMB, V, SCR>: c-sched's trip-pop scheduler
//    (cv_wqj2_kernel, layouts with POP2 and NL = 1, no POPX) with the
//    endpoint's mask formed by c-screen's cooperative routine
//    (scr_mask_setup / scr_word, cuda/screen.cuh): SCR_ROWS uses the (12, 9)
//    row table instead of the mod-9 row, SCR_RANGE scans only completion words
//    [w_first, w_end).  Lane i of a screening pass forms completion word
//    chunk * 32 + i exactly as the per-thread kernels do; words outside the
//    range are 0.  Everything else (queue, ring, pops, accounting) is
//    cv_wqj2_kernel's line for line.
//
// Host: wq_launch_jd() runs jd codes (unscreened) and wq_launch_scr() any
// supported code with a screen; both are reached from wq_launch() /
// cv_gpu_launch_impl.
#pragma once

template<int TV, bool FOUR, bool DIG = false>
struct JdWqTail3 {
    using J = JDTraits<TV>;
    static_assert(J::valid && J::lay == 1 && J::floor == 0 && (J::ka == 1 || J::ka == 2),
                  "warp-queue jd tails use FD3 codes with KA = 1 or 2");
    static constexpr int W = J::w;
    static constexpr int K = x_growth_bits(W);
    static constexpr bool THREE = J::three && !FOUR;
    static constexpr bool DEFERS = true;   // step() defers flags through pend
    static constexpr u32 TABLE_BYTES = jd_jtab_bytes<TV>();
    static constexpr u32 LIM3 = 1u << (32 - K);

    struct Lane {
        u32 x0, x1, x2, x3;
        u32 b0, b1, b2, b3;
        u32 pend;
        int h;               // H << 13
        u32 m0, d0;
        int g0;              // entry of the next jump
        u32 m1, d1;
        int g1;              // KA 2: entry of the jump after
        u32 chash, cjumps;   // DIG only
        u64 dsum, dxor;      // DIG only
        u32 dcount;          // DIG only
    };

    __device__ __forceinline__ static JdTables<TV> tabs(const u32* s_tab) {
        JdTables<TV> t;
        t.jtab = s_tab;
        t.dip = nullptr;
        return t;
    }

    __device__ __forceinline__ static void idle(Lane& L) {
        L.x0 = L.x1 = L.x2 = L.x3 = 0;
        L.b0 = L.b1 = L.b2 = L.b3 = 0;
        L.pend = 0;
        L.h = 0;
        L.m0 = L.d0 = 0; L.g0 = 0;
        L.m1 = L.d1 = 0; L.g1 = 0;
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

    __device__ __forceinline__ static void flag(const Lane& L, const WqSink* sink) {
        wq_flag_out(L.b0, L.b1, L.b2, L.b3, sink);
    }

    __device__ __forceinline__ static void settle(Lane& L, const WqSink* sink) {
        if (L.pend) {
            flag(L, sink);
            L.pend = 0;
        }
    }

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
        const u64 sh = (u64)aval << w.m_shift;
        rec1 = ep.n0;
        if constexpr (MLIMB == 0) { rec1.x |= (u32)sh; rec1.y |= (u32)(sh >> 32); }
        if constexpr (MLIMB == 1) { rec1.y |= (u32)sh; rec1.z |= (u32)(sh >> 32); }
        if constexpr (MLIMB == 2) { rec1.z |= (u32)sh; rec1.w |= (u32)(sh >> 32); }
    }

    __device__ __forceinline__ static void jump4(const JdTables<TV>& t, u32& x0, u32& x1, u32& x2, u32& x3) {
        u32 m, d;
        int g;
        t.decode(x0, m, d, g);
        const u32 t0 = __funnelshift_r(x0, x1, W);
        const u32 t1 = __funnelshift_r(x1, x2, W);
        const u32 t2 = __funnelshift_r(x2, x3, W);
        const u32 t3 = x3 >> W;
        const u64 p0 = (u64)t0 * m;
        const u64 p1 = (u64)t1 * m;
        const u64 p2 = (u64)t2 * m;
        const u32 q3 = t3 * m;
        asm("add.cc.u32 %0, %4, %5;\n\t"
            "addc.cc.u32 %1, %6, %7;\n\t"
            "addc.cc.u32 %2, %8, %9;\n\t"
            "addc.u32 %3, %10, %11;"
            : "=r"(x0), "=r"(x1), "=r"(x2), "=r"(x3)
            : "r"((u32)p0), "r"(d), "r"((u32)p1), "r"((u32)(p0 >> 32)), "r"((u32)p2), "r"((u32)(p1 >> 32)),
              "r"(q3), "r"((u32)(p2 >> 32)));
    }

    // Pop.  Returns the lane's new amask (zero if the case already finished).
    __device__ __forceinline__ static u32 start(Lane& L, const uint4& r0, const uint4& r1, u32& hash, u32& iters,
                                                const u32* s_tab, const WqParams& w, const WqSink* sink) {
        settle(L, sink);   // a finished flagged case still owns L
        const JdTables<TV> t = tabs(s_tab);
        L.x0 = r0.x; L.x1 = r0.y; L.x2 = r0.z; L.x3 = r0.w;
        L.b0 = r1.x; L.b1 = r1.y; L.b2 = r1.z; L.b3 = r1.w;
        if constexpr (DIG) { L.chash = 0; L.cjumps = 0; }
        const bool small = ((L.b1 | L.b2) == 0) | (L.b3 != 0);
        if ((L.x3 >= w.guard_hi) | small) {
            // Rare: guard before the first jump, or a whole exact-descent case.
            while (true) {
                if (L.x3 >= w.guard_hi) {
                    flag(L, sink);
                    break;
                }
                jump4(t, L.x0, L.x1, L.x2, L.x3);
                hash ^= L.x0;
                iters += 1;
                if constexpr (DIG) { L.chash ^= L.x0; L.cjumps += 1; }
                if (x_below4(L.x0, L.x1, L.x2, L.x3, L.b0, L.b1, L.b2, L.b3)) break;
            }
            if constexpr (DIG) digest(L);
            L.x0 = L.x1 = L.x2 = L.x3 = 0;
            return 0u;
        }
        L.h = (int)w.cap * 8192;   // H0 << 13
        t.split(t.entry(L.x0), L.m0, L.d0, L.g0);
        if constexpr (J::ka == 2) {
            const u32 y1 = L.m0 * __funnelshift_r(L.x0, L.x1, W) + L.d0;
            t.split(t.entry(y1), L.m1, L.d1, L.g1);
        }
        return WQJ_FULL;
    }

    // One boundary on every lane: the test, then (if not certified) one exact jump.
    __device__ __forceinline__ static void step(Lane& L, u32& amask, u32& hash, u32& iters, const u32* s_tab,
                                                const WqParams& w, const WqSink* = nullptr) {
        const JdTables<TV> t = tabs(s_tab);
        bool four = true;
        if constexpr (THREE) {
            const bool need4 = (amask != 0) & ((L.x3 != 0) | (L.x2 >= LIM3));
            four = __any_sync(WQJ_FULL, need4);
        }
        const u32 t0 = __funnelshift_r(L.x0, L.x1, W);
        const u64 p0 = (u64)t0 * L.m0;
        const u32 y1 = (u32)p0 + L.d0;               // x_{j+1} mod 2^32
        const int h1 = L.h + L.g0;
        u32 e;
        bool fire;
        if constexpr (J::ka == 1) {
            e = t.entry(y1);
            fire = JdTables<TV>::dipfield(e) > h1;
        } else {
            const u32 y2 = L.m1 * (y1 >> W) + L.d1;  // x_{j+2} mod 2^(32-W)
            e = t.entry(y2);
            fire = JdTables<TV>::dipfield(e) > h1 + L.g1;
        }
        const u32 m = L.m0, d = L.d0;
        if (!four) {
            const u32 t1 = __funnelshift_r(L.x1, L.x2, W);
            const u32 t2 = L.x2 >> W;
            const u64 p1 = (u64)t1 * m;
            const u32 q2 = t2 * m;
            asm("add.cc.u32 %0, %3, %4;\n\t"
                "addc.cc.u32 %1, %5, %6;\n\t"
                "addc.u32 %2, %7, %8;"
                : "=r"(L.x0), "=r"(L.x1), "=r"(L.x2)
                : "r"((u32)p0), "r"(d), "r"((u32)p1), "r"((u32)(p0 >> 32)), "r"(q2), "r"((u32)(p1 >> 32)));
        } else {
            const u32 t1 = __funnelshift_r(L.x1, L.x2, W);
            const u32 t2 = __funnelshift_r(L.x2, L.x3, W);
            const u32 t3 = L.x3 >> W;
            const u64 p1 = (u64)t1 * m;
            const u64 p2 = (u64)t2 * m;
            const u32 q3 = t3 * m;
            asm("add.cc.u32 %0, %4, %5;\n\t"
                "addc.cc.u32 %1, %6, %7;\n\t"
                "addc.cc.u32 %2, %8, %9;\n\t"
                "addc.u32 %3, %10, %11;"
                : "=r"(L.x0), "=r"(L.x1), "=r"(L.x2), "=r"(L.x3)
                : "r"((u32)p0), "r"(d), "r"((u32)p1), "r"((u32)(p0 >> 32)), "r"((u32)p2), "r"((u32)(p1 >> 32)),
                  "r"(q3), "r"((u32)(p2 >> 32)));
        }
        const u32 live = fire ? 0u : amask;
        hash ^= L.x0 & live;
        iters -= live;
        if constexpr (DIG) { L.chash ^= L.x0 & live; L.cjumps -= live; }
        L.h = h1;
        if constexpr (J::ka == 1) {
            t.split(e, L.m0, L.d0, L.g0);
        } else {
            L.m0 = L.m1; L.d0 = L.d1; L.g0 = L.g1;
            t.split(e, L.m1, L.d1, L.g1);
        }
        const bool over = L.x3 >= w.guard_hi;        // guard of the next boundary
        L.pend |= (u32)(over & (live != 0));
        const bool stop = fire | over;
        if constexpr (DIG) { if (stop && amask) digest(L); }
        amask = stop ? 0u : amask;
    }
};

// ---------------------------------------------------------------------------
// Trip-pop scheduler with cooperative screens (see the header comment).
// Same size as WqWarp (80 bytes): with 20 warps/block, 2 blocks of the
// trip-pop layouts (33.6 KB static + 16.5 KB table each) just fit an SM.
struct WqWarpS {
    WqEp e;
    const u32* bvw;     // window words of the current endpoint
    const u32* mw;      // its mask words: mod-9 row, or row-table row + phase / 32
    u32 offset, oee_mask;
    u32 ep, ep_end;     // current endpoint, end of the fetched chunk
    u32 chunk, qbase;   // 32-word chunk and first rank of the current pass
    u32 have_ep;
    u32 sg;             // row funnel offset (bits 0-4) | h = 0 guard << 8
};
static_assert(sizeof(WqWarpS) == sizeof(WqWarp), "keep the per-warp screening state at 80 bytes");

template<class Tail, int MLIMB, int V, int SCR>
__global__ void __launch_bounds__(32 * WqCfg<V>::WARPS, WqCfg<V>::MINB)
cv_wqs_kernel(const u32* __restrict__ bv, const u32* __restrict__ mod9, const u32* __restrict__ affine,
              const WqParams w, KOut* out, u32* flags, const u32* __restrict__ xtab, const ScreenParams sp) {
    using C = WqCfg<V>;
    static_assert(C::POP2 && C::NL == 1 && !C::POPX && !C::PT && !C::INSTR, "trip-pop layouts only");
    constexpr bool ROWS = (SCR & SCR_ROWS) != 0;
    constexpr bool RANGE = (SCR & SCR_RANGE) != 0;
    static_assert((SCR & ~(SCR_ROWS | SCR_RANGE)) == 0, "screen flavours: rows and range");
    constexpr int WARPS = C::WARPS, RING = C::RING, QCAP = C::QCAP;
    static_assert((RING & (RING - 1)) == 0 && RING <= 32, "ring size");
    extern __shared__ __align__(16) u32 s_tab[];
    __shared__ uint4 s_rec[WARPS][RING][2];
    __shared__ WqWarpS s_w[WARPS];
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
    // Screened completion words [w_lo, w_hi) and their 32-word chunks.
    const u32 w_lo = RANGE ? sp.w_first : 0u;
    const u32 w_hi = RANGE ? sp.w_end : w.words32;
    const u32 chunk_lo = w_lo >> 5;
    const u32 chunk_hi = (w_hi + 31u) >> 5;
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
    u32 qhead = 0, qcount = 0;   // queued completion indices of the current pass
    u32 rhead = 0, rcount = 0;   // ring: records [rhead, rhead + rcount)
    bool more = true;

    while (true) {
        u32 idle = __ballot_sync(WQJ_FULL, amask == 0);
        u32 nidle = __popc(idle);
        if (nidle >= w.kmin) {
            const u32 want_records = nidle < (u32)RING ? nidle : (u32)RING;
            while (rcount < want_records && more) {
                WqWarpS& sw = s_w[wib];
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
                        const u32* mw;
                        u32 soff, guard0;
                        scr_mask_setup<ROWS>(rb.x, rb.y, rb.z, b3, mod9, w.words32, sp, mw, soff, guard0);
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
                            sw.mw = mw;
                            sw.offset = (u32)(index & 31);
                            sw.oee_mask = oee_mask;
                            sw.sg = soff | (guard0 << 8);
                            sw.ep = ep;
                            sw.ep_end = ep_end;
                            sw.chunk = chunk_lo;
                            sw.qbase = 0;
                            sw.have_ep = 1;
                        }
                        __syncwarp();
                    }
                    const u32 chunk = sw.chunk, qbase = sw.qbase;
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
                            if (chunk + 1 >= chunk_hi) {
                                sw.chunk = chunk_lo;
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
                const u32 space = (u32)RING - rcount;
                const u32 m = qcount < space ? qcount : space;
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
                const u32 rank = __popc(idle & lt_mask);
                if (amask == 0 && rank < rcount) {
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
    }
}

// ---------------------------------------------------------------------------
// Host side.

// jd FD3 codes the warp queue runs (KA 1 / 2; W12 and W11).
static bool wq_jd_code(u32 tv) { return tv == 91 || tv == 92 || tv == 94 || tv == 95; }

// Layouts for unscreened jd tails: trip-pop 43 (U = 4) and 17 (U = 2) and
// their digest audit copies 52 and 26, and 12 (c-tail's choice in the
// snapshot: pop rounds, 48-register bound); limbs as WqAllLimbs.

template<class Tail, int MLIMB, int V, int SCR>
static int wq_launch_scr_kernel(WqState& st, const u32* bv, const u32* mod9, const u32* affine, u32 n, u64 magic,
                                const u32* tp, const KParams& p, KOut* out, u32* flags, const u32* table,
                                cudaStream_t stream, const ScreenParams& sp) {
    auto kernel = cv_wqs_kernel<Tail, MLIMB, V, SCR>;
    constexpr u32 threads = 32 * WqCfg<V>::WARPS;
    constexpr u32 dyn = Tail::TABLE_BYTES;
    auto it = st.per_sm.find((const void*)kernel);
    int per_sm;
    if (it == st.per_sm.end()) {
        if (dyn > 0 && cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn) != cudaSuccess)
            return -1;
        per_sm = 0;
        if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, threads, dyn) != cudaSuccess || per_sm <= 0)
            return -1;
        st.per_sm[(const void*)kernel] = per_sm;
        if (getenv("CV_WQ_VERBOSE")) {
            cudaFuncAttributes fa;
            cudaFuncGetAttributes(&fa, kernel);
            fprintf(stderr, "wqs kernel layout %d tail %d limb %d scr %d: %d regs, %zu B static + %u B dynamic smem, %d blocks/SM (%d warps)\n",
                    V, (int)(p.pad >> 24), MLIMB, SCR, fa.numRegs, fa.sharedSizeBytes, dyn, per_sm, per_sm * WqCfg<V>::WARPS);
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
    if (w.kmin == 0) w.kmin = 1u;
    w.fetch = 1u << ((p.pad >> 10) & 7u);
    w.cap = p.tail_budget;
    const u64 grid = (u64)st.sms * (u64)per_sm;
    const u64 need = ((u64)n + (u64)w.fetch * WqCfg<V>::WARPS - 1) / ((u64)w.fetch * WqCfg<V>::WARPS);
    const u32 blocks = (u32)(need < grid ? need : grid);
    kernel<<<blocks, threads, dyn, stream>>>(bv, mod9, affine, w, out, flags, table, sp);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

template<class Tail, int V, int SCR>
static int wq_launch_scr_m(WqState& st, const u32* bv, const u32* mod9, const u32* affine, u32 n, u64 magic,
                           const u32* tp, const KParams& p, KOut* out, u32* flags, const u32* table,
                           cudaStream_t stream, const ScreenParams& sp) {
    // Production M = 62 (limb 1) and the validation shapes N32/N40 (limb 0) and M = 64 (limb 2).
    if (p.m_limb == 1) return wq_launch_scr_kernel<Tail, 1, V, SCR>(st, bv, mod9, affine, n, magic, tp, p, out, flags, table, stream, sp);
    if (p.m_limb == 0) return wq_launch_scr_kernel<Tail, 0, V, SCR>(st, bv, mod9, affine, n, magic, tp, p, out, flags, table, stream, sp);
    if (p.m_limb == 2) return wq_launch_scr_kernel<Tail, 2, V, SCR>(st, bv, mod9, affine, n, magic, tp, p, out, flags, table, stream, sp);
    return -1;
}

static const u32* wq_tail_table(WqState& st, int tv) {
    if (wq_jd_code((u32)tv)) {
        if (!st.dt) return nullptr;   // c3-hyg: jd tables are the context's (cv_gpu_init sets st.dt)
        switch (tv) {
#define WJT(C) case C: return jd_device_emb(*st.dt, JDTraits<C>::w, JDTraits<C>::dq, JDTraits<C>::desc);
            WJT(91) WJT(92) WJT(94) WJT(95)
#undef WJT
            default: return nullptr;
        }
    }
    return wq_table(st, tv);
}

// Screened warp-queue launch: jd codes WQS_JD (KA 1/2) on layouts 43
// (production trip-pop, U = 4) and 52 (its per-case digest audit copy),
// SCR in {1 rows, 2 range, 3 rows + range} (an unscreened launch takes
// c-sched's own schedulers through wq_launch).
#define WQS_JD(X) X(91) X(92) X(95)
static bool wq_scr_code(u32 tv) {
#define WQSC(C) if (tv == C) return true;
    WQS_JD(WQSC)
#undef WQSC
    return false;
}

template<int TV>
static int wq_launch_scr_tv(WqState& st, u32 scr, const u32* bv, const u32* mod9, const u32* affine, u32 n, u64 magic,
                            const u32* tp, const KParams& p, KOut* out, u32* flags, const u32* table,
                            cudaStream_t stream, const ScreenParams& sp) {
    const u32 v = wq_layout_of(p.pad);
#define WSV(V, SCR_) if (v == V && scr == SCR_) { \
        using T_ = JdWqTail3<TV, WqCfg<V>::FOUR, WqCfg<V>::DIG>; \
        return wq_launch_scr_m<T_, V, SCR_>(st, bv, mod9, affine, n, magic, tp, p, out, flags, table, stream, sp); }
#define WSL(V) WSV(V, 1) WSV(V, 2) WSV(V, 3)
    WSL(43) WSL(52)
#undef WSL
#undef WSV
    return -1;
}

static int wq_launch_scr(WqState& st, u32 scr, const u32* bv, const u32* mod9, const u32* affine, u32 n, u64 magic,
                         const u32* tp, const KParams& p, KOut* out, u32* flags, cudaStream_t stream,
                         const ScreenParams& sp) {
    const int tv = (int)(p.pad >> 24);
    if ((p.pad >> 21) & 1u) return -1;   // no capped test tail here
    if (!wq_scr_code((u32)tv) || scr == 0 || scr > 3) return -1;
    const u32* table = wq_tail_table(st, tv);
    if (!table) return -1;
    switch (tv) {
#define WSC(C) case C: return wq_launch_scr_tv<C>(st, scr, bv, mod9, affine, n, magic, tp, p, out, flags, table, stream, sp);
        WQS_JD(WSC)
#undef WSC
        default: return -1;
    }
}

// Unscreened jd codes through c-sched's own schedulers (layouts WqJdLayout).
template<int TV>
static int wq_launch_jd_tv(WqState& st, const u32* bv, const u32* mod9, const u32* affine, u32 n, u64 magic,
                           const u32* tp, const KParams& p, KOut* out, u32* flags, const u32* table,
                           cudaStream_t stream) {
    switch (wq_layout_of(p.pad)) {
#define WLJ(V) case V: { using T_ = JdWqTail3<TV, WqCfg<V>::FOUR, WqCfg<V>::DIG>; \
        return wq_launch_m<T_, V>(st, bv, mod9, affine, n, magic, tp, p, out, flags, table, stream); }
        WLJ(12) WLJ(17) WLJ(26) WLJ(43) WLJ(52)
#undef WLJ
        default: return -1;
    }
}

static int wq_launch_jd(WqState& st, const u32* bv, const u32* mod9, const u32* affine, u32 n, u64 magic,
                        const u32* tp, const KParams& p, KOut* out, u32* flags, cudaStream_t stream) {
    const int tv = (int)(p.pad >> 24);
    if ((p.pad >> 21) & 1u) return -1;   // the capped test tail is refused for jd codes
    const u32* table = wq_tail_table(st, tv);
    if (!table) return -1;
    switch (tv) {
#define WJD(C) case C: return wq_launch_jd_tv<C>(st, bv, mod9, affine, n, magic, tp, p, out, flags, table, stream);
        WJD(91) WJD(92) WJD(94) WJD(95)
#undef WJD
        default: return -1;
    }
}
