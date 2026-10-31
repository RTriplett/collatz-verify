// c-final: the per-thread jd tail (cuda/jdip.cuh, c-tail) behind the
// completion screens of cuda/screen.cuh (c-screen).
//
// cv_kernel_jds<TV, SCR> is cv_kernel_xs<TV', SCR> (exact-iterate MODE 2
// input, bitvector-word carry and prefetch, optional (12, 9) row table and
// completion-word range, diagnostic flavours) with the case tail replaced by
// c-tail's JdLane: exact W-step jumps on x = T^j(n) with the window-dip
// certificate tested at every boundary against H = 485 F' - 306 M'
// (H0 = 485 F - 306 M of the class in KParams.tail_budget).  Small-base cases
// (n < 2^32 or n >= 2^96) run the exact-descent jump loop, as in cv_kernel_jd.
//
// Per-case semantics (CPU reference cv::jdip::jump_dip_until_k through
// cv::jump::TailConfig::run_h) are exactly cv_kernel_jd's; the case set is
// exactly cv_kernel_xs's for the same SCR (CPU reference: jump::for_each_case
// and tail::endpoint_work with the same TailParams::screen).  Nothing new is
// certified here: the screen removes completions by c-screen's proven row
// exclusions (or by the declared h range), the tail stops cases by c-tail's
// proven window-dip criterion.
//
// SCR bits as cuda/screen.cuh: SCR_ROWS 1, SCR_RANGE 2, diagnostics SCR_SCAN 8
// and SCR_SETUP 16 (not verifications; full range only).  SCR_LOAD, SCR_M9PF
// and SCR_LAZY are not instantiated for jd codes (measured negative or
// redundant in c-screen; the load-only floor is the jump kernels' SCR_LOAD).
#pragma once

// Launch-bound option (SCR_LB5, carried like the scan options): the same body
// compiled with __launch_bounds__(256, 5), i.e. at most 48 registers, so that
// 5 blocks of 256 threads (40 warps) fit an SM even with the row table.
enum : int { SCR_LB5 = 128 };

template<int TV, int SCR>
__device__ __forceinline__ void jds_body(const u32* __restrict__ bv, const u32* __restrict__ mod9,
                                         const u32* __restrict__ pairs,
                                         u32 n, u64 magic, u32 tp_0, u32 tp_1, u32 tp_2, u32 tp_3,
                                         const KParams& p, KOut* out, u32* flags, const u32* __restrict__ jtab_g,
                                         const ScreenParams& sp) {
    using J = JDTraits<TV>;
    static_assert(J::valid && J::lay == 1 && J::floor == 0 && J::pref, "jds kernels take the FD3 jd codes");
    constexpr bool ROWS = (SCR & SCR_ROWS) != 0;
    constexpr bool RANGE = (SCR & SCR_RANGE) != 0;
    constexpr bool D_SCAN = (SCR & SCR_SCAN) != 0;
    constexpr bool D_SETUP = (SCR & SCR_SETUP) != 0;
    static_assert((SCR & ~(SCR_ROWS | SCR_RANGE | SCR_SCAN | SCR_SETUP)) == 0 && !(D_SCAN && D_SETUP),
                  "jds flavour");
    extern __shared__ __align__(16) u32 s_jd[];
    {
        constexpr u32 JB = jd_jtab_bytes<TV>();
        const uint4* src = reinterpret_cast<const uint4*>(jtab_g);
        uint4* dst = reinterpret_cast<uint4*>(s_jd);
        for (u32 i = threadIdx.x; i < JB / 16; i += blockDim.x) dst[i] = __ldg(src + i);
        __syncthreads();
    }
    JdTables<TV> tabs;
    tabs.jtab = s_jd;
    tabs.dip = nullptr;

    u32 rolling_hash = 0;
    u32 cases = 0;
    u64 iters = 0;
    const int h0 = (int)p.tail_budget;
    constexpr int K = x_growth_bits(J::w);
    const u32 guard_hi = p.cutoff_hi >> K;

    for (u32 idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += gridDim.x * blockDim.x) {
        const uint4 in = reinterpret_cast<const uint4*>(pairs)[(size_t)idx * 2];
        const u32 base0_0 = in.x, base0_1 = in.y, base0_2 = in.z;
        const u32 oee = in.w >> 30;
        const u32 family = (in.w >> 29) & 1u;
        const u32 base0_3 = in.w & 0x1FFFFFFFu;
        const uint4 iterate = reinterpret_cast<const uint4*>(pairs)[(size_t)idx * 2 + 1];
        const u64 lo = (u64)iterate.x | ((u64)iterate.y << 32);
        const u64 hi = (u64)iterate.z | ((u64)iterate.w << 32);
        // Jump tails start from T^M(n0) itself.
        const u32 s0 = (u32)lo, s1 = (u32)(lo >> 32), s2 = (u32)hi, s3 = (u32)(hi >> 32);
        const u64 lo1 = lo + 1;
        const u32 number0_0 = (u32)lo1;

        const u32* m9;
        u32 soff = 0, guard0 = 0;
        scr_mask_setup<ROWS>(base0_0, base0_1, base0_2, base0_3, mod9, p.words32, sp, m9, soff, guard0);

        const u64 index = (magic * lo1) & p.b_mask;
        const u32 offset = (u32)(index & 31);
        const u32* bvw = bv + family * p.fam_stride + (index >> 5);

        u32 oee_mask = 0xFFFFFFFFu;
        if (p.tail_oee) {
            if (oee == 2) {
                oee_mask = (number0_0 & 1) ? 0xAAAAAAAAu : 0x55555555u;
            } else if (oee == 1) {
                const u32 forbidden = ((1u - number0_0) * tp_0) & 3u;
                oee_mask = ~(0x11111111u << forbidden);
            }
        }

        u32 w = RANGE ? sp.w_first : 0u;
        const u32 w_end = RANGE ? sp.w_end : p.words32;
        u32 bv_lo = bvw[w], bv_hi = bvw[w + 1];
        u32 bv_next = 0;
        if (w + 2 <= p.words32) bv_next = bvw[w + 2];
        u32 r_lo = 0, r_hi = 0, r_next = 0, temp;
        if constexpr (ROWS) {
            // Rows are padded: m9[w + 2] exists for every w < words32.
            r_lo = m9[w]; r_hi = m9[w + 1]; r_next = m9[w + 2];
            temp = (__funnelshift_r(r_lo, r_hi, soff) | (w == 0 ? guard0 : 0u))
                 & __funnelshift_r(bv_lo, bv_hi, offset) & oee_mask;
        } else {
            temp = m9[w] & __funnelshift_r(bv_lo, bv_hi, offset) & oee_mask;
        }
        if constexpr (D_SCAN) {
            // Diagnostic: every word, no case work.
            while (true) {
                cases += __popc(temp);
                if (++w == w_end) break;
                bv_lo = bv_hi; bv_hi = bv_next;
                if (w + 2 <= p.words32) bv_next = bvw[w + 2];
                u32 mw;
                if constexpr (ROWS) { r_lo = r_hi; r_hi = r_next; r_next = m9[w + 2]; mw = __funnelshift_r(r_lo, r_hi, soff); }
                else mw = m9[w];
                temp = mw & __funnelshift_r(bv_lo, bv_hi, offset) & oee_mask;
            }
            rolling_hash ^= temp;
            continue;
        }
        while (true) {
            while (temp == 0) {
                if (++w == w_end) goto next_endpoint;
                // Window words [0, words32] exist; bvw[w + 2] only while w + 2 <= words32.
                bv_lo = bv_hi;
                bv_hi = bv_next;
                if (w + 2 <= p.words32) bv_next = bvw[w + 2];
                u32 mw;
                if constexpr (ROWS) { r_lo = r_hi; r_hi = r_next; r_next = m9[w + 2]; mw = __funnelshift_r(r_lo, r_hi, soff); }
                else mw = m9[w];
                temp = mw & __funnelshift_r(bv_lo, bv_hi, offset) & oee_mask;
            }
            const u32 aval = 32 * w + (__ffs(temp) - 1);
            temp &= temp - 1;

            u64 q0 = (u64)aval * tp_0 + s0;
            u64 q1 = (u64)aval * tp_1 + s1;
            u64 q2 = (u64)aval * tp_2 + s2;
            u32 q3 = aval * tp_3 + s3;
            u32 num_0 = (u32)q0; u32 carry = q0 >> 32;
            u64 t = q1 + carry; u32 num_1 = (u32)t; carry = t >> 32;
            t = q2 + carry; u32 num_2 = (u32)t; carry = t >> 32;
            u32 num_3 = q3 + carry;

            u32 base_0, base_1, base_2, base_3;
            add_aval(base0_0, base0_1, base0_2, base0_3, aval, p, base_0, base_1, base_2, base_3);
            cases += 1;
            if constexpr (D_SETUP) {
                rolling_hash ^= num_0 ^ num_1 ^ num_2 ^ num_3 ^ base_1 ^ base_3;
                continue;
            }

            const u64 bh = ((u64)base_2 << 32) | base_1;
            const bool small_base = (bh == 0) | (base_3 != 0);
            bool finished = false;
            u32 jumps = 0;
            if (!small_base) {
                JdLane<TV> s;
                s.init(tabs, num_0, num_1, num_2, num_3, h0);
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
                jumps = s.jumps;
                rolling_hash ^= s.hash;
            } else {
                // Exact descent loop (n < 2^32 or n >= 2^96): no dip certificate.
                while (num_3 < guard_hi) {
                    u32 m, d; int g;
                    tabs.decode(num_0, m, d, g);
                    constexpr int W = J::w;
                    const u32 t0 = __funnelshift_r(num_0, num_1, W);
                    const u32 t1 = __funnelshift_r(num_1, num_2, W);
                    const u32 t2 = __funnelshift_r(num_2, num_3, W);
                    const u32 t3 = num_3 >> W;
                    const u64 p0 = (u64)t0 * m;
                    const u64 p1 = (u64)t1 * m;
                    const u64 p2 = (u64)t2 * m;
                    const u32 q3b = t3 * m;
                    asm("add.cc.u32 %0, %4, %5;\n\t"
                        "addc.cc.u32 %1, %6, %7;\n\t"
                        "addc.cc.u32 %2, %8, %9;\n\t"
                        "addc.u32 %3, %10, %11;"
                        : "=r"(num_0), "=r"(num_1), "=r"(num_2), "=r"(num_3)
                        : "r"((u32)p0), "r"(d), "r"((u32)p1), "r"((u32)(p0 >> 32)), "r"((u32)p2), "r"((u32)(p1 >> 32)),
                          "r"(q3b), "r"((u32)(p2 >> 32)));
                    rolling_hash ^= num_0;
                    jumps++;
                    if (x_below4(num_0, num_1, num_2, num_3, base_0, base_1, base_2, base_3)) { finished = true; break; }
                }
            }
            if (!finished) x_flag(out, flags, p, base_0, base_1, base_2, base_3);
            iters += jumps;
        }
    next_endpoint:;
    }
    u64 c = cases;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        rolling_hash ^= __shfl_xor_sync(0xFFFFFFFFu, rolling_hash, o);
        c += __shfl_xor_sync(0xFFFFFFFFu, c, o);
        iters += __shfl_xor_sync(0xFFFFFFFFu, iters, o);
    }
    if ((threadIdx.x & 31) == 0) {
        atomicXor(&out->hash, rolling_hash);
        atomicAdd(&out->cases, (unsigned long long)c);
        atomicAdd(&out->iters, (unsigned long long)iters);
    }
}

template<int TV, int SCR>
__global__ void cv_kernel_jds(const u32* __restrict__ bv, const u32* __restrict__ mod9,
                              const u32* __restrict__ pairs,
                              u32 n, u64 magic, u32 tp_0, u32 tp_1, u32 tp_2, u32 tp_3,
                              KParams p, KOut* out, u32* flags, const u32* __restrict__ jtab_g,
                              ScreenParams sp) {
    jds_body<TV, SCR>(bv, mod9, pairs, n, magic, tp_0, tp_1, tp_2, tp_3, p, out, flags, jtab_g, sp);
}

template<int TV, int SCR>
__global__ void __launch_bounds__(256, 5)
cv_kernel_jds_lb(const u32* __restrict__ bv, const u32* __restrict__ mod9,
                 const u32* __restrict__ pairs,
                 u32 n, u64 magic, u32 tp_0, u32 tp_1, u32 tp_2, u32 tp_3,
                 KParams p, KOut* out, u32* flags, const u32* __restrict__ jtab_g,
                 ScreenParams sp) {
    jds_body<TV, SCR>(bv, mod9, pairs, n, magic, tp_0, tp_1, tp_2, tp_3, p, out, flags, jtab_g, sp);
}

template<int TV, int SCR>
static bool jds_launch_one(DevTables& tabs, const u32* bv, const u32* mod9, const u32* input, u32 n, u64 magic, const u32* tp,
                           KParams p, KOut* out, u32* flags, u32 blocks, u32 threads, cudaStream_t stream,
                           const ScreenParams& sp) {
    using J = JDTraits<TV>;
    constexpr u32 bytes = jd_jtab_bytes<TV>();
    static_assert(bytes <= 48 * 1024, "jds kernels use at most 48 KB of shared tables");
    const u32* jt = jd_device_emb(tabs, J::w, J::dq, J::desc);   // c3-hyg: this context's FD3 image
    if (!jt) {
        fprintf(stderr, "jds variant %d: dip table (w %d, q %d, desc %d) not uploaded to this context\n",
                TV, J::w, J::dq, (int)J::desc);
        return false;
    }
    if constexpr ((SCR & SCR_LB5) != 0) {
        if (threads > 256) return false;
        cv_kernel_jds_lb<TV, (SCR & ~SCR_LB5)><<<blocks, threads, bytes, stream>>>(bv, mod9, input, n, magic,
            tp[0], tp[1], tp[2], tp[3], p, out, flags, jt, sp);
    } else {
        cv_kernel_jds<TV, SCR><<<blocks, threads, bytes, stream>>>(bv, mod9, input, n, magic, tp[0], tp[1], tp[2], tp[3],
                                                                   p, out, flags, jt, sp);
    }
    return true;
}

// Production FD3 codes with screens (c-tail's per-class choices and their
// neighbours): 90/91 (W12 KA0/KA1), 93/95 (W11 KA0/KA2), 96 (W10 KA0),
// 103 (W12 4-limb KA0), 111 (W12 4-limb, speculative KA1).
#define JDS_CODES(X) X(90) X(91) X(93) X(95) X(96) X(103) X(111)

static bool jds_code_valid(u32 tv) {
#define JDSV(C) if (tv == C) return true;
    JDS_CODES(JDSV)
#undef JDSV
    return false;
}

// Flavours: 0-3 (rows x range), the diagnostics 8/9 (screen only) and
// 16/17 (screen + case setup), full range, and rows (+ range) with the
// 48-register launch bound (129, 131).
static bool jds_scr_valid(u32 scr) {
    return scr <= 3 || scr == 8 || scr == 9 || scr == 16 || scr == 17 ||
           scr == (SCR_LB5 | 1) || scr == (SCR_LB5 | 3);
}

// Returns false (nothing launched) for an unsupported code or flavour.
static bool jds_launch(DevTables& tabs, u32 tv, u32 scr, const u32* bv, const u32* mod9, const u32* input, u32 n, u64 magic,
                       const u32* tp, KParams p, KOut* out, u32* flags, u32 blocks, u32 threads,
                       cudaStream_t stream, const ScreenParams& sp) {
    if (!jds_code_valid(tv) || !jds_scr_valid(scr)) return false;
#define JDS_ONE(TV_, SCR_) case SCR_: return jds_launch_one<TV_, SCR_>(tabs, bv, mod9, input, n, magic, tp, p, out, flags, blocks, threads, stream, sp);
#define JDS_TV(TV_) case TV_: switch (scr) { \
        JDS_ONE(TV_, 0) JDS_ONE(TV_, 1) JDS_ONE(TV_, 2) JDS_ONE(TV_, 3) \
        JDS_ONE(TV_, 8) JDS_ONE(TV_, 9) JDS_ONE(TV_, 16) JDS_ONE(TV_, 17) \
        JDS_ONE(TV_, 129) JDS_ONE(TV_, 131) \
        default: return false; }
    switch (tv) {
        JDS_CODES(JDS_TV)
        default: return false;
    }
#undef JDS_TV
#undef JDS_ONE
}
