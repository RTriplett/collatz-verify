// c-screen: endpoint-side screening on the jump tails (cuda/xtail.cuh).
//
// cv_kernel_xs<TV, SCR> is the exact-iterate (MODE 2) jump kernel of
// xtail.cuh for the production codes 49..53 (coarse descent test, 3-limb then
// 4-limb W-step jumps, bitvector-word carry and prefetch), with these
// compile-time screening options (SCR bits):
//
//   SCR_ROWS  1  the (12, 9) row table replaces the mod-9 mask (src/rowsieve.rs):
//                mask word w = funnelshift(row[w], row[w + 1], phase & 31) with
//                row = rows + rowmap[n0 mod 2^K] * row_words + (phase >> 5),
//                phase = n0 (2^M)^-1 mod 3^9; bit h = 0 is always kept.
//                Row words are carried and prefetched like the bitvector words.
//   SCR_RANGE 2  scan only completion words [w_first, w_end) (32 completions
//                each); w_first = 2^(A-6) restricts to the top half n >= 2^(N-1).
//   SCR_LOAD  4  diagnostic: load the endpoint records only (no screening).
//   SCR_SCAN  8  diagnostic: screen every word and count surviving bits (no case work).
//   SCR_SETUP 16 diagnostic: screen and build each case's start (no jumps).
//   SCR_M9PF  32 mod-9 words carried and prefetched like the bitvector words
//                (work-preserving; without SCR_ROWS only).
//   SCR_LAZY  64 with SCR_ROWS: fetch the two row words of word w only when the
//                bitvector & OEE word is nonzero (no carry; same case set).
//
// With SCR = 0 the kernel is the jump agent's cv_kernel_x<2, TV> code path,
// line for line (checked by SASS diff).  The tail itself, its hash, iteration
// and flag semantics are unchanged (CPU reference: cv::jump::jump_until with
// coarse descent); only the case set changes, by sound exclusions (rows) or by
// the declared range (h range), and the CPU reference applies the same
// TailParams::screen (src/tail.rs, src/jump.rs).
//
// The diagnostic flavours are not verifications: they report the case counts
// they screened but no iterations, and hashes that mean nothing.
#pragma once

enum : int { SCR_ROWS = 1, SCR_RANGE = 2, SCR_LOAD = 4, SCR_SCAN = 8, SCR_SETUP = 16, SCR_M9PF = 32, SCR_LAZY = 64 };

// Device-side screening parameters (filled by the host launch from the
// context's uploaded row table and the caller's ScreenSpec).
struct ScreenParams {
    const u32* rows;     // compact rows, row_words u32 words each; row 0 keeps everything
    const u16* rowmap;   // compact row of n0 mod 2^K
    u32 row_words;
    u32 kmask;           // 2^K - 1
    u32 d;               // 3^J
    u32 minv;            // floor(2^32 / 3^J)
    u32 c0, c1, c2, c3;  // 2^(32 i) (2^M)^-1 mod 3^J
    u32 w_first, w_end;  // completion words [w_first, w_end)
};

// Host-side request (Rust ScreenSpec, same layout).
struct ScreenSpec {
    u32 rows;     // 1: apply the uploaded row table
    u32 w_first;  // first completion word (32 completions per word)
    u32 w_end;    // one past the last word; 0 means words32
    u32 diag;     // 0, SCR_LOAD, SCR_SCAN or SCR_SETUP
};

// x mod d for any u32 x (d < 2^16, minv = floor(2^32 / d)): q >= floor(x/d) - 1,
// so one correction suffices.
__device__ __forceinline__ u32 scr_mod(u32 x, u32 d, u32 minv) {
    const u32 q = __umulhi(x, minv);
    const u32 r = x - q * d;
    return r >= d ? r - d : r;
}

// phase = n0 (2^M)^-1 mod 3^J from the four u32 limbs of n0 (each product
// is below d^2 < 2^29, the sum below 2^31).
__device__ __forceinline__ u32 scr_phase(u32 b0, u32 b1, u32 b2, u32 b3, const ScreenParams& sp) {
    const u32 s = scr_mod(b0, sp.d, sp.minv) * sp.c0 + scr_mod(b1, sp.d, sp.minv) * sp.c1
                + scr_mod(b2, sp.d, sp.minv) * sp.c2 + scr_mod(b3, sp.d, sp.minv) * sp.c3;
    return scr_mod(s, sp.d, sp.minv);
}

// ---------------------------------------------------------------------------
// Cooperative per-lane-word screening for warp-queue schedulers (c-sched).
//
// Per endpoint (warp-uniform inputs), scr_mask_setup returns the mask-word
// pointer `mw`, its funnel offset `soff` and the h = 0 guard; then lane i
// calls scr_word(..., w = chunk * 32 + i) and gets completion word w exactly as
// the per-thread scan of cv_kernel_xs forms it:
//     word(w) = mask(w) & funnelshift(bvw[w], bvw[w + 1], offset) & oee_mask
// with mask(w) = mod9 row word (ROWS = false) or the row-table word
// funnelshift(mw[w], mw[w + 1], soff) | (w == 0) (ROWS = true).  Lanes with
// w outside [w_first, w_end) must use 0.  b0..b3 are n0's limbs (OEE/family
// bits removed), as in every kernel here.
template<bool ROWS>
__device__ __forceinline__ void scr_mask_setup(u32 b0, u32 b1, u32 b2, u32 b3, const u32* mod9, u32 words32,
                                               const ScreenParams& sp, const u32*& mw, u32& soff, u32& guard0) {
    if constexpr (ROWS) {
        const u32 phase = scr_phase(b0, b1, b2, b3, sp);
        const u32 row = __ldg(sp.rowmap + (b0 & sp.kmask));
        mw = sp.rows + (size_t)row * sp.row_words + (phase >> 5);
        soff = phase & 31u;
        guard0 = 1u;
    } else {
        u32 redmod9 = (b0 & 0x3FFFF) + (b0 >> 18) + 4 * (b1 & 0x3FFFF) + 4 * (b1 >> 18)
                    + 7 * (b2 & 0x3FFFF) + 7 * (b2 >> 18) + (b3 & 0x3FFFF) + (b3 >> 18);
        redmod9 %= 9;
        mw = mod9 + words32 * redmod9;
        soff = 0;
        guard0 = 0;
    }
}

template<bool ROWS>
__device__ __forceinline__ u32 scr_word(const u32* bvw, u32 offset, const u32* mw, u32 soff, u32 guard0,
                                        u32 oee_mask, u32 w) {
    u32 m;
    if constexpr (ROWS) m = __funnelshift_r(mw[w], mw[w + 1], soff) | (w == 0 ? guard0 : 0u);
    else m = mw[w];
    return m & __funnelshift_r(bvw[w], bvw[w + 1], offset) & oee_mask;
}

// Reference use of the cooperative routine (also its test): one warp per
// endpoint, lane i forms completion words i, i + 32, ... exactly as a warp-queue
// scheduler would, and writes them out (words outside [w_first, w_end) are 0).
// MODE 2 records.  Compared bit for bit with the CPU enumeration by
// `screenbench words`.
template<bool ROWS>
__global__ void cv_screen_words(const u32* __restrict__ bv, const u32* __restrict__ mod9,
                                const u32* __restrict__ pairs, u32 n, u64 magic, u32 tp_0,
                                KParams p, ScreenParams sp, u32 w_first, u32 w_end, u32* __restrict__ out) {
    const u64 e = ((u64)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (e >= n) return;  // warp-uniform
    const u32 lane = threadIdx.x & 31u;
    const uint4 rb = reinterpret_cast<const uint4*>(pairs)[(size_t)e * 2];
    const uint4 ri = reinterpret_cast<const uint4*>(pairs)[(size_t)e * 2 + 1];
    const u32 oee = rb.w >> 30, family = (rb.w >> 29) & 1u, b3 = rb.w & 0x1FFFFFFFu;
    const u64 lo1 = ((u64)ri.x | ((u64)ri.y << 32)) + 1;  // low 64 bits of T^M(n0) + 1
    const u64 index = (magic * lo1) & p.b_mask;
    const u32 offset = (u32)(index & 31);
    const u32* bvw = bv + family * p.fam_stride + (index >> 5);
    u32 oee_mask = 0xFFFFFFFFu;
    if (p.tail_oee) {
        if (oee == 2) oee_mask = ((u32)lo1 & 1) ? 0xAAAAAAAAu : 0x55555555u;
        else if (oee == 1) oee_mask = ~(0x11111111u << (((1u - (u32)lo1) * tp_0) & 3u));
    }
    const u32* mw;
    u32 soff, guard0;
    scr_mask_setup<ROWS>(rb.x, rb.y, rb.z, b3, mod9, p.words32, sp, mw, soff, guard0);
    for (u32 w = lane; w < p.words32; w += 32) {
        const u32 word = (w >= w_first && w < w_end) ? scr_word<ROWS>(bvw, offset, mw, soff, guard0, oee_mask, w) : 0u;
        out[(size_t)e * p.words32 + w] = word;
    }
}

// ---------------------------------------------------------------------------
// Per-thread jump kernel with screening (one endpoint per thread, MODE 2).

template<int TV, int SCR>
__global__ void cv_kernel_xs(const u32* __restrict__ bv, const u32* __restrict__ mod9,
                             const u32* __restrict__ pairs,
                             u32 n, u64 magic, u32 tp_0, u32 tp_1, u32 tp_2, u32 tp_3,
                             KParams p, KOut* out, u32* flags, const u32* __restrict__ xtab,
                             ScreenParams sp) {
    using X = XTraits<TV>;
    static_assert(X::valid && X::jump && X::opt && X::pref && !X::skip && X::prefix == 0 && X::rep == 1,
                  "screen kernels cover the production jump codes 49..53");
    constexpr bool ROWS = (SCR & SCR_ROWS) != 0;
    constexpr bool RANGE = (SCR & SCR_RANGE) != 0;
    constexpr bool D_LOAD = (SCR & SCR_LOAD) != 0;
    constexpr bool D_SCAN = (SCR & SCR_SCAN) != 0;
    constexpr bool D_SETUP = (SCR & SCR_SETUP) != 0;
    constexpr bool M9PF = (SCR & SCR_M9PF) != 0;
    constexpr bool LAZY = (SCR & SCR_LAZY) != 0;
    static_assert((int)D_LOAD + (int)D_SCAN + (int)D_SETUP <= 1, "one diagnostic flavour at a time");
    static_assert(!(M9PF && ROWS) && !(LAZY && !ROWS) && !(LAZY && (D_SCAN || D_LOAD)), "flavour combination");
    extern __shared__ __align__(16) u32 s_tab[];
    constexpr u32 TABLE_BYTES = x_table_bytes<TV>();
    {
        const uint4* src = reinterpret_cast<const uint4*>(xtab);
        uint4* dst = reinterpret_cast<uint4*>(s_tab);
        for (u32 i = threadIdx.x; i < TABLE_BYTES / 16; i += blockDim.x) dst[i] = __ldg(src + i);
        __syncthreads();
    }

    u32 rolling_hash = 0;
    u32 cases = 0;
    u64 iters = 0;

    for (u32 idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += gridDim.x * blockDim.x) {
        const uint4 in = reinterpret_cast<const uint4*>(pairs)[(size_t)idx * 2];
        const u32 base0_0 = in.x, base0_1 = in.y, base0_2 = in.z;
        const u32 oee = in.w >> 30;
        const u32 family = (in.w >> 29) & 1u;
        const u32 base0_3 = in.w & 0x1FFFFFFFu;
        const uint4 iterate = reinterpret_cast<const uint4*>(pairs)[(size_t)idx * 2 + 1];
        if constexpr (D_LOAD) {
            rolling_hash ^= in.x ^ in.y ^ in.z ^ in.w ^ iterate.x ^ iterate.y ^ iterate.z ^ iterate.w;
            continue;
        }
        u64 lo = (u64)iterate.x | ((u64)iterate.y << 32);
        u64 hi = (u64)iterate.z | ((u64)iterate.w << 32);
        // Jump tails start from T^M(n0) itself.
        const u64 slo = lo, shi = hi;
        const u64 lo1 = lo + 1;
        const u32 s0 = (u32)slo, s1 = (u32)(slo >> 32), s2 = (u32)shi, s3 = (u32)(shi >> 32);
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
        constexpr int K = x_growth_bits(X::w);
        const u32 guard_hi = p.cutoff_hi >> K;

        u32 w = RANGE ? sp.w_first : 0u;
        const u32 w_end = RANGE ? sp.w_end : p.words32;
        u32 bv_lo = bvw[w], bv_hi = bvw[w + 1];
        u32 bv_next = 0;
        if (w + 2 <= p.words32) bv_next = bvw[w + 2];
        u32 r_lo = 0, r_hi = 0, r_next = 0, temp;
        if constexpr (ROWS && LAZY) {
            temp = __funnelshift_r(bv_lo, bv_hi, offset) & oee_mask;
            if (temp) temp &= __funnelshift_r(m9[w], m9[w + 1], soff) | (w == 0 ? guard0 : 0u);
        } else if constexpr (ROWS) {
            // Rows are padded: m9[w + 2] exists for every w < words32.
            r_lo = m9[w]; r_hi = m9[w + 1]; r_next = m9[w + 2];
            temp = (__funnelshift_r(r_lo, r_hi, soff) | (w == 0 ? guard0 : 0u))
                 & __funnelshift_r(bv_lo, bv_hi, offset) & oee_mask;
        } else if constexpr (M9PF) {
            // mod-9 row words: m9[w + 1] exists only while w + 1 < words32.
            r_lo = m9[w];
            if (w + 1 < p.words32) r_hi = m9[w + 1];
            temp = r_lo & __funnelshift_r(bv_lo, bv_hi, offset) & oee_mask;
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
                if constexpr (ROWS && LAZY) {
                    temp = __funnelshift_r(bv_lo, bv_hi, offset) & oee_mask;
                    if (temp) temp &= __funnelshift_r(m9[w], m9[w + 1], soff);
                } else {
                    u32 mw;
                    if constexpr (ROWS) { r_lo = r_hi; r_hi = r_next; r_next = m9[w + 2]; mw = __funnelshift_r(r_lo, r_hi, soff); }
                    else if constexpr (M9PF) { mw = r_hi; if (w + 1 < p.words32) r_hi = m9[w + 1]; }
                    else mw = m9[w];
                    temp = mw & __funnelshift_r(bv_lo, bv_hi, offset) & oee_mask;
                }
            }
            const u32 aval = 32 * w + (__ffs(temp) - 1);
            temp &= temp - 1;

            u64 p0 = (u64)aval * tp_0 + s0;
            u64 p1 = (u64)aval * tp_1 + s1;
            u64 p2 = (u64)aval * tp_2 + s2;
            u32 p3 = aval * tp_3 + s3;
            u32 num_0 = (u32)p0; u32 carry = p0 >> 32;
            u64 t = p1 + carry; u32 num_1 = (u32)t; carry = t >> 32;
            t = p2 + carry; u32 num_2 = (u32)t; carry = t >> 32;
            u32 num_3 = p3 + carry;

            u32 base_0, base_1, base_2, base_3;
            add_aval(base0_0, base0_1, base0_2, base0_3, aval, p, base_0, base_1, base_2, base_3);
            cases += 1;
            if constexpr (D_SETUP) {
                rolling_hash ^= num_0 ^ num_1 ^ num_2 ^ num_3 ^ base_1 ^ base_3;
                continue;
            }

            {
                // Descent test D: if n >= 2^32, (x >> 32) < (n >> 32); else x < n.
                // Both imply x < n.  CPU reference: jump::descends_coarse.
                const u64 bh = ((u64)base_2 << 32) | base_1;
                const bool small_base = (bh == 0) | (base_3 != 0);
                const u32 lane_c = 0u;
                u32 jumps = 0;
                bool finished = false;
                if constexpr (X::three) {
                    constexpr u32 LIM3 = 1u << (32 - K);
                    if (num_3 == 0 && !small_base && num_2 < LIM3) {
                        if (num_2 < LIM3) {
                            do {
                                x_jump3o<TV>(s_tab, lane_c, num_0, num_1, num_2);
                                rolling_hash ^= num_0;
                                jumps++;
                                if ((((u64)num_2 << 32) | num_1) < bh) {
                                    finished = true;
                                    break;
                                }
                            } while (num_2 < LIM3);
                        }
                    }
                }
                if (!finished) {
                    if (!small_base) {
                        // num_3 < guard_hi tested before every jump.
                        while (num_3 < guard_hi) {
                            x_jump4o<TV>(s_tab, lane_c, num_0, num_1, num_2, num_3);
                            rolling_hash ^= num_0;
                            jumps++;
                            if (num_3 == 0 && (((u64)num_2 << 32) | num_1) < bh) {
                                finished = true;
                                break;
                            }
                        }
                    } else {
                        while (num_3 < guard_hi) {
                            x_jump4o<TV>(s_tab, lane_c, num_0, num_1, num_2, num_3);
                            rolling_hash ^= num_0;
                            jumps++;
                            if (x_below4(num_0, num_1, num_2, num_3, base_0, base_1, base_2, base_3)) {
                                finished = true;
                                break;
                            }
                        }
                    }
                    if (!finished) x_flag(out, flags, p, base_0, base_1, base_2, base_3);
                }
                iters += jumps;
            }
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
static bool xs_launch_one(DevTables& tabs, const u32* bv, const u32* mod9, const u32* input, u32 n, u64 magic, const u32* tp,
                          KParams p, KOut* out, u32* flags, u32 blocks, u32 threads, cudaStream_t stream,
                          const ScreenParams& sp) {
    using X = XTraits<TV>;
    constexpr u32 bytes = x_table_bytes<TV>();
    static_assert(bytes <= 48 * 1024, "screen kernels use at most 48 KB of shared tables");
    bool ok = true;
    const u32* table = x_device_table(tabs, TV, X::w, X::layout, X::shpow, X::rep, ok);   // c3-hyg: per context
    if (!ok) return false;
    cv_kernel_xs<TV, SCR><<<blocks, threads, bytes, stream>>>(bv, mod9, input, n, magic, tp[0], tp[1], tp[2], tp[3],
                                                              p, out, flags, table, sp);
    return true;
}

// Supported screen flavours: SCR in {0, 1, 2, 3} plus one diagnostic bit,
// or the scan options M9PF (no rows) / LAZY (rows) without diagnostics.
static bool xs_scr_valid(u32 scr) {
    const u32 diag = scr & (SCR_LOAD | SCR_SCAN | SCR_SETUP);
    const u32 opt = scr & (SCR_M9PF | SCR_LAZY);
    if ((scr & ~127u) != 0) return false;
    if (opt == SCR_M9PF) return diag == 0 && !(scr & SCR_ROWS);
    if (opt == SCR_LAZY) return diag == 0 && (scr & SCR_ROWS);
    if (opt != 0) return false;
    return diag == 0 || diag == SCR_LOAD || diag == SCR_SCAN || diag == SCR_SETUP;
}

static bool xs_code_valid(u32 tv) { return tv >= 49 && tv <= 53; }

// Returns false (nothing launched) for an unsupported code or flavour.
static bool xs_launch(DevTables& tabs, u32 tv, u32 scr, const u32* bv, const u32* mod9, const u32* input, u32 n, u64 magic,
                      const u32* tp, KParams p, KOut* out, u32* flags, u32 blocks, u32 threads,
                      cudaStream_t stream, const ScreenParams& sp) {
    if (!xs_code_valid(tv) || !xs_scr_valid(scr)) return false;
#define XS_ONE(TV_, SCR_) case SCR_: return xs_launch_one<TV_, SCR_>(tabs, bv, mod9, input, n, magic, tp, p, out, flags, blocks, threads, stream, sp);
#define XS_TV(TV_) case TV_: switch (scr) { \
        XS_ONE(TV_, 0) XS_ONE(TV_, 1) XS_ONE(TV_, 2) XS_ONE(TV_, 3) \
        XS_ONE(TV_, 4) XS_ONE(TV_, 8) XS_ONE(TV_, 9) XS_ONE(TV_, 10) XS_ONE(TV_, 11) \
        XS_ONE(TV_, 16) XS_ONE(TV_, 17) XS_ONE(TV_, 18) XS_ONE(TV_, 19) \
        XS_ONE(TV_, 32) XS_ONE(TV_, 34) XS_ONE(TV_, 65) XS_ONE(TV_, 67) \
        default: return false; }
    switch (tv) {
        XS_TV(49) XS_TV(50) XS_TV(51) XS_TV(52) XS_TV(53)
        default: return false;
    }
#undef XS_TV
#undef XS_ONE
}
