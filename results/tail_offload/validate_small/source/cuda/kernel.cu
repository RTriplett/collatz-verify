// Steps 2+3 on the GPU: one thread per depth-M endpoint.
//
// The arithmetic core (128-bit values as four u32 limbs, Barina's alternating
// odd-run/even-run loop on T(n)+1) is Angeltveit's `compute_kernel` from
// collatz_cudav7.cu, unchanged.  Differences: runtime parameters instead of
// compile-time MAXBITS/LASTBITS, lazy set-bit extraction instead of a
// 570-entry local array, the OEE state carried into the window, warp-reduced
// checksum and work counters, and an overflow buffer instead of printf.
//
// Endpoint layout: 4 x u32 = limbs of n0; bits 30-31 of limb 3 carry the OEE
// state and bit 29 the bitvector family (parity of trailing even steps).
// The default transport recomputes T^M(n0) with M plain steps. Experimental
// transports pack the residue into 8 bytes, supply its iterate in 32 bytes,
// or expand shared parent/suffix records on the device before verification.

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>

typedef uint32_t u32;
typedef uint64_t u64;
typedef uint16_t u16;

#include "grouped_tail.cuh"
#include "compact_chain.cuh"

__constant__ u16 THREEPOWERSU16[9] = {1, 3, 9, 27, 81, 243, 729, 2187, 6561};
// Experimental alternative: lane-varying indices share one read-only sector.
__device__ const u16 THREEPOWERSRO[9] = {1, 3, 9, 27, 81, 243, 729, 2187, 6561};

struct KParams {
    u64 b_mask;      // 2^B - 1 (B up to 39: 64-bit bitvector indices)
    u64 fam_stride;  // u32 words from a family-0 array to its family-1 twin
    u32 m;           // M = N - A
    u32 words32;     // 2^A / 32 completion words per endpoint
    u32 m_limb;      // M / 32
    u32 m_shift;     // M % 32
    u32 cutoff_hi;   // flag when limb 3 >= 2^(cutoff_bits - 96)
    u32 tail_oee;    // apply the carried OEE state to completions
    u32 flag_cap;    // capacity of the flag buffer (entries of 4 u32)
    u32 pad;
    u32 tail_budget; // completed double iterations before CPU continuation (0 disables)
    u32 tail_cap;    // continuation capacity; supplied by the context, not the caller
};
static_assert(sizeof(KParams) == 56, "kernel parameter ABI must be 56 bytes");

struct KOut {
    unsigned long long cases;
    unsigned long long iters;
    u32 hash;
    u32 flags;
};

// Exact state at the same double-iteration boundary used by the checksum and
// descent test. number is the alternating loop's T(n)+1 representation.
struct TailContinuation {
    u32 base[4];
    u32 number[4];
};
static_assert(sizeof(TailContinuation) == 32, "continuation ABI must be 32 bytes");

__device__ __forceinline__ void add_aval(u32 b0, u32 b1, u32 b2, u32 b3, u32 aval, const KParams& p,
                                         u32& o0, u32& o1, u32& o2, u32& o3) {
    o0 = b0; o1 = b1; o2 = b2; o3 = b3;
    u32 lo = aval << p.m_shift;
    u32 hi = p.m_shift ? (aval >> (32 - p.m_shift)) : 0u;
    switch (p.m_limb) {
        case 0: o0 |= lo; o1 |= hi; break;
        case 1: o1 |= lo; o2 |= hi; break;
        case 2: o2 |= lo; o3 |= hi; break;
        default: o3 |= lo; break;
    }
}

template<int MODE, bool OVERFLOW_VOTE = true, bool READONLY_POWERS = false, bool OEE_MASK = false,
         bool TAIL_JUMP = false, bool TAIL_DEFER = false>
__global__ void cv_kernel(const u32* __restrict__ bv,     // this f-class's bitvector (u32 words)
                          const u32* __restrict__ mod9,   // 9 x words32
                          const u32* __restrict__ pairs,
                          u32 n, u64 magic, u32 tp_0, u32 tp_1, u32 tp_2, u32 tp_3,
                          KParams p, KOut* out, u32* flags,
                          const TailJumpEntry* __restrict__ jump_table = nullptr,
                          TailContinuation* __restrict__ deferred = nullptr,
                          u32* deferred_count = nullptr) {
    const u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    const bool live = idx < n;

    u32 rolling_hash = 0;
    u32 cases = 0;
    u64 iters = 0;

    if (live) {
        uint4 in;
        if constexpr (MODE == 1) {
            const uint2 packed = reinterpret_cast<const uint2*>(pairs)[idx];
            in = make_uint4(packed.x, packed.y & 0x3FFFFFFFu, 0, packed.y & 0xC0000000u);
        } else {
            in = reinterpret_cast<const uint4*>(pairs)[(size_t)idx * (MODE == 2 ? 2 : 1)];
        }
        const u32 base0_0 = in.x, base0_1 = in.y, base0_2 = in.z;
        const u32 oee = in.w >> 30;
        const u32 family = (in.w >> 29) & 1u;
        const u32 base0_3 = in.w & 0x1FFFFFFFu;

        // number0 = T^M(n0) + 1. MODE 2 receives the exact iterate;
        // the two residue-only modes replay M steps on two u64 halves.
        u64 lo, hi;
        if constexpr (MODE == 2) {
            const uint4 iterate = reinterpret_cast<const uint4*>(pairs)[(size_t)idx * 2 + 1];
            lo = (u64)iterate.x | ((u64)iterate.y << 32);
            hi = (u64)iterate.z | ((u64)iterate.w << 32);
        } else {
            lo = (u64)base0_0 | ((u64)base0_1 << 32);
            hi = (u64)base0_2 | ((u64)base0_3 << 32);
            for (u32 j = 0; j < p.m; j++) {
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
        }
        lo += 1;
        if (lo == 0) hi += 1;
        const u32 number0_0 = (u32)lo, number0_1 = (u32)(lo >> 32);
        const u32 number0_2 = (u32)hi, number0_3 = (u32)(hi >> 32);

        // n0 mod 9 from limbs (2^18 = 1, 2^32 = 4, 2^64 = 7, 2^96 = 1 mod 9).
        u32 redmod9 = (base0_0 & 0x3FFFF) + (base0_0 >> 18)
                    + 4 * (base0_1 & 0x3FFFF) + 4 * (base0_1 >> 18)
                    + 7 * (base0_2 & 0x3FFFF) + 7 * (base0_2 >> 18)
                    + (base0_3 & 0x3FFFF) + (base0_3 >> 18);
        redmod9 %= 9;

        const u64 index = (magic * lo) & p.b_mask;  // lo = low 64 bits of number0
        const u32 offset = (u32)(index & 31);
        const u32* bvw = bv + family * p.fam_stride + (index >> 5);
        const u32* m9 = mod9 + p.words32 * redmod9;

        // Carried OEE depends only on aval modulo four. Applying it before
        // set-bit extraction avoids constructing candidates discarded below.
        u32 oee_mask = 0xFFFFFFFFu;
        if constexpr (OEE_MASK) {
            if (p.tail_oee) {
                if (oee == 2) {
                    oee_mask = (number0_0 & 1) ? 0xAAAAAAAAu : 0x55555555u;
                } else if (oee == 1) {
                    // Every odd power of three is its own inverse modulo 4.
                    const u32 forbidden = ((1u - number0_0) * tp_0) & 3u;
                    oee_mask = ~(0x11111111u << forbidden);
                }
            }
        }

        u32 w = 0;
        u32 temp = m9[0] & __funnelshift_r(bvw[0], bvw[1], offset) & oee_mask;
        while (true) {
            while (temp == 0) {
                if (++w == p.words32) goto done;
                temp = m9[w] & __funnelshift_r(bvw[w], bvw[w + 1], offset) & oee_mask;
            }
            const u32 aval = 32 * w + (__ffs(temp) - 1);
            temp &= temp - 1;

            // num = number0 + aval * 3^f  (T^M(n) + 1 for n = n0 + aval 2^M)
            u64 p0 = (u64)aval * tp_0 + number0_0;
            u64 p1 = (u64)aval * tp_1 + number0_1;
            u64 p2 = (u64)aval * tp_2 + number0_2;
            u32 p3 = aval * tp_3 + number0_3;
            u32 num_0 = (u32)p0; u32 carry = p0 >> 32;
            u64 t = p1 + carry; u32 num_1 = (u32)t; carry = t >> 32;
            t = p2 + carry; u32 num_2 = (u32)t; carry = t >> 32;
            u32 num_3 = p3 + carry;

            if (!OEE_MASK && p.tail_oee) {
                // T even <=> num odd; T = 0 mod 4 <=> num = 1 mod 4.
                if (oee == 2 && (num_0 & 1)) continue;
                if (oee == 1 && (num_0 & 3) == 1) continue;
            }

            u32 base_0, base_1, base_2, base_3;
            add_aval(base0_0, base0_1, base0_2, base0_3, aval, p, base_0, base_1, base_2, base_3);
            cases += 1;

            float base_upper = 0, cutoff = 0;
            u32 jump_mask = 0;
            u32 tail_remaining = 0;
            if constexpr (TAIL_DEFER) tail_remaining = p.tail_budget;
            if constexpr (TAIL_JUMP) {
                base_upper = tail_float_upper(base_0, base_1, base_2, base_3);
                cutoff = __fmul_rd(__uint2float_rd(p.cutoff_hi), 0x1p96f);
                jump_mask = (1u << ((p.pad >> 16) & 31u)) - 1;
            }

            while (true) {
                if constexpr (TAIL_JUMP) {
                    if (tail_jump_try(jump_table, jump_mask, base_upper, cutoff,
                                      num_0, num_1, num_2, num_3, rolling_hash, iters))
                        continue;
                }
#pragma unroll
                for (int half = 0; half < 2; half++) {
                    // odd run
                    u32 a = __ffs(num_0 | (1u << 8)) - 1;
                    u32 mm;
                    if constexpr (READONLY_POWERS) mm = __ldg(THREEPOWERSRO + a);
                    else mm = THREEPOWERSU16[a];
                    u32 t0 = __funnelshift_r(num_0, num_1, a);
                    u64 tt = (u64)t0 * mm;
                    num_0 = (u32)tt; carry = tt >> 32;
                    t0 = __funnelshift_r(num_1, num_2, a);
                    tt = (u64)t0 * mm + carry;
                    num_1 = (u32)tt; carry = tt >> 32;
                    t0 = __funnelshift_r(num_2, num_3, a);
                    tt = (u64)t0 * mm + carry;
                    num_2 = (u32)tt; carry = tt >> 32;
                    num_3 = (num_3 >> a) * mm + carry;

                    bool overflow = num_3 >= p.cutoff_hi;
                    bool handle_overflow = overflow;
                    if constexpr (OVERFLOW_VOTE)
                        handle_overflow = __any_sync(__activemask(), overflow);
                    if (handle_overflow) {
                        if (overflow) {
                            u32 slot = atomicAdd(&out->flags, 1u);
                            if (slot < p.flag_cap) {
                                flags[4 * slot] = base_0;
                                flags[4 * slot + 1] = base_1;
                                flags[4 * slot + 2] = base_2;
                                flags[4 * slot + 3] = base_3;
                            }
                            num_0 = 2; num_1 = 0; num_2 = 0; num_3 = 0;
                        }
                    }

                    // even run
                    num_0 -= 1;
                    u32 b = __ffs(num_0 | (1u << 31)) - 1;
                    num_0 = __funnelshift_r(num_0, num_1, b) + 1;
                    num_1 = __funnelshift_r(num_1, num_2, b);
                    num_2 = __funnelshift_r(num_2, num_3, b);
                    num_3 = num_3 >> b;
                    if (num_0 == 0 && b > 0) {
                        num_1 += 1;
                        if (num_1 == 0) {
                            num_2 += 1;
                            if (num_2 == 0) num_3 += 1;
                        }
                    }
                }
                rolling_hash ^= num_0;
                iters += 1;

                bool below = (num_3 < base_3)
                           | (num_3 == base_3 && num_2 < base_2)
                           | (num_3 == base_3 && num_2 == base_2 && num_1 < base_1)
                           | (num_3 == base_3 && num_2 == base_2 && num_1 == base_1 && num_0 < base_0);
                if (below) break;
                if constexpr (TAIL_DEFER) {
                    // A full queue retains this case on the GPU. Counting
                    // down to zero ensures at most one enqueue attempt per
                    // candidate, including trajectories longer than 2^32
                    // double iterations.
                    if (tail_remaining && --tail_remaining == 0) {
                        const u32 slot = atomicAdd(deferred_count, 1u);
                        if (slot < p.tail_cap) {
                            deferred[slot] = {{base_0, base_1, base_2, base_3},
                                              {num_0, num_1, num_2, num_3}};
                            break;
                        }
                    }
                }
            }
        }
    }
done:
    // Warp-reduce, one atomic per warp.
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

// Select template instantiations on the host; no tuning branches remain in
// the hot tail loop. pad bits 0..2 disable votes / use read-only powers /
// prefilter OEE; bits 8..9 select 256(default), 128, or 512 threads;
// bits 16..20 select the experimental chained transition width (0 disables).
template<int MODE>
void cv_launch_tuned(const u32* bv, const u32* mod9, const u32* input, u32 n,
                     u64 magic, const u32* tp, KParams p, KOut* out, u32* flags,
                     u32 blocks, u32 threads, cudaStream_t stream,
                     const TailJumpEntry* jump_table,
                     TailContinuation* deferred, u32* deferred_count) {
    if constexpr (MODE == 2) {
        if (p.tail_budget) {
            cv_kernel<2, false, false, true, false, true><<<blocks, threads, 0, stream>>>(
                bv, mod9, input, n, magic, tp[0], tp[1], tp[2], tp[3], p, out, flags,
                nullptr, deferred, deferred_count);
            return;
        }
    }
    if (jump_table) {
        cv_kernel<MODE, false, false, true, true><<<blocks, threads, 0, stream>>>(
            bv, mod9, input, n, magic, tp[0], tp[1], tp[2], tp[3], p, out, flags, jump_table);
        return;
    }
#define LAUNCH_TUNED(VOTE, RO, OEE) \
    cv_kernel<MODE, VOTE, RO, OEE><<<blocks, threads, 0, stream>>>( \
        bv, mod9, input, n, magic, tp[0], tp[1], tp[2], tp[3], p, out, flags)
    switch (p.pad & 7u) {
        case 0: LAUNCH_TUNED(true, false, false); break;
        case 1: LAUNCH_TUNED(false, false, false); break;
        case 2: LAUNCH_TUNED(true, true, false); break;
        case 3: LAUNCH_TUNED(false, true, false); break;
        case 4: LAUNCH_TUNED(true, false, true); break;
        case 5: LAUNCH_TUNED(false, false, true); break;
        case 6: LAUNCH_TUNED(true, true, true); break;
        case 7: LAUNCH_TUNED(false, true, true); break;
    }
#undef LAUNCH_TUNED
}

// One block per parent/class descriptor. Only its actual terminal survivors
// are emitted, densely, so the verification kernel does no padded leaf work.
// Descriptor: n0[2], m[4], 3^f[4], d, table_start, count, out_start, width, k.
// Terminal record: r, q_coeff, iterate, pow3, f, oee, ne, reserved.
// The host has checked the exact-128-bit headroom for every parent group.
__global__ void cv_expand_parents(const u32* __restrict__ parents,
                                  const u32* __restrict__ terminal,
                                  u32* __restrict__ affine, u64 fam_stride) {
    const u32* parent = parents + (size_t)blockIdx.x * 16;
    const u64 parent_n0 = (u64)parent[0] | ((u64)parent[1] << 32);
    const u64 parent_mlo = (u64)parent[2] | ((u64)parent[3] << 32);
    const u64 parent_mhi = (u64)parent[4] | ((u64)parent[5] << 32);
    const u64 parent_p3lo = (u64)parent[6] | ((u64)parent[7] << 32);
    const u64 parent_p3hi = (u64)parent[8] | ((u64)parent[9] << 32);
    const u32 width = parent[14], k = parent[15];
    const u32 mask = (1u << width) - 1;
    for (u32 i = threadIdx.x; i < parent[12]; i += blockDim.x) {
        const u32* entry = terminal + (size_t)(parent[11] + i) * 8;
        const u32 q = (entry[1] - parent[10]) & mask;
        const u64 n0 = parent_n0 | ((u64)q << k);

        // x = m + 3^f*q - r; x is divisible by 2^width.
        u64 lo = parent_p3lo * q;
        u64 hi = parent_p3hi * q + __umul64hi(parent_p3lo, (u64)q);
        u64 sum = lo + parent_mlo;
        hi += parent_mhi + (sum < lo);
        lo = sum;
        hi -= lo < entry[0];
        lo -= entry[0];
        const u64 shifted_lo = (lo >> width) | (hi << (64 - width));
        const u64 shifted_hi = hi >> width;

        // T^M(n0) = (x >> width)*3^(suffix odd steps) + T^width(r).
        lo = shifted_lo * entry[3];
        hi = shifted_hi * entry[3] + __umul64hi(shifted_lo, (u64)entry[3]);
        sum = lo + entry[2];
        hi += sum < lo;
        lo = sum;
        const u32 family = fam_stride ? (entry[6] & 1u) : 0u;
        uint4* out = reinterpret_cast<uint4*>(affine) + (size_t)(parent[13] + i) * 2;
        out[0] = make_uint4((u32)n0, (u32)(n0 >> 32), 0,
                           (entry[5] << 30) | (family << 29));
        out[1] = make_uint4((u32)lo, (u32)(lo >> 32), (u32)hi, (u32)(hi >> 32));
    }
}


// ---------------------------------------------------------------------------
// Bitvector construction on the device (any B <= 39).  Mirrors
// lookahead.rs::window_dip_ext step for step; tests compare the result with
// the host-built arrays bit for bit.

__constant__ u64 MIX_OFF[17];   // start of the table for modulus 3^j in the flat mix array
__constant__ u32 POW3U32[17];

struct BuildParams {
    u64 nbits;      // 2^B
    u64 magic;      // 3^-f mod 2^64
    u64 tail_bits;  // 2^A: indices below this are mirrored past 2^B
    u32 b;
    int cutoff;
    u32 start_evens;
    u32 pm_inf;
    u32 mix_j;      // 0 = no mixed-word credits
    u32 pad;
};

__device__ __forceinline__ bool odd_run_then_two_evens(u64 updown, u32 i, u32 b) {
    u32 a = __ffsll((long long)~(updown >> i)) - 1;  // odd steps in the run starting at i
    if (i + a + 2 >= b) return false;
    return ((updown >> (i + a)) & 3ull) == 0;
}

__device__ __forceinline__ u32 v3u64(u64 x) {
    u32 t = 0;
    while (x != 0 && x % 3 == 0) { x /= 3; t++; }
    return t;
}

__global__ void cv_build_kernel(u32* out, BuildParams p, const u64* __restrict__ mix) {
    const u64 stride = (u64)gridDim.x * blockDim.x;
    const bool extra = p.pm_inf || p.mix_j;
    for (u64 x0 = (u64)blockIdx.x * blockDim.x + threadIdx.x; x0 < p.nbits; x0 += stride) {
        // pattern 0 stands for T^M(n)+1 = 2^B (mod 2^B); iterate the
        // representative 2^B so that x - 1 >= 0 carries the true residues
        const u64 xr = x0 ? x0 : p.nbits;
        u64 updown = 0, x = xr;
        for (u32 i = 0; i < p.b; i++) {
            if ((x & 1) == 0) { x = x + x / 2; updown |= 1ull << i; } else { x = x / 2 + 1; }
        }
        int cur = 0, mx = 0;
        const bool known = p.start_evens != 2;  // 2: parity unknown, no credit before an odd step
        u32 evens = known ? p.start_evens : 0, fw = 0;
        x = xr;
        for (u32 i = 0; i < p.b; i++) {
            if ((updown >> i) & 1) {
                if (cur + 306 > mx && odd_run_then_two_evens(updown, i, p.b)) mx = cur + 306;
                cur -= 179; evens = 0; fw++; x = x + x / 2;
            } else {
                cur += 306; evens++;
                if ((evens & 1) == 0 && (known || fw > 0) && cur + 179 > mx) mx = cur + 179;
                x = x / 2 + 1;
            }
            mx = max(mx, cur);
            if (extra && fw > 0) {
                if (p.pm_inf) mx = max(mx, cur + 179 * (int)min(v3u64(x), fw));
                if (p.mix_j) {
                    const u32 jp = min(fw, p.mix_j);
                    const u64 e = mix[MIX_OFF[jp] + (x - 1) % (u64)POW3U32[jp]];
                    if (e) {
                        const u64 raw = e - 1;
                        mx = max(mx, cur + 485 * (int)((raw >> 6) & 31) - 306 * (int)(raw & 63));
                    }
                }
            }
        }
        if (mx <= p.cutoff) {
            const u64 idx = (x0 * p.magic) & (p.nbits - 1);
            atomicOr(&out[idx >> 5], 1u << (idx & 31));
            if (idx < p.tail_bits) {
                const u64 j = idx + p.nbits;
                atomicOr(&out[j >> 5], 1u << (j & 31));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Host API (extern "C", called from Rust).

struct Slot {
    cudaStream_t stream;
    cudaEvent_t start, stop;
    u32* d_pairs;
    u32* d_affine;  // experimental parent expansion output (32 bytes/endpoint)
    KOut* d_out;
    KOut* h_out;    // pinned
    u32* d_flags;
    u32* h_flags;   // pinned
    TailContinuation* d_deferred;
    TailContinuation* h_deferred; // pinned, filled by cv_gpu_wait
    u32* d_deferred_count;
    u32* h_deferred_count; // pinned
    size_t cap;     // endpoints
    size_t input_bytes;
    size_t affine_bytes;
    int busy;
    bool tail_active;
    bool tail_ready; // must be consumed before the slot can be reused
};

struct Ctx {
    u32* d_bv;
    size_t bv_stride;  // u32 words per array
    size_t bv_words;   // total u32 words allocated
    unsigned long long* d_mix;
    u32* d_mod9;
    u32* d_terminal;
    size_t terminal_records;
    ChainProgramGpu* d_chain_programs;
    ChainLeafGpu* d_chain_leaves;
    u32* d_chain_words;
    std::vector<ChainProgramGpu> h_chain_programs;
    TailJumpEntry* d_tail_jump[4];
    int nslots;
    Slot slots[8];
    u32 flag_cap;
    u32 tail_cap;
};

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); return -1; } } while (0)

extern "C" int cv_gpu_device_name(char* buf, int len) {
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return -1;
    snprintf(buf, len, "%s (sm_%d%d, %d SMs)", prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    return 0;
}

extern "C" void* cv_gpu_init(const u32* bv, size_t bv_words, size_t bv_stride, const u32* mod9, size_t mod9_words,
                             int nslots, size_t cap) {
    if (nslots <= 0 || cap == 0 || cap > SIZE_MAX / 16) return nullptr;
    Ctx* c = new Ctx();
    c->bv_stride = bv_stride;
    c->bv_words = bv_words;
    c->d_bv = nullptr;
    c->d_mix = nullptr;
    c->nslots = nslots > 8 ? 8 : nslots;
    c->flag_cap = 1 << 16;
    cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
    if (bv_words > 0) {  // otherwise arrays are built on the device later
        if (cudaMalloc(&c->d_bv, bv_words * 4) != cudaSuccess) return nullptr;
        if (cudaMemcpy(c->d_bv, bv, bv_words * 4, cudaMemcpyHostToDevice) != cudaSuccess) return nullptr;
    }
    if (cudaMalloc(&c->d_mod9, mod9_words * 4) != cudaSuccess) return nullptr;
    if (cudaMemcpy(c->d_mod9, mod9, mod9_words * 4, cudaMemcpyHostToDevice) != cudaSuccess) return nullptr;
    for (int i = 0; i < c->nslots; i++) {
        Slot& s = c->slots[i];
        s.cap = cap;
        s.input_bytes = cap * 16;
        s.busy = 0;
        cudaStreamCreateWithFlags(&s.stream, cudaStreamNonBlocking);
        cudaEventCreate(&s.start);
        cudaEventCreate(&s.stop);
        if (cudaMalloc(&s.d_pairs, cap * 16) != cudaSuccess) return nullptr;
        cudaMalloc(&s.d_out, sizeof(KOut));
        cudaHostAlloc(&s.h_out, sizeof(KOut), cudaHostAllocDefault);
        cudaMalloc(&s.d_flags, (size_t)c->flag_cap * 16);
        cudaHostAlloc(&s.h_flags, (size_t)c->flag_cap * 16, cudaHostAllocDefault);
    }
    return c;
}

extern "C" u32* cv_gpu_host_alloc(size_t bytes) {
    u32* p = nullptr;
    if (cudaHostAlloc(&p, bytes, cudaHostAllocDefault) != cudaSuccess) return nullptr;
    return p;
}

extern "C" void cv_gpu_host_free(u32* p) { cudaFreeHost(p); }

// Allocate only for experiments which request continuations. All slots are
// replaced transactionally; unread continuations cannot be discarded by
// reconfiguration. Capacity zero releases the buffers after all work drains.
extern "C" int cv_gpu_set_tail_capacity(void* ctx, u32 capacity) {
    Ctx* c = (Ctx*)ctx;
    if (!c || (size_t)capacity > SIZE_MAX / sizeof(TailContinuation)) return -1;
    for (int i = 0; i < c->nslots; i++)
        if (c->slots[i].busy || c->slots[i].tail_ready) return -1;
    if (capacity == c->tail_cap) return 0;
    TailContinuation* device[8] = {};
    TailContinuation* host[8] = {};
    u32* device_count[8] = {};
    u32* host_count[8] = {};
    if (capacity) {
        for (int i = 0; i < c->nslots; i++) {
            cudaError_t error = cudaMalloc(&device[i], (size_t)capacity * sizeof(TailContinuation));
            if (error == cudaSuccess)
                error = cudaHostAlloc(&host[i], (size_t)capacity * sizeof(TailContinuation), cudaHostAllocDefault);
            if (error == cudaSuccess) error = cudaMalloc(&device_count[i], sizeof(u32));
            if (error == cudaSuccess)
                error = cudaHostAlloc(&host_count[i], sizeof(u32), cudaHostAllocDefault);
            if (error != cudaSuccess) {
                for (int j = 0; j <= i; j++) {
                    cudaFree(device[j]);
                    cudaFreeHost(host[j]);
                    cudaFree(device_count[j]);
                    cudaFreeHost(host_count[j]);
                }
                fprintf(stderr, "CUDA continuation allocation: %s\n", cudaGetErrorString(error));
                return -1;
            }
        }
    }
    for (int i = 0; i < c->nslots; i++) {
        Slot& s = c->slots[i];
        cudaFree(s.d_deferred);
        cudaFreeHost(s.h_deferred);
        cudaFree(s.d_deferred_count);
        cudaFreeHost(s.h_deferred_count);
        s.d_deferred = device[i];
        s.h_deferred = host[i];
        s.d_deferred_count = device_count[i];
        s.h_deferred_count = host_count[i];
        s.tail_active = false;
    }
    c->tail_cap = capacity;
    return 0;
}

// Reserve equal buffers for all transport experiments. cap is the maximum
// number of logical endpoints, not the number of input u32 words. No slot may
// be in flight while buffers or resident terminal records are replaced.
extern "C" int cv_gpu_enable_experimental(void* ctx, size_t cap) {
    Ctx* c = (Ctx*)ctx;
    if (!c || cap == 0 || cap > UINT32_MAX || cap > SIZE_MAX / 64) return -1;
    for (int i = 0; i < c->nslots; i++) if (c->slots[i].busy) return -1;
    u32* inputs[8] = {};
    u32* outputs[8] = {};
    for (int i = 0; i < c->nslots; i++) {
        cudaError_t error = cudaMalloc(&inputs[i], cap * 64);
        if (error == cudaSuccess) error = cudaMalloc(&outputs[i], cap * 32);
        if (error != cudaSuccess) {
            fprintf(stderr, "CUDA experimental buffer allocation: %s\n", cudaGetErrorString(error));
            for (int j = 0; j <= i; j++) {
                cudaFree(inputs[j]);
                cudaFree(outputs[j]);
            }
            return -1;
        }
    }
    for (int i = 0; i < c->nslots; i++) {
        Slot& s = c->slots[i];
        cudaFree(s.d_pairs);
        cudaFree(s.d_affine);
        s.d_pairs = inputs[i];
        s.d_affine = outputs[i];
        s.cap = cap;
        s.input_bytes = cap * 64;
        s.affine_bytes = cap * 32;
    }
    return 0;
}

// record_count counts 32-byte TerminalWord records (eight u32 each).
extern "C" int cv_gpu_upload_terminal(void* ctx, const u32* records, size_t record_count) {
    Ctx* c = (Ctx*)ctx;
    if (!c || !records || record_count == 0 || record_count > UINT32_MAX || record_count > SIZE_MAX / 32)
        return -1;
    for (int i = 0; i < c->nslots; i++) if (c->slots[i].busy) return -1;
    u32* uploaded = nullptr;
    CK(cudaMalloc(&uploaded, record_count * 32));
    cudaError_t error = cudaMemcpy(uploaded, records, record_count * 32, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        cudaFree(uploaded);
        fprintf(stderr, "CUDA terminal upload: %s\n", cudaGetErrorString(error));
        return -1;
    }
    cudaFree(c->d_terminal);
    c->d_terminal = uploaded;
    c->terminal_records = record_count;
    return 0;
}

static bool cv_chain_word_valid(const u32* word, u32 width) {
    const u32 mask = (1u << width) - 1;
    if (word[0] > mask || word[1] > mask || word[3] == 0 ||
        word[4] > 64 || word[5] > 2 || word[6] > 3 || word[7] != 0)
        return false;
    u32 power = word[3], odd_steps = 0;
    while (power > 1 && power % 3 == 0) { power /= 3; odd_steps++; }
    return power == 1 && odd_steps <= width && word[4] >= odd_steps;
}

// All three resident arrays are replaced atomically after validation and
// successful upload. Programs use dense leaf references into separate short
// word tables; no full-width composed affine words are uploaded.
extern "C" int cv_gpu_upload_chain(void* ctx,
                                    const u32* programs, size_t program_count,
                                    const u32* leaves, size_t leaf_count,
                                    const u32* words, size_t word_count) {
    Ctx* c = (Ctx*)ctx;
    if (!c || !programs || !leaves || !words || program_count == 0 ||
        leaf_count == 0 || word_count == 0 || program_count > UINT32_MAX ||
        leaf_count > UINT32_MAX || word_count > UINT32_MAX ||
        program_count > SIZE_MAX / sizeof(ChainProgramGpu) ||
        leaf_count > SIZE_MAX / sizeof(ChainLeafGpu) || word_count > SIZE_MAX / 32)
        return -1;
    for (int i = 0; i < c->nslots; i++) if (c->slots[i].busy) return -1;
    const auto* host_programs = reinterpret_cast<const ChainProgramGpu*>(programs);
    const auto* host_leaves = reinterpret_cast<const ChainLeafGpu*>(leaves);
    for (size_t i = 0; i < program_count; i++) {
        const auto& p = host_programs[i];
        if (p.first_width == 0 || p.first_width > 6 ||
            p.second_width == 0 || p.second_width > 12 ||
            p.first_len > 64 || p.first_len > (1u << p.first_width) ||
            p.route_len > p.first_len || p.first_start > word_count ||
            p.first_len > word_count - p.first_start ||
            p.leaf_start > leaf_count || p.endpoints > leaf_count - p.leaf_start ||
            (p.endpoints && !p.first_len)) return -1;
        for (u32 j = 0; j < p.first_len; j++) {
            if (!cv_chain_word_valid(words + ((size_t)p.first_start + j) * 8, p.first_width))
                return -1;
        }
        for (u32 j = 0; j < p.endpoints; j++) {
            const auto& leaf = host_leaves[(size_t)p.leaf_start + j];
            if (leaf.first_slot >= p.first_len || leaf.second_index >= word_count) return -1;
            const u32* first = words + ((size_t)p.first_start + leaf.first_slot) * 8;
            const u32* second = words + (size_t)leaf.second_index * 8;
            if (!cv_chain_word_valid(second, p.second_width) ||
                second[4] < first[4] || second[4] - first[4] > p.second_width)
                return -1;
            u32 expected_power = 1;
            for (u32 odd = first[4]; odd < second[4]; odd++) expected_power *= 3;
            if (second[3] != expected_power) return -1;
        }
    }
    ChainProgramGpu* uploaded_programs = nullptr;
    ChainLeafGpu* uploaded_leaves = nullptr;
    u32* uploaded_words = nullptr;
    cudaError_t error = cudaMalloc(&uploaded_programs, program_count * sizeof(ChainProgramGpu));
    if (error == cudaSuccess)
        error = cudaMalloc(&uploaded_leaves, leaf_count * sizeof(ChainLeafGpu));
    if (error == cudaSuccess) error = cudaMalloc(&uploaded_words, word_count * 32);
    if (error == cudaSuccess)
        error = cudaMemcpy(uploaded_programs, programs, program_count * sizeof(ChainProgramGpu), cudaMemcpyHostToDevice);
    if (error == cudaSuccess)
        error = cudaMemcpy(uploaded_leaves, leaves, leaf_count * sizeof(ChainLeafGpu), cudaMemcpyHostToDevice);
    if (error == cudaSuccess)
        error = cudaMemcpy(uploaded_words, words, word_count * 32, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        cudaFree(uploaded_programs);
        cudaFree(uploaded_leaves);
        cudaFree(uploaded_words);
        fprintf(stderr, "CUDA compact chain upload: %s\n", cudaGetErrorString(error));
        return -1;
    }
    cudaFree(c->d_chain_programs);
    cudaFree(c->d_chain_leaves);
    cudaFree(c->d_chain_words);
    c->d_chain_programs = uploaded_programs;
    c->d_chain_leaves = uploaded_leaves;
    c->d_chain_words = uploaded_words;
    c->h_chain_programs.assign(host_programs, host_programs + program_count);
    return 0;
}

// Preload tables outside benchmark timing; cached widths may be interleaved.
extern "C" int cv_gpu_set_tail_jump(void* ctx, u32 width) {
    Ctx* c = (Ctx*)ctx;
    if (!c) return -1;
    if (width == 0) return 0;
    const int slot = tail_jump_slot(width);
    if (slot < 0) return -1;
    if (c->d_tail_jump[slot]) return 0;
    for (int i = 0; i < c->nslots; i++) if (c->slots[i].busy) return -1;
    const auto table = tail_jump_build(width);
    TailJumpEntry* uploaded = nullptr;
    CK(cudaMalloc(&uploaded, table.size() * sizeof(TailJumpEntry)));
    cudaError_t error = cudaMemcpy(uploaded, table.data(), table.size() * sizeof(TailJumpEntry), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        cudaFree(uploaded);
        fprintf(stderr, "CUDA tail jump upload: %s\n", cudaGetErrorString(error));
        return -1;
    }
    c->d_tail_jump[slot] = uploaded;
    size_t usable = 0, doubles = 0;
    u32 maximum = 0;
    for (const auto& row : table) {
        usable += row.count != 0;
        doubles += row.count;
        if (row.count > maximum) maximum = row.count;
    }
    fprintf(stderr, "tail-jump W=%u rows=%zu usable=%zu mean_doubles=%.6f max_doubles=%u bytes=%zu\n",
            width, table.size(), usable, (double)doubles / table.size(), maximum,
            table.size() * sizeof(TailJumpEntry));
    return 0;
}

// MODE 0: replay16, 1: packed8 (shared windows, M<=62), 2: affine32,
// 3: terminal parents, 4: compact two-stage parents. Both expand 64-byte
// descriptors to dense affine32 records before the unchanged verifier.
// n is always the number of logical endpoints; n_records counts input records.
// The pinned input remains owned by this slot until cv_gpu_wait returns.
extern "C" int cv_gpu_launch_ex(void* ctx, int slot, const u32* h_pairs, u32 n, u32 n_records, u32 mode,
                                size_t bv_offset, u64 magic, const u32* tp, const KParams* params) {
    Ctx* c = (Ctx*)ctx;
    if (!c || slot < 0 || slot >= c->nslots || !h_pairs || !tp || !params || n == 0 || mode > 4)
        return -1;
    Slot& s = c->slots[slot];
    if (s.busy || s.tail_ready || n > s.cap || bv_offset >= c->bv_words) return -1;
    KParams p = *params;
    if ((p.pad & ~0x1F0307u) != 0 || ((p.pad >> 8) & 3u) == 3u) return -1;
    const u32 tail_width = (p.pad >> 16) & 31u;
    const TailJumpEntry* jump_table = nullptr;
    if (tail_width) {
        const int jump_slot = tail_jump_slot(tail_width);
        if (jump_slot < 0 || (p.pad & 7u) != 5u || !c->d_tail_jump[jump_slot]) return -1;
        jump_table = c->d_tail_jump[jump_slot];
    }
    if (p.tail_budget) {
        // Only the measured affine/parent no-vote, constant-powers, early-OEE
        // specialization supports deferral. Bound enqueue attempts so its
        // u32 counter cannot wrap, even if every completion reaches budget.
        if (mode < 2 || jump_table || (p.pad & 7u) != 5u || !c->tail_cap ||
            p.words32 == 0 || (u64)n * p.words32 > UINT32_MAX / 32u)
            return -1;
    }
    p.flag_cap = c->flag_cap;
    p.tail_cap = c->tail_cap;
    const size_t record_bytes = mode == 0 ? 16 : mode == 1 ? 8 : mode == 2 ? 32 : 64;
    const size_t input_bytes = (size_t)n_records * record_bytes;
    if (input_bytes > s.input_bytes) return -1;
    if (mode < 3 && n_records != n) return -1;
    if (mode == 1 && (p.m > 62 || p.fam_stride != 0)) return -1;
    if (mode == 3) {
        if (!c->d_terminal || !s.d_affine || n_records == 0 || n_records > n || (size_t)n * 32 > s.affine_bytes)
            return -1;
        // Reject malformed descriptor ranges before they could leave stale
        // output records or write beyond the per-slot expansion buffer.
        size_t written = 0;
        for (u32 i = 0; i < n_records; i++) {
            const u32* parent = h_pairs + (size_t)i * 16;
            const size_t start = parent[11], count = parent[12];
            const u32 width = parent[14], k = parent[15];
            if (width == 0 || width > 31 || k >= 64 || k + width != p.m || p.m > 64 ||
                count == 0 || start > c->terminal_records || count > c->terminal_records - start ||
                parent[13] != written || written > n || count > n - written) return -1;
            written += count;
        }
        if (written != n) return -1;
    }
    if (mode == 4) {
        if (!c->d_chain_programs || !c->d_chain_leaves || !c->d_chain_words ||
            !s.d_affine || n_records == 0 || n_records > n || (size_t)n * 32 > s.affine_bytes)
            return -1;
        size_t written = 0;
        for (u32 i = 0; i < n_records; i++) {
            const u32* parent = h_pairs + (size_t)i * 16;
            const size_t program_id = parent[11], count = parent[12], offset = parent[14];
            const u32 k = parent[15];
            if (program_id >= c->h_chain_programs.size()) return -1;
            const auto& program = c->h_chain_programs[program_id];
            if (k >= 64 || k + program.first_width + program.second_width != p.m || p.m > 64 ||
                count == 0 || offset > program.endpoints || count > program.endpoints - offset ||
                parent[13] != written || written > n || count > n - written ||
                !(parent[6] & 1u) || parent[6] * parent[10] != 1u)
                return -1;
            written += count;
        }
        if (written != n) return -1;
    }
    CK(cudaMemcpyAsync(s.d_pairs, h_pairs, input_bytes, cudaMemcpyHostToDevice, s.stream));
    CK(cudaMemsetAsync(s.d_out, 0, sizeof(KOut), s.stream));
    if (p.tail_budget) CK(cudaMemsetAsync(s.d_deferred_count, 0, sizeof(u32), s.stream));
    const u32 block_choice = (p.pad >> 8) & 3u;
    const u32 threads = block_choice == 1 ? 128 : block_choice == 2 ? 512 : 256;
    const u32 blocks = (n - 1) / threads + 1;
    CK(cudaEventRecord(s.start, s.stream));
    const u32* input = s.d_pairs;
    if (mode == 3) {
        cv_expand_parents<<<n_records, 128, 0, s.stream>>>(s.d_pairs, c->d_terminal, s.d_affine, p.fam_stride);
        CK(cudaGetLastError());
        input = s.d_affine;
    } else if (mode == 4) {
        cv_expand_chain<<<n_records, 128, 0, s.stream>>>(
            s.d_pairs, c->d_chain_programs, c->d_chain_leaves, c->d_chain_words,
            s.d_affine, p.fam_stride);
        CK(cudaGetLastError());
        input = s.d_affine;
    }
    if (mode == 0) {
        cv_launch_tuned<0>(c->d_bv + bv_offset, c->d_mod9, input, n, magic, tp, p,
                           s.d_out, s.d_flags, blocks, threads, s.stream, jump_table,
                           s.d_deferred, s.d_deferred_count);
    } else if (mode == 1) {
        cv_launch_tuned<1>(c->d_bv + bv_offset, c->d_mod9, input, n, magic, tp, p,
                           s.d_out, s.d_flags, blocks, threads, s.stream, jump_table,
                           s.d_deferred, s.d_deferred_count);
    } else {
        cv_launch_tuned<2>(c->d_bv + bv_offset, c->d_mod9, input, n, magic, tp, p,
                           s.d_out, s.d_flags, blocks, threads, s.stream, jump_table,
                           s.d_deferred, s.d_deferred_count);
    }
    CK(cudaGetLastError());
    CK(cudaEventRecord(s.stop, s.stream));
    CK(cudaMemcpyAsync(s.h_out, s.d_out, sizeof(KOut), cudaMemcpyDeviceToHost, s.stream));
    if (p.tail_budget)
        CK(cudaMemcpyAsync(s.h_deferred_count, s.d_deferred_count, sizeof(u32), cudaMemcpyDeviceToHost, s.stream));
    s.tail_active = p.tail_budget != 0;
    s.busy = 1;
    return 0;
}

// Original ABI and representation remain the default for production callers.
extern "C" int cv_gpu_launch(void* ctx, int slot, const u32* h_pairs, u32 n, size_t bv_offset, u64 magic,
                             const u32* tp, const KParams* params) {
    return cv_gpu_launch_ex(ctx, slot, h_pairs, n, n, 0, bv_offset, magic, tp, params);
}

// Wait for `slot`; returns kernel milliseconds, fills totals and flags.
extern "C" int cv_gpu_wait(void* ctx, int slot, float* ms, unsigned long long* cases, unsigned long long* iters,
                           u32* hash, u32* nflags, u32* flags_out, u32 flags_out_cap) {
    Ctx* c = (Ctx*)ctx;
    if (!c || slot < 0 || slot >= c->nslots || !ms || !cases || !iters || !hash || !nflags ||
        (flags_out_cap && !flags_out)) return -1;
    Slot& s = c->slots[slot];
    if (!s.busy) { *ms = 0; *cases = 0; *iters = 0; *hash = 0; *nflags = 0; return 0; }
    CK(cudaStreamSynchronize(s.stream));
    CK(cudaEventElapsedTime(ms, s.start, s.stop));
    *cases = s.h_out->cases;
    *iters = s.h_out->iters;
    *hash = s.h_out->hash;
    u32 nf = s.h_out->flags;
    if (nf > c->flag_cap) nf = c->flag_cap;
    if (nf > flags_out_cap) nf = flags_out_cap;
    if (nf) {
        CK(cudaMemcpy(flags_out, s.d_flags, (size_t)nf * 16, cudaMemcpyDeviceToHost));
    }
    *nflags = s.h_out->flags;
    if (s.tail_active) {
        const u32 stored = *s.h_deferred_count < c->tail_cap ? *s.h_deferred_count : c->tail_cap;
        if (stored) {
            CK(cudaMemcpyAsync(s.h_deferred, s.d_deferred, (size_t)stored * sizeof(TailContinuation),
                               cudaMemcpyDeviceToHost, s.stream));
            CK(cudaStreamSynchronize(s.stream));
        }
        s.tail_ready = true;
    }
    s.busy = 0;
    return 0;
}

// Consume the continuation result after ordinary wait. attempts may exceed
// capacity: only min(attempts, configured capacity) records were deferred;
// the remaining candidates completed on the GPU. A null zero-capacity
// destination queries the count without consuming a nonempty result; an
// empty result is consumed. An undersized non-query destination returns an
// error without consuming anything, allowing a safe retry.
extern "C" int cv_gpu_wait_deferred(void* ctx, int slot, u32* attempts,
                                    u32* records, size_t record_capacity) {
    Ctx* c = (Ctx*)ctx;
    if (!c || slot < 0 || slot >= c->nslots || !attempts) return -1;
    Slot& s = c->slots[slot];
    if (s.busy || !s.tail_ready || !s.h_deferred_count) return -1;
    const u32 count = *s.h_deferred_count;
    const u32 stored = count < c->tail_cap ? count : c->tail_cap;
    if (!records && record_capacity == 0 && stored) {
        *attempts = count;
        return 0;
    }
    if (record_capacity < stored || (stored && !records)) return -1;
    if (stored) std::memcpy(records, s.h_deferred, (size_t)stored * sizeof(TailContinuation));
    *attempts = count;
    s.tail_ready = false;
    s.tail_active = false;
    return 0;
}

// Upload the flat mixed-word tables (u64 entries; offsets[j] = start of 3^j table).
extern "C" int cv_gpu_upload_mix(void* ctx, const unsigned long long* flat, size_t words, const unsigned long long* offsets) {
    Ctx* c = (Ctx*)ctx;
    if (c->d_mix) cudaFree(c->d_mix);
    CK(cudaMalloc(&c->d_mix, words * 8));
    CK(cudaMemcpy(c->d_mix, flat, words * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpyToSymbol(MIX_OFF, offsets, 17 * 8));
    u32 p3[17];
    p3[0] = 1;
    for (int j = 1; j < 17; j++) p3[j] = p3[j - 1] * 3;
    CK(cudaMemcpyToSymbol(POW3U32, p3, sizeof(p3)));
    return 0;
}

// (Re)allocate `words` zeroed u32 words of array memory.
extern "C" int cv_gpu_alloc_bv(void* ctx, size_t words) {
    Ctx* c = (Ctx*)ctx;
    if (c->d_bv) cudaFree(c->d_bv);
    c->d_bv = nullptr;
    c->bv_words = words;
    CK(cudaMalloc(&c->d_bv, c->bv_words * 4));
    CK(cudaMemset(c->d_bv, 0, c->bv_words * 4));
    return 0;
}

// Build one array (2^B + 2^A bits) at u32-word `offset`; returns milliseconds.
extern "C" int cv_gpu_build_array(void* ctx, size_t offset, u32 b, u32 a, u64 magic, int cutoff, u32 start_evens,
                                  u32 pm_inf, u32 mix_j, float* ms) {
    Ctx* c = (Ctx*)ctx;
    BuildParams p;
    p.nbits = 1ull << b;
    p.magic = magic;
    p.tail_bits = 1ull << a;
    p.b = b;
    p.cutoff = cutoff;
    p.start_evens = start_evens;
    p.pm_inf = pm_inf;
    p.mix_j = c->d_mix ? mix_j : 0;
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    cudaEventRecord(e0);
    int sms = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    cv_build_kernel<<<sms * 32, 256>>>(c->d_bv + offset, p, (const u64*)c->d_mix);
    CK(cudaGetLastError());
    cudaEventRecord(e1);
    CK(cudaEventSynchronize(e1));
    cudaEventElapsedTime(ms, e0, e1);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return 0;
}

// Copy `words` u32 words of array memory (from word `start`) back to the host.
extern "C" int cv_gpu_download_bv(void* ctx, size_t start, u32* host, size_t words) {
    Ctx* c = (Ctx*)ctx;
    CK(cudaMemcpy(host, c->d_bv + start, words * 4, cudaMemcpyDeviceToHost));
    return 0;
}

extern "C" int cv_gpu_mem_info(size_t* free_bytes, size_t* total_bytes) {
    return cudaMemGetInfo(free_bytes, total_bytes) == cudaSuccess ? 0 : -1;
}

extern "C" void cv_gpu_free(void* ctx) {
    Ctx* c = (Ctx*)ctx;
    for (int i = 0; i < c->nslots; i++) {
        Slot& s = c->slots[i];
        cudaStreamDestroy(s.stream);
        cudaEventDestroy(s.start);
        cudaEventDestroy(s.stop);
        cudaFree(s.d_pairs);
        cudaFree(s.d_affine);
        cudaFree(s.d_out);
        cudaFreeHost(s.h_out);
        cudaFree(s.d_flags);
        cudaFreeHost(s.h_flags);
        cudaFree(s.d_deferred);
        cudaFreeHost(s.h_deferred);
        cudaFree(s.d_deferred_count);
        cudaFreeHost(s.h_deferred_count);
    }
    if (c->d_bv) cudaFree(c->d_bv);
    if (c->d_mix) cudaFree(c->d_mix);
    if (c->d_terminal) cudaFree(c->d_terminal);
    cudaFree(c->d_chain_programs);
    cudaFree(c->d_chain_leaves);
    cudaFree(c->d_chain_words);
    for (auto* table : c->d_tail_jump) cudaFree(table);
    cudaFree(c->d_mod9);
    delete c;
}
