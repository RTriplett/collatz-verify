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
// state and bit 29 the bitvector family (parity of trailing even steps).  The kernel recomputes T^M(n0) itself (M plain steps, ~1% of the
// work), which halves host-to-device traffic.

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

typedef uint32_t u32;
typedef uint64_t u64;
typedef uint16_t u16;

__constant__ u16 THREEPOWERSU16[9] = {1, 3, 9, 27, 81, 243, 729, 2187, 6561};

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
};

struct KOut {
    unsigned long long cases;
    unsigned long long iters;
    u32 hash;
    u32 flags;
};

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

__global__ void cv_kernel(const u32* __restrict__ bv,     // this f-class's bitvector (u32 words)
                          const u32* __restrict__ mod9,   // 9 x words32
                          const u32* __restrict__ pairs,
                          u32 n, u64 magic, u32 tp_0, u32 tp_1, u32 tp_2, u32 tp_3,
                          KParams p, KOut* out, u32* flags) {
    const u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    const bool live = idx < n;

    u32 rolling_hash = 0;
    u32 cases = 0;
    u64 iters = 0;

    if (live) {
        const uint4 in = reinterpret_cast<const uint4*>(pairs)[idx];
        const u32 base0_0 = in.x, base0_1 = in.y, base0_2 = in.z;
        const u32 oee = in.w >> 30;
        const u32 family = (in.w >> 29) & 1u;
        const u32 base0_3 = in.w & 0x1FFFFFFFu;

        // number0 = T^M(n0) + 1, by M plain steps on two u64 halves.
        u64 lo = (u64)base0_0 | ((u64)base0_1 << 32);
        u64 hi = (u64)base0_2 | ((u64)base0_3 << 32);
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

        u32 w = 0;
        u32 temp = m9[0] & __funnelshift_r(bvw[0], bvw[1], offset);
        while (true) {
            while (temp == 0) {
                if (++w == p.words32) goto done;
                temp = m9[w] & __funnelshift_r(bvw[w], bvw[w + 1], offset);
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

            if (p.tail_oee) {
                // T even <=> num odd; T = 0 mod 4 <=> num = 1 mod 4.
                if (oee == 2 && (num_0 & 1)) continue;
                if (oee == 1 && (num_0 & 3) == 1) continue;
            }

            u32 base_0, base_1, base_2, base_3;
            add_aval(base0_0, base0_1, base0_2, base0_3, aval, p, base_0, base_1, base_2, base_3);
            cases += 1;

            while (true) {
#pragma unroll
                for (int half = 0; half < 2; half++) {
                    // odd run
                    u32 a = __ffs(num_0 | (1u << 8)) - 1;
                    u32 mm = THREEPOWERSU16[a];
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
                    if (__any_sync(__activemask(), overflow)) {
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
    KOut* d_out;
    KOut* h_out;    // pinned
    u32* d_flags;
    u32* h_flags;   // pinned
    size_t cap;     // endpoints
    int busy;
};

struct Ctx {
    u32* d_bv;
    size_t bv_stride;  // u32 words per array
    size_t bv_words;   // total u32 words allocated
    unsigned long long* d_mix;
    u32* d_mod9;
    int nslots;
    Slot slots[8];
    u32 flag_cap;
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

// Launch on `slot` the `n` endpoints in pinned host buffer `h_pairs`.  The
// buffer may be reused once `cv_gpu_wait` has returned for this slot.
extern "C" int cv_gpu_launch(void* ctx, int slot, const u32* h_pairs, u32 n, size_t bv_offset, u64 magic,
                             const u32* tp, const KParams* params) {
    Ctx* c = (Ctx*)ctx;
    Slot& s = c->slots[slot];
    KParams p = *params;
    p.flag_cap = c->flag_cap;
    CK(cudaMemcpyAsync(s.d_pairs, h_pairs, (size_t)n * 16, cudaMemcpyHostToDevice, s.stream));
    CK(cudaMemsetAsync(s.d_out, 0, sizeof(KOut), s.stream));
    const u32 threads = 256;
    const u32 blocks = (n + threads - 1) / threads;
    CK(cudaEventRecord(s.start, s.stream));
    cv_kernel<<<blocks, threads, 0, s.stream>>>(c->d_bv + bv_offset, c->d_mod9, s.d_pairs, n,
                                                magic, tp[0], tp[1], tp[2], tp[3], p, s.d_out, s.d_flags);
    CK(cudaGetLastError());
    CK(cudaEventRecord(s.stop, s.stream));
    CK(cudaMemcpyAsync(s.h_out, s.d_out, sizeof(KOut), cudaMemcpyDeviceToHost, s.stream));
    s.busy = 1;
    return 0;
}

// Wait for `slot`; returns kernel milliseconds, fills totals and flags.
extern "C" int cv_gpu_wait(void* ctx, int slot, float* ms, unsigned long long* cases, unsigned long long* iters,
                           u32* hash, u32* nflags, u32* flags_out, u32 flags_out_cap) {
    Ctx* c = (Ctx*)ctx;
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
    s.busy = 0;
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
        cudaFree(s.d_out);
        cudaFreeHost(s.h_out);
        cudaFree(s.d_flags);
        cudaFreeHost(s.h_flags);
    }
    if (c->d_bv) cudaFree(c->d_bv);
    if (c->d_mix) cudaFree(c->d_mix);
    cudaFree(c->d_mod9);
    delete c;
}
