// Experimental tail variants (KParams.pad bits 24..31 select one; 0 keeps the
// original kernels).  Screening (bitvector, mod 9, early OEE mask) is the
// tuned kernel's (no warp vote, constant powers, OEE before extraction), so
// every variant iterates exactly the same case set.
//
// Variant families
//   * BASE*: the original double-iteration loop on y = T(n)+1.  BASE, SHPOW,
//     OVF1, FAST3 and FAST3_SH are work-preserving: identical cases, double
//     iterations, hash and flags.  CAP5* cap odd runs at five steps so that
//     3^a comes from a register byte permute; that changes the iteration and
//     checksum definition (CPU reference: jump::step_until_below_cap).
//   * J*: exact W-step jumps on the value x = T^j(n) itself (CPU reference:
//     jump::jump_until_below).  T^W(2^W q + r) = 3^f(r) q + T^W(r); one shared
//     memory lookup per W steps.  Before each jump x >= 2^(cutoff-K) flags the
//     case (2^K >= 1.5^W), so no intermediate value of an unflagged jump can
//     reach 2^cutoff and 128-bit arithmetic cannot wrap.  The descent test
//     x < n follows every jump after the first `skip` (KParams.tail_budget).
#pragma once

#include <map>
#include <mutex>
#include <vector>

enum XLayout : int { XL_NONE = 0, XL_PACKED = 1, XL_FD = 2, XL_U64 = 3 };

template<int TV> struct XTraits {
    static constexpr bool valid = false;
};
#define XT_BASE(CODE, SHPOW_, OVF1_, FAST3_, CAP_) template<> struct XTraits<CODE> { \
    static constexpr bool valid = true, jump = false, shpow = SHPOW_, ovf1 = OVF1_, fast3 = FAST3_; \
    static constexpr int cap = CAP_, w = 0, layout = XL_NONE; static constexpr bool three = false, skip = false; \
    static constexpr bool opt = false; static constexpr int rep = 1; static constexpr bool pref = false; static constexpr int prefix = 0; };
#define XT_JUMP(CODE, W_, LAYOUT_, THREE_, SKIP_) template<> struct XTraits<CODE> { \
    static constexpr bool valid = true, jump = true, shpow = false, ovf1 = false, fast3 = false; \
    static constexpr int cap = 8, w = W_, layout = LAYOUT_; static constexpr bool three = THREE_, skip = SKIP_; \
    static constexpr bool opt = false; static constexpr int rep = 1; static constexpr bool pref = false; static constexpr int prefix = 0; };
// Optimized jump loop: coarse 64-bit descent test, do-while guard, add-carry chain.
#define XT_JOPT(CODE, W_, LAYOUT_, THREE_, SKIP_, REP_) template<> struct XTraits<CODE> { \
    static constexpr bool valid = true, jump = true, shpow = false, ovf1 = false, fast3 = false; \
    static constexpr int cap = 8, w = W_, layout = LAYOUT_; static constexpr bool three = THREE_, skip = SKIP_; \
    static constexpr bool opt = true; static constexpr int rep = REP_; static constexpr bool pref = false; static constexpr int prefix = 0; };

XT_BASE(1, false, false, false, 8)   // original loop (sanity: must equal pad 5)
XT_BASE(2, true,  false, false, 8)   // powers of three from shared memory
XT_BASE(3, false, true,  false, 8)   // overflow tested once per double iteration
XT_BASE(4, false, true,  true,  8)   // + 3-limb double iterations below 2^86
XT_BASE(5, true,  true,  true,  8)   // + shared powers
XT_BASE(6, false, false, false, 5)   // odd runs capped at 5, powers by PRMT
XT_BASE(7, false, true,  true,  5)   // cap 5 + once-per-double check + 3 limbs
XT_JUMP(16, 10, XL_PACKED, false, false)
XT_JUMP(17, 10, XL_PACKED, true,  false)
XT_JUMP(18, 10, XL_PACKED, true,  true)
XT_JUMP(19,  8, XL_PACKED, true,  false)
XT_JUMP(20, 11, XL_FD,     true,  false)
XT_JUMP(21, 12, XL_FD,     true,  false)
XT_JUMP(22, 13, XL_FD,     true,  false)
XT_JUMP(23, 14, XL_FD,     true,  false)
XT_JUMP(24, 12, XL_U64,    true,  false)
XT_JUMP(25, 13, XL_FD,     true,  true)
XT_JUMP(26, 12, XL_FD,     true,  true)
XT_JUMP(27, 11, XL_FD,     true,  true)
XT_JUMP(28,  9, XL_PACKED, true,  false)
XT_JUMP(29, 10, XL_FD,     true,  false)
XT_JUMP(30, 13, XL_FD,     false, false)
XT_JUMP(31, 12, XL_U64,    true,  true)
XT_JOPT(32, 10, XL_PACKED, true,  false, 1)
XT_JOPT(33, 11, XL_FD,     true,  false, 1)
XT_JOPT(34, 12, XL_FD,     true,  false, 1)
XT_JOPT(35, 12, XL_U64,    true,  false, 1)
XT_JOPT(36, 10, XL_PACKED, true,  true,  1)
XT_JOPT(37, 11, XL_FD,     true,  true,  1)
XT_JOPT(38, 12, XL_FD,     true,  true,  1)
XT_JOPT(39, 12, XL_U64,    true,  true,  1)
XT_JOPT(40, 13, XL_FD,     false, false, 1)
XT_JOPT(41, 10, XL_PACKED, false, false, 1)
XT_JOPT(42, 11, XL_FD,     false, false, 1)
XT_JOPT(43, 12, XL_U64,    false, false, 1)
XT_JOPT(44, 10, XL_PACKED, true,  false, 4)
XT_JOPT(45, 13, XL_FD,     true,  false, 1)
XT_JOPT(46, 13, XL_FD,     true,  true,  1)
XT_JOPT(47, 11, XL_FD,     true,  false, 4)
#undef XT_BASE
#undef XT_JUMP
#undef XT_JOPT
// Prefetching versions: the completion-word scan carries the high bitvector
// word and loads the next one before the current word's cases run.
template<int BASE_TV> struct XPref : XTraits<BASE_TV> { static constexpr bool pref = true; };
#define XT_PREF(CODE, BASE) template<> struct XTraits<CODE> : XPref<BASE> {};
// Unrolled prefix of P jumps without guard or descent test (entry check
// x < 2^(96 - P*K)); descent tests start after jump P (CPU: fixed skip P).
template<int BASE_TV, int P, bool PREF> struct XPrefix : XTraits<BASE_TV> {
    static constexpr bool skip = true, pref = PREF; static constexpr int prefix = P; };
#define XT_PFX(CODE, BASE, P, PREF) template<> struct XTraits<CODE> : XPrefix<BASE, P, PREF> {};
XT_PREF(48, 1)
XT_PREF(49, 32)
XT_PREF(50, 33)
XT_PREF(51, 34)
XT_PREF(52, 35)
XT_PREF(53, 43)
XT_PREF(54, 40)
XT_PREF(55, 42)
XT_PFX(56, 35, 2, false)
XT_PFX(57, 34, 2, false)
XT_PFX(58, 33, 2, false)
XT_PFX(59, 32, 2, false)
XT_PFX(60, 35, 2, true)
XT_PFX(61, 33, 2, true)
XT_PFX(62, 34, 2, true)
#undef XT_PREF
#undef XT_PFX

// Smallest K with 2^(K+W) >= 3^W.
__host__ __device__ constexpr int x_growth_bits(int w) {
    unsigned long long p = 1;
    for (int i = 0; i < w; i++) p *= 3;
    int k = 0;
    while ((1ull << (k + w)) < p) k++;
    return k;
}

template<int TV> __host__ __device__ constexpr u32 x_table_bytes() {
    using X = XTraits<TV>;
    if constexpr (X::jump) {
        if constexpr (X::layout == XL_PACKED) return (4u * X::rep) << X::w;
        else if constexpr (X::layout == XL_FD) return ((4u * X::rep) << X::w) + 64u;
        else return (8u * X::rep) << X::w;
    } else if constexpr (X::shpow) {
        return 64u;
    } else {
        return 0u;
    }
}

// Host table images (u32 words), identical to jump::JumpTable on the CPU.
static std::vector<u32> x_build_table1(int tv, int w, int layout, bool shpow);
// Replicated layout: word (i * rep + c) holds entry i for lanes with lane % rep == c,
// so lanes of different residue classes mod rep use disjoint bank sets.
static std::vector<u32> x_build_table(int tv, int w, int layout, bool shpow, int rep) {
    std::vector<u32> one = x_build_table1(tv, w, layout, shpow);
    if (rep == 1 || layout == XL_NONE) return one;
    const u32 size = 1u << w;
    const u32 ew = layout == XL_U64 ? 2 : 1;
    std::vector<u32> out((size_t)size * ew * rep + (one.size() - (size_t)size * ew));
    for (u32 i = 0; i < size; i++)
        for (int c = 0; c < rep; c++)
            for (u32 k = 0; k < ew; k++) out[((size_t)i * rep + c) * ew + k] = one[(size_t)i * ew + k];
    for (size_t k = (size_t)size * ew; k < one.size(); k++) out[(size_t)size * ew * rep + (k - (size_t)size * ew)] = one[k];
    return out;
}

static std::vector<u32> x_build_table1(int tv, int w, int layout, bool shpow) {
    std::vector<u32> out;
    if (layout == XL_NONE) {
        if (shpow) {
            out.assign(16, 0);
            u32 p = 1;
            for (int i = 0; i < 9; i++) { out[i] = p; p *= 3; }
        }
        return out;
    }
    const u32 size = 1u << w;
    std::vector<u32> mult(size), add(size), odd(size);
    for (u32 r = 0; r < size; r++) {
        u64 x = r;
        u32 f = 0;
        for (int i = 0; i < w; i++) {
            if (x & 1) { x = (3 * x + 1) / 2; f++; } else { x /= 2; }
        }
        u32 p = 1;
        for (u32 i = 0; i < f; i++) p *= 3;
        mult[r] = p; add[r] = (u32)x; odd[r] = f;
    }
    if (layout == XL_PACKED) {
        out.resize(size);
        for (u32 r = 0; r < size; r++) {
            if (mult[r] > 0xFFFFu || add[r] > 0xFFFFu) { fprintf(stderr, "packed jump table overflow\n"); abort(); }
            out[r] = (mult[r] << 16) | add[r];
        }
    } else if (layout == XL_FD) {
        out.resize(size + 16);
        for (u32 r = 0; r < size; r++) {
            if (add[r] >= (1u << 25) || odd[r] > 31) { fprintf(stderr, "fd jump table overflow\n"); abort(); }
            out[r] = add[r] | ((4u * odd[r]) << 25);
        }
        u32 p = 1;
        for (int i = 0; i < 16; i++) { out[size + i] = i <= w ? p : 0; p *= 3; }
    } else {
        out.resize(2 * size);
        for (u32 r = 0; r < size; r++) { out[2 * r] = mult[r]; out[2 * r + 1] = add[r]; }
    }
    (void)tv;
    return out;
}

// c3-hyg: the table cache is the launching context's (cuda/devtables.cuh),
// not process-global.  Returns nullptr (and the launch is refused) when the
// upload fails.
static const u32* x_device_table(DevTables& tabs, int tv, int w, int layout, bool shpow, int rep, bool& ok) {
    std::lock_guard<std::mutex> guard(tabs.lock);
    ok = true;
    auto it = tabs.xtail.find(tv);
    if (it != tabs.xtail.end()) return it->second;
    const auto host = x_build_table(tv, w, layout, shpow, rep);
    u32* device = nullptr;
    if (!host.empty()) {
        if (cudaMalloc(&device, host.size() * 4) != cudaSuccess ||
            cudaMemcpy(device, host.data(), host.size() * 4, cudaMemcpyHostToDevice) != cudaSuccess) {
            fprintf(stderr, "xtail table upload failed\n");
            cudaFree(device);
            ok = false;
            return nullptr;
        }
    }
    tabs.xtail[tv] = device;
    return device;
}

// ---------------------------------------------------------------------------
// Device helpers.

__device__ __forceinline__ u32 x_pow3_prmt5(u32 a) {
    // bytes 0..5 = 1,3,9,27,81,243; selector nibbles 1..3 pick byte 7 = 0
    return __byte_perm(0x1B090301u, 0x0000F351u, 0x7770u | a);
}

template<int TV>
__device__ __forceinline__ u32 x_pow3(u32 a, const u32* s_pow) {
    using X = XTraits<TV>;
    if constexpr (X::cap == 5) return x_pow3_prmt5(a);
    else if constexpr (X::shpow) return s_pow[a];
    else return THREEPOWERSU16[a];
}

// One original half (odd run + even run) on 4 limbs, with the cutoff test.
template<int TV>
__device__ __forceinline__ void x_half4(u32& n0, u32& n1, u32& n2, u32& n3, const u32* s_pow,
                                        bool check, u32 cutoff_hi, bool& over) {
    using X = XTraits<TV>;
    const u32 a = __ffs(n0 | (1u << X::cap)) - 1;
    const u32 mm = x_pow3<TV>(a, s_pow);
    u32 t0 = __funnelshift_r(n0, n1, a);
    u64 tt = (u64)t0 * mm;
    n0 = (u32)tt; u32 carry = tt >> 32;
    t0 = __funnelshift_r(n1, n2, a);
    tt = (u64)t0 * mm + carry;
    n1 = (u32)tt; carry = tt >> 32;
    t0 = __funnelshift_r(n2, n3, a);
    tt = (u64)t0 * mm + carry;
    n2 = (u32)tt; carry = tt >> 32;
    n3 = (n3 >> a) * mm + carry;
    if (check && n3 >= cutoff_hi) {
        over = true;
        n0 = 2; n1 = 0; n2 = 0; n3 = 0;
    }
    n0 -= 1;
    const u32 b = __ffs(n0 | (1u << 31)) - 1;
    n0 = __funnelshift_r(n0, n1, b) + 1;
    n1 = __funnelshift_r(n1, n2, b);
    n2 = __funnelshift_r(n2, n3, b);
    n3 = n3 >> b;
    if (n0 == 0 && b > 0) {
        n1 += 1;
        if (n1 == 0) {
            n2 += 1;
            if (n2 == 0) n3 += 1;
        }
    }
}

// The same half on 3 limbs; valid while the value stays below 2^96.
template<int TV>
__device__ __forceinline__ void x_half3(u32& n0, u32& n1, u32& n2, const u32* s_pow) {
    using X = XTraits<TV>;
    const u32 a = __ffs(n0 | (1u << X::cap)) - 1;
    const u32 mm = x_pow3<TV>(a, s_pow);
    u32 t0 = __funnelshift_r(n0, n1, a);
    u64 tt = (u64)t0 * mm;
    n0 = (u32)tt; u32 carry = tt >> 32;
    t0 = __funnelshift_r(n1, n2, a);
    tt = (u64)t0 * mm + carry;
    n1 = (u32)tt; carry = tt >> 32;
    n2 = (n2 >> a) * mm + carry;
    n0 -= 1;
    const u32 b = __ffs(n0 | (1u << 31)) - 1;
    n0 = __funnelshift_r(n0, n1, b) + 1;
    n1 = __funnelshift_r(n1, n2, b);
    n2 = n2 >> b;
    if (n0 == 0 && b > 0) {
        n1 += 1;
        if (n1 == 0) n2 += 1;
    }
}

__device__ __forceinline__ bool x_below4(u32 x0, u32 x1, u32 x2, u32 x3, u32 b0, u32 b1, u32 b2, u32 b3) {
    return (x3 < b3) | (x3 == b3 && x2 < b2) | (x3 == b3 && x2 == b2 && x1 < b1)
         | (x3 == b3 && x2 == b2 && x1 == b1 && x0 < b0);
}

__device__ __forceinline__ bool x_below3(u32 x0, u32 x1, u32 x2, u32 b0, u32 b1, u32 b2) {
    return (x2 < b2) | (x2 == b2 && x1 < b1) | (x2 == b2 && x1 == b1 && x0 < b0);
}

// Decode a jump entry: multiplier 3^f and additive T^W(r).
template<int TV>
__device__ __forceinline__ void x_entry(const u32* s_tab, u32 x0, u32& m, u32& d) {
    using X = XTraits<TV>;
    constexpr u32 mask = (1u << X::w) - 1;
    if constexpr (X::layout == XL_PACKED) {
        const u32 e = s_tab[x0 & mask];
        m = e >> 16;
        d = e & 0xFFFFu;
    } else if constexpr (X::layout == XL_FD) {
        const u32 e = s_tab[x0 & mask];
        d = e & 0x01FFFFFFu;
        m = *reinterpret_cast<const u32*>(reinterpret_cast<const char*>(s_tab + (1u << X::w)) + (e >> 25));
    } else {
        const uint2 e = reinterpret_cast<const uint2*>(s_tab)[x0 & mask];
        m = e.x;
        d = e.y;
    }
}

template<int TV>
__device__ __forceinline__ void x_jump4(const u32* s_tab, u32& x0, u32& x1, u32& x2, u32& x3) {
    constexpr int W = XTraits<TV>::w;
    u32 m, d;
    x_entry<TV>(s_tab, x0, m, d);
    const u32 t0 = __funnelshift_r(x0, x1, W);
    const u32 t1 = __funnelshift_r(x1, x2, W);
    const u32 t2 = __funnelshift_r(x2, x3, W);
    const u32 t3 = x3 >> W;
    u64 pr = (u64)t0 * m + d;
    x0 = (u32)pr;
    pr = (u64)t1 * m + (pr >> 32);
    x1 = (u32)pr;
    pr = (u64)t2 * m + (pr >> 32);
    x2 = (u32)pr;
    x3 = t3 * m + (u32)(pr >> 32);
}

template<int TV>
__device__ __forceinline__ void x_jump3(const u32* s_tab, u32& x0, u32& x1, u32& x2) {
    constexpr int W = XTraits<TV>::w;
    u32 m, d;
    x_entry<TV>(s_tab, x0, m, d);
    const u32 t0 = __funnelshift_r(x0, x1, W);
    const u32 t1 = __funnelshift_r(x1, x2, W);
    const u32 t2 = x2 >> W;
    u64 pr = (u64)t0 * m + d;
    x0 = (u32)pr;
    pr = (u64)t1 * m + (pr >> 32);
    x1 = (u32)pr;
    x2 = t2 * m + (u32)(pr >> 32);
}

// Optimized entry decode with replicated tables (lane offset c = lane % rep).
template<int TV>
__device__ __forceinline__ void x_entry_o(const u32* s_tab, u32 x0, u32 c, u32& m, u32& d) {
    using X = XTraits<TV>;
    constexpr u32 mask = (1u << X::w) - 1;
    if constexpr (X::layout == XL_PACKED) {
        const u32 e = s_tab[(x0 & mask) * X::rep + c];
        m = e >> 16;
        d = e & 0xFFFFu;
    } else if constexpr (X::layout == XL_FD) {
        const u32 e = s_tab[(x0 & mask) * X::rep + c];
        d = e & 0x01FFFFFFu;
        m = *reinterpret_cast<const u32*>(reinterpret_cast<const char*>(s_tab + (X::rep << X::w)) + (e >> 25));
    } else {
        const uint2 e = reinterpret_cast<const uint2*>(s_tab)[(x0 & mask) * X::rep + c];
        m = e.x;
        d = e.y;
    }
}

// x = (x >> W) * m + d on three limbs: two independent wide products and an
// explicit carry chain (no 64-bit addend pairs to assemble).
template<int TV>
__device__ __forceinline__ void x_jump3o(const u32* s_tab, u32 c, u32& x0, u32& x1, u32& x2) {
    constexpr int W = XTraits<TV>::w;
    u32 m, d;
    x_entry_o<TV>(s_tab, x0, c, m, d);
    const u32 t0 = __funnelshift_r(x0, x1, W);
    const u32 t1 = __funnelshift_r(x1, x2, W);
    const u32 t2 = x2 >> W;
    const u64 p0 = (u64)t0 * m;
    const u64 p1 = (u64)t1 * m;
    const u32 q2 = t2 * m;
    asm("add.cc.u32 %0, %3, %4;\n\t"
        "addc.cc.u32 %1, %5, %6;\n\t"
        "addc.u32 %2, %7, %8;"
        : "=r"(x0), "=r"(x1), "=r"(x2)
        : "r"((u32)p0), "r"(d), "r"((u32)p1), "r"((u32)(p0 >> 32)), "r"(q2), "r"((u32)(p1 >> 32)));
}

template<int TV>
__device__ __forceinline__ void x_jump4o(const u32* s_tab, u32 c, u32& x0, u32& x1, u32& x2, u32& x3) {
    constexpr int W = XTraits<TV>::w;
    u32 m, d;
    x_entry_o<TV>(s_tab, x0, c, m, d);
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

__device__ __forceinline__ void x_flag(KOut* out, u32* flags, const KParams& p,
                                       u32 b0, u32 b1, u32 b2, u32 b3) {
    const u32 slot = atomicAdd(&out->flags, 1u);
    if (slot < p.flag_cap) {
        flags[4 * slot] = b0;
        flags[4 * slot + 1] = b1;
        flags[4 * slot + 2] = b2;
        flags[4 * slot + 3] = b3;
    }
}

template<int MODE, int TV>
__global__ void cv_kernel_x(const u32* __restrict__ bv, const u32* __restrict__ mod9,
                            const u32* __restrict__ pairs,
                            u32 n, u64 magic, u32 tp_0, u32 tp_1, u32 tp_2, u32 tp_3,
                            KParams p, KOut* out, u32* flags, const u32* __restrict__ xtab) {
    using X = XTraits<TV>;
    static_assert(X::valid, "unknown tail variant");
    extern __shared__ __align__(16) u32 s_tab[];
    constexpr u32 TABLE_BYTES = x_table_bytes<TV>();
    if constexpr (TABLE_BYTES > 0) {
        const uint4* src = reinterpret_cast<const uint4*>(xtab);
        uint4* dst = reinterpret_cast<uint4*>(s_tab);
        for (u32 i = threadIdx.x; i < TABLE_BYTES / 16; i += blockDim.x) dst[i] = __ldg(src + i);
        __syncthreads();
    }
    const u32* s_pow = s_tab;

    u32 rolling_hash = 0;
    u32 cases = 0;
    u64 iters = 0;

    // Grid-stride loop: a full grid runs one endpoint per thread; a persistent
    // grid (pad bit 23) amortizes the shared-table load over many endpoints.
    for (u32 idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += gridDim.x * blockDim.x) {
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

        u64 lo, hi;
        if constexpr (MODE == 2) {
            const uint4 iterate = reinterpret_cast<const uint4*>(pairs)[(size_t)idx * 2 + 1];
            lo = (u64)iterate.x | ((u64)iterate.y << 32);
            hi = (u64)iterate.z | ((u64)iterate.w << 32);
        } else {
            lo = (u64)base0_0 | ((u64)base0_1 << 32);
            hi = (u64)base0_2 | ((u64)base0_3 << 32);
            for (u32 j = 0; j < p.m; j++) {
                if (lo & 1) {
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
        // Jump tails start from T^M(n0) itself; the original loop from +1.
        u64 slo = lo, shi = hi;
        u64 lo1 = lo + 1;
        u64 hi1 = hi + (lo1 == 0);
        if constexpr (!X::jump) { slo = lo1; shi = hi1; }
        const u32 s0 = (u32)slo, s1 = (u32)(slo >> 32), s2 = (u32)shi, s3 = (u32)(shi >> 32);
        const u32 number0_0 = (u32)lo1;

        u32 redmod9 = (base0_0 & 0x3FFFF) + (base0_0 >> 18)
                    + 4 * (base0_1 & 0x3FFFF) + 4 * (base0_1 >> 18)
                    + 7 * (base0_2 & 0x3FFFF) + 7 * (base0_2 >> 18)
                    + (base0_3 & 0x3FFFF) + (base0_3 >> 18);
        redmod9 %= 9;

        const u64 index = (magic * lo1) & p.b_mask;
        const u32 offset = (u32)(index & 31);
        const u32* bvw = bv + family * p.fam_stride + (index >> 5);
        const u32* m9 = mod9 + p.words32 * redmod9;

        u32 oee_mask = 0xFFFFFFFFu;
        if (p.tail_oee) {
            if (oee == 2) {
                oee_mask = (number0_0 & 1) ? 0xAAAAAAAAu : 0x55555555u;
            } else if (oee == 1) {
                const u32 forbidden = ((1u - number0_0) * tp_0) & 3u;
                oee_mask = ~(0x11111111u << forbidden);
            }
        }
        // Jump guards: 3-limb loop while x < 2^(96-K); 4-limb flag at 2^(cutoff-K).
        constexpr int K = X::jump ? x_growth_bits(X::w) : 0;
        const u32 guard_hi = p.cutoff_hi >> K;
        const u32 skip = X::prefix > 0 ? (u32)X::prefix : X::skip ? p.tail_budget : 0u;

        u32 w = 0;
        u32 bv_lo = bvw[0], bv_hi = bvw[1];
        u32 bv_next = 0;
        if constexpr (X::pref) { if (p.words32 > 1) bv_next = bvw[2]; }
        u32 temp = m9[0] & __funnelshift_r(bv_lo, bv_hi, offset) & oee_mask;
        while (true) {
            while (temp == 0) {
                if (++w == p.words32) goto next_endpoint;
                if constexpr (X::pref) {
                    // Window words [0, words32] exist; bvw[w + 2] only while w + 2 <= words32.
                    bv_lo = bv_hi;
                    bv_hi = bv_next;
                    if (w + 2 <= p.words32) bv_next = bvw[w + 2];
                    temp = m9[w] & __funnelshift_r(bv_lo, bv_hi, offset) & oee_mask;
                } else {
                    temp = m9[w] & __funnelshift_r(bvw[w], bvw[w + 1], offset) & oee_mask;
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

            if constexpr (X::jump && X::opt) {
                // Descent test D: if n >= 2^32, (x >> 32) < (n >> 32); else x < n.
                // Both imply x < n.  CPU reference: jump::descends_coarse.
                const u64 bh = ((u64)base_2 << 32) | base_1;
                const bool small_base = (bh == 0) | (base_3 != 0);
                const u32 lane_c = X::rep > 1 ? (threadIdx.x % X::rep) : 0u;
                u32 jumps = 0;
                bool finished = false;
                if constexpr (X::prefix > 0) {
                    static_assert(X::three && 32 - X::prefix * K > 0, "prefix needs 3-limb headroom");
                    if (num_3 == 0 && !small_base && num_2 < (1u << (32 - X::prefix * K))) {
                        // P jumps from x < 2^(96-PK) stay below 2^96: no guard needed.
#pragma unroll
                        for (int j = 0; j < X::prefix; j++) {
                            x_jump3o<TV>(s_tab, lane_c, num_0, num_1, num_2);
                            rolling_hash ^= num_0;
                        }
                        jumps = X::prefix;
                    }
                }
                if constexpr (X::three) {
                    constexpr u32 LIM3 = 1u << (32 - K);
                    if (num_3 == 0 && !small_base && num_2 < LIM3) {
                        if constexpr (X::skip) {
                            // No descent test during the first `skip` jumps.
                            while (jumps < skip) {
                                x_jump3o<TV>(s_tab, lane_c, num_0, num_1, num_2);
                                rolling_hash ^= num_0;
                                jumps++;
                                if (num_2 >= LIM3) break;
                            }
                        }
                        if (num_2 < LIM3) {
                            do {
                                x_jump3o<TV>(s_tab, lane_c, num_0, num_1, num_2);
                                rolling_hash ^= num_0;
                                jumps++;
                                if ((((u64)num_2 << 32) | num_1) < bh && (!X::skip || jumps > skip)) {
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
                            if (num_3 == 0 && (((u64)num_2 << 32) | num_1) < bh && (!X::skip || jumps > skip)) {
                                finished = true;
                                break;
                            }
                        }
                    } else {
                        while (num_3 < guard_hi) {
                            x_jump4o<TV>(s_tab, lane_c, num_0, num_1, num_2, num_3);
                            rolling_hash ^= num_0;
                            jumps++;
                            if (x_below4(num_0, num_1, num_2, num_3, base_0, base_1, base_2, base_3) && (!X::skip || jumps > skip)) {
                                finished = true;
                                break;
                            }
                        }
                    }
                    if (!finished) x_flag(out, flags, p, base_0, base_1, base_2, base_3);
                }
                iters += jumps;
            } else if constexpr (X::jump) {
                u32 jumps = 0;
                bool finished = false;
                if constexpr (X::three) {
                    if (num_3 == 0 && base_3 == 0) {
                        constexpr u32 LIM3 = 1u << (32 - K);
                        if constexpr (X::skip) {
                            while (jumps < skip && num_2 < LIM3) {
                                x_jump3<TV>(s_tab, num_0, num_1, num_2);
                                rolling_hash ^= num_0;
                                jumps++;
                            }
                        }
                        while (num_2 < LIM3) {
                            x_jump3<TV>(s_tab, num_0, num_1, num_2);
                            rolling_hash ^= num_0;
                            jumps++;
                            if (x_below3(num_0, num_1, num_2, base_0, base_1, base_2) && jumps > skip) {
                                finished = true;
                                break;
                            }
                        }
                    }
                }
                if (!finished) {
                    while (true) {
                        if (num_3 >= guard_hi) {
                            x_flag(out, flags, p, base_0, base_1, base_2, base_3);
                            break;
                        }
                        x_jump4<TV>(s_tab, num_0, num_1, num_2, num_3);
                        rolling_hash ^= num_0;
                        jumps++;
                        if (x_below4(num_0, num_1, num_2, num_3, base_0, base_1, base_2, base_3) && jumps > skip)
                            break;
                    }
                }
                iters += jumps;
            } else {
                // Original double iterations; fast variants are work-preserving.
                const u32 safe_hi = p.cutoff_hi >> 10;  // one double iteration grows y < 2^9.36
                while (true) {
                    bool over = false;
                    if constexpr (X::fast3) {
                        // y < 2^86: both halves stay below 2^96 < 2^cutoff.
                        if (num_3 == 0 && num_2 < (1u << 22)) {
                            x_half3<TV>(num_0, num_1, num_2, s_pow);
                            x_half3<TV>(num_0, num_1, num_2, s_pow);
                            rolling_hash ^= num_0;
                            iters += 1;
                            if (base_3 != 0 || x_below3(num_0, num_1, num_2, base_0, base_1, base_2)) break;
                            continue;
                        }
                    }
                    const bool check = X::ovf1 ? (num_3 >= safe_hi) : true;
                    x_half4<TV>(num_0, num_1, num_2, num_3, s_pow, check, p.cutoff_hi, over);
                    bool over2 = false;
                    x_half4<TV>(num_0, num_1, num_2, num_3, s_pow, check, p.cutoff_hi, over2);
                    if (over || over2) {
                        // The original records each crossing; a second crossing
                        // within one double iteration is impossible after reset.
                        x_flag(out, flags, p, base_0, base_1, base_2, base_3);
                    }
                    rolling_hash ^= num_0;
                    iters += 1;
                    if (x_below4(num_0, num_1, num_2, num_3, base_0, base_1, base_2, base_3)) break;
                }
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

template<int MODE, int TV>
static bool x_launch_one(DevTables& tabs, const u32* bv, const u32* mod9, const u32* input, u32 n, u64 magic, const u32* tp,
                         KParams p, KOut* out, u32* flags, u32 blocks, u32 threads, cudaStream_t stream) {
    using X = XTraits<TV>;
    constexpr u32 bytes = x_table_bytes<TV>();
    bool ok = true;
    const u32* table = x_device_table(tabs, TV, X::w, X::layout, X::shpow, X::rep, ok);
    if (!ok) return false;
    const void* kernel = (const void*)cv_kernel_x<MODE, TV>;
    if constexpr (bytes > 48 * 1024) {
        if (!devtables_big_smem(tabs, kernel, bytes)) return false;
    }
    if ((p.pad >> 23) & 1u) {
        // Persistent grid: at most the resident blocks of this context's device.
        const int resident = devtables_resident(tabs, kernel, threads, bytes);
        const u32 cap = (u32)(resident > 0 ? resident : 1) * (u32)(tabs.sms > 0 ? tabs.sms : 1);
        if (blocks > cap) blocks = cap;
    }
    cv_kernel_x<MODE, TV><<<blocks, threads, bytes, stream>>>(bv, mod9, input, n, magic, tp[0], tp[1], tp[2], tp[3],
                                                               p, out, flags, table);
    return true;
}

static bool x_variant_valid(u32 tv) {
    return (tv >= 1 && tv <= 7) || (tv >= 16 && tv <= 62);
}

// Jump guards need cutoff >= 96 + K (3-limb loop below 2^(96-K) must not reach
// the flag threshold, and the 4-limb guard limb cutoff_hi >> K must be nonzero).
static int x_variant_width(u32 tv) {
#define XW(CODE) case CODE: return XTraits<CODE>::jump ? XTraits<CODE>::w : 0;
    switch (tv) {
        XW(1) XW(2) XW(3) XW(4) XW(5) XW(6) XW(7)
        XW(16) XW(17) XW(18) XW(19) XW(20) XW(21) XW(22) XW(23) XW(24) XW(25) XW(26) XW(27)
        XW(28) XW(29) XW(30) XW(31) XW(32) XW(33) XW(34) XW(35) XW(36) XW(37) XW(38) XW(39)
        XW(40) XW(41) XW(42) XW(43) XW(44) XW(45) XW(46) XW(47) XW(48) XW(49) XW(50) XW(51)
        XW(52) XW(53) XW(54) XW(55) XW(56) XW(57) XW(58) XW(59) XW(60) XW(61) XW(62)
        default: return -1;
    }
#undef XW
}

static bool x_cutoff_ok(u32 tv, u32 cutoff_hi) {
    const int w = x_variant_width(tv);
    if (w < 0) return false;
    if (w == 0) return cutoff_hi != 0;
    return (cutoff_hi >> x_growth_bits(w)) != 0;
}

// Returns false for an unknown variant code.
template<int MODE>
static bool x_launch(DevTables& tabs, u32 tv, const u32* bv, const u32* mod9, const u32* input, u32 n, u64 magic, const u32* tp,
                     KParams p, KOut* out, u32* flags, u32 blocks, u32 threads, cudaStream_t stream) {
#define XL(CODE) case CODE: return x_launch_one<MODE, CODE>(tabs, bv, mod9, input, n, magic, tp, p, out, flags, blocks, threads, stream);
    switch (tv) {
        XL(1) XL(2) XL(3) XL(4) XL(5) XL(6) XL(7)
        XL(16) XL(17) XL(18) XL(19) XL(20) XL(21) XL(22) XL(23) XL(24) XL(25) XL(26) XL(27)
        XL(28) XL(29) XL(30) XL(31)
        XL(32) XL(33) XL(34) XL(35) XL(36) XL(37) XL(38) XL(39) XL(40) XL(41) XL(42) XL(43) XL(44)
        XL(45) XL(46) XL(47) XL(48) XL(49) XL(50) XL(51) XL(52) XL(53) XL(54) XL(55)
        XL(56) XL(57) XL(58) XL(59) XL(60) XL(61) XL(62)
        default: return false;
    }
#undef XL
}
