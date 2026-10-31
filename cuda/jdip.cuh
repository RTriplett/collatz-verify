// c-tail: exact W-step jump tails with a window-dip certificate tested at
// every jump boundary ("jd" variants, KParams.pad bits 24..31 = code >= 64).
// CPU reference: src/jdip.rs (cv::jdip::jump_dip_until_k).
//
// State x = T^j(n) (3 or 4 u32 limbs), per-lane H = 485 F' - 306 M' (the
// bitvector cutoff at the current depth; H0 = 485 F - 306 M arrives in
// KParams.tail_budget).  At every boundary, before jump j:
//   * x >= 2^(cutoff - K) flags the case (4-limb loop guard; 2^K >= 1.5^W);
//   * the window that starts KA jumps later is tested:
//       KA = 0: dip[x mod 2^WD]                     > H
//       KA = 1: dip[x_{j+1} mod 2^WD]                > H + g_j
//       KA = 2: dip[x_{j+2} mod 2^WD]                > H + g_j + g_{j+1}
//     where x_{j+1} mod 2^32 is the low limb of the jump itself
//     (m_j * ((x >> W) mod 2^32) + d_j) and x_{j+2} mod 2^(32 - W) is one
//     32-bit mini jump from it (exact for WD <= 32 - W);
//   * otherwise x = 3^f(r) (x >> W) + T^W(r), H += g = 485 f(r) - 306 W,
//     hash ^= x mod 2^32, jumps += 1.
// A firing test is the bitvector's own exclusion criterion at that depth, so
// n descends or merges with a smaller positive partner (proof in jdip.rs and
// the c-tail README).  The dip test runs only for 2^32 <= n < 2^96 (n >=
// 99781 is the theorem's precondition); other cases take the plain jump loop
// with the exact descent test.
//
// Tables.  Jump table "FD2": u32 entry d | (8 f) << 25 for r < 2^W (d =
// T^W(r) < 2^25), followed by 16 uint2 (3^f, g_f) at byte offset 8 f, in
// shared memory.  Dip table: 2^WD entries, u16 (DQ = 1) or u8 = min(255,
// dip / DQ) (DQ a power of two), in shared memory (DLOC 0) or read through
// the L1/L2 with __ldg (DLOC 1); uploaded per context by cv_jd_upload
// (kernel.cu; quanta built by cv::jdip::DipTable, production window rules).
#pragma once

#include <map>
#include <mutex>
#include <vector>

template<int TV> struct JDTraits { static constexpr bool valid = false; };
// CODE, W, WD, DQ, DLOC, KA, THREE (3-limb fast loop), PREF (bitvector prefetch), DESC (descent-only dips)
#define JD_DEF(CODE, W_, WD_, DQ_, DLOC_, KA_, THREE_, PREF_, DESC_) template<> struct JDTraits<CODE> { \
    static constexpr bool valid = true; static constexpr int w = W_, wd = WD_, dq = DQ_, dloc = DLOC_, ka = KA_; \
    static constexpr bool three = THREE_, pref = PREF_, desc = DESC_; static constexpr int floor = 0, lay = 0; \
    static constexpr bool spec = false; };
// Embedded layout "FD3" (lay 1): the dip of the jump's own window sits in the
// jump entry: e = d | (dip / Q) << DB | (8 f) << 25 with DB = bits of d
// (20/18/16 for W = 12/11/10) and Q = 2^(DB - 13) (128/32/8), so the test is
// (e & DMASK) > H << 13 for every W; the pair table holds (3^f, g << 13).
// WD = W, dip table in the entry (DLOC unused).
template<int W> struct JdEmb { static constexpr int db = W == 12 ? 20 : W == 11 ? 18 : 16; static constexpr int q = 1 << (db - 13); };
#define JD_EMB(CODE, W_, KA_, THREE_, DESC_) template<> struct JDTraits<CODE> { \
    static constexpr bool valid = true; static constexpr int w = W_, wd = W_, dq = JdEmb<W_>::q, dloc = 0, ka = KA_; \
    static constexpr bool three = THREE_, pref = true, desc = DESC_; static constexpr int floor = 0, lay = 1; \
    static constexpr bool spec = false; };
// Speculative order (KA 1/2): the jump is computed before the dip test; a
// firing step ends the case without counting that jump (same semantics).
#define JD_EMBS(CODE, W_, KA_, THREE_) template<> struct JDTraits<CODE> { \
    static constexpr bool valid = true; static constexpr int w = W_, wd = W_, dq = JdEmb<W_>::q, dloc = 0, ka = KA_; \
    static constexpr bool three = THREE_, pref = true, desc = false; static constexpr int floor = 0, lay = 1; \
    static constexpr bool spec = true; };
JD_EMBS(106, 12, 1, true)
JD_EMBS(107, 12, 2, true)
JD_EMBS(108, 11, 1, true)
JD_EMBS(109, 11, 2, true)
JD_EMBS(110, 10, 2, true)
JD_EMBS(111, 12, 1, false)
JD_EMBS(112, 12, 2, false)
JD_EMB(90, 12, 0, true, false)
JD_EMB(91, 12, 1, true, false)
JD_EMB(92, 12, 2, true, false)
JD_EMB(93, 11, 0, true, false)
JD_EMB(94, 11, 1, true, false)
JD_EMB(95, 11, 2, true, false)
JD_EMB(96, 10, 0, true, false)
JD_EMB(97, 10, 1, true, false)
JD_EMB(98, 10, 2, true, false)
JD_EMB(99, 12, 1, false, false)
JD_EMB(100, 12, 2, false, false)
JD_EMB(101, 12, 1, true, true)    // descent-only dips (capture mode)
JD_EMB(102, 12, 2, true, true)
JD_EMB(103, 12, 0, false, false)  // 4-limb only, test at the boundary (high classes)
JD_EMB(104, 11, 0, false, false)
JD_EMB(105, 12, 0, true, true)    // descent-only, test at the boundary
// Measurement-only floors (NOT verification): FLOOR 1 = screening + per-case
// setup (case counted, num/base folded into the hash, no tail); FLOOR 2 = as 1
// plus the tail's first entry decode and one dip lookup.
#define JD_FLOOR(CODE, FL_) JD_FLOORW(CODE, FL_, 12)
#define JD_FLOORW(CODE, FL_, W_) template<> struct JDTraits<CODE> { \
    static constexpr bool valid = true; static constexpr int w = W_, wd = 12, dq = 16, dloc = 0, ka = 0; \
    static constexpr bool three = true, pref = true, desc = false; static constexpr int floor = FL_, lay = 0; \
    static constexpr bool spec = false; };
JD_FLOOR(126, 2)
JD_FLOOR(127, 1)
JD_FLOORW(125, 1, 10)   // as 127 with a 4 KB W10 table (smaller shared-memory footprint)
// --- W12, 3-limb + 4-limb, prefetch ---
JD_DEF(64, 12, 12, 16, 0, 0, true, true, false)   // dip12 u8 shared, test at the boundary
JD_DEF(65, 12, 12, 16, 0, 1, true, true, false)   // one window ahead
JD_DEF(66, 12, 12, 16, 0, 2, true, true, false)   // two windows ahead
JD_DEF(67, 12, 12,  1, 1, 2, true, true, false)   // dip12 u16 global
JD_DEF(68, 12, 16, 32, 1, 2, true, true, false)   // dip16 u8 global (64 KB)
JD_DEF(69, 12, 20, 32, 1, 2, true, true, false)   // dip20 u8 global (1 MB)
JD_DEF(70, 12, 20, 32, 1, 1, true, true, false)
JD_DEF(71, 12, 16, 32, 1, 1, true, true, false)
JD_DEF(72, 12, 24, 32, 1, 1, true, true, false)   // dip24 u8 global (16 MB), one ahead
JD_DEF(73, 12, 12, 16, 1, 2, true, true, false)   // dip12 u8 global
// --- W12, 4-limb only (high classes) ---
JD_DEF(76, 12, 12, 16, 0, 2, false, true, false)
JD_DEF(77, 12, 16, 32, 1, 2, false, true, false)
JD_DEF(78, 12, 20, 32, 1, 2, false, true, false)
// --- W11 / W10 ---
JD_DEF(80, 11, 12, 16, 0, 2, true, true, false)
JD_DEF(81, 11, 16, 32, 1, 2, true, true, false)
JD_DEF(82, 11, 20, 32, 1, 2, true, true, false)
JD_DEF(84, 10, 12, 16, 0, 2, true, true, false)
JD_DEF(85, 10, 16, 32, 1, 2, true, true, false)
JD_DEF(86, 10, 20, 32, 1, 2, true, true, false)
// --- descent-only dips (capture mode), W12 ---
JD_DEF(88, 12, 12, 16, 0, 2, true, true, true)
JD_DEF(89, 12, 20, 32, 1, 2, true, true, true)
#undef JD_DEF

#define JD_CODES(X) X(64) X(65) X(66) X(67) X(68) X(69) X(70) X(71) X(72) X(73) \
                    X(76) X(77) X(78) X(80) X(81) X(82) X(84) X(85) X(86) X(88) X(89) X(125) X(126) X(127) \
                    X(90) X(91) X(92) X(93) X(94) X(95) X(96) X(97) X(98) X(99) X(100) X(101) X(102) X(103) X(104) X(105) \
                    X(106) X(107) X(108) X(109) X(110) X(111) X(112)

__host__ __device__ constexpr int jd_log2(int q) { return q <= 1 ? 0 : 1 + jd_log2(q / 2); }

template<int TV> __host__ __device__ constexpr u32 jd_jtab_bytes() { return (4u << JDTraits<TV>::w) + 128u; }
template<int TV> __host__ __device__ constexpr u32 jd_dip_bytes() {
    using J = JDTraits<TV>;
    return (J::dq == 1 ? 2u : 1u) << J::wd;
}
template<int TV> __host__ __device__ constexpr u32 jd_smem_bytes() {
    return jd_jtab_bytes<TV>() + (JDTraits<TV>::dloc == 0 && JDTraits<TV>::lay == 0 ? jd_dip_bytes<TV>() : 0u);
}

// ---------------------------------------------------------------------------
// Host tables.

// FD2 jump table image (u32 words): 2^w entries then 16 x (3^f, g_f).
static std::vector<u32> jd_build_jtab(int w) {
    const u32 size = 1u << w;
    std::vector<u32> out(size + 32, 0);
    for (u32 r = 0; r < size; r++) {
        u64 x = r;
        u32 f = 0;
        for (int i = 0; i < w; i++) {
            if (x & 1) { x = (3 * x + 1) / 2; f++; } else { x /= 2; }
        }
        if (x >= (1u << 25) || f > 15) { fprintf(stderr, "jd jump table overflow\n"); abort(); }
        out[r] = (u32)x | ((8u * f) << 25);
    }
    u32 p = 1;
    for (int f = 0; f < 16; f++) {
        out[size + 2 * f] = f <= w ? p : 0;
        out[size + 2 * f + 1] = (u32)(485 * f - 306 * w);
        p *= 3;
    }
    return out;
}

// c3-hyg: the uploaded tables belong to one context (DevTables, cuda/devtables.cuh);
// there is no process-global cache any more.
//
// Dip tables arrive as u16 quanta `floor(dip / dq)` (dq = 1: the exact dip),
// already capped by the host (cv::jdip::DipTable) at the format's capacity:
//   * lay-0 tables (codes 64..89, 125..127): a u16 image for dq = 1, else a u8
//     image, so at most 255 quanta;
//   * FD3 keys (wd, dq) = (12, 128), (11, 32), (10, 8) (codes 90..112): the
//     embedded field of 25 - DB bits (31 / 127 / 511 quanta for W12 / W11 /
//     W10).  Until round 2 these were derived from the u8 image, which clamped
//     W10 dips (field 511) at 255 quanta.
// The FD3 image is built at upload time; a quantum above the format's
// capacity is refused (the upload fails), never truncated.

static bool jd_fd3_key(u32 wd, u32 dq) {
    return (wd == 12 && dq == 128) || (wd == 11 && dq == 32) || (wd == 10 && dq == 8);
}
static u32 jd_key(u32 wd, u32 dq, u32 desc) { return wd | (dq << 8) | (desc << 16); }
// Largest storable quantum of a dip table key (see above).
static u32 jd_quanta_cap(u32 wd, u32 dq) {
    if (jd_fd3_key(wd, dq)) {
        const u32 db = wd == 12 ? 20 : wd == 11 ? 18 : 16;
        return (1u << (25 - db)) - 1u;
    }
    return dq == 1 ? 0xFFFFu : 0xFFu;
}

// FD2 jump table of width w (no dips), built on first use per context.
static const u32* jd_device_jtab(DevTables& tabs, int w) {
    std::lock_guard<std::mutex> guard(tabs.lock);
    auto it = tabs.jd_jtab.find(w);
    if (it != tabs.jd_jtab.end()) return it->second;
    const auto host = jd_build_jtab(w);
    u32* device = nullptr;
    if (cudaMalloc(&device, host.size() * 4) != cudaSuccess ||
        cudaMemcpy(device, host.data(), host.size() * 4, cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "jd table upload failed\n");
        cudaFree(device);
        return nullptr;
    }
    tabs.jd_jtab[w] = device;
    return device;
}

// Lay-0 dip image of a key, or nullptr when it was never uploaded.
static const void* jd_device_dip(DevTables& tabs, int wd, int dq, bool desc) {
    std::lock_guard<std::mutex> guard(tabs.lock);
    auto it = tabs.jd_dip.find(jd_key((u32)wd, (u32)dq, (u32)desc));
    return it == tabs.jd_dip.end() ? nullptr : it->second;
}

// FD3 image of width w (quantum q), or nullptr when its dips were never uploaded.
static const u32* jd_device_emb(DevTables& tabs, int w, int q, bool desc) {
    std::lock_guard<std::mutex> guard(tabs.lock);
    auto it = tabs.jd_emb.find(jd_key((u32)w, (u32)q, (u32)desc));
    return it == tabs.jd_emb.end() ? nullptr : it->second;
}

// Upload one dip table (2^wd u16 quanta) into a context; replaces any previous
// table of the same key.  Returns 0, or a negative code without changing the
// context: -1 bad shape, -2 a quantum above the key's capacity, -3 allocation
// or copy failure, -4 FD3 jump field overflow.
static int jd_upload(DevTables& tabs, u32 wd, u32 dq, u32 desc, const unsigned short* quanta, u64 n) {
    if (!quanta || wd < 1 || wd > 24 || dq == 0 || (dq & (dq - 1)) != 0 || desc > 1 || n != (1ull << wd)) return -1;
    const u32 cap = jd_quanta_cap(wd, dq);
    for (u64 r = 0; r < n; r++) if (quanta[r] > cap) return -2;
    const u32 key = jd_key(wd, dq, desc);
    if (jd_fd3_key(wd, dq)) {
        const int w = (int)wd;
        const int db = w == 12 ? 20 : w == 11 ? 18 : 16;
        std::vector<u32> img = jd_build_jtab(w);
        const u32 size = 1u << w;
        for (u32 r = 0; r < size; r++) {
            const u32 d = img[r] & 0x01FFFFFFu;
            const u32 dip = quanta[r];
            if (d >= (1u << db) || dip >= (1u << (25 - db))) return -4;
            img[r] = (img[r] & 0xFE000000u) | d | (dip << db);
        }
        for (int f = 0; f < 16; f++) img[size + 2 * f + 1] = (u32)((int)img[size + 2 * f + 1] * 8192);
        u32* device = nullptr;
        if (cudaMalloc(&device, img.size() * 4) != cudaSuccess ||
            cudaMemcpy(device, img.data(), img.size() * 4, cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaFree(device);
            return -3;
        }
        std::lock_guard<std::mutex> guard(tabs.lock);
        auto it = tabs.jd_emb.find(key);
        if (it != tabs.jd_emb.end()) cudaFree(it->second);
        tabs.jd_emb[key] = device;
        return 0;
    }
    // Lay-0 image: u16 exact dips (dq = 1) or u8 quanta.
    std::vector<unsigned char> bytes;
    if (dq == 1) {
        bytes.resize(n * 2);
        for (u64 r = 0; r < n; r++) { bytes[2 * r] = (unsigned char)(quanta[r] & 0xFF); bytes[2 * r + 1] = (unsigned char)(quanta[r] >> 8); }
    } else {
        bytes.resize(n);
        for (u64 r = 0; r < n; r++) bytes[r] = (unsigned char)quanta[r];
    }
    void* device = nullptr;
    if (cudaMalloc(&device, bytes.size()) != cudaSuccess ||
        cudaMemcpy(device, bytes.data(), bytes.size(), cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(device);
        return -3;
    }
    std::lock_guard<std::mutex> guard(tabs.lock);
    auto it = tabs.jd_dip.find(key);
    if (it != tabs.jd_dip.end()) cudaFree(it->second);
    tabs.jd_dip[key] = device;
    return 0;
}

static bool jd_present(DevTables& tabs, u32 wd, u32 dq, u32 desc) {
    std::lock_guard<std::mutex> guard(tabs.lock);
    const u32 key = jd_key(wd, dq, desc);
    return jd_fd3_key(wd, dq) ? tabs.jd_emb.count(key) != 0 : tabs.jd_dip.count(key) != 0;
}

// ---------------------------------------------------------------------------
// Per-lane tail interface (composable: a warp-queue scheduler can call it).
//
//   JdTables<TV> t;  t.bind(s_jtab, dip)          table pointers (shared jump table)
//   JdLane<TV> s;    s.init(t, x0..x3, h0)        x = T^M(n), H0 = 485F - 306M
//   s.step<LIMBS>(t) -> true when the case is certified (stop)
//   s.flag_due(guard_hi)                           true when x >= 2^(cutoff-K): flag and stop
//   s.fits3()                                      true while the 3-limb step is valid
//   s.jumps, s.hash                                per-case counters (iterations, checksum)
// Only cases with 2^32 <= n < 2^96 may use step(); others need the exact
// descent loop (jd_plain_until below).

template<int TV> struct JdTables {
    const u32* jtab;    // shared: 2^W FD2/FD3 entries, then (3^f, g_f) pairs (FD3: g_f << 13)
    const void* dip;    // shared or global dip table (lay 0)
    __device__ __forceinline__ u32 entry(u32 r) const {
        constexpr u32 MJ = (1u << JDTraits<TV>::w) - 1;
        return jtab[r & MJ];
    }
    // FD3: split an entry into d, the scaled dip field, and the pair (3^f, g << 13).
    __device__ __forceinline__ void split(u32 e, u32& m, u32& d, int& g) const {
        constexpr int DB = JdEmb<JDTraits<TV>::w>::db;
        d = e & ((1u << DB) - 1u);
        const uint2 mg = *reinterpret_cast<const uint2*>(reinterpret_cast<const char*>(jtab + (1u << JDTraits<TV>::w)) + (e >> 25));
        m = mg.x;
        g = (int)mg.y;
    }
    __device__ __forceinline__ static int dipfield(u32 e) {
        constexpr int DB = JdEmb<JDTraits<TV>::w>::db;
        return (int)(e & (((1u << (25 - DB)) - 1u) << DB));
    }
    __device__ __forceinline__ void decode(u32 r, u32& m, u32& d, int& g) const {
        constexpr u32 MJ = (1u << JDTraits<TV>::w) - 1;
        const u32 e = jtab[r & MJ];
        if constexpr (JDTraits<TV>::lay == 1) { split(e, m, d, g); return; }
        d = e & 0x01FFFFFFu;
        const uint2 mg = *reinterpret_cast<const uint2*>(reinterpret_cast<const char*>(jtab + (1u << JDTraits<TV>::w)) + (e >> 25));
        m = mg.x;
        g = (int)mg.y;
    }
    // Dip of the window whose low WD bits are y, scaled back to 1/306-bit units.
    __device__ __forceinline__ int dip_of(u32 y) const {
        using J = JDTraits<TV>;
        constexpr u32 MD = (1u << J::wd) - 1;
        if constexpr (J::dq == 1) {
            const unsigned short* t = reinterpret_cast<const unsigned short*>(dip);
            if constexpr (J::dloc == 0) return (int)t[y & MD];
            else return (int)__ldg(t + (y & MD));
        } else {
            const unsigned char* t = reinterpret_cast<const unsigned char*>(dip);
            u32 v;
            if constexpr (J::dloc == 0) v = t[y & MD];
            else v = __ldg(t + (y & MD));
            return (int)(v << jd_log2(J::dq));
        }
    }
};

template<int TV> struct JdLane {
    u32 x0, x1, x2, x3;
    int h;              // lay 0: H; lay 1 (FD3): H << 13
    u32 m0, d0;
    int g0;
    u32 m1, d1;         // FD3, KA = 2: entry of jump j+1
    int g1;
    int dip0;           // FD3, KA = 0: dip field of the current window
    u32 jumps, hash;

    __device__ __forceinline__ void init(const JdTables<TV>& t, u32 a0, u32 a1, u32 a2, u32 a3, int h0) {
        using J = JDTraits<TV>;
        x0 = a0; x1 = a1; x2 = a2; x3 = a3;
        jumps = 0;
        hash = 0;
        if constexpr (J::lay == 1) {
            h = h0 * 8192;
            const u32 e = t.entry(x0);
            t.split(e, m0, d0, g0);
            dip0 = JdTables<TV>::dipfield(e);
            if constexpr (J::ka == 2) {
                const u32 y1 = m0 * __funnelshift_r(x0, x1, J::w) + d0;
                t.split(t.entry(y1), m1, d1, g1);
            }
        } else {
            h = h0;
            t.decode(x0, m0, d0, g0);
        }
    }
    __device__ __forceinline__ bool fits3() const {
        constexpr int K = x_growth_bits(JDTraits<TV>::w);
        return x3 == 0 && x2 < (1u << (32 - K));
    }
    __device__ __forceinline__ bool flag_due(u32 guard_hi) const { return x3 >= guard_hi; }

    // Full jump j with (m0, d0) given p0 = t0 * m0 (low 64 bits of the first product).
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

    // FD3 boundary (embedded dips): test, then (if not certified) one exact jump.
    template<int LIMBS>
    __device__ __forceinline__ bool step_emb(const JdTables<TV>& t) {
        using J = JDTraits<TV>;
        constexpr int W = J::w;
        const u32 t0 = __funnelshift_r(x0, x1, W);
        const u64 p0 = (u64)t0 * m0;
        const u32 y1 = (u32)p0 + d0;                 // x_{j+1} mod 2^32
        if constexpr (J::ka == 0) {
            if (dip0 > h) return true;
            const u32 e1 = t.entry(y1);
            finish<LIMBS>(p0);
            h += g0;
            t.split(e1, m0, d0, g0);
            dip0 = JdTables<TV>::dipfield(e1);
        } else if constexpr (J::ka == 1 && J::spec) {
            const u32 e1 = t.entry(y1);
            finish<LIMBS>(p0);
            const int h1 = h + g0;
            if (JdTables<TV>::dipfield(e1) > h1) return true;
            h = h1;
            t.split(e1, m0, d0, g0);
        } else if constexpr (J::ka == 2 && J::spec) {
            static_assert(J::wd <= 32 - W, "two-ahead index needs WD <= 32 - W");
            const u32 y2 = m1 * (y1 >> W) + d1;
            const u32 e2 = t.entry(y2);
            finish<LIMBS>(p0);
            const int h1 = h + g0;
            if (JdTables<TV>::dipfield(e2) > h1 + g1) return true;
            h = h1;
            m0 = m1; d0 = d1; g0 = g1;
            t.split(e2, m1, d1, g1);
        } else if constexpr (J::ka == 1) {
            const u32 e1 = t.entry(y1);
            const int h1 = h + g0;
            if (JdTables<TV>::dipfield(e1) > h1) return true;
            finish<LIMBS>(p0);
            h = h1;
            t.split(e1, m0, d0, g0);
        } else {
            static_assert(J::wd <= 32 - W, "two-ahead index needs WD <= 32 - W");
            const u32 y2 = m1 * (y1 >> W) + d1;      // x_{j+2} mod 2^(32-W)
            const u32 e2 = t.entry(y2);
            const int h1 = h + g0;
            if (JdTables<TV>::dipfield(e2) > h1 + g1) return true;
            finish<LIMBS>(p0);
            h = h1;
            m0 = m1; d0 = d1; g0 = g1;
            t.split(e2, m1, d1, g1);
        }
        hash ^= x0;
        jumps += 1;
        return false;
    }

    // One boundary: test, then (if not certified) one exact jump.
    template<int LIMBS>
    __device__ __forceinline__ bool step(const JdTables<TV>& t) {
        using J = JDTraits<TV>;
        if constexpr (J::lay == 1) return step_emb<LIMBS>(t);
        constexpr int W = J::w;
        const u32 t0 = __funnelshift_r(x0, x1, W);
        const u64 p0 = (u64)t0 * m0;
        const u32 y1 = (u32)p0 + d0;                 // x_{j+1} mod 2^32
        u32 m1, d1;
        int g1;
        t.decode(y1, m1, d1, g1);                    // entry of the next jump (carried)
        const int h1 = h + g0;
        bool fire;
        if constexpr (J::ka == 0) {
            fire = t.dip_of(x0) > h;
        } else if constexpr (J::ka == 1) {
            fire = t.dip_of(y1) > h1;
        } else {
            static_assert(J::wd <= 32 - W, "two-ahead index needs WD <= 32 - W");
            const u32 y2 = m1 * (y1 >> W) + d1;      // x_{j+2} mod 2^(32-W)
            fire = t.dip_of(y2) > h1 + g1;
        }
        if (fire) return true;
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
        h = h1;
        m0 = m1; d0 = d1; g0 = g1;
        hash ^= x0;
        jumps += 1;
        return false;
    }
};

// ---------------------------------------------------------------------------
// Kernel: screening identical to cv_kernel_x (tuned no-vote kernel, OEE mask
// before extraction, optional bitvector-word prefetch), tail = JdLane.

template<int MODE, int TV>
__global__ void cv_kernel_jd(const u32* __restrict__ bv, const u32* __restrict__ mod9,
                             const u32* __restrict__ pairs,
                             u32 n, u64 magic, u32 tp_0, u32 tp_1, u32 tp_2, u32 tp_3,
                             KParams p, KOut* out, u32* flags,
                             const u32* __restrict__ jtab_g, const void* __restrict__ dip_g) {
    using J = JDTraits<TV>;
    static_assert(J::valid, "unknown jd variant");
    extern __shared__ __align__(16) u32 s_jd[];
    {
        constexpr u32 JB = jd_jtab_bytes<TV>();
        const uint4* src = reinterpret_cast<const uint4*>(jtab_g);
        uint4* dst = reinterpret_cast<uint4*>(s_jd);
        for (u32 i = threadIdx.x; i < JB / 16; i += blockDim.x) dst[i] = __ldg(src + i);
        if constexpr (J::dloc == 0 && J::lay == 0) {
            constexpr u32 DB = jd_dip_bytes<TV>();
            const uint4* dsrc = reinterpret_cast<const uint4*>(dip_g);
            uint4* ddst = reinterpret_cast<uint4*>(reinterpret_cast<char*>(s_jd) + JB);
            for (u32 i = threadIdx.x; i < DB / 16; i += blockDim.x) ddst[i] = __ldg(dsrc + i);
        }
        __syncthreads();
    }
    JdTables<TV> tabs;
    tabs.jtab = s_jd;
    tabs.dip = (J::dloc == 0 && J::lay == 0) ? (const void*)(reinterpret_cast<const char*>(s_jd) + jd_jtab_bytes<TV>()) : dip_g;

    u32 rolling_hash = 0;
    u32 cases = 0;
    u64 iters = 0;
    const int h0 = (int)p.tail_budget;
    constexpr int K = x_growth_bits(J::w);
    const u32 guard_hi = p.cutoff_hi >> K;

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
        const u32 s0 = (u32)lo, s1 = (u32)(lo >> 32), s2 = (u32)hi, s3 = (u32)(hi >> 32);
        const u64 lo1 = lo + 1;
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

        u32 w = 0;
        u32 bv_lo = bvw[0], bv_hi = bvw[1];
        u32 bv_next = 0;
        if constexpr (J::pref) { if (p.words32 > 1) bv_next = bvw[2]; }
        u32 temp = m9[0] & __funnelshift_r(bv_lo, bv_hi, offset) & oee_mask;
        while (true) {
            while (temp == 0) {
                if (++w == p.words32) goto next_endpoint;
                if constexpr (J::pref) {
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

            if constexpr (J::floor == 1) {
                rolling_hash ^= num_0 ^ num_1 ^ num_2 ^ num_3 ^ base_0 ^ base_1 ^ base_2 ^ base_3;
                continue;
            }
            if constexpr (J::floor == 2) {
                u32 m, d; int g;
                tabs.decode(num_0, m, d, g);
                rolling_hash ^= num_0 ^ num_1 ^ num_2 ^ num_3 ^ base_0 ^ base_1 ^ base_2 ^ base_3 ^ m ^ d
                              ^ (u32)tabs.dip_of(m * __funnelshift_r(num_0, num_1, 12) + d);
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

template<int MODE, int TV>
static bool jd_launch_one(DevTables& tabs, const u32* bv, const u32* mod9, const u32* input, u32 n, u64 magic, const u32* tp,
                          KParams p, KOut* out, u32* flags, u32 blocks, u32 threads, cudaStream_t stream) {
    using J = JDTraits<TV>;
    constexpr u32 bytes = jd_smem_bytes<TV>();
    // c3-hyg: this context's tables.  FD3 codes (lay 1) read only their FD3
    // image (the kernel's dip_g is unused; the image is passed to keep it
    // non-null); lay-0 codes need the FD2 jump table and the uploaded dips.
    const u32* jt = J::lay == 1 ? jd_device_emb(tabs, J::w, J::dq, J::desc) : jd_device_jtab(tabs, J::w);
    const void* dip = J::lay == 1 ? (const void*)jt : jd_device_dip(tabs, J::wd, J::dq, J::desc);
    if (!dip || !jt) {
        fprintf(stderr, "jd variant %d: dip table (wd %d, dq %d, desc %d) not uploaded to this context\n",
                TV, J::wd, J::dq, (int)J::desc);
        return false;
    }
    const void* kernel = (const void*)cv_kernel_jd<MODE, TV>;
    if constexpr (bytes > 48 * 1024) {
        if (!devtables_big_smem(tabs, kernel, bytes)) return false;
    }
    if ((p.pad >> 23) & 1u) {
        const int resident = devtables_resident(tabs, kernel, threads, bytes);
        const u32 cap = (u32)(resident > 0 ? resident : 1) * (u32)(tabs.sms > 0 ? tabs.sms : 1);
        if (blocks > cap) blocks = cap;
    }
    cv_kernel_jd<MODE, TV><<<blocks, threads, bytes, stream>>>(bv, mod9, input, n, magic, tp[0], tp[1], tp[2], tp[3],
                                                                p, out, flags, jt, dip);
    return true;
}

static bool jd_variant_valid(u32 tv) {
#define JDV(C) if (tv == C) return true;
    JD_CODES(JDV)
#undef JDV
    return false;
}

// c3-hyg: measurement-only floors (JDTraits<code>::floor != 0, codes 125..127)
// count cases without running tails; verification launches refuse them.
static bool jd_variant_floor(u32 tv) {
#define JDF(C) if (tv == C) return JDTraits<C>::floor != 0;
    JD_CODES(JDF)
#undef JDF
    return false;
}

template<int MODE>
static bool jd_launch(DevTables& tabs, u32 tv, const u32* bv, const u32* mod9, const u32* input, u32 n, u64 magic, const u32* tp,
                      KParams p, KOut* out, u32* flags, u32 blocks, u32 threads, cudaStream_t stream) {
#define JDL(C) case C: return jd_launch_one<MODE, C>(tabs, bv, mod9, input, n, magic, tp, p, out, flags, blocks, threads, stream);
    switch (tv) {
        JD_CODES(JDL)
        default: return false;
    }
#undef JDL
}

// Jump width of a jd code (for the host's cutoff check), or -1.
static int jd_variant_width(u32 tv) {
#define JDW(C) if (tv == C) return JDTraits<C>::w;
    JD_CODES(JDW)
#undef JDW
    return -1;
}

// c3-hyg: the CUDA traits of a jd code, for the host's consistency check
// against cv::jump::jd_kind (src/jump.rs).  out[0..8] = valid, w, wd, dq, ka,
// desc, floor, lay.  Host-only; touches no CUDA state.
extern "C" int cv_jd_code_traits(u32 tv, int* out) {
    if (!out) return -1;
    for (int i = 0; i < 8; i++) out[i] = 0;
#define JDT(C) if (tv == C) { using J = JDTraits<C>; out[0] = 1; out[1] = J::w; out[2] = J::wd; out[3] = J::dq; \
        out[4] = J::ka; out[5] = J::desc; out[6] = J::floor; out[7] = J::lay; return 0; }
    JD_CODES(JDT)
#undef JDT
    return 0;
}
