// Experimental chained low-bit transitions for the ORIGINAL GPU tail loop.
// A row skips only whole two-half iterations, retaining every checksum point.
// It is usable for arbitrarily large high parts; only its low W bits are a key.
// Cutoff and descent guards are conservative bounds, never probabilistic tests.
#pragma once

#include <cmath>
#include <vector>
#include <cassert>

struct TailJumpPrefix {
    u32 shift;
    u32 multiplier;
    u32 add;
};

struct TailJumpEntry {
    u32 count;
    float min_ratio;
    float max_ratio;
    float max_add;
    // W <= 14 permits at most four complete original double-iterations.
    TailJumpPrefix prefix[4];
};

static int tail_jump_slot(u32 width) {
    return width == 8 ? 0 : width == 10 ? 1 : width == 12 ? 2 : width == 14 ? 3 : -1;
}

// An uncapped trailing-zero count needs its terminating one to be known.
// In particular, zero remaining bits cannot establish even a zero-bit run.
static bool tail_known_ctz(u64 value, u32 known, u32 cap, u32& result) {
    if (!known) return false;
    const u32 inspect = known < cap ? known : cap;
    const u64 low = value & ((1ull << inspect) - 1);
    if (low) {
        result = (u32)__builtin_ctzll(low);
        return true;
    }
    if (known < cap) return false;
    result = cap;
    return true;
}

static std::vector<TailJumpEntry> tail_jump_build(u32 width) {
    assert(tail_jump_slot(width) >= 0);
    std::vector<TailJumpEntry> table(1u << width);
    for (u32 r = 0; r < (1u << width); r++) {
        TailJumpEntry& row = table[r];
        const u64 seed = r + (1ull << width);
        u64 sample = seed;
        u32 shift = 0, power = 1;
        double add = 0;
        double min_ratio = 1e300, max_ratio = 1, max_add = 0;
        while (row.count < 4) {
            u64 next = sample;
            u32 d = shift, p = power;
            double c = add, peak_ratio = max_ratio, peak_add = max_add;
            bool known = true;
            for (u32 half = 0; half < 2; half++) {
                u32 a, b;
                if (!tail_known_ctz(next, width - d, 8, a)) { known = false; break; }
                u32 mm = 1;
                for (u32 j = 0; j < a; j++) mm *= 3;
                next = (next >> a) * mm;
                d += a;
                p *= mm;
                c = std::ldexp(c * mm, -(int)a);
                const double ratio = std::ldexp((double)p, -(int)d);
                if (ratio > peak_ratio) peak_ratio = ratio;
                if (c > peak_add) peak_add = c;
                if (!tail_known_ctz(next - 1, width - d, 31, b)) { known = false; break; }
                next = ((next - 1) >> b) + 1;
                d += b;
                c = std::ldexp(c - 1, -(int)b) + 1;
            }
            if (!known) break;
            assert(d <= width && d > shift);
            const u64 bias = next - (u64)p * (seed >> d);
            assert(bias <= UINT32_MAX);
            row.prefix[row.count++] = {d, p, (u32)bias};
            const double ratio = std::ldexp((double)p, -(int)d);
            if (ratio < min_ratio) min_ratio = ratio;
            sample = next; shift = d; power = p; add = c;
            max_ratio = peak_ratio; max_add = peak_add;
        }
        if (row.count) {
            // Powers <= 3^14 have <= 23 significant bits; dyadic ratios are
            // exact binary32 values. Affine additives are dyadic binary64.
            row.min_ratio = (float)min_ratio;
            row.max_ratio = (float)max_ratio;
            row.max_add = (float)max_add;
            if ((double)row.max_add < max_add)
                row.max_add = std::nextafter(row.max_add, INFINITY);
            assert((double)row.min_ratio == min_ratio);
            assert((double)row.max_ratio == max_ratio);
            assert((double)row.max_add >= max_add);
        }
    }
    return table;
}

#ifdef __CUDACC__
__device__ __forceinline__ float tail_float_upper(u32 x0, u32 x1, u32 x2, u32 x3) {
    const u64 lo = (u64)x0 | ((u64)x1 << 32);
    const u64 hi = (u64)x2 | ((u64)x3 << 32);
    return __fmaf_ru(__ull2float_ru(hi), 0x1p64f, __ull2float_ru(lo));
}

__device__ __forceinline__ bool tail_jump_try(
        const TailJumpEntry* table, u32 mask, float base_upper, float cutoff,
        u32& x0, u32& x1, u32& x2, u32& x3, u32& hash, u64& iters) {
    const TailJumpEntry& row = table[x0 & mask];
    const u32 count = row.count;
    if (!count) return false;
    const u64 lo = (u64)x0 | ((u64)x1 << 32);
    const u64 hi = (u64)x2 | ((u64)x3 << 32);
    const float lower = __fmaf_rd(__ull2float_rd(hi), 0x1p64f, __ull2float_rd(lo));
    const float upper = __fmaf_ru(__ull2float_ru(hi), 0x1p64f, __ull2float_ru(lo));
    // Every whole double endpoint is >= min_ratio*y (additive >= 0),
    // and every checked odd-run peak is <= max_ratio*y + max_add.
    // Strict cutoff preserves all exact CPU fallback/path-record candidates.
    if (__fmul_rd(lower, row.min_ratio) < base_upper ||
        !(__fmaf_ru(upper, row.max_ratio, row.max_add) < cutoff)) return false;
    u32 d = 0, p = 0, c = 0;
    for (u32 j = 0; j < count; j++) {
        const TailJumpPrefix prefix = row.prefix[j];
        d = prefix.shift; p = prefix.multiplier; c = prefix.add;
        hash ^= (u32)(lo >> d) * p + c;
    }
    // Update the wide exact state only at the LAST checksum point. Right
    // shifts consume known bits, so the next iteration uses a fresh row key.
    u64 product = (u64)__funnelshift_r(x0, x1, d) * p + c;
    x0 = (u32)product;
    product = (u64)__funnelshift_r(x1, x2, d) * p + (product >> 32);
    x1 = (u32)product;
    product = (u64)__funnelshift_r(x2, x3, d) * p + (product >> 32);
    x2 = (u32)product;
    x3 = (x3 >> d) * p + (u32)(product >> 32);
    iters += count;
    return true;
}

// Isolated arithmetic validation: no bitvector or endpoint generator can
// hide near-cutoff, short-descent, capped-run, or carry boundary inputs.
struct TailJumpTestInput {
    u32 x[4], base[4], cutoff_hi;
};
struct TailJumpTestOutput {
    u64 iters, jumps;
    u32 hash, flags, exhausted;
};

template<bool JUMP>
__global__ void tail_jump_test_kernel(const TailJumpTestInput* inputs,
        TailJumpTestOutput* outputs, u32 n, const TailJumpEntry* table, u32 width) {
    const u32 index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) return;
    const TailJumpTestInput in = inputs[index];
    u32 x0 = in.x[0], x1 = in.x[1], x2 = in.x[2], x3 = in.x[3];
    TailJumpTestOutput out = {};
    const float base_upper = tail_float_upper(in.base[0], in.base[1], in.base[2], in.base[3]);
    const float cutoff = __fmul_rd(__uint2float_rd(in.cutoff_hi), 0x1p96f);
    for (u32 outer = 0; outer < 10000; outer++) {
        if constexpr (JUMP) {
            if (tail_jump_try(table, (1u << width) - 1, base_upper, cutoff,
                              x0, x1, x2, x3, out.hash, out.iters)) {
                out.jumps++;
                continue;
            }
        }
#pragma unroll
        for (u32 half = 0; half < 2; half++) {
            const u32 a = __ffs(x0 | (1u << 8)) - 1;
            u32 mm = 1;
            for (u32 j = 0; j < a; j++) mm *= 3;
            u64 product = (u64)__funnelshift_r(x0, x1, a) * mm;
            x0 = (u32)product;
            product = (u64)__funnelshift_r(x1, x2, a) * mm + (product >> 32);
            x1 = (u32)product;
            product = (u64)__funnelshift_r(x2, x3, a) * mm + (product >> 32);
            x2 = (u32)product;
            x3 = (x3 >> a) * mm + (u32)(product >> 32);
            if (x3 >= in.cutoff_hi) {
                out.flags++;
                x0 = 2; x1 = x2 = x3 = 0;
            }
            x0 -= 1;
            const u32 b = __ffs(x0 | (1u << 31)) - 1;
            x0 = __funnelshift_r(x0, x1, b) + 1;
            x1 = __funnelshift_r(x1, x2, b);
            x2 = __funnelshift_r(x2, x3, b);
            x3 >>= b;
            if (!x0 && b) {
                x1++;
                if (!x1) {
                    x2++;
                    if (!x2) x3++;
                }
            }
        }
        out.hash ^= x0;
        out.iters++;
        const bool below = x3 < in.base[3]
            || (x3 == in.base[3] && x2 < in.base[2])
            || (x3 == in.base[3] && x2 == in.base[2] && x1 < in.base[1])
            || (x3 == in.base[3] && x2 == in.base[2] && x1 == in.base[1] && x0 < in.base[0]);
        if (below) { outputs[index] = out; return; }
    }
    out.exhausted = 1;
    outputs[index] = out;
}

extern "C" int cv_gpu_tail_jump_self_test() {
    const u32 count = 16384;
    std::vector<TailJumpTestInput> inputs(count);
    std::vector<TailJumpTestOutput> baseline(count), actual(count);
    u64 rng = 0x68dfe311215abc71ull;
    auto random = [&]() { rng ^= rng >> 12; rng ^= rng << 25; rng ^= rng >> 27; return rng * 0x2545F4914F6CDD1Dull; };
    for (u32 i = 0; i < count; i++) {
        const u32 cutoffs[4] = {97, 104, 112, 120};
        const u32 cutoff_bits = cutoffs[i % 4];
        const unsigned __int128 cutoff = (unsigned __int128)1 << cutoff_bits;
        unsigned __int128 x = ((unsigned __int128)random() << 64) | random();
        x &= cutoff - 1;
        if (i % 8 == 0) x = cutoff - 2 - (i & 127);
        if (i % 8 == 1) x = ((x >> 32) << 32);   // capped odd run
        if (i % 8 == 2) x = ((x >> 32) << 32) + 1; // capped even run
        if (i % 8 == 3) x = ((x >> 32) << 32) + UINT32_MAX;
        if (x < 4) x = 4;
        unsigned __int128 base;
        switch ((i / 8) % 5) {
            case 0: base = x - 1; break;
            case 1: base = x + 1; break;
            case 2: base = x >> 1; break;
            case 3: base = x >> 20; break;
            default: base = (unsigned __int128)1 << 72; break;
        }
        if (base < 3) base = 3;
        for (u32 j = 0; j < 4; j++) {
            inputs[i].x[j] = (u32)(x >> (32 * j));
            inputs[i].base[j] = (u32)(base >> (32 * j));
        }
        inputs[i].cutoff_hi = 1u << (cutoff_bits - 96);
    }
    TailJumpTestInput* d_input = nullptr;
    TailJumpTestOutput* d_output = nullptr;
    TailJumpEntry* d_table = nullptr;
    int status = -1;
    do {
        if (cudaMalloc(&d_input, count * sizeof(TailJumpTestInput)) != cudaSuccess) break;
        if (cudaMalloc(&d_output, count * sizeof(TailJumpTestOutput)) != cudaSuccess) break;
        if (cudaMemcpy(d_input, inputs.data(), count * sizeof(TailJumpTestInput), cudaMemcpyHostToDevice) != cudaSuccess) break;
        tail_jump_test_kernel<false><<<(count + 127) / 128, 128>>>(d_input, d_output, count, nullptr, 0);
        if (cudaMemcpy(baseline.data(), d_output, count * sizeof(TailJumpTestOutput), cudaMemcpyDeviceToHost) != cudaSuccess) break;
        bool passed = true;
        for (u32 width = 8; width <= 14 && passed; width += 2) {
            const auto table = tail_jump_build(width);
            if (cudaMalloc(&d_table, table.size() * sizeof(TailJumpEntry)) != cudaSuccess) { passed = false; break; }
            if (cudaMemcpy(d_table, table.data(), table.size() * sizeof(TailJumpEntry), cudaMemcpyHostToDevice) != cudaSuccess) { passed = false; break; }
            tail_jump_test_kernel<true><<<(count + 127) / 128, 128>>>(d_input, d_output, count, d_table, width);
            if (cudaMemcpy(actual.data(), d_output, count * sizeof(TailJumpTestOutput), cudaMemcpyDeviceToHost) != cudaSuccess) { passed = false; break; }
            u64 jumps = 0, flagged = 0, iters = 0;
            for (u32 i = 0; i < count; i++) {
                const auto& a = actual[i];
                const auto& b = baseline[i];
                if (a.hash != b.hash || a.flags != b.flags || a.iters != b.iters || a.exhausted || b.exhausted) {
                    fprintf(stderr, "tail-jump validation failed W=%u input=%u hash=%u/%u flags=%u/%u iters=%llu/%llu exhausted=%u/%u\n",
                            width, i, a.hash, b.hash, a.flags, b.flags, (unsigned long long)a.iters,
                            (unsigned long long)b.iters, a.exhausted, b.exhausted);
                    passed = false; break;
                }
                jumps += a.jumps; flagged += a.flags; iters += a.iters;
            }
            if (passed) fprintf(stderr, "tail-jump GPU self-test W=%u cases=%u iters=%llu flags=%llu accepted_jumps=%llu exact_match=true\n",
                    width, count, (unsigned long long)iters, (unsigned long long)flagged, (unsigned long long)jumps);
            cudaFree(d_table); d_table = nullptr;
        }
        if (passed) status = 0;
    } while (false);
    cudaFree(d_table); cudaFree(d_output); cudaFree(d_input);
    return status;
}
#endif
