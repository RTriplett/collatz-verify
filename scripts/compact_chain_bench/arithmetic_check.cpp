// Independently check the actual CUDA compact-chain arithmetic against exact
// checked 128-bit Collatz iteration. Build with a normal C++20 compiler:
//   c++ -O2 -std=c++20 scripts/compact_chain_bench/arithmetic_check.cpp -o /tmp/compact-chain-check
// The CUDA stubs only make the header parse; the tested arithmetic is unchanged.
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <optional>
#include <vector>

using u32 = uint32_t;
using u64 = uint64_t;
using U128 = unsigned __int128;
#define __device__
#define __forceinline__ inline
#define __global__
#define __shared__ static
struct Dim { u32 x; };
static Dim threadIdx, blockIdx, blockDim;
struct uint4 { u32 x, y, z, w; };
static uint4 make_uint4(u32 x, u32 y, u32 z, u32 w) { return {x, y, z, w}; }
static u64 __umul64hi(u64 a, u64 b) { return (U128(a) * b) >> 64; }
static void __syncthreads() {}
#include "../../cuda/compact_chain.cuh"

static void require(bool condition, const char* message) {
    if (!condition) { std::fprintf(stderr, "%s\n", message); std::exit(1); }
}

static std::optional<U128> iterate(U128 value, u32 width, u32* odd_count = nullptr) {
    u32 odd = 0;
    for (u32 i = 0; i < width; ++i) {
        if ((value & 1) != 0) {
            const U128 extra = (value >> 1) + 1;
            if (value > ~U128(0) - extra) return {};
            value += extra;
            ++odd;
        } else value >>= 1;
    }
    if (odd_count) *odd_count = odd;
    return value;
}

static U128 pow3(u32 power) {
    U128 value = 1;
    while (power--) value *= 3;
    return value;
}

static std::array<u32, 8> word(U128 m, U128 p3, u32 q, u32 width, u32 f) {
    const u32 mask = (1u << width) - 1;
    const u32 residue = (u32(m) + u32(p3) * q) & mask;
    u32 odd;
    const u32 end = u32(*iterate(residue, width, &odd));
    return {residue, (residue * chain_inverse(u32(p3))) & mask,
            end, u32(pow3(odd)), f + odd, 0, 0, 0};
}

static U128 step(U128 m, U128 p3, u32 q, const std::array<u32, 8>& w, u32 width) {
    u64 lo, hi;
    chain_iterate(u64(m), u64(m >> 64), u64(p3), u64(p3 >> 64),
                  q, w.data(), width, lo, hi);
    return U128(lo) | (U128(hi) << 64);
}

static u64 check_kernel_slices(std::mt19937_64& random) {
    u64 checked = 0;
    // A single first-stage survivor permits the unmodified kernel to execute
    // serially under the stubs. This covers descriptor offsets, metadata, and
    // high residue bits. Multi-prefix shared-memory synchronization is covered
    // by the real GPU validation profiles, not by this serial host check.
    for (u32 trial = 0; trial < 16; ++trial) {
        const u32 width1 = 1 + trial % 6, width2 = 5 + trial % 8;
        const u32 k = 64 - width1 - width2;
        const u64 n0 = (trial % 2 == 0 ? ~u64(0) : random()) & ((u64(1) << k) - 1);
        u32 f;
        const U128 m = *iterate(n0, k, &f), p3 = pow3(f);
        const u32 q1 = random() & ((1u << width1) - 1);
        const auto first = word(m, p3, q1, width1, f);
        const U128 middle = step(m, p3, q1, first, width1);
        std::vector<u32> words(first.begin(), first.end());
        std::vector<ChainLeafGpu> leaves;
        for (u32 q2 = 0; q2 < (1u << width2); ++q2) {
            auto second = word(middle, p3 * first[3], q2, width2, first[4]);
            second[5] = q2 % 3;
            second[6] = (q2 / 3) % 4;
            leaves.push_back({0, u32(words.size() / 8)});
            words.insert(words.end(), second.begin(), second.end());
        }
        const ChainProgramGpu program = {0, 1, u32(leaves.size()), width1, width2, 0, 1, 0};
        u32 parent[16] = {u32(n0), u32(n0 >> 32), u32(m), u32(m >> 32),
            u32(m >> 64), u32(m >> 96), u32(p3), u32(p3 >> 32), u32(p3 >> 64),
            u32(p3 >> 96), chain_inverse(u32(p3)), 0, 0, 13, 0, k};
        const u32 endpoints = leaves.size();
        for (u32 offset : {0u, 1u, endpoints / 2, endpoints - 1}) {
            const u32 count = std::min(endpoints - offset, 257u);
            parent[12] = count;
            parent[14] = offset;
            std::vector<u32> output((13 + count + 5) * 8, 0xdeadbeef);
            threadIdx.x = blockIdx.x = 0;
            blockDim.x = 1;
            cv_expand_chain(parent, &program, leaves.data(), words.data(), output.data(), trial % 2);
            for (u32 i = 0; i < 13 * 8; ++i)
                require(output[i] == 0xdeadbeef, "kernel overwrote leading slice boundary");
            for (u32 i = (13 + count) * 8; i < output.size(); ++i)
                require(output[i] == 0xdeadbeef, "kernel overwrote trailing slice boundary");
            for (u32 i = 0; i < count; ++i) {
                const u32 q2 = offset + i;
                const u64 expected_n = n0 | (u64(q1) << k) | (u64(q2) << (k + width1));
                const U128 expected_m = *iterate(expected_n, 64);
                const u32* record = output.data() + (13 + i) * 8;
                const u32 flags = (q2 % 3) << 30
                    | (trial % 2 ? ((q2 / 3) % 4 & 1u) << 29 : 0);
                require(record[0] == u32(expected_n) && record[1] == u32(expected_n >> 32)
                    && record[2] == 0 && record[3] == flags, "kernel residue/metadata mismatch");
                for (u32 limb = 0; limb < 4; ++limb)
                    require(record[4 + limb] == u32(expected_m >> (32 * limb)),
                            "kernel endpoint iterate mismatch");
                ++checked;
            }
        }
    }
    return checked;
}

int main() {
    std::mt19937_64 random(0xc011a72c4a1ULL);
    const U128 maximum = ~U128(0);
    u64 checks = 0, wide_states = 0, carrying_products = 0, near_limit = 0;
    // Include arbitrary wide affine states as well as states derived from real
    // start residues. The former exercise carry, borrow, and shift boundaries.
    for (u32 trial = 0; trial < 30000; ++trial) {
        const u32 width1 = 1 + random() % 6, width2 = 1 + random() % 12;
        const u32 k = 2 + random() % (63 - width1 - width2);
        const u64 start_mask = (u64(1) << k) - 1;
        const u64 n0 = (trial % 7 == 0 ? ~u64(0) : random()) & start_mask;
        u32 f;
        U128 m = *iterate(n0, k, &f);
        if (trial % 3 == 0) {
            f = random() % (k + 1);
            m = U128(random()) | (U128(random() & ((u64(1) << 47) - 1)) << 64);
        } else if (trial % 11 == 0) {
            f = random() % (k + 1);
            m = ~U128(0) - (random() & 0xffff);
        }
        const U128 p3 = pow3(f);
        const u32 q1 = random() & ((1u << width1) - 1);
        const auto first = word(m, p3, q1, width1, f);
        if (m > maximum - p3 * q1) continue;
        const U128 x1 = m + p3 * q1;
        const auto expected1 = iterate(x1, width1);
        if (!expected1) continue;
        const U128 middle = step(m, p3, q1, first, width1);
        require(middle == *expected1, "first-stage arithmetic mismatch");
        const U128 middle_p3 = p3 * first[3];
        const u32 q2 = random() & ((1u << width2) - 1);
        const auto second = word(middle, middle_p3, q2, width2, first[4]);
        if (middle > maximum - middle_p3 * q2) continue;
        const U128 x2 = middle + middle_p3 * q2;
        const auto expected2 = iterate(x2, width2);
        if (!expected2) continue;
        const U128 final = step(middle, middle_p3, q2, second, width2);
        require(final == *expected2, "second-stage arithmetic mismatch");
        const u32 inverse1 = chain_inverse(u32(p3));
        const u32 inverse2 = inverse1 * chain_inverse(first[3]);
        require(u32(p3) * inverse1 == 1u, "first modular inverse mismatch");
        require(u32(middle_p3) * inverse2 == 1u, "chained modular inverse mismatch");
        require(((first[1] - u32(m) * inverse1) & ((1u << width1) - 1)) == q1,
                "first translation mismatch");
        require(((second[1] - u32(middle) * inverse2) & ((1u << width2) - 1)) == q2,
                "second translation mismatch");
        if (trial % 3 != 0 && trial % 11 != 0) {
            const u64 extended = n0 | (u64(q1) << k) | (u64(q2) << (k + width1));
            require(final == *iterate(extended, k + width1 + width2),
                    "composed endpoint differs from direct Collatz iteration");
        }
        ++checks;
        wide_states += (m >> 64) != 0;
        carrying_products += (p3 * q1 >> 64) != 0;
        near_limit += (m >> 127) != 0;
    }
    require(checks > 25000 && wide_states > 9000 && carrying_products > 500 && near_limit > 0,
            "insufficient arithmetic boundary coverage");
    const u64 kernel_slices = check_kernel_slices(random);
    std::printf("{\"checks\":%llu,\"wide_states\":%llu,\"carrying_products\":%llu,\"near_limit\":%llu,\"kernel_slice_endpoints\":%llu}\n",
                (unsigned long long)checks, (unsigned long long)wide_states,
                (unsigned long long)carrying_products, (unsigned long long)near_limit,
                (unsigned long long)kernel_slices);
}
