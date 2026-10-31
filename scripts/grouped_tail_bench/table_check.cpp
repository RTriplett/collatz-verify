// Independent, exhaustive-key validation of the chained tail table.
// Run from the repository root:
//   g++ -O2 -std=c++17 scripts/grouped_tail_bench/table_check.cpp -o /tmp/collatz-tail-table-check
//   /tmp/collatz-tail-table-check
// This is a host-only arithmetic check; CUDA integration needs separate checks.
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>

using u32 = uint32_t;
using u64 = uint64_t;
using u128 = unsigned __int128;

#include "../../cuda/grouped_tail.cuh"

struct Fraction {
    u64 numerator;
    u32 shift;
};

static void require(bool condition, const char* what, u32 width, u32 key) {
    if (!condition) {
        std::cerr << "FAIL width=" << width << " key=" << key << ": " << what << '\n';
        std::exit(1);
    }
}

// Decode the actual stored binary32 bit pattern, avoiding floating arithmetic
// in the oracle. The table has nonnegative, finite, normal values or zero.
static Fraction fraction(float value) {
    u32 bits;
    std::memcpy(&bits, &value, sizeof(bits));
    if (bits == 0) return {0, 0};
    const u32 exponent = (bits >> 23) & 255;
    if ((bits >> 31) || !exponent || exponent == 255) std::abort();
    u64 numerator = (bits & 0x7fffff) | 0x800000;
    int shift = 150 - (int)exponent;
    while (shift > 0 && !(numerator & 1)) {
        numerator >>= 1;
        --shift;
    }
    if (shift < 0) {
        numerator <<= -shift;
        shift = 0;
    }
    if (shift > 32) std::abort();
    return {numerator, (u32)shift};
}

// Exact floor/ceil of an affine bound, splitting before multiplication to
// avoid any transient 128-bit overflow. Inputs stay below 2^120, and all
// committed W<=14 table peaks stay below 2^128.
static u128 affine_bound(u128 x, Fraction ratio, Fraction add, bool ceiling) {
    const u32 shift = ratio.shift > add.shift ? ratio.shift : add.shift;
    const u64 multiplier = ratio.numerator << (shift - ratio.shift);
    const u64 bias = add.numerator << (shift - add.shift);
    const u128 mask = ((u128)1 << shift) - 1;
    const u128 remainder = (x & mask) * multiplier + bias;
    const u128 whole = (x >> shift) * multiplier + (remainder >> shift);
    return whole + (ceiling && (remainder & mask));
}

static u32 trailing_zeros_capped(u128 value, u32 cap) {
    const u32 low = (u32)value;
    return (u32)__builtin_ctz(low | (1u << cap));
}

static u64 random_word(u64& state) {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    return state * 2685821657736338717ull;
}

int main() {
    u64 random_state = 0x6f6c6c61747aULL;
    u64 total_states = 0, total_prefixes = 0, total_peaks = 0;
    u64 cutoff_safe = 0, cutoff_rejected = 0;
    const u128 cutoff = (u128)1 << 120;
    const u128 mask120 = cutoff - 1;
    for (u32 width : {8u, 10u, 12u, 14u}) {
        const auto table = tail_jump_build(width);
        const u128 low_mask = ((u128)1 << width) - 1;
        u64 rows = 0, states = 0, prefixes = 0;
        u32 max_count = 0;
        for (u32 key = 0; key < table.size(); ++key) {
            const auto& row = table[key];
            if (!row.count) continue;
            ++rows;
            if (row.count > max_count) max_count = row.count;
            const auto min_ratio = fraction(row.min_ratio);
            const auto max_ratio = fraction(row.max_ratio);
            const auto max_add = fraction(row.max_add);
            for (u32 sample = 0; sample < 24; ++sample) {
                u128 high;
                switch (sample) {
                    case 0: high = 0; break;
                    case 1: high = (u128)1 << width; break;
                    case 2: high = (u128)3 << width; break;
                    case 3: high = (u128)1 << 32; break;
                    case 4: high = ((u128)1 << 64) - 1; break;
                    case 5: high = (u128)1 << 64; break;
                    case 6: high = ((u128)1 << 96) - 1; break;
                    case 7: high = (u128)1 << 96; break;
                    case 8: high = (u128)1 << 119; break;
                    case 9: high = mask120; break;
                    default:
                        high = (((u128)random_word(random_state) << 64) |
                                 random_word(random_state)) & mask120;
                        break;
                }
                const u128 input = (high & ~low_mask) | key;
                if (input == 0) continue;
                const u128 lower_endpoint = affine_bound(input, min_ratio, {0, 0}, true);
                const u128 upper_peak = affine_bound(input, max_ratio, max_add, false);
                bool every_peak_safe = true;
                u128 exact = input;
                u32 exact_hash = 0, table_hash = 0;
                u32 previous_shift = 0;
                for (u32 i = 0; i < row.count; ++i) {
                    for (u32 half = 0; half < 2; ++half) {
                        const u32 a = trailing_zeros_capped(exact, 8);
                        u32 power = 1;
                        for (u32 j = 0; j < a; ++j) power *= 3;
                        // Detect overflow before the oracle's own multiply.
                        require((exact >> a) <= (~(u128)0) / power,
                                "oracle headroom", width, key);
                        exact = (exact >> a) * power;
                        require(exact <= upper_peak, "odd-peak upper bound", width, key);
                        if (exact >= cutoff) every_peak_safe = false;
                        const u32 b = trailing_zeros_capped(exact - 1, 31);
                        exact = ((exact - 1) >> b) + 1;
                        ++total_peaks;
                    }
                    const auto& prefix = row.prefix[i];
                    require(prefix.shift > previous_shift && prefix.shift < width,
                            "known-bit boundary", width, key);
                    previous_shift = prefix.shift;
                    const u128 projected = (input >> prefix.shift) * prefix.multiplier + prefix.add;
                    require(projected == exact, "exact prefix state", width, key);
                    require(exact >= lower_endpoint, "descent lower bound", width, key);
                    exact_hash ^= (u32)exact;
                    table_hash ^= (u32)((u64)input >> prefix.shift) * prefix.multiplier + prefix.add;
                    require(exact_hash == table_hash, "original per-double checksum", width, key);
                    ++prefixes;
                }
                if (upper_peak < cutoff) {
                    require(every_peak_safe, "cutoff guard admitted overflow", width, key);
                    ++cutoff_safe;
                } else {
                    ++cutoff_rejected;
                }
                ++states;
            }
        }
        total_states += states;
        total_prefixes += prefixes;
        std::cout << "width=" << width << " rows=" << table.size()
                  << " usable=" << rows << " max_count=" << max_count
                  << " states=" << states << " prefixes=" << prefixes << '\n';
    }
    std::cout << "PASS states=" << total_states << " prefixes=" << total_prefixes
              << " odd_peaks=" << total_peaks << " cutoff_safe=" << cutoff_safe
              << " cutoff_rejected=" << cutoff_rejected << '\n';
}
