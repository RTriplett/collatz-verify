// Experimental compact two-stage affine expansion. A descriptor crosses both
// table boundaries without materializing the intermediate frontier on the CPU
// or in device global memory. First-stage states are shared within each CTA.
#pragma once

struct ChainProgramGpu {
    u32 route_start, route_len, endpoints;
    u32 first_width, second_width, first_start, first_len, leaf_start;
};

struct ChainLeafGpu {
    u32 first_slot, second_index;
};

static_assert(sizeof(ChainProgramGpu) == 32, "chain program ABI");
static_assert(sizeof(ChainLeafGpu) == 8, "chain leaf ABI");

struct alignas(16) ChainPrefixGpu {
    u64 n0, mlo, mhi, p3lo, p3hi;
    u32 delta, reserved;
};

__device__ __forceinline__ u32 chain_inverse(u32 odd) {
    // Newton's inverse in Z/(2^32), starting with one correct low bit.
    u32 inverse = 1;
#pragma unroll
    for (int i = 0; i < 5; i++) inverse *= 2u - odd * inverse;
    return inverse;
}

__device__ __forceinline__ void chain_iterate(
        u64 mlo, u64 mhi, u64 p3lo, u64 p3hi, u32 q,
        const u32* entry, u32 width, u64& out_lo, u64& out_hi) {
    // (m + 3^f*q - r)/2^width is integral. The host checks the exact
    // 128-bit range before accepting a descriptor for either stage.
    u64 lo = p3lo * q;
    u64 hi = p3hi * q + __umul64hi(p3lo, (u64)q);
    u64 sum = lo + mlo;
    hi += mhi + (sum < lo);
    lo = sum;
    hi -= lo < entry[0];
    lo -= entry[0];
    const u64 shifted_lo = (lo >> width) | (hi << (64 - width));
    const u64 shifted_hi = hi >> width;
    lo = shifted_lo * entry[3];
    hi = shifted_hi * entry[3] + __umul64hi(shifted_lo, (u64)entry[3]);
    sum = lo + entry[2];
    out_hi = hi + (sum < lo);
    out_lo = sum;
}

// Descriptor words: n0[2], m[4], 3^f[4], inverse(3^f), program,
// logical count, output start, logical leaf slice offset, initial k.
__global__ void cv_expand_chain(const u32* __restrict__ parents,
                               const ChainProgramGpu* __restrict__ programs,
                               const ChainLeafGpu* __restrict__ leaves,
                               const u32* __restrict__ words,
                               u32* __restrict__ affine, u64 fam_stride) {
    __shared__ ChainPrefixGpu prefix[64];
    const u32* parent = parents + (size_t)blockIdx.x * 16;
    const ChainProgramGpu program = programs[parent[11]];
    const u32 k = parent[15], width1 = program.first_width;
    const u32 width2 = program.second_width;
    const u32 mask1 = (1u << width1) - 1, mask2 = (1u << width2) - 1;
    if (threadIdx.x < program.first_len) {
        const u64 n0 = (u64)parent[0] | ((u64)parent[1] << 32);
        const u64 mlo = (u64)parent[2] | ((u64)parent[3] << 32);
        const u64 mhi = (u64)parent[4] | ((u64)parent[5] << 32);
        const u64 p3lo = (u64)parent[6] | ((u64)parent[7] << 32);
        const u64 p3hi = (u64)parent[8] | ((u64)parent[9] << 32);
        const u32* first = words + (size_t)(program.first_start + threadIdx.x) * 8;
        const u32 delta1 = (u32)mlo * parent[10];
        const u32 q1 = (first[1] - delta1) & mask1;
        ChainPrefixGpu state;
        state.n0 = n0 | ((u64)q1 << k);
        chain_iterate(mlo, mhi, p3lo, p3hi, q1, first, width1, state.mlo, state.mhi);
        state.p3lo = p3lo * first[3];
        state.p3hi = p3hi * first[3] + __umul64hi(p3lo, (u64)first[3]);
        const u32 inverse2 = parent[10] * chain_inverse(first[3]);
        state.delta = (u32)state.mlo * inverse2;
        state.reserved = 0;
        prefix[threadIdx.x] = state;
    }
    __syncthreads();
    for (u32 i = threadIdx.x; i < parent[12]; i += blockDim.x) {
        const ChainLeafGpu leaf = leaves[(size_t)program.leaf_start + parent[14] + i];
        const ChainPrefixGpu state = prefix[leaf.first_slot];
        const u32* second = words + (size_t)leaf.second_index * 8;
        const u32 q2 = (second[1] - state.delta) & mask2;
        const u64 n0 = state.n0 | ((u64)q2 << (k + width1));
        u64 lo, hi;
        chain_iterate(state.mlo, state.mhi, state.p3lo, state.p3hi,
                      q2, second, width2, lo, hi);
        const u32 family = fam_stride ? (second[6] & 1u) : 0u;
        uint4* out = reinterpret_cast<uint4*>(affine) + (size_t)(parent[13] + i) * 2;
        out[0] = make_uint4((u32)n0, (u32)(n0 >> 32), 0,
                           (second[5] << 30) | (family << 29));
        out[1] = make_uint4((u32)lo, (u32)(lo >> 32), (u32)hi, (u32)(hi >> 32));
    }
}
