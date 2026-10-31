// Generated diagnostic overlay. Production source is unchanged.
// clock64 deltas measure lane residence, including stalls and scheduling.
#include <cstdlib>
#include <cerrno>

struct CvPhaseCounters {
    unsigned long long sampled_lanes;
    unsigned long long sampled_blocks;
    unsigned long long cases;
    unsigned long long double_iterations;
    unsigned long long setup_cycles;
    unsigned long long screening_cycles;
    unsigned long long tail_cycles;
    unsigned long long reduction_cycles;
    unsigned long long total_cycles;
};
__device__ CvPhaseCounters cv_phase_counters;
static u32 cv_phase_stride = 0;
static u32 cv_phase_seed = 0;
static unsigned long long cv_phase_sequence = 0;
static unsigned long long cv_phase_capture = 0;

__device__ __forceinline__ unsigned long long cv_phase_clock() {
    unsigned long long value;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(value) :: "memory");
    return value;
}

static bool cv_phase_env(const char* name, u32 fallback, u32* result) {
    const char* value = std::getenv(name);
    if (!value) { *result = fallback; return true; }
    if (!*value || *value == '-') return false;
    char* end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (errno || !end || *end || parsed > 0xffffffffull) return false;
    *result = (u32)parsed;
    return true;
}

extern "C" int cv_gpu_phase_start() {
    if (!cv_phase_env("CV_GPU_PHASE_STRIDE", 128, &cv_phase_stride) ||
        !cv_phase_env("CV_GPU_PHASE_SEED", 0, &cv_phase_seed)) return -1;
    if (cudaDeviceSynchronize() != cudaSuccess) return -1;
    CvPhaseCounters zero = {};
    if (cudaMemcpyToSymbol(cv_phase_counters, &zero, sizeof(zero)) != cudaSuccess) return -1;
    cv_phase_sequence = 0;
    // Use different deterministic offsets for successive aggregate trials.
    cv_phase_seed += (u32)(53ull * cv_phase_capture);
    cv_phase_capture += 1;
    return 0;
}

extern "C" int cv_gpu_phase_stop() {
    if (cudaDeviceSynchronize() != cudaSuccess) return -1;
    CvPhaseCounters counts = {};
    if (cudaMemcpyFromSymbol(&counts, cv_phase_counters, sizeof(counts)) != cudaSuccess) return -1;
    const auto sum = counts.setup_cycles + counts.screening_cycles + counts.tail_cycles + counts.reduction_cycles;
    if (sum != counts.total_cycles) return -1;
    std::fprintf(stderr,
        "{\"type\":\"gpu_phase_profile\",\"capture\":%llu,"
        "\"sampling_stride_blocks\":%u,\"starting_offset_seed\":%u,"
        "\"verifier_launches\":%llu,\"sampled_lanes\":%llu,\"sampled_blocks\":%llu,"
        "\"sampled_cases\":%llu,\"sampled_double_iterations\":%llu,"
        "\"cycles\":{\"endpoint_setup\":%llu,\"completion_screening\":%llu,"
        "\"exact_trajectory\":%llu,\"checksum_reduction\":%llu},"
        "\"total_lane_cycles\":%llu,\"partition_exact\":true,"
        "\"metric\":\"sampled_lane_elapsed_cycles_including_stalls_and_instrumentation\"}\n",
        cv_phase_capture, cv_phase_stride, cv_phase_seed, cv_phase_sequence,
        counts.sampled_lanes, counts.sampled_blocks, counts.cases, counts.double_iterations,
        counts.setup_cycles, counts.screening_cycles, counts.tail_cycles,
        counts.reduction_cycles, counts.total_cycles);
    std::fflush(stderr);
    cv_phase_stride = 0;
    return 0;
}
