// c3-hyg: per-context device tables and launch caches.
//
// Until round 2 the jump tables of cuda/xtail.cuh, the jd FD2/FD3 and dip
// tables of cuda/jdip.cuh, and a few launch attributes were cached in
// process-global statics.  A second context in the same process (another
// cv_gpu_init on another device, or a context created after cudaDeviceReset)
// would then have launched with device pointers that belong to a different,
// possibly destroyed, context.  Every such cache now lives in one DevTables
// owned by the context that created it (Ctx::tables in kernel.cu): it is
// filled lazily (jump tables) or by an explicit upload (dip tables,
// cv_jd_upload), consulted only by launches on that context, and freed by
// cv_gpu_free.  Launch entry points also refuse to run when the calling
// thread's current device or CUDA context differs from the one the context
// was created on (kernel.cu, cv_ctx_current).
//
// Host-only code: nothing here is visible to device code, so no kernel's
// machine code depends on it.
#pragma once

#include <map>
#include <mutex>
#include <set>
#include <utility>
#include <vector>

struct DevTables {
    int device = -1;                          // ordinal of the device the context was created on
    int sms = 0;                              // multiprocessors of that device
    std::mutex lock;                          // guards every map below
    std::map<int, u32*> xtail;                // cuda/xtail.cuh table images, keyed by code
    std::map<int, u32*> jd_jtab;              // cuda/jdip.cuh FD2 jump tables, keyed by W
    std::map<u32, void*> jd_dip;              // lay-0 dip images (u8 or u16), key wd | dq << 8 | desc << 16
    std::map<u32, u32*> jd_emb;               // FD3 images (dips in the jump entries), same keys (wd = w)
    std::set<const void*> big_smem;           // kernels opted in to more than 48 KB of dynamic shared memory
    std::map<std::pair<const void*, u32>, int> resident;   // (kernel, threads) -> resident blocks per SM
};

// Frees every device table of `t` (the owning context must be current).
static void devtables_free(DevTables& t) {
    std::lock_guard<std::mutex> guard(t.lock);
    for (auto& kv : t.xtail) cudaFree(kv.second);
    for (auto& kv : t.jd_jtab) cudaFree(kv.second);
    for (auto& kv : t.jd_dip) cudaFree(kv.second);
    for (auto& kv : t.jd_emb) cudaFree(kv.second);
    t.xtail.clear();
    t.jd_jtab.clear();
    t.jd_dip.clear();
    t.jd_emb.clear();
    t.big_smem.clear();
    t.resident.clear();
}

// Opt a kernel in to `bytes` of dynamic shared memory once per context.
static bool devtables_big_smem(DevTables& t, const void* kernel, u32 bytes) {
    {
        std::lock_guard<std::mutex> guard(t.lock);
        if (t.big_smem.count(kernel)) return true;
    }
    if (cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)bytes) != cudaSuccess)
        return false;
    std::lock_guard<std::mutex> guard(t.lock);
    t.big_smem.insert(kernel);
    return true;
}

// Resident blocks per SM of `kernel` at `threads` threads (cached per context).
static int devtables_resident(DevTables& t, const void* kernel, u32 threads, u32 smem) {
    {
        std::lock_guard<std::mutex> guard(t.lock);
        auto it = t.resident.find({kernel, threads});
        if (it != t.resident.end()) return it->second;
    }
    int per_sm = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, (int)threads, smem) != cudaSuccess) per_sm = 0;
    std::lock_guard<std::mutex> guard(t.lock);
    t.resident[{kernel, threads}] = per_sm;
    return per_sm;
}
