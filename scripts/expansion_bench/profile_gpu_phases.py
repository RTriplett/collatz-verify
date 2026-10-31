#!/usr/bin/env python3
"""Build a diagnostic GPU-phase overlay without changing production sources.

The overlay samples every live lane in selected CUDA blocks. Four clock64
intervals partition each sampled lane's residence in the verifier: endpoint
setup, completion screening/assembly, trajectory verification, and checksum
reduction. These clocks include stalls, other-warp scheduling, and profiling
instructions. They are NOT additive GPU execution seconds.

Only warm aggregate trials are recorded, through the benchmark's existing
CV_FULLRANGE_PROFILE_AGGREGATE=1 hook. CV_GPU_PHASE_STRIDE defaults to 128;
CV_GPU_PHASE_SEED changes the starting block offset. The offset advances for
each verifier launch. Set stride to zero for an instrumented-build control.

Example:
  python3 scripts/expansion_bench/profile_gpu_phases.py --build

The script never runs a benchmark or changes profiling permissions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
MARKER = ".collatz-gpu-phase-overlay"

PHASE_HEADER = r'''// Generated diagnostic overlay. Production source is unchanged.
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
'''


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one source anchor, found {count}")
    return text.replace(old, new, 1)


def instrument_kernel(text: str) -> str:
    text = replace_once(text, 'typedef uint16_t u16;\n',
                        'typedef uint16_t u16;\n\n#include "phase_profile.cuh"\n', "header")
    text = replace_once(text, 'u32* deferred_count = nullptr) {',
                        'u32* deferred_count = nullptr,\n'
                        '                          u32 phase_stride = 0, u32 phase_offset = 0) {', "parameters")
    text = replace_once(text, '    const bool live = idx < n;\n', '''    const bool live = idx < n;
    const bool phase_sample = live && phase_stride && blockIdx.x % phase_stride == phase_offset;
    u64 phase_setup = 0, phase_screening = 0, phase_tail = 0;
    const u64 phase_begin = phase_sample ? cv_phase_clock() : 0;
    u64 phase_mark = phase_begin;
''', "begin")
    text = replace_once(text, '        u32 w = 0;\n', '''        if (phase_sample) {
            const u64 now = cv_phase_clock();
            phase_setup = now - phase_mark;
            phase_mark = now;
        }
        u32 w = 0;
''', "setup end")
    text = replace_once(text, '            float base_upper = 0, cutoff = 0;\n', '''            if (phase_sample) {
                const u64 now = cv_phase_clock();
                phase_screening += now - phase_mark;
                phase_mark = now;
            }
            float base_upper = 0, cutoff = 0;
''', "tail begin")
    text = replace_once(text, '''            }
        }
    }
done:
    // Warp-reduce, one atomic per warp.
''', '''            }
            if (phase_sample) {
                const u64 now = cv_phase_clock();
                phase_tail += now - phase_mark;
                phase_mark = now;
            }
        }
    }
done:
    const u32 phase_cases = cases;
    const u64 phase_iters = iters;
    if (phase_sample) {
        const u64 now = cv_phase_clock();
        phase_screening += now - phase_mark;
        phase_mark = now;
    }
    // Warp-reduce, one atomic per warp.
''', "tail end / reduction begin")
    text = replace_once(text, '''        atomicAdd(&out->iters, (unsigned long long)iters);
    }
}
''', '''        atomicAdd(&out->iters, (unsigned long long)iters);
    }
    if (phase_sample) {
        const u64 now = cv_phase_clock();
        const u64 reduction = now - phase_mark;
        // All diagnostic global writes occur after the final timestamp.
        atomicAdd(&cv_phase_counters.sampled_lanes, 1ull);
        if (threadIdx.x == 0) atomicAdd(&cv_phase_counters.sampled_blocks, 1ull);
        atomicAdd(&cv_phase_counters.cases, (unsigned long long)phase_cases);
        atomicAdd(&cv_phase_counters.double_iterations, (unsigned long long)phase_iters);
        atomicAdd(&cv_phase_counters.setup_cycles, (unsigned long long)phase_setup);
        atomicAdd(&cv_phase_counters.screening_cycles, (unsigned long long)phase_screening);
        atomicAdd(&cv_phase_counters.tail_cycles, (unsigned long long)phase_tail);
        atomicAdd(&cv_phase_counters.reduction_cycles, (unsigned long long)reduction);
        atomicAdd(&cv_phase_counters.total_cycles, (unsigned long long)(now - phase_begin));
    }
}
''', "reduction end")
    text = replace_once(text, '''                     TailContinuation* deferred, u32* deferred_count) {
''', '''                     TailContinuation* deferred, u32* deferred_count) {
    const u32 phase_stride = cv_phase_stride;
    const u32 phase_offset = phase_stride
        ? (u32)((cv_phase_sequence + cv_phase_seed) % phase_stride) : 0;
    cv_phase_sequence += 1;
''', "launch offset")
    text = replace_once(text, '                nullptr, deferred, deferred_count);',
                        '                nullptr, deferred, deferred_count, phase_stride, phase_offset);', "defer launch")
    text = replace_once(text, 'bv, mod9, input, n, magic, tp[0], tp[1], tp[2], tp[3], p, out, flags, jump_table);',
                        'bv, mod9, input, n, magic, tp[0], tp[1], tp[2], tp[3], p, out, flags,\n'
                        '            jump_table, nullptr, nullptr, phase_stride, phase_offset);', "jump launch")
    text = replace_once(text, 'bv, mod9, input, n, magic, tp[0], tp[1], tp[2], tp[3], p, out, flags)\n',
                        'bv, mod9, input, n, magic, tp[0], tp[1], tp[2], tp[3], p, out, flags, \\\n'
                        '        nullptr, nullptr, nullptr, phase_stride, phase_offset)\n', "ordinary launch")
    return text


def summarize_profile(directory: Path, confirmation: Path) -> None:
    from summarize_fullrange import summarize, work_signature

    audited = summarize(directory)
    previous = summarize(confirmation)
    if audited["aggregate_corpus_sha256"] != previous["aggregate_corpus_sha256"]:
        raise RuntimeError("phase diagnostic corpus differs from confirmation")
    rows = [json.loads(line) for line in (directory / "pipeline.jsonl").read_text().splitlines()]
    prior_rows = [json.loads(line) for line in (confirmation / "pipeline.jsonl").read_text().splitlines()]
    aggregates = [row for row in rows if row["type"] == "aggregate_trial"]
    phases = [json.loads(line) for line in (directory / "pipeline.stderr.txt").read_text().splitlines()
              if line.startswith('{"type":"gpu_phase_profile"')]
    if len(phases) != len(aggregates) or not phases:
        raise RuntimeError("phase records do not match aggregate trials")
    labels = {
        "endpoint_setup": "Endpoint state and filter setup",
        "completion_screening": "Step 2: bitvector / mod-9 / OEE screening and candidate assembly",
        "exact_trajectory": "Step 3: exact trajectory stepping and handoff",
        "checksum_reduction": "Warp completion and checksum reduction",
    }
    descriptions = {
        "endpoint_setup": "Load endpoint and exact iterate, derive bitvector offset and mod-9/OEE masks.",
        "completion_screening": "Read lookahead words, combine masks, enumerate set bits and construct each exact candidate.",
        "exact_trajectory": "Alternating odd/even runs, arithmetic-cutoff and descent checks, per-iteration checksum, plus offload queue handling when enabled.",
        "checksum_reduction": "Waiting for other lanes at warp completion, warp reductions and atomic accumulation of final verification counters. This is not pure checksum arithmetic. Diagnostic counter writes are excluded.",
    }
    grouped = {}
    for capture, (phase, row) in enumerate(zip(phases, aggregates), 1):
        if phase["capture"] != capture or set(phase["cycles"]) != set(labels):
            raise RuntimeError("unexpected phase capture order or phase set")
        if sum(phase["cycles"].values()) != phase["total_lane_cycles"]:
            raise RuntimeError("lane cycle partition does not sum exactly")
        gpu = row["report"]["gpu"]
        launches = sum(item["launches"] for item in gpu["classes"])
        if phase["verifier_launches"] != launches:
            raise RuntimeError("captured verifier launch count differs from report")
        if phase["sampled_lanes"] <= 0 or phase["sampled_blocks"] <= 0 or phase["total_lane_cycles"] <= 0:
            raise RuntimeError("empty phase sample")
        if not (phase["sampled_blocks"] <= phase["sampled_lanes"] <= phase["sampled_blocks"] * gpu["tuning"]["block_threads"]):
            raise RuntimeError("sampled block/lane accounting invalid")
        full_cases = sum(item["cases"] for item in gpu["classes"])
        full_iters = sum(item["iters"] for item in gpu["classes"]) - gpu["deferred_iterations"]
        if not (phase["sampled_cases"] <= full_cases and phase["sampled_double_iterations"] <= full_iters):
            raise RuntimeError("sampled work exceeds full GPU work")
        prior = [item for item in prior_rows if item.get("type") == "aggregate_trial" and item["variant"] == row["variant"]]
        if not prior or not all(work_signature(item["report"]) == work_signature(row["report"]) for item in prior):
            raise RuntimeError("phase diagnostic work differs from confirmation")
        grouped.setdefault(row["variant"], []).append({
            **phase, "repetition": row["repetition"], "wall_seconds": row["wall_seconds"],
            "shares": {key: value / phase["total_lane_cycles"] for key, value in phase["cycles"].items()},
            "sampled_case_fraction": phase["sampled_cases"] / full_cases,
            "sampled_iteration_fraction": phase["sampled_double_iterations"] / full_iters,
        })
    variants = []
    for variant, captures in grouped.items():
        cycles = {key: sum(capture["cycles"][key] for capture in captures) for key in labels}
        total = sum(cycles.values())
        prior_wall = [item["wall_seconds"] for item in prior_rows if item.get("type") == "aggregate_trial" and item["variant"] == variant]
        measured_wall = statistics.mean(capture["wall_seconds"] for capture in captures)
        entry = {
            "variant": variant,
            "gpu_phases": {
                "name": "Fused GPU verifier: diagnostic lane residence",
                "metric": "gpu_cycles", "cycles": total,
                "sample_count": sum(capture["sampled_lanes"] for capture in captures),
                "children": [{"name": labels[key], "cycles": cycles[key], "description": descriptions[key]} for key in labels],
                "description": "Summed sampled lane clock64 deltas across warm aggregate repetitions. Includes stalls, warp scheduling and instrumentation. These are diagnostic shares, not additive GPU execution seconds.",
                "source_file": "gpu_phase_profile/pipeline.stderr.txt",
                "attribution_method": "instrumented_lane_elapsed_cycles",
            },
            "captures": captures,
            "mean_instrumented_wall_seconds": measured_wall,
            "mean_saved_confirmation_wall_seconds": statistics.mean(prior_wall),
            "wall_ratio_to_saved_confirmation": measured_wall / statistics.mean(prior_wall),
        }
        control_dir = directory / "control"
        if (control_dir / "metadata.json").exists():
            control_audit = summarize(control_dir)
            if control_audit["aggregate_corpus_sha256"] != audited["aggregate_corpus_sha256"]:
                raise RuntimeError("fresh control corpus differs")
            control_rows = [json.loads(line) for line in (control_dir / "pipeline.jsonl").read_text().splitlines()]
            controls = [row for row in control_rows if row["type"] == "aggregate_trial" and row["variant"] == variant]
            captured = [row for row in aggregates if row["variant"] == variant]
            if not controls or any(work_signature(row["report"]) != work_signature(captured[0]["report"]) for row in controls):
                raise RuntimeError("fresh control work differs")
            control_wall = statistics.mean(row["wall_seconds"] for row in controls)
            entry["mean_fresh_control_wall_seconds"] = control_wall
            entry["wall_ratio_to_fresh_control"] = measured_wall / control_wall
            entry["fresh_control_repetitions"] = [{"repetition": row["repetition"], "wall_seconds": row["wall_seconds"]} for row in controls]
        variants.append(entry)
    result = {
        "schema_version": 1, "aggregate_corpus_sha256": audited["aggregate_corpus_sha256"],
        "correctness_audit_passed": True, "matches_confirmation_work": True,
        "variants": variants,
        "notes": [
            "Exact lane-cycle partitions; shares describe the instrumented program, not uninstrumented GPU seconds.",
            "One sampled block per 128 block indices, with offset advancing per launch and by 53 per capture; all live lanes in selected blocks are measured.",
            "Repeated aggregate trials reuse identical roots. Different offsets diagnose sensitivity; two captures are not a precision estimate.",
            "Clock reads, branches and additional register pressure perturb execution. Separate runtime controls quantify observed runtime changes.",
            "Warp-completion time includes reconvergence and waiting for longer trajectories in other lanes; it is not pure checksum arithmetic. Other phase intervals can also include SIMT waiting.",
            "Global diagnostic counter atomics occur after each sampled lane's final timestamp, but can affect other warps' scheduling.",
        ],
        "raw_sha256": {name: sha256(directory / name) for name in (
            "pipeline.jsonl", "pipeline.stderr.txt", "metadata.json", "build_metadata.json",
            "control/pipeline.jsonl", "control/pipeline.stderr.txt", "control/metadata.json")
            if (directory / name).is_file()},
    }
    output = directory / "phase_trees.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(output)
    for entry in variants:
        print(entry["variant"], {key: round(sum(capture["cycles"][key] for capture in entry["captures"]) / entry["gpu_phases"]["cycles"], 5) for key in labels})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("/tmp/collatz-gpu-phase-source"))
    parser.add_argument("--target-dir", type=Path, default=Path("/tmp/collatz-gpu-phase-target"))
    parser.add_argument("--provenance-dir", type=Path, default=ROOT / "results/tail_offload/gpu_phase_profile")
    parser.add_argument("--refresh", action="store_true", help="replace a previously generated overlay")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--summarize", action="store_true", help="audit completed captures and write phase_trees.json")
    parser.add_argument("--confirmation", type=Path, default=ROOT / "results/tail_offload/confirmation")
    args = parser.parse_args()
    source = args.source_dir.resolve()
    target = args.target_dir.resolve()
    provenance = args.provenance_dir.resolve()
    if args.summarize:
        summarize_profile(provenance, args.confirmation.resolve())
        return
    if source == ROOT or ROOT in source.parents or Path("/tmp") not in source.parents:
        parser.error("source-dir must be an isolated directory under /tmp")
    if Path("/tmp") not in target.parents:
        parser.error("target-dir must be under /tmp")
    if source.exists():
        if not args.refresh or not (source / MARKER).is_file():
            parser.error("source-dir already exists; --refresh requires the overlay marker")
        shutil.rmtree(source)
    source.mkdir(parents=True)
    (source / MARKER).write_text("Generated by profile_gpu_phases.py\n")
    inputs = [ROOT / name for name in ("Cargo.toml", "Cargo.lock", "build.rs")]
    inputs += sorted((ROOT / "src").rglob("*"))
    inputs += sorted((ROOT / "cuda").rglob("*"))
    bench = ROOT / "scripts/expansion_bench"
    inputs += sorted(bench.glob("*.rs"))
    inputs += [bench / "Cargo.toml", bench / "Cargo.lock"]
    hashes = {}
    for original in inputs:
        if not original.is_file():
            continue
        relative = original.relative_to(ROOT)
        dest = source / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, dest)
        hashes[str(relative)] = sha256(original)
    kernel = source / "cuda/kernel.cu"
    kernel.write_text(instrument_kernel(kernel.read_text()))
    header = source / "cuda/phase_profile.cuh"
    header.write_text(PHASE_HEADER)
    build = source / "build.rs"
    build.write_text(replace_once(build.read_text(), '.args(["-O3", "-std=c++17",',
                                 '.args(["-O3", "-lineinfo", "-Xptxas=-v", "-std=c++17",', "nvcc flags"))
    benchmark = source / "scripts/expansion_bench/fullrange_bench.rs"
    bench_text = benchmark.read_text()
    for original, replacement in (("cudaProfilerStart", "cv_gpu_phase_start"),
                                  ("cudaProfilerStop", "cv_gpu_phase_stop")):
        if bench_text.count(original) != 3:
            raise RuntimeError(f"expected three occurrences of {original} in benchmark")
        bench_text = bench_text.replace(original, replacement)
    benchmark.write_text(bench_text)
    provenance.mkdir(parents=True, exist_ok=True)
    for path in (kernel, header, build, benchmark):
        dest = provenance / "source" / path.relative_to(source)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    command = ["cargo", "build", "--release", "--offline", "--locked", "--manifest-path",
               str(source / "scripts/expansion_bench/Cargo.toml"), "--features", "cuda", "--bin", "fullrange_bench"]
    metadata = {
        "schema_version": 1, "generator": str(Path(__file__).relative_to(ROOT)),
        "generator_sha256": sha256(Path(__file__)), "source_dir": str(source),
        "target_dir": str(target), "original_source_sha256": hashes,
        "overlay_sha256": {str(path.relative_to(source)): sha256(path) for path in (kernel, header, build, benchmark)},
        "build_command": command, "build_environment": {"CUDA_ARCH": "sm_89", "CARGO_TARGET_DIR": str(target)},
        "metric": "sampled lane elapsed cycles, including stalls and instrumentation; not GPU seconds",
        "sampling": "Every live lane in blocks whose index modulo stride equals (launch sequence + seed + 53*capture_index) modulo stride",
        "partition": ["endpoint_setup", "completion_screening", "exact_trajectory", "checksum_reduction"],
        "excluded_from_clocks": "profiler counter atomics after final timestamp; sampling predicate and clock setup before first timestamp",
    }
    if args.build:
        env = os.environ.copy()
        env.update(metadata["build_environment"])
        env["CARGO_PROFILE_RELEASE_DEBUG"] = "1"
        metadata["build_environment"]["CARGO_PROFILE_RELEASE_DEBUG"] = "1"
        metadata["compiler_versions"] = {}
        for label, version_command in (("rustc", ["rustc", "--version"]),
                                       ("cargo", ["cargo", "--version"]),
                                       ("nvcc", [str(Path(env.get("CUDA_HOME", "/usr/local/cuda")) / "bin/nvcc"), "--version"])):
            version = subprocess.run(version_command, env=env, text=True, capture_output=True, check=True)
            metadata["compiler_versions"][label] = version.stdout.strip()
        with (provenance / "build.log").open("w") as log:
            completed = subprocess.run(command, env=env, cwd=source, stdout=log, stderr=subprocess.STDOUT)
        metadata["build_returncode"] = completed.returncode
        if completed.returncode == 0:
            binary = target / "release/fullrange_bench"
            metadata["binary_sha256"] = sha256(binary)
    (provenance / "build_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"source_dir": str(source), "target_dir": str(target),
                      "metadata": str(provenance / "build_metadata.json"),
                      "build_returncode": metadata.get("build_returncode")}, indent=2))
    if args.build and metadata["build_returncode"]:
        sys.exit(metadata["build_returncode"])


if __name__ == "__main__":
    main()
