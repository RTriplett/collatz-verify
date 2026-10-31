#!/usr/bin/env bash
# Run from repository root; output files are replaced.
CV_FULLRANGE_COHORTS=4 CV_FULLRANGE_CUTOFF=50 CV_FULLRANGE_ENTRY=52 CV_FULLRANGE_AGGREGATE=2 CV_FULLRANGE_PROFILE_AGGREGATE=0 taskset -c 0-7 /tmp/collatz-tail-offload-target/release/fullrange_bench target 128 1 2 20261101 novote_oee,defer16_cap262k_w4 32,35,35,35,35,34,32,32,27,27,27,27,27,27,27,27 10 > results/tail_offload/gpu_phase_profile/control/pipeline.jsonl 2> results/tail_offload/gpu_phase_profile/control/pipeline.stderr.txt
