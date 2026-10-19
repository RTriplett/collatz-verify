#!/usr/bin/env bash
# End-to-end check of the 24 GB configuration: B=34 for classes 0..7 and B=28
# for the rest, shared bitvectors built on the GPU (16.25 GiB resident),
# window credits, step-1 pm_inf + mix(3^13).  Prediction first, then the run.
set -euo pipefail
cd "$(dirname "$0")"
CV=../../target/release/cv
CFG="--n 56 --a 10 --b 28 --b-hi 34 --hi-classes 8 --bv-shared --bv-pm-inf --bv-mix-j 13 --rules improved --mix-j 13"
$CV measure $CFG --gpu --per-class 50000 --gpu-per-class 2097152 --tree-samples 50000 --json measure_n56_bhi34.json > measure_n56_bhi34.txt
/usr/bin/time -f "wall=%e user=%U maxrss=%MKB" $CV verify $CFG --gpu --json verify_n56_bhi34.json > verify_n56_bhi34.txt 2>&1
echo done
