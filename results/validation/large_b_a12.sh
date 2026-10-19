#!/usr/bin/env bash
# The recommended 24 GB configuration on a 16-thread host: A=12 (3.6x less
# step-1 CPU work), per-class windows 32,35,35,35,35,35,32,32 (21.5 GiB of
# shared bitvectors built on the GPU), window credits, step-1 pm_inf + mix(3^13).
set -euo pipefail
cd "$(dirname "$0")"
CV=../../target/release/cv
CFG="--n 56 --a 12 --b 28 --class-b 32,35,35,35,35,35,32,32 --bv-shared --bv-pm-inf --bv-mix-j 13 --rules improved --mix-j 13"
$CV measure $CFG --gpu --per-class 30000 --gpu-per-class 2097152 --tree-samples 50000 --json measure_n56_a12_cls.json > measure_n56_a12_cls.txt
/usr/bin/time -f "wall=%e user=%U maxrss=%MKB" $CV verify $CFG --gpu --json verify_n56_a12_cls.json > verify_n56_a12_cls.txt 2>&1
echo done
