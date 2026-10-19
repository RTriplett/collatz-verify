#!/usr/bin/env bash
# Sweep of `cv measure` over N for the configurations in docs/COST_MODEL.md.
# Each run: exact DP counts x uniform draws (CPU work per endpoint) x
# serialized RTX 4090 kernel timing (2^21 draws per class, median of 3).
set -euo pipefail
cd "$(dirname "$0")/../.."
CV=./target/release/cv
COMMON="--gpu --per-class 50000 --gpu-per-class 2097152 --tree-samples 50000"
for N in ${SWEEP_N:-64 68 70 71 72 73 74 75 76 77 78 80}; do
  [ -s results/sweep/n${N}_a12b28w.json ] && continue
  case $N in 64|72|76|80) $CV measure --n $N $COMMON --a 10 --b 24 --bv-legacy \
      --json results/sweep/n${N}_ref-legacy.json > results/sweep/n${N}_ref-legacy.txt ;; esac
  $CV measure --n $N $COMMON --a 10 --b 24 --rules improved --mix-j 13 \
      --json results/sweep/n${N}_a10b24.json > results/sweep/n${N}_a10b24.txt
  $CV measure --n $N $COMMON --a 10 --b 28 --bv-pm-inf --bv-mix-j 13 --rules improved --mix-j 13 \
      --json results/sweep/n${N}_a10b28w.json > results/sweep/n${N}_a10b28w.txt
  $CV measure --n $N $COMMON --a 12 --b 28 --bv-pm-inf --bv-mix-j 13 --rules improved --mix-j 13 \
      --json results/sweep/n${N}_a12b28w.json > results/sweep/n${N}_a12b28w.txt
  echo "done N=$N $(date +%T)"
done
