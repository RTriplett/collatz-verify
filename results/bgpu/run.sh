#!/usr/bin/env bash
# Real RTX 4090 kernel timing at large lookahead windows (device-built
# per-class arrays, 2 x 2^B bits resident), N=72.  Each run also checks that
# the GPU reproduces the CPU's exact cases/iterations on the CPU draws.
set -euo pipefail
cd "$(dirname "$0")/../.."
CV=./target/release/cv
for AB in "10 30" "10 32" "10 34" "10 36" "14 32" "14 36" "12 36"; do
  set -- $AB; A=$1; B=$2
  out=results/bgpu/n72_a${A}_b${B}.json
  [ -s "$out" ] && continue
  $CV measure --n 72 --a $A --b $B --bv-pm-inf --bv-mix-j 13 --rules improved --mix-j 13 --direct --gpu \
    --per-class $([ $A = 10 ] && echo 20000 || echo 4000) --gpu-per-class 2097152 --tree-samples 0 \
    --json "$out" > "${out%.json}.txt" 2>&1
  echo "done A=$A B=$B $(date +%T)"
done
