#!/usr/bin/env bash
# Lookahead-window sweep at N=72: cases and iterations for B up to 38, by
# direct window evaluation (no arrays), with window credits and step-1 mix.
set -euo pipefail
cd "$(dirname "$0")/../.."
CV=./target/release/cv
N=${N:-72}
for A in 10 12 14; do
  case $A in 10) PC=20000;; 12) PC=8000;; 14) PC=3000;; esac
  for B in 24 26 28 30 32 34 36 38; do
    out=results/bsweep/n${N}_a${A}_b${B}.json
    [ -s "$out" ] && continue
    $CV measure --n $N --a $A --b $B --bv-pm-inf --bv-mix-j 13 --rules improved --mix-j 13 \
      --direct --per-class $PC --tree-samples 20000 --json "$out" > "${out%.json}.txt"
    echo "done A=$A B=$B $(date +%T)"
  done
done
