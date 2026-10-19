#!/usr/bin/env bash
# Regression anchors: published counts reproduced exactly, CPU/GPU agreement,
# and the work-package decomposition.  ~6 minutes on a 16-thread CPU + GPU.
# Usage: scripts/reproduce.sh [--no-gpu]
set -euo pipefail
cd "$(dirname "$0")/.."
GPU=1
[ "${1:-}" = "--no-gpu" ] && GPU=0
CV=./target/release/cv
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
fail=0
check() { # name expected actual
  if [ "$2" = "$3" ]; then echo "PASS  $1  ($3)"; else echo "FAIL  $1  expected $2 got $3"; fail=1; fi
}
field() { tr ' ' '\n' | sed -n "s/^$1=//p" | head -1; }
survivors() { awk -F, -v k="$1" '$1==k {print $3}'; }

echo "== step-1 survivor counts"
check "A076227 (descent only), k=30" 12771274 "$($CV census --depth 30 --rules descent | survivors 30)"
check "Angeltveit set, k=37 (collatz-probe CSV)" 575349866 "$($CV census --depth 37 | survivors 37)"
check "Angeltveit CPU default (unarmed OEE), k=30" 8555400 "$($CV census --depth 30 --no-root-oee | survivors 30)"
check "+pm_inf+mix(3^10), k=37 (CSV)" 548786150 "$($CV census --depth 37 --rules improved --mix-j 10 | survivors 37)"
check "+pm_inf+mix(3^16), k=37 (CSV)" 543518371 "$($CV census --depth 37 --rules improved --mix-j 16 | survivors 37)"
check "residue-sieve P_inf+M8, k=30" 7360216 "$($CV census --depth 30 --rules improved --mix-j 8 --no-pm1 --no-oee | survivors 30)"
check "residue-sieve P_inf+M16, k=30" 6921256 "$($CV census --depth 30 --rules improved --mix-j 16 --no-pm1 --no-oee | survivors 30)"
check "canonical walked W_26" 602780 "$($CV census --depth 26 --rules improved --mix-j 0 --no-pm1 --no-oee --walkback-depth 26 --walkback-states 10000000 | survivors 26)"

echo "== cases to check (Angeltveit Fig. 5.1, A=6, 8 bitvectors, reference bitvectors)"
check "N=38" 205942265 "$($CV verify --n 38 --a 6 --nbv 8 --bv-legacy | field cases)"
check "N=40" 756583624 "$($CV verify --n 40 --a 6 --nbv 8 --bv-legacy | field cases)"

echo "== exact DP prediction vs full run (sound bitvectors, A=10)"
run=$($CV verify --n 40)
pred=$($CV predict --from 40 --to 40 | cut -d, -f6)
check "N=40 cases within 0.01% of prediction" ok "$(python3 -c "print('ok' if abs($pred/$(echo "$run" | field cases)-1)<1e-4 else 'off')")"

if [ $GPU = 1 ]; then
  echo "== GPU-built bitvectors equal the host build"
  for opts in "--b 24 --bv-pm-inf --bv-mix-j 13" "--b 28 --bv-shared --bv-pm-inf --bv-mix-j 13" "--b 26 --bv-legacy"; do
    if $CV gpu-check --n 60 $opts > "$TMP/gc.txt"; then echo "PASS  gpu-check $opts"; else echo "FAIL  gpu-check $opts"; cat "$TMP/gc.txt"; fail=1; fi
  done
  echo "== CPU and GPU agree bit for bit (host arrays, and device-built per-class windows)"
  for opts in "--n 40" "--n 42 --a 12 --b 28 --bv-pm-inf --bv-mix-j 13 --rules improved --mix-j 13" \
              "--n 40 --b 26 --class-b 31,31,30,30,29,29,28,27 --bv-shared --bv-pm-inf --bv-mix-j 13 --rules improved --mix-j 13"; do
    c=$($CV verify $opts); g=$($CV verify $opts --gpu)
    check "$opts: cases" "$(echo "$c" | field cases)" "$(echo "$g" | field cases)"
    check "$opts: hash" "$(echo "$c" | field hash)" "$(echo "$g" | field hash)"
  done
fi

echo "== work packages reproduce the full run"
$CV measure --n 40 --per-class 20000 --tree-samples 0 --json "$TMP/m.json" > /dev/null
$CV plan --n 40 --costs "$TMP/m.json" --target-hours 0.0000002 --out "$TMP/plan.json" > /dev/null
np=$(python3 -c "import json;print(len(json.load(open('$TMP/plan.json'))['packages']))")
for i in $(seq 0 $((np - 1))); do $CV --threads 4 verify --n 40 --plan "$TMP/plan.json" --package "$i"; done > "$TMP/pk.txt"
sum=$(python3 -c "
import sys
c=0; h=0
for l in open('$TMP/pk.txt'):
    d=dict(x.split('=',1) for x in l.split() if '=' in x); c+=int(d['cases']); h^=int(d['hash'],16)
print(f'{c}/{h:08X}')")
check "$np packages: cases/hash" "$(echo "$run" | field cases)/$(echo "$run" | field hash)" "$sum"

echo "== unit and audit tests"
if cargo test --release --quiet > "$TMP/test.txt" 2>&1; then
  echo "PASS  cargo test (incl. bitvector soundness audit)"
else
  echo "FAIL  cargo test"; grep -E "FAILED|panicked" "$TMP/test.txt" | head; fail=1
fi

exit $fail
