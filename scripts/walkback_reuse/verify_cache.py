"""Independent arbitrary-precision audit of every serialized certificate."""
import argparse
import hashlib
import json
import struct
import time
from pathlib import Path

if not __debug__:
    raise SystemExit("Certificate audit requires enabled assertions; rerun Python without -O or -OO.")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("results_dir", nargs="?", type=Path,
                    default=Path("results/walkback_reuse/cache-k30"))
root = parser.parse_args().results_dir
meta = json.loads((root / "metadata.json").read_text())
raw = (root / "certificates-k30.bin").read_bytes()
assert hashlib.sha256(raw).hexdigest() == meta["binary_sha256"]
assert len(raw) == meta["binary_bytes"] == 32 * meta["certificate_count"]
k = meta["checkpoint_depth"]

def iterate(n, steps):
    odd = 0
    for _ in range(steps):
        if n % 2:
            n = (3 * n + 1) // 2
            odd += 1
        else:
            n //= 2
    return n, odd

start = time.perf_counter()
previous = -1
checked = 0
for offset in range(0, len(raw), 32):
    n0 = int.from_bytes(raw[offset:offset + 8], "little")
    p = int.from_bytes(raw[offset + 8:offset + 24], "little")
    s, o = struct.unpack_from("<II", raw, offset + 24)
    assert previous < n0 < 2**k
    previous = n0
    m, f = iterate(n0, k)
    assert 0 < p < n0 and s >= 1 and 0 <= o <= f
    assert iterate(p, s) == (m, o)
    slope = 2**s * 3**(f - o)
    assert slope <= 2**k
    checked += 1

out = {
    "verifier": "Independent Python arbitrary-precision forward iteration and slope checks",
    "records_checked": checked,
    "sha256": hashlib.sha256(raw).hexdigest(),
    "all_passed": True,
    "seconds": time.perf_counter() - start,
}
(root / "python-audit.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
