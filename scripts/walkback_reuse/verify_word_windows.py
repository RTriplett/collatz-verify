#!/usr/bin/env python3
"""Independently audit cutoff-changing examples and their harvested origin."""

import argparse
import hashlib
import json
import struct
from pathlib import Path

if not __debug__:
    raise SystemExit("Run without -O/-OO: this auditor requires assertions.")


def trajectory(n, steps):
    odd_count = bits = constant = 0
    for i in range(steps):
        if n & 1:
            n = (3 * n + 1) // 2
            odd_count += 1
            bits |= 1 << i
            constant = 3 * constant + (1 << i)
        else:
            n //= 2
    return n, odd_count, bits, constant


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results_dir", nargs="?", default="results/walkback_reuse/word-window"
    )
    parser.add_argument(
        "--cache", default="results/walkback_reuse/cache-k30/certificates-k30.bin"
    )
    args = parser.parse_args()
    filenames = sorted(Path(args.results_dir).glob("b*-10m-mix*.json"))
    assert filenames, "No word-window sample results found"
    raw = Path(args.cache).read_bytes()
    assert len(raw) % 32 == 0

    certificates = []
    for i in range(0, len(raw), 32):
        n0 = int.from_bytes(raw[i : i + 8], "little")
        p = int.from_bytes(raw[i + 8 : i + 24], "little")
        s, o = struct.unpack("<II", raw[i + 24 : i + 32])
        m, _, _, _ = trajectory(n0, 30)
        certificates.append(
            (
                i // 32,
                n0,
                p,
                s,
                o,
                (m << s) - 3**o * p,
                hashlib.sha256(raw[i : i + 32]).hexdigest(),
            )
        )

    results = []
    for filename in filenames:
        data = json.loads(filename.read_text())
        example = data["example"]
        word = example["word"]
        s, o, constant = word["s"], word["o"], int(word["b"])
        assert (
            example["base_dip"]
            <= example["cutoff"]
            < example["word_dip"]
            <= example["extended_dip"]
        )
        assert o <= example["fw"]
        assert example["current_dip"] == 306 * example["j"] - 485 * example["fw"]
        assert example["word_dip"] == example["current_dip"] + 485 * o - 306 * s
        assert (word["residue"] * (1 << s) - constant) % 3**o == 0

        for check in example["checks"]:
            start, q = int(check["window_start"]), int(check["q"])
            rep = example["x0_plus_one_pattern"] or (1 << example["window_b"])
            assert start == rep - 1 + q * (1 << example["window_b"])
            y, fw, _, _ = trajectory(start, example["j"])
            assert y == int(check["merge_y"]) and fw == example["fw"]
            assert y % 3**o == word["residue"]
            numerator = (y << s) - constant
            assert numerator > 0 and numerator % 3**o == 0
            p = numerator // 3**o
            assert p == int(check["partner_p"])
            merged, odds, bits, recovered_constant = trajectory(p, s)
            assert merged == y and odds == o
            assert bits == word["forward_bits"] and recovered_constant == constant

        origin = next(c for c in certificates if c[3:6] == (s, o, constant))
        results.append(
            {
                "result_file": filename.name,
                "base_mix_j": data["base_mix_j"],
                "window_b": data["window_b"],
                "translated_examples_checked": len(example["checks"]),
                "origin_record": {
                    "index": origin[0],
                    "n0": origin[1],
                    "p": origin[2],
                    "s": origin[3],
                    "o": origin[4],
                    "b": origin[5],
                    "record_sha256": origin[6],
                },
                "passed": True,
            }
        )
    print(
        json.dumps(
            {"cache_sha256": hashlib.sha256(raw).hexdigest(), "results": results},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
