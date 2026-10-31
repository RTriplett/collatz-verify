#!/usr/bin/env python3
"""Regression checks for missing work, bad timing accounting and fixture identity."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from summarize_early_cpu import audit_rows, fixture_identity, summarize_run


def valid_rows():
    config = {"type": "config", "experiment": "early_cpu_replay", "states": 4,
              "available_states": 4, "input_bytes": 128, "workers": 1,
              "cutoff_bits": 120, "repetitions": 3, "modes": [0, 2, 3, 4],
              "widths": [8], "chunks": [4]}
    oracle = {"type": "oracle", "cases": 4, "iters": 12, "hash": 23,
              "flags": 2, "unique_wide_rechecks": 1, "all_wide_rechecks_passed": True}
    rows = [config, oracle, {"type": "table_setup", "width": 8, "wall_seconds": 0.01,
                            "cpu_seconds": 0.01, "table_setup_seconds": 0.009}]
    modes = config["modes"]
    for rep in range(3):
        order = [(i + 13 * rep) % 4 for i in range(4)]
        if rep % 2:
            order.reverse()
        for position, index in enumerate(order):
            mode = modes[index]
            cpu = 0.8 + 0.1 * mode
            rows.append({"type": "trial", "mode": mode, "width": 0 if mode == 0 else 8,
                         "mode_name": ["scalar", "bucket_scalar", "table", "initial_group_table", "regrouped_table"][mode],
                         "chunk_size": 4, "repetition": rep, "order_position": position,
                         "cases": 4, "iters": 12, "hash": 23, "flags": 2,
                         "buckets": {0: 0, 2: 0, 3: 2, 4: 4}[mode],
                         "jumps": 0 if mode == 0 else 3,
                         "jumped_iterations": 0 if mode == 0 else 6,
                         "scalar_iterations": 12 if mode == 0 else 6,
                         "table_lookups": {0: 0, 2: 6, 3: 4, 4: 4}[mode],
                         "wall_seconds": 1.0 + mode * 0.1, "cpu_seconds": cpu,
                         "grouping_seconds": 0.1 if mode in (3, 4) else 0.0,
                         "verification_seconds": 0.4, "table_setup_seconds": 0.0,
                         "cases_per_cpu_second": 4 / cpu, "iterations_per_cpu_second": 12 / cpu,
                         "exact_match": True})
    rows.append({"type": "complete", "configs": 4, "trials": 12, "exact_match": True})
    return rows


class AuditTests(unittest.TestCase):
    def test_accepts_complete_balanced_replay(self):
        _, _, _, groups = audit_rows(valid_rows())
        self.assertEqual(len(groups), 4)

    def test_rejects_missing_or_duplicate_trials_and_false_work(self):
        for mutation in (lambda rows: rows.pop(4),
                         lambda rows: rows.insert(4, copy.deepcopy(rows[3])),
                         lambda rows: rows[3].update(hash=24),
                         lambda rows: rows[3].update(flags=1),
                         lambda rows: rows[3].update(exact_match=False),
                         lambda rows: rows[3].update(order_position=3),
                         lambda rows: rows[-1].update(trials=11)):
            rows = valid_rows()
            mutation(rows)
            with self.assertRaises(ValueError):
                audit_rows(rows)

    def test_rejects_invalid_timing_rates_and_jump_counts(self):
        for update in ({"wall_seconds": float("nan")}, {"cpu_seconds": 0},
                       {"verification_seconds": 100}, {"cases_per_cpu_second": 1},
                       {"jumped_iterations": 10}, {"table_lookups": 1}):
            rows = valid_rows()
            rows[3].update(update)
            with self.assertRaises(ValueError):
                audit_rows(rows)

    def test_rejects_inconsistent_regrouped_lookups(self):
        rows = valid_rows()
        next(r for r in rows if r.get("mode") == 4)["table_lookups"] += 1
        with self.assertRaisesRegex(ValueError, "regrouped lookup"):
            audit_rows(rows)

    def test_fixture_digest_and_metadata_are_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            fixture = directory / "fixture.bin"
            fixture.write_bytes(bytes(range(128)))
            identity = fixture_identity(fixture, {})
            rows = valid_rows()
            rows[0].update(input=str(fixture), input_fingerprint_fnv1a64=identity["fnv1a64"])
            replay = directory / "replay.jsonl"
            replay.write_text("".join(json.dumps(row) + "\n" for row in rows))
            metadata = directory / "metadata.json"
            metadata.write_text(json.dumps({"fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest()}))
            result = summarize_run(replay, metadata=metadata)
            self.assertTrue(result["exact_work_audit_passed"])
            self.assertEqual(result["variants"][2]["table_lookup_reduction_vs_independent"], 1 - 4 / 6)
            metadata.write_text(json.dumps({"fixture_sha256": "0" * 64}))
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                summarize_run(replay, metadata=metadata)
            fixture.write_bytes(bytes(128))
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                summarize_run(replay)


if __name__ == "__main__":
    unittest.main()
