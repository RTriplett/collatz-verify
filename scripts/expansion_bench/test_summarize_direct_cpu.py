#!/usr/bin/env python3
"""Malformed-report tests for direct CPU experiment accounting."""

import copy
import statistics
import unittest

from summarize_direct_cpu import audit_rows, fingerprint, iterate_source


def valid_rows():
    source = 4065654739835021807
    sources = [source, source]
    n0 = list(map(str, sources))
    fnv, _ = fingerprint(sources)
    classes, aa, bb, reps = [40], [10, 6], [0, 12], 3
    combinations = [(a, b, f) for a in aa for b in bb for f in classes]
    count = len(sources)
    preview = [{"n0": str(source), "m": str(iterate_source(source)[0]), "k": 62, "f": 40}] * count
    config = {"type": "config", "experiment": "direct_cpu_complete_source_families",
              "n": 72, "source_m": 62, "source_a": 10, "timing_threads": 1,
              "bv_mode": "Shared", "cutoff_bits": 120, "setup_threads": 1, "seed": 17,
              "count_per_class": count, "repetitions": reps, "audit_sources_per_class": count,
              "classes": classes, "a_values": aa, "b_values": bb,
              "cohorts": [{"source_f": 40, "count": count, "source_n0": n0,
                           "ordered_n0_fnv1a64_le128": fnv, "duplicates_retained": 1,
                           "draws": 3, "rejected_before_or_at_k50": 1, "baseline_population": "12345",
                           "sampling_wall_seconds": 0.01, "first_8_sources": preview}]}
    rows = [config]
    work = {}
    for a, b, f in combinations:
        cases = (4 if a == 10 else 8) - (2 if b else 0)
        work[(a, b, f)] = {"source_endpoints": count, "endpoints": 2 if a == 10 else 8,
                           "cases": cases, "iters": cases * 2, "hash": f"{cases:08X}", "flagged": []}
        rows.append({"type": "setup", "a": a, "m": 72-a, "b": b,
                     "wall_seconds": 0.01, "table_bytes": 123, "lookahead_bytes": 456})
        prefix, b_cull = (0 if a == 10 else 100), (1 if b else 0)
        rows.append({"type": "validation", "a": a, "b": b, "source_f": f,
                     "audit": {"exact_equal": True, "sources": count, "original_starts": count*1024,
                               "prefix_culled": prefix, "mod9_culled": count*1024-cases-prefix-b_cull,
                               "oee_culled": 0, "b_culled": b_cull, "verified": cases,
                               "all_starts_independent_wide_steps": count*2048,
                               "work": copy.deepcopy(work[(a, b, f)])}})
    def timing(rep):
        return {"wall_seconds": 0.1 + rep * 0.01, "thread_cpu_seconds": 0.09 + rep * 0.01,
                "expansion_screen_tail_wall_seconds": 0.08, "expansion_screen_tail_thread_cpu_seconds": 0.07,
                "wide_recheck_wall_seconds": 0.005, "wide_recheck_steps": 0}
    for a, b, f in combinations:
        rows.append({"type": "warmup", "a": a, "b": b, "source_f": f,
                     "work": copy.deepcopy(work[(a, b, f)]), "timing": timing(0)})
    for rep in range(reps):
        offset = rep * 11 % len(combinations)
        order = combinations[offset:] + combinations[:offset]
        if rep % 2:
            order.reverse()
        for position, (a, b, f) in enumerate(order):
            rows.append({"type": "trial", "a": a, "m": 72-a, "b": b, "source_f": f,
                         "repetition": rep, "position": position,
                         "work": copy.deepcopy(work[(a, b, f)]), "timing": timing(rep)})
    times = [timing(rep)["wall_seconds"] for rep in range(reps)]
    for a, b, f in combinations:
        rows.append({"type": "summary", "a": a, "m": 72-a, "b": b, "source_f": f,
                     "repetitions": reps, "median_wall_seconds": statistics.median(times),
                     "min_wall_seconds": min(times), "max_wall_seconds": max(times),
                     "median_ns_per_source": statistics.median(times)*1e9/count,
                     "work": copy.deepcopy(work[(a, b, f)])})
    rows.append({"type": "complete", "exact_equal": True,
                 "configurations": len(combinations), "trials": len(combinations)*reps})
    return rows


def first(rows, kind):
    return next(row for row in rows if row["type"] == kind)


class AuditTests(unittest.TestCase):
    def test_complete_policy_local_signatures_and_duplicates(self):
        config, setup, identities, groups = audit_rows(valid_rows())
        self.assertEqual(len(setup), 4)
        self.assertEqual(identities[40]["duplicates_retained"], 1)
        self.assertNotEqual(groups[(10, 0, 40)][0]["work"], groups[(6, 0, 40)][0]["work"])

    def test_incomplete_duplicate_and_wrong_order_trials(self):
        for mutation in (lambda rows: rows.pop(),
                         lambda rows: rows.insert(-1, copy.deepcopy(first(rows, "trial"))),
                         lambda rows: rows.remove(first(rows, "trial")),
                         lambda rows: first(rows, "trial").update(position=1)):
            rows = valid_rows()
            mutation(rows)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                audit_rows(rows)

    def test_source_fingerprints_and_duplicate_counts(self):
        for field, value in (("ordered_n0_fnv1a64_le128", "0"*16), ("duplicates_retained", 0),
                             ("draws", 99), ("source_n0", ["3", "3"])):
            rows = valid_rows()
            rows[0]["cohorts"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                audit_rows(rows)

    def test_wrong_source_class_even_with_updated_fingerprint(self):
        rows = valid_rows()
        cohort = rows[0]["cohorts"][0]
        cohort["source_n0"] = ["3", "3"]
        cohort["ordered_n0_fnv1a64_le128"] = fingerprint([3, 3])[0]
        with self.assertRaisesRegex(ValueError, "another F class"):
            audit_rows(rows)

    def test_changed_work_or_unchecked_flags(self):
        for mutation in (lambda row: row["work"].update(hash="12345678"),
                         lambda row: row["timing"].update(wide_recheck_steps=123)):
            rows = valid_rows()
            mutation(first(rows, "trial"))
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                audit_rows(rows)

    def test_validation_coverage_and_summary_recalculation(self):
        for kind, mutation in (("validation", lambda row: row["audit"].update(prefix_culled=1)),
                               ("validation", lambda row: row["audit"].update(exact_equal=False)),
                               ("summary", lambda row: row.update(median_ns_per_source=1))):
            rows = valid_rows()
            mutation(first(rows, kind))
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                audit_rows(rows)

    def test_phase_overlap_and_nonfinite_time(self):
        for timing in ({"expansion_screen_tail_wall_seconds": 2}, {"wall_seconds": float("nan")},
                       {"expansion_screen_tail_thread_cpu_seconds": 2}):
            rows = valid_rows()
            first(rows, "trial")["timing"].update(timing)
            with self.subTest(timing=timing), self.assertRaises(ValueError):
                audit_rows(rows)


if __name__ == "__main__":
    unittest.main()
