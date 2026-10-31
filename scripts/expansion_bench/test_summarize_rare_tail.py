"""Focused audits for conditional rare-tail accounting; no timed workload."""
import unittest

from summarize_rare_tail import audit_timing, audit_work


class RareTailAuditTests(unittest.TestCase):
    def test_class_totals_flags_and_duplicate_origin_multiplicity(self):
        cfg = {"n": 72, "m": 62, "f_min": 56, "f_max": 62}
        roots = [{"n0": "7", "f": 56}, {"n0": "7", "f": 56}, {"n0": "11", "f": 57}]
        rows = [{"f": f, "endpoints": 0, "cases": 0, "iters": 0} for f in range(56, 63)]
        rows[0].update(endpoints=2, cases=4, iters=6)
        rows[1].update(endpoints=1, cases=1, iters=3)
        work = {"endpoints": 3, "cases": 5, "iters": 9, "hash": "00000011", "by_f": rows,
                "flagged": [str(11 + (1 << 62)), "7", "7"], "flagged_rechecked": 3,
                "unresolved_flags": 0, "wide_recheck_steps": 100}
        checked = audit_work(work, roots, cfg)
        self.assertEqual(checked["flagged"], ["7", "7", str(11 + (1 << 62))])
        for altered in ({**work, "cases": 6}, {**work, "flagged_rechecked": 2},
                        {**work, "unresolved_flags": 1}, {**work, "hash": "0000001Z"},
                        {**work, "flagged": ["7", "7", "7"]},
                        {**work, "flagged": ["7", "7", "15"]},
                        {**work, "by_f": rows[:-1]}):
            with self.assertRaises(ValueError):
                audit_work(altered, roots, cfg)

    def test_component_timings_are_included_not_added_twice(self):
        timing = {"wall_seconds": 5, "process_cpu_seconds": 4.5,
                  "tail_wall_seconds": 3, "tail_process_cpu_seconds": 2.5,
                  "wide_recheck_wall_seconds": 1.9, "wide_recheck_process_cpu_seconds": 1.9,
                  "average_busy_cpu_cores": 0.9}
        audit_timing(timing)
        for altered in ({**timing, "wide_recheck_wall_seconds": 3},
                        {**timing, "wide_recheck_process_cpu_seconds": 3},
                        {**timing, "average_busy_cpu_cores": 1},
                        {**timing, "tail_wall_seconds": -1}):
            with self.assertRaises(ValueError):
                audit_timing(altered)


if __name__ == "__main__":
    unittest.main()
