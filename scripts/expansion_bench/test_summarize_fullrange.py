"""Small statistical/accounting tests; no GPU or benchmark work."""
import unittest

from summarize_fullrange import (aggregate_expected, audit_aggregate_draws, audit_cohort,
                                 audit_deferred, audit_early, audit_flags, audit_metadata, audit_variant,
                                 early_settings, estimate,
                                 paired_ratio, work_signature, check_report)
from summarize_offload import costs as offload_costs, derived as offload_derived


class EstimatorTests(unittest.TestCase):
    def test_uses_baseline_draw_denominator(self):
        # Two cohorts of 100 draws. Each accepted draw costs one second;
        # rejected draws cost zero. Accepted-count scaling would wrongly give
        # 1000 seconds, rather than the correct estimated 500 seconds.
        result = estimate([40 / 100, 60 / 100], 1000)
        self.assertEqual(result["estimate"], 500)
        self.assertEqual(result["degrees_of_freedom"], 1)
        self.assertAlmostEqual(result["standard_error"], 100)

    def test_rejections_and_duplicate_draws_are_retained(self):
        root = {"n0": "3", "k": 38}
        row = {"drawn": 4, "cheap_rejected": 1, "walk_rejected": 1,
               "accepted": 2, "roots": [root, root]}
        config = {"draws_per_cohort": 4, "checkpoint": 38}
        self.assertEqual(len(audit_cohort(row, config)), 64)
        with self.assertRaises(ValueError):
            audit_cohort({**row, "drawn": 2}, config)
        with self.assertRaises(ValueError):
            audit_cohort({**row, "cheap_rejected": 0}, config)

    def test_paired_ratio_is_ratio_of_totals(self):
        result = paired_ratio([1, 100], [2, 100])
        self.assertAlmostEqual(result["ratio"], 101 / 102)
        self.assertNotAlmostEqual(result["ratio"], (0.5 + 1) / 2)
        exact = paired_ratio([2, 4, 6], [1, 2, 3])
        self.assertEqual(exact["ci95"], [2, 2])

    def test_work_adds_but_checksum_xors(self):
        def report(n, h):
            return {"tree": {"generated": [0, n], "survived": [0, n],
                             "culls": [[0, n], [0, 0]]},
                    "tail": {"endpoints": [n, 0], "cases": [2*n, 0],
                             "iters": [3*n, 0], "hash": h},
                    "hash": f"{h:08X}", "flagged": [], "flagged_rechecked": 0}
        result = aggregate_expected([report(2, 5), report(3, 6), report(2, 5)])
        self.assertEqual(result["tail"]["endpoints"], [7, 0])
        self.assertEqual(result["tree"]["culls"], [[0, 7], [0, 0]])
        self.assertEqual(result["tail"]["hash"], 6)
        self.assertEqual(result["hash"], "00000006")

    def test_variant_label_checks_actual_tuning_and_transport(self):
        for name, block in (("novote_oee", 256), ("novote_oee128", 128), ("novote_oee512", 512)):
            gpu = {"transport": "parents", "tuning": {
                "block_threads": block, "overflow_vote": False,
                "readonly_powers": False, "oee_mask": True}}
            audit_variant(name, gpu)
            with self.assertRaises(ValueError):
                audit_variant(name, {**gpu, "transport": "replay16"})
            for key, value in (("overflow_vote", True), ("readonly_powers", True),
                               ("oee_mask", False), ("block_threads", 192)):
                with self.subTest(name=name, field=key), self.assertRaises(ValueError):
                    audit_variant(name, {**gpu, "tuning": {**gpu["tuning"], key: value}})
        with self.assertRaises(ValueError):
            audit_variant("unknown", {})

    def test_jump_labels_require_the_exact_width_kernel_and_transport(self):
        variants = (("jump8", 8, "parents"), ("jump10", 10, "parents"),
                    ("jump12", 12, "parents"), ("jump14", 14, "parents"),
                    ("group4_jump12", 12, "affine32"))
        for name, width, transport in variants:
            tuning = {"block_threads": 256, "overflow_vote": False,
                      "readonly_powers": False, "oee_mask": True,
                      "tail_jump_bits": width}
            gpu = {"transport": transport, "tuning": tuning}
            with self.subTest(name=name):
                audit_variant(name, gpu)
                missing_width = {key: value for key, value in tuning.items()
                                 if key != "tail_jump_bits"}
                with self.assertRaises(ValueError):
                    audit_variant(name, {**gpu, "tuning": missing_width})
            for changed_width in (0, 6, 8, 10, 12, 14, 16):
                if changed_width != width:
                    with self.subTest(name=name, width=changed_width), self.assertRaises(ValueError):
                        audit_variant(name, {**gpu, "tuning": {**tuning, "tail_jump_bits": changed_width}})
            for key, value in (("overflow_vote", True), ("readonly_powers", True),
                               ("oee_mask", False), ("block_threads", 128)):
                with self.subTest(name=name, field=key), self.assertRaises(ValueError):
                    audit_variant(name, {**gpu, "tuning": {**tuning, key: value}})
            for other_transport in ("replay16", "packed8", "parents", "affine32"):
                if other_transport != transport:
                    with self.subTest(name=name, transport=other_transport), self.assertRaises(ValueError):
                        audit_variant(name, {**gpu, "transport": other_transport})

    def test_nonjump_labels_keep_the_historical_tuning_shape(self):
        tuning = {"block_threads": 256, "overflow_vote": False,
                  "readonly_powers": False, "oee_mask": True}
        for name, transport in (("novote_oee", "parents"), ("affine32", "affine32"),
                                ("chain4", "affine32"), ("group4", "affine32"),
                                ("chain6", "affine32"), ("group6", "affine32")):
            gpu = {"transport": transport, "tuning": tuning}
            with self.subTest(name=name):
                audit_variant(name, gpu)
            # Serialization omits a disabled jump field. Reject both falsely
            # enabled jumps and unexpected changes to the archived JSON shape.
            for width in (0, 8, 12):
                with self.subTest(name=name, width=width), self.assertRaises(ValueError):
                    audit_variant(name, {**gpu, "tuning": {**tuning, "tail_jump_bits": width}})

    def test_compact_chain_labels_require_grouped_transport_and_unchanged_tail(self):
        tuning = {"block_threads": 256, "overflow_vote": False,
                  "readonly_powers": False, "oee_mask": True}
        for name in ("compact66", "compact48"):
            gpu = {"transport": "compact_chain", "tuning": tuning}
            with self.subTest(name=name):
                audit_variant(name, gpu)
            for transport in ("parents", "affine32", "replay16", "packed8"):
                with self.subTest(name=name, transport=transport), self.assertRaises(ValueError):
                    audit_variant(name, {**gpu, "transport": transport})
            for field, value in (("block_threads", 128), ("overflow_vote", True),
                                  ("readonly_powers", True), ("oee_mask", False),
                                  ("tail_jump_bits", 12)):
                with self.subTest(name=name, field=field), self.assertRaises(ValueError):
                    audit_variant(name, {**gpu, "tuning": {**tuning, field: value}})

    def test_compact_chain_transfer_counts_descriptors_and_excludes_cpu_fallback(self):
        # Three compact descriptors represent 23 GPU endpoints; another five
        # endpoints are processed by the CPU and require no transfer records.
        report = {"n": 72, "num_bv": 1, "backend": "gpu",
                  "endpoints": 28, "cases": 42, "iters": 56,
                  "tree_nodes": 28, "tree": {"generated": [0, 28]},
                  "tail": {"endpoints": [23, 5], "cases": [36, 6],
                           "iters": [49, 7], "hash": 17},
                  "hash": "00000011", "flagged": [], "flagged_rechecked": 0,
                  "gpu": {"classes": [{"endpoints": 23, "cases": 36, "iters": 49}],
                          "transport": "compact_chain", "payload_bytes_per_endpoint": None,
                          "input_records": 3, "transfer_bytes": 3 * 64, "flags": 0}}
        check_report(report)
        for changes in ({"transfer_bytes": 23 * 32},  # Expanded GPU records are not PCIe input.
                        {"transfer_bytes": 28 * 64},  # CPU fallback never contributes input.
                        {"input_records": 0, "transfer_bytes": 0},
                        {"input_records": 24, "transfer_bytes": 24 * 64},
                        {"payload_bytes_per_endpoint": 64}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                check_report({**report, "gpu": {**report["gpu"], **changes}})

    def test_group_chunk_labels_require_exact_recorded_host_limit(self):
        ordinary = {"block_threads": 256, "overflow_vote": False,
                    "readonly_powers": False, "oee_mask": True}
        for base, transport in (("novote_oee", "parents"),
                                ("early_parents", "parents"),
                                ("compact66", "compact_chain"),
                                ("compact48", "compact_chain")):
            default_gpu = {"transport": transport, "tuning": ordinary}
            small_gpu = {"transport": transport,
                         "tuning": {**ordinary, "group_chunk_records": 1024}}
            with self.subTest(base=base):
                audit_variant(base, default_gpu)
                audit_variant(base, {**default_gpu,
                                    "tuning": {**ordinary, "group_chunk_records": 4096}})
                audit_variant(base + "_1k", small_gpu)
                with self.assertRaises(ValueError):
                    audit_variant(base + "_1k", default_gpu)
                with self.assertRaises(ValueError):
                    audit_variant(base, small_gpu)
            for value in (0, 1, 2048, 4096, 4097, None):
                with self.subTest(base=base, value=value), self.assertRaises(ValueError):
                    audit_variant(base + "_1k", {**small_gpu,
                        "tuning": {**ordinary, "group_chunk_records": value}})
            for other in ("parents", "compact_chain", "affine32", "replay16"):
                if other != transport:
                    with self.subTest(base=base, transport=other), self.assertRaises(ValueError):
                        audit_variant(base + "_1k", {**small_gpu, "transport": other})

    def test_tail_offload_labels_require_exact_budget_capacity_workers_and_transport(self):
        ordinary = {"block_threads": 256, "overflow_vote": False,
                    "readonly_powers": False, "oee_mask": True}
        variants = (("defer1", 1, 65536, 1, "parents"),
                    ("defer8", 8, 65536, 1, "parents"),
                    ("defer16", 16, 65536, 1, "parents"),
                    ("defer24", 24, 65536, 1, "parents"),
                    ("defer32", 32, 65536, 1, "parents"),
                    ("defer64", 64, 65536, 1, "parents"),
                    ("defer128", 128, 65536, 1, "parents"),
                    ("defer256", 256, 65536, 1, "parents"),
                    ("defer512", 512, 65536, 1, "parents"),
                    ("defer_count", (1 << 32) - 1, 65536, 1, "parents"),
                    ("defer1_cap1", 1, 1, 1, "parents"),
                    ("defer16_w2", 16, 65536, 2, "parents"),
                    ("defer24_w2", 24, 65536, 2, "parents"),
                    ("defer24_cap262k_w2", 24, 262144, 2, "parents"),
                    ("defer24_cap262k_w4", 24, 262144, 4, "parents"),
                    ("defer16_cap262k_w4", 16, 262144, 4, "parents"),
                    ("defer32_w2", 32, 65536, 2, "parents"),
                    ("defer64_w2", 64, 65536, 2, "parents"),
                    ("defer128_w2", 128, 65536, 2, "parents"),
                    ("defer256_w2", 256, 65536, 2, "parents"),
                    ("compact_defer24", 24, 65536, 1, "compact_chain"),
                    ("compact_defer24_cap262k_w2", 24, 262144, 2, "compact_chain"),
                    ("compact_defer32", 32, 65536, 1, "compact_chain"),
                    ("compact_defer128", 128, 65536, 1, "compact_chain"),
                    ("compact_defer256", 256, 65536, 1, "compact_chain"))
        for name, budget, capacity, workers, transport in variants:
            tuning = {**ordinary, "tail_budget": budget,
                      "tail_queue_capacity": capacity, "tail_cpu_workers": workers}
            gpu = {"transport": transport, "tuning": tuning}
            with self.subTest(name=name):
                audit_variant(name, gpu)
                # Default fields can be omitted in both new and archived logs.
                sparse = {k: v for k, v in tuning.items()
                          if (k, v) not in (("tail_queue_capacity", 65536),
                                             ("tail_cpu_workers", 1))}
                audit_variant(name, {**gpu, "tuning": sparse})
                with self.assertRaises(ValueError):
                    audit_variant(name, {**gpu, "tuning": ordinary})
            for field, value in (("tail_budget", 0),
                                  ("tail_budget", budget + 1),
                                  ("tail_queue_capacity", capacity + 1),
                                  ("tail_cpu_workers", workers + 1),
                                  ("overflow_vote", True),
                                  ("oee_mask", False),
                                  ("tail_jump_bits", 12)):
                with self.subTest(name=name, field=field), self.assertRaises(ValueError):
                    audit_variant(name, {**gpu, "tuning": {**tuning, field: value}})
            for other in ("parents", "compact_chain", "affine32", "replay16"):
                if other != transport:
                    with self.subTest(name=name, transport=other), self.assertRaises(ValueError):
                        audit_variant(name, {**gpu, "transport": other})

    def test_archived_variants_normalize_only_disabled_tail_offload_defaults(self):
        tuning = {"block_threads": 256, "overflow_vote": False,
                  "readonly_powers": False, "oee_mask": True}
        gpu = {"transport": "parents", "tuning": tuning}
        audit_variant("novote_oee", gpu)
        defaults = {"tail_budget": 0, "tail_queue_capacity": 65536, "tail_cpu_workers": 1}
        audit_variant("novote_oee", {**gpu, "tuning": {**tuning, **defaults}})
        for field, value in (("tail_budget", 128), ("tail_queue_capacity", 1),
                              ("tail_cpu_workers", 2)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                audit_variant("novote_oee", {**gpu, "tuning": {**tuning, field: value}})

    def test_tail_continuations_are_audited_as_subset_of_combined_gpu_origin_work(self):
        gpu = {"tuning": {"tail_budget": 32}, "classes": [{"cases": 10, "iters": 100}],
               "flags": 2, "deferred_attempts": 4, "deferred_cases": 3,
               "deferred_rejected": 1, "deferred_iterations": 50,
               "deferred_hash": 17, "deferred_flags": 1, "deferred_transfer_bytes": 96,
               "deferred_cpu_seconds": 0.01, "deferred_queue_wait_seconds": 0.0,
               "deferred_drain_seconds": 0.002}
        audit_deferred(gpu)
        for change in ({"deferred_attempts": 5}, {"deferred_cases": 2},
                       {"deferred_transfer_bytes": 128}, {"deferred_flags": 3},
                       {"deferred_iterations": 101}, {"deferred_hash": 1 << 32},
                       {"deferred_cases": True}, {"deferred_cpu_seconds": float("nan")},
                       {"deferred_drain_seconds": -0.01}, {"tuning": {"tail_budget": 0}},
                       {"classes": [{"cases": 3, "iters": 100}]}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                audit_deferred({**gpu, **change})
        for missing in tuple(key for key in gpu if key.startswith("deferred_")):
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                audit_deferred({key: value for key, value in gpu.items() if key != missing})
        # Disabled historical logs have no new fields. Enabled measurements
        # can legitimately move zero cases when their threshold is too high.
        archived = {"classes": [{"cases": 10, "iters": 100}], "flags": 0}
        audit_deferred(archived)
        zero = {key: 0 if key.startswith("deferred_") else value for key, value in gpu.items()}
        audit_deferred(zero)
        for key in ("deferred_iterations", "deferred_hash", "deferred_flags"):
            with self.subTest(field=key), self.assertRaises(ValueError):
                audit_deferred({**zero, key: 1})

    @staticmethod
    def early_report():
        # Selected launches are 0, 3 and 6. Class totals include CPU-finished
        # survivors, while input PCIe bytes remain distinct from return bytes.
        return {"a": 10, "gpu": {
            "tuning": {"early_every": 3},
            "classes": [{"launches": 4, "endpoints": 10, "cases": 100, "iters": 150},
                        {"launches": 3, "endpoints": 20, "cases": 200, "iters": 400}],
            "flags": 2, "regular_batches": 4,
            "early_batches": 3, "early_endpoints": 9, "early_cases": 100,
            "early_iterations": 170, "early_hash": 17, "early_flags": 1,
            "early_transfer_bytes": 9 * (32 + 128), "early_capture_cases": 20,
            "early_class_endpoints": [4, 5], "early_class_cases": [40, 60],
            "early_class_iterations": [70, 100],
            "early_buckets": 10, "early_jumps": 20, "early_jumped_iterations": 50,
            "early_scalar_iterations": 120, "early_table_lookups": 30,
            "early_cpu_seconds": 0.4, "early_group_seconds": 0.05,
            "early_verification_seconds": 0.3, "early_table_setup_seconds": 0.01,
            "early_decode_seconds": 0.02, "early_queue_wait_seconds": 0.001,
            "early_drain_seconds": 0.1, "early_copy_seconds": 0.01,
            "early_host_copy_seconds": 0.001}}

    def test_early_labels_require_exact_method_width_interval_workers_and_transport(self):
        ordinary = {"block_threads": 256, "overflow_vote": False,
                    "readonly_powers": False, "oee_mask": True}
        methods = [("scalar", 0, 10)] + [
            (label + str(width), mode, width)
            for label, mode in (("bucket", 1), ("table", 2), ("group", 3), ("regroup", 4))
            for width in (8, 10, 12, 14)]
        for method, mode, width in methods:
            for every, workers in ((1, 1), (64, 4), ((1 << 32) - 1, 8)):
                name = f"early{every}_{method}_w{workers}"
                expected = (every, mode, width, workers)
                self.assertEqual(early_settings(name), expected)
                tuning = {**ordinary, **dict(zip(
                    ("early_every", "early_mode", "early_group_bits", "early_cpu_workers"), expected))}
                gpu = {"transport": "parents", "tuning": tuning}
                with self.subTest(name=name):
                    audit_variant(name, gpu)
                    sparse = {key: value for key, value in tuning.items()
                              if (key, value) not in (("early_mode", 0),
                                                     ("early_group_bits", 10), ("early_cpu_workers", 1))}
                    audit_variant(name, {**gpu, "tuning": sparse})
                    for key in ("early_every", "early_mode", "early_group_bits", "early_cpu_workers"):
                        with self.subTest(field=key), self.assertRaises(ValueError):
                            audit_variant(name, {**gpu, "tuning": {**tuning, key: tuning[key] + 1}})
                    for transport in ("affine32", "compact_chain", "replay16", "packed8"):
                        with self.subTest(transport=transport), self.assertRaises(ValueError):
                            audit_variant(name, {**gpu, "transport": transport})
                    for key, value in (("tail_budget", 16), ("tail_jump_bits", 10),
                                       ("overflow_vote", True), ("early_every", float(every))):
                        with self.subTest(field=key), self.assertRaises(ValueError):
                            audit_variant(name, {**gpu, "tuning": {**tuning, key: value}})
        for invalid in ("early0_scalar_w1", "early01_scalar_w1", "early1_scalar10_w1",
                        "early1_group9_w1", "early1_regroup10_w0", "early1_table10_w9",
                        "early4294967296_scalar_w1", "early_parents", "early1_group10_w1_extra"):
            with self.subTest(name=invalid):
                self.assertIsNone(early_settings(invalid))

    def test_early_accounting_preserves_logical_work_and_separate_return_payload(self):
        report = self.early_report()
        audit_early(report)
        for a in (8, 10, 12):
            gpu = {**report["gpu"], "early_transfer_bytes": 9 * (32 + (1 << a) // 8)}
            audit_early({"a": a, "gpu": gpu})
            for wrong_bytes in (9 * 32, 9 * (1 << a) // 8, 100 * 32):
                with self.subTest(a=a, bytes=wrong_bytes), self.assertRaises(ValueError):
                    audit_early({"a": a, "gpu": {**gpu, "early_transfer_bytes": wrong_bytes}})
        # Timing sums need not add to wall time: CPU workers and CUDA streams overlap.
        audit_early({**report, "gpu": {**report["gpu"], "early_copy_seconds": 10,
                                      "early_cpu_seconds": 20, "early_group_seconds": 15}})
        for change in ({"early_jumped_iterations": 49}, {"early_scalar_iterations": 121},
                       {"early_flags": 3}, {"early_capture_cases": 101},
                       {"early_hash": 1 << 32}, {"early_endpoints": 31},
                       {"early_cases": 301}, {"early_iterations": 551},
                       {"early_batches": True}, {"early_cpu_seconds": float("nan")},
                       {"early_copy_seconds": float("inf")}, {"early_drain_seconds": -1},
                       {"early_group_seconds": True}, {"tuning": {"early_every": 0}},
                       {"tuning": {"early_every": 3, "tail_budget": 16}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                audit_early({**report, "gpu": {**report["gpu"], **change}})

    def test_early_enabled_accounting_requires_every_counter_and_full_class_vectors(self):
        report = self.early_report()
        for missing in (key for key in report["gpu"]
                        if key.startswith("early_") or key == "regular_batches"):
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                audit_early({**report, "gpu": {key: value for key, value in report["gpu"].items()
                                              if key != missing}})
        for key in ("early_class_endpoints", "early_class_cases", "early_class_iterations"):
            for vector in ([], None, False, [0], [0, 0, 0], [0, 0],
                           [True, 0], [1.0, 0], [-1, 1], "00"):
                with self.subTest(field=key, vector=vector), self.assertRaises(ValueError):
                    audit_early({**report, "gpu": {**report["gpu"], key: vector}})
        # A correct aggregate does not excuse attributing work to the wrong class.
        for key, vector in (("early_class_cases", [101, -1]),
                            ("early_class_iterations", [160, 10])):
            with self.subTest(field=key), self.assertRaises(ValueError):
                audit_early({**report, "gpu": {**report["gpu"], key: vector}})

    def test_early_launch_selection_uses_zero_origin_and_partitions_ordinary_batches(self):
        report = self.early_report()
        # Seven launches at stride three select 0, 3, 6, not just 3 and 6.
        for changes in ({"regular_batches": 5},
                        {"regular_batches": 5, "early_batches": 2},
                        {"regular_batches": 3, "early_batches": 4},
                        {"regular_batches": True},
                        {"tuning": {"early_every": True}},
                        {"tuning": {"early_every": 3.0}}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                audit_early({**report, "gpu": {**report["gpu"], **changes}})

    def test_all_screened_batches_require_all_logical_work_to_finish_on_cpu(self):
        report = self.early_report()
        gpu = report["gpu"]
        gpu.update(tuning={"early_every": 1}, regular_batches=0, early_batches=7,
                   early_endpoints=30, early_cases=300, early_iterations=550,
                   early_scalar_iterations=500, early_flags=2,
                   early_transfer_bytes=30 * 160, early_class_endpoints=[10, 20],
                   early_class_cases=[100, 200], early_class_iterations=[150, 400])
        audit_early(report)
        for changes in ({"early_endpoints": 29, "early_class_endpoints": [9, 20],
                         "early_transfer_bytes": 29 * 160},
                        {"early_cases": 299, "early_class_cases": [99, 200]},
                        {"early_iterations": 549, "early_class_iterations": [149, 400],
                         "early_scalar_iterations": 499},
                        {"early_flags": 1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                audit_early({**report, "gpu": {**gpu, **changes}})

    def test_early_archived_reports_and_zero_survivor_exports_are_valid(self):
        archived = {"gpu": {"classes": [{"endpoints": 2, "cases": 3, "iters": 4}], "flags": 0}}
        audit_early(archived)
        for key in ("early_cases", "early_hash", "early_cpu_seconds"):
            with self.subTest(field=key), self.assertRaises(ValueError):
                audit_early({"gpu": {**archived["gpu"], key: 1}})
        report = self.early_report()
        gpu = report["gpu"]
        # An exported endpoint may have no surviving completions. Return masks
        # still occupy bytes and launches still execute, but no case is invented.
        for key in ("early_cases", "early_iterations", "early_hash", "early_flags",
                    "early_capture_cases", "early_jumped_iterations", "early_scalar_iterations"):
            gpu[key] = 0
        gpu["early_class_cases"] = [0, 0]
        gpu["early_class_iterations"] = [0, 0]
        audit_early(report)
        with self.assertRaises(ValueError):
            audit_early({**report, "gpu": {**gpu, "early_hash": 1}})

    def test_offload_summary_averages_repeated_work_without_double_counting(self):
        def row(wall, cpu, deferred):
            return {"wall_seconds": wall, "process_cpu_seconds": cpu,
                    "report": {"gpu": {"classes": [{"cases": 10, "iters": 100, "endpoints": 5}],
                                       "transfer_bytes": 64, "deferred_cases": deferred,
                                       "deferred_attempts": 4, "deferred_rejected": 4 - deferred,
                                       "deferred_iterations": deferred * 10,
                                       "deferred_cpu_seconds": deferred * 0.01}}}
        # Queue occupancy can differ between repeats while exact total work
        # stays fixed. Mean accepted counts may therefore be fractional.
        result = offload_derived(offload_costs([row(2, 1, 2), row(4, 2, 3)]))
        self.assertEqual(result["wall_seconds"], 3)
        self.assertEqual(result["gpu_origin_cases"], 10)
        self.assertEqual(result["deferred_cases"], 2.5)
        self.assertEqual(result["gpu_executed_iterations"], 75)
        self.assertEqual(result["cpu_remainder_iteration_fraction"], 0.25)
        self.assertEqual(result["threshold_exceeding_case_fraction"], 0.4)
        self.assertEqual(result["queue_rejection_fraction"], 0.375)

    def test_runner_arguments_and_effective_environment_must_match(self):
        fields = {"roots": "draws_per_cohort", "cohorts": "cohort_count",
                  "repetitions": "repetitions", "a": "a", "seed": "seed", "threads": "threads",
                  "cutoff": "cutoff", "entry": "terminal_entry", "aggregate_runs": "aggregate_repetitions"}
        cfg = {"draws_per_cohort": 512, "cohort_count": 16, "repetitions": 2,
               "a": 10, "seed": 20261026, "threads": 2, "cutoff": 50,
               "terminal_entry": 52, "aggregate_repetitions": 1,
               "variants": ["parents", "novote_oee"], "class_b": [32, 35]}
        args = {arg: cfg[field] for arg, field in fields.items()}
        args.update(variants="parents,novote_oee", class_b="32,35")
        env = {"CV_FULLRANGE_COHORTS": "16", "CV_FULLRANGE_CUTOFF": "50",
               "CV_FULLRANGE_ENTRY": "52", "CV_FULLRANGE_AGGREGATE": "1"}
        metadata = {"arguments": args, "fullrange_environment": env}
        audit_metadata(metadata, cfg)
        audit_metadata({}, cfg)  # Legacy/minimal runner metadata is optional.
        for arg in fields:
            with self.subTest(argument=arg), self.assertRaises(ValueError):
                audit_metadata({**metadata, "arguments": {**args, arg: args[arg] + 1}}, cfg)
        for key in env:
            with self.subTest(environment=key), self.assertRaises(ValueError):
                audit_metadata({**metadata, "fullrange_environment": {**env, key: "99"}}, cfg)
        for key, value in (("variants", "parents"), ("class_b", "35,32")):
            with self.subTest(argument=key), self.assertRaises(ValueError):
                audit_metadata({**metadata, "arguments": {**args, key: value}}, cfg)

    def test_aggregate_trial_keeps_baseline_draw_denominator(self):
        row = {"drawn": 8192, "accepted": 7800}
        audit_aggregate_draws(row, 16, 512, 7800)
        with self.assertRaises(ValueError):
            audit_aggregate_draws({**row, "drawn": 7800}, 16, 512, 7800)
        with self.assertRaises(ValueError):
            audit_aggregate_draws({**row, "accepted": 8192}, 16, 512, 7800)

    def test_successfully_rechecked_flags_preserve_origin_multiplicity(self):
        report = {"n": 72, "flagged": ["31", "7", "31"], "flagged_rechecked": 3,
                  "gpu": {"flags": 2, "flagged": ["31", "31"]},
                  "tail": {"cases": [10, 1]}}
        audit_flags(report)
        # Old logs omit the GPU-only list; count consistency remains auditable.
        audit_flags({**report, "gpu": {"flags": 2}})
        for altered in (
            {**report, "flagged_rechecked": 2},
            {**report, "gpu": {"flags": 4}},
            {**report, "gpu": {"flags": 1}},  # Two CPU flags, only one CPU case.
            {**report, "gpu": {"flags": 2, "flagged": ["7", "7"]}},
            {**report, "gpu": {"flags": 2, "flagged": ["31"]}},
            {**report, "flagged": ["31", "7", str(1 << 72)]},
        ):
            with self.assertRaises(ValueError):
                audit_flags(altered)

    def test_flagged_signature_and_aggregate_ignore_order_not_multiplicity(self):
        first = {"tree": {"generated": [1]}, "tail": {"cases": [10, 1], "hash": 5,
                 "flagged": ["31", "7", "31"]}, "hash": "00000005",
                 "flagged": ["7", "31", "31"], "flagged_rechecked": 3}
        second = {**first, "tail": {**first["tail"], "flagged": ["7", "31", "31"]},
                  "flagged": ["31", "7", "31"]}
        self.assertEqual(work_signature(first), work_signature(second))
        altered = {**second, "flagged": ["7", "31"]}
        self.assertNotEqual(work_signature(first), work_signature(altered))
        combined = aggregate_expected([first, second])
        self.assertEqual(combined["flagged"], ["7", "7", "31", "31", "31", "31"])
        self.assertEqual(combined["tail"]["flagged"], combined["flagged"])
        self.assertEqual(combined["flagged_rechecked"], 6)

    def test_full_report_accepts_rechecked_flags_but_preserves_accounting_checks(self):
        report = {"n": 72, "num_bv": 1, "backend": "gpu",
                  "endpoints": 2, "cases": 3, "iters": 5,
                  "tree_nodes": 2, "tree": {"generated": [0, 2]},
                  "tail": {"endpoints": [2, 0], "cases": [3, 0],
                           "iters": [5, 0], "hash": 17},
                  "hash": "00000011", "flagged": ["7"], "flagged_rechecked": 1,
                  "gpu": {"classes": [{"endpoints": 2, "cases": 3, "iters": 5}],
                          "transport": "replay16", "payload_bytes_per_endpoint": 16,
                          "input_records": 2, "transfer_bytes": 32, "flags": 1}}
        check_report(report)
        for altered in ({**report, "flagged_rechecked": 0},
                        {**report, "cases": 4},
                        {**report, "gpu": {**report["gpu"], "transfer_bytes": 64}}):
            with self.assertRaises(ValueError):
                check_report(altered)


if __name__ == "__main__":
    unittest.main()
