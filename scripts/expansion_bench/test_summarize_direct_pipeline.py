"""Malformed-accounting tests for mixed-A source ownership; no GPU work."""

import copy
import unittest

from summarize_direct_pipeline import (add_fingerprints, audit_result, audit_rows,
                                       audit_scope, variant_settings)


def vec(**entries):
    values = [0] * 81
    for key, value in entries.items():
        values[int(key[1:])] = value
    return values


def configuration():
    return dict(type="config", experiment="direct_pipeline", profile="target",
                n=72, gpu_a=10, gpu_m=62, gpu_class_b=[12, 12], num_bv=2,
                rules="fixture", window="fixture", cutoff_bits=120, threads=2,
                draws_per_cohort=1, cohort_count=1, repetitions=1,
                aggregate_repetitions=1, seed=1, checkpoint=38, walkback_cap=65536,
                cutoff=50, terminal_entry=52, selected_original_f=[40],
                queue_capacity_groups=8, startup_gpu_endpoint_guard=0,
                variants=[dict(name="gpu", cpu_a=10, cpu_b=None, stride=0, workers=0, wrapper=False),
                          dict(name="route_control", cpu_a=10, cpu_b=None, stride=0, workers=0, wrapper=True),
                          dict(name="direct_a8_b12_s4_w2", cpu_a=8, cpu_b=12, stride=4, workers=2, wrapper=True)])


def root():
    return dict(n0=str((1 << 38)-1), k=38, f=38, m=str(3**38-1), m3=3**16-1, oee=1, ne=0)


def result(name):
    direct = name.startswith("direct_")
    endpoints = [72, 20, 2] if direct else [80, 20, 2]
    cases = [180, 40, 4] if direct else [200, 40, 4]
    iters = [360, 100, 8] if direct else [400, 100, 8]
    tree = dict(generated=[0]*62+[102], survived=[0]*62+[102], culls=[[0]*7 for _ in range(63)])
    tail = dict(endpoints=endpoints, cases=cases, iters=iters, hash=5 if direct else 7)
    report = dict(n=72, a=10, b=27, num_bv=2, bv_mode="Shared", class_b=[12,12],
                  rules="fixture", window="fixture", cutoff_bits=120, tail_oee=True,
                  threads=2, units=1, tree_nodes=102, endpoints=sum(endpoints), cases=sum(cases),
                  iters=sum(iters), hash="00000005" if direct else "00000007",
                  flagged=[], flagged_rechecked=0, seconds=1.0, tree=tree, tail=tail, backend="gpu",
                  verification_scope="selected_work_units")
    report["gpu"] = dict(classes=[dict(endpoints=e,cases=c,iters=i,kernel_seconds=0.4,launches=1)
                                  for e,c,i in zip(endpoints[:2],cases[:2],iters[:2])],
                         transport="parents", payload_bytes_per_endpoint=None,
                         input_records=10 if direct else 12, transfer_bytes=640 if direct else 768,
                         flags=0, kernel_seconds=0.8, setup_included_in_report_seconds=False,
                         tuning=dict(block_threads=256,overflow_vote=False,readonly_powers=False,oee_mask=True))
    cpu = dict(source_endpoints=8,endpoints=20,cases=40,iters=90,hash="00000003",flagged=[]) if direct else dict(
        source_endpoints=0,endpoints=0,cases=0,iters=0,hash="00000000",flagged=[])
    zero = dict(count=0,xor=0,sum=0)
    groups = dict(count=12,xor=0x11,sum=70)
    cpu_groups = dict(count=2,xor=0x33,sum=40) if direct else zero.copy()
    gpu_groups = dict(count=10,xor=0x22,sum=30) if direct else groups.copy()
    routing = None
    if name != "gpu":
        routing = dict(original_by_f=vec(f40=80,f41=20,f42=2),
                       gpu_by_f=vec(f40=72 if direct else 80,f41=20,f42=2),
                       cpu_by_f=vec(f40=8 if direct else 0), groups=groups, gpu_groups=gpu_groups,
                       cpu_groups=cpu_groups.copy(), eligible_groups=8,
                       selected_groups=4 if direct else 0,queue_full_groups=1 if direct else 0,
                       startup_blocked_groups=1 if direct else 0,
                       copied_descriptor_bytes=384 if direct else 0,
                       successful_descriptor_bytes=384 if direct else 0,rejected_descriptor_bytes=0)
    return dict(report=report,routing=routing,cpu_native=cpu,cpu_original_by_f=vec(f40=8 if direct else 0),
                cpu_groups=cpu_groups,wall_seconds=1.1,process_cpu_seconds=2.2,average_busy_cpu_cores=2.0,
                cpu_worker_seconds=1.0 if direct else 0.0,cpu_reconstruction_seconds=0.1 if direct else 0.0,
                cpu_verification_seconds=0.9 if direct else 0.0,cpu_drain_seconds=0.01,
                cpu_wide_recheck_seconds=0.0,startup_gpu_endpoints_forwarded=0,
                cpu_flagged_rechecked=0,partition_equal=True,cpu_source_capture_enabled=False,
                captured_cpu_sources=0,cpu_max_inflight_groups=2 if direct else 0,cpu_inflight_capacity_groups=8)


def rows():
    cfg = configuration()
    names = [value["name"] for value in cfg["variants"]]
    saved = root()
    rows = [cfg, dict(type="cohort",cohort=0,seed=2,drawn=1,cheap_rejected=0,walk_rejected=0,
                      accepted=1,duplicates_retained=0,roots=[saved]),
            dict(type="gpu_setup",wall_seconds=0.1)]
    rows += [dict(type="warmup",variant=name,result=result(name)) for name in names]
    rows += [dict(type="trial",cohort=0,repetition=0,order_position=i,variant=name,drawn=1,result=result(name))
             for i,name in enumerate(names)]
    rows += [dict(type="aggregate_config",drawn=1,accepted=1,roots=[saved])]
    rows += [dict(type="aggregate_trial",repetition=0,order_position=i,variant=name,drawn=1,result=result(name))
             for i,name in enumerate(names)]
    rows += [dict(type="complete",cohorts=1,variants=3,repetitions=1,aggregate_repetitions=1)]
    return rows


class PipelineAccountingTests(unittest.TestCase):
    def test_mixed_a_native_counts_and_hashes_may_differ(self):
        audited = audit_rows(rows())
        self.assertEqual(audited["accepted_draws"], 1)
        direct = result("direct_a8_b12_s4_w2")
        baseline = result("gpu")
        self.assertNotEqual(direct["report"]["iters"]+direct["cpu_native"]["iters"],baseline["report"]["iters"])
        self.assertNotEqual(int(direct["report"]["hash"],16)^int(direct["cpu_native"]["hash"],16),int(baseline["report"]["hash"],16))

    def test_descriptor_sum_wraps_and_xor_preserves_duplicates(self):
        values = [dict(count=1,xor=(1<<64)-1,sum=(1<<64)-1),dict(count=1,xor=3,sum=3)]
        self.assertEqual(add_fingerprints(values),dict(count=2,xor=((1<<64)-1)^3,sum=2))
        self.assertEqual(add_fingerprints([values[0],values[0]])["xor"],0)

    def test_original_partition_rejects_missing_and_duplicated_sources(self):
        cfg = configuration(); policy = cfg["variants"][2]
        for field in ("original_by_f", "gpu_by_f", "cpu_by_f"):
            broken = result(policy["name"])
            broken["routing"][field][40] += 1
            with self.subTest(field=field), self.assertRaises(ValueError):
                audit_result(broken,cfg,policy,[root()])

    def test_digest_checks_count_xor_sum_and_worker_completion(self):
        cfg = configuration(); policy = cfg["variants"][2]
        for outer in ("groups","gpu_groups","cpu_groups"):
            for key in ("count","xor","sum"):
                broken = result(policy["name"])
                broken["routing"][outer][key] += 1
                with self.subTest(outer=outer,key=key), self.assertRaises(ValueError):
                    audit_result(broken,cfg,policy,[root()])
        broken = result(policy["name"]); broken["cpu_original_by_f"][40] -= 1
        with self.assertRaises(ValueError): audit_result(broken,cfg,policy,[root()])

    def test_selected_dispositions_bytes_and_capacity_must_balance(self):
        cfg = configuration(); policy = cfg["variants"][2]
        for key in ("selected_groups","queue_full_groups","startup_blocked_groups",
                    "copied_descriptor_bytes","successful_descriptor_bytes","rejected_descriptor_bytes"):
            broken = result(policy["name"]); broken["routing"][key] += 1
            with self.subTest(key=key), self.assertRaises(ValueError): audit_result(broken,cfg,policy,[root()])
        broken = result(policy["name"]); broken["cpu_max_inflight_groups"] = 9
        with self.assertRaises(ValueError): audit_result(broken,cfg,policy,[root()])

    def test_ineligible_f_and_existing_fallback_cannot_move(self):
        cfg = configuration(); policy = cfg["variants"][2]
        broken = result(policy["name"])
        cfg["selected_original_f"] = [41]
        with self.assertRaises(ValueError): audit_result(broken,cfg,policy,[root()])
        records = rows()
        for row in records:
            if row.get("type") == "trial" and row["variant"].startswith("direct"):
                row["result"]["report"]["tail"]["cases"][-1] += 1
                row["result"]["report"]["cases"] += 1
        with self.assertRaises(ValueError): audit_rows(records)

    def test_control_checksums_must_match(self):
        records = rows()
        row = next(row for row in records if row.get("type") == "trial" and row["variant"] == "route_control")
        row["result"]["report"]["tail"]["hash"] ^= 1
        row["result"]["report"]["hash"] = "00000006"
        with self.assertRaises(ValueError): audit_rows(records)

    def test_unchanged_selected_identity_requires_exact_work(self):
        cfg = configuration(); variants = {v["name"]:v for v in cfg["variants"]}
        original = dict(variant=cfg["variants"][2]["name"],result=result(cfg["variants"][2]["name"]))
        changed = copy.deepcopy(original)
        changed["result"]["cpu_native"]["hash"] = "12345678"
        with self.assertRaises(ValueError):
            audit_scope([dict(variant="gpu",result=result("gpu")),original,changed],cfg,variants,[root()])

    def test_different_admitted_descriptor_identity_may_change_native_work(self):
        cfg = configuration(); variants = {v["name"]:v for v in cfg["variants"]}
        original = dict(variant=cfg["variants"][2]["name"],result=result(cfg["variants"][2]["name"]))
        changed = copy.deepcopy(original)
        changed["result"]["cpu_native"]["hash"] = "12345678"
        changed["result"]["cpu_groups"]["xor"] ^= 5
        changed["result"]["cpu_groups"]["sum"] += 7
        changed["result"]["routing"]["cpu_groups"] = changed["result"]["cpu_groups"].copy()
        changed["result"]["routing"]["gpu_groups"]["xor"] ^= 5
        changed["result"]["routing"]["gpu_groups"]["sum"] -= 7
        audit_scope([dict(variant="gpu",result=result("gpu")),original,changed],cfg,variants,[root()])

    def test_capture_and_unresolved_flags_rejected(self):
        cfg = configuration(); policy = cfg["variants"][2]
        for key,value in (("cpu_source_capture_enabled",True),("captured_cpu_sources",1)):
            broken = result(policy["name"]); broken[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): audit_result(broken,cfg,policy,[root()])
        broken = result(policy["name"]); broken["cpu_native"]["flagged"] = [root()["n0"]]
        with self.assertRaises(ValueError): audit_result(broken,cfg,policy,[root()])

    def test_missing_trials_completion_and_wrong_aggregate_rejected(self):
        records = rows(); records.pop(-1)
        with self.assertRaises(ValueError): audit_rows(records)
        records = rows(); records.pop(6)
        with self.assertRaises(ValueError): audit_rows(records)
        records = rows()
        for row in records:
            if row.get("type") == "aggregate_trial":
                row["result"]["report"]["tree"]["generated"][60] += 1
                row["result"]["report"]["tree_nodes"] += 1
        with self.assertRaises(ValueError): audit_rows(records)

    def test_aggregate_corpus_and_affine_root_state_checked(self):
        records = rows()
        next(row for row in records if row.get("type") == "aggregate_config")["roots"] = []
        with self.assertRaises(ValueError): audit_rows(records)
        records = rows()
        records[1]["roots"][0]["m"] = "3"
        with self.assertRaises(ValueError): audit_rows(records)

    def test_aggregate_checksum_is_cohort_xor(self):
        records = rows()
        for row in records:
            if row.get("type") == "aggregate_trial":
                report = row["result"]["report"]
                report["tail"]["hash"] ^= 1
                report["hash"] = f'{report["tail"]["hash"]:08X}'
        with self.assertRaisesRegex(ValueError,"checksum XOR"):
            audit_rows(records)

    def test_label_cannot_hide_different_cpu_policy(self):
        policy = configuration()["variants"][2]
        self.assertEqual(variant_settings(policy,10),policy)
        for key,value in (("cpu_a",6),("cpu_b",16),("stride",8),("workers",4),("wrapper",False)):
            with self.subTest(key=key), self.assertRaises(ValueError): variant_settings({**policy,key:value},10)


if __name__ == "__main__":
    unittest.main()
