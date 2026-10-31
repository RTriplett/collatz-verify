# Exported path candidates

A subsequent [complete verification below 2^56](n56/README.md) captured 2,236
candidates at cutoff 97 and recovered all five expected published records
above that threshold, including the additional record in `[2^55, 2^56)`.

[`candidates.json`](candidates.json) contains exact full-trajectory metrics for
the eight overflow starts captured by the saved N72/A10 `parents` aggregate
trial. No GPU verification was repeated to produce this artifact.

[`verification.json`](verification.json) is the unchanged `report` object from
the `aggregate_trial` entry with `variant: "parents"`, repetition 0, in
[`confirmation_a10_v2/pipeline.jsonl`](../fullrange_pipeline/confirmation_a10_v2/pipeline.jsonl).
That trial processed 8,192 sampled baseline K38 draws (7,724 accepted), rather
than enumerating every start below 2^72. Its checksum is `7C68AE2E`.

The historical report predates explicit arithmetic-cutoff and coverage fields,
so these fields remain `null` in the artifact. In particular, the enclosing
benchmark's `cutoff: 50` means the depth at which expensive tree rules stop;
it is not an arithmetic cutoff.

Reproduce from the repository root with a CPU-only build:

```sh
cargo build --no-default-features
target/debug/cv export-paths --report results/path_records/verification.json \
    --out results/path_records/candidates.json
```

Every exported maximum and step count was checked against an independent
Python integer replay of the ordinary map, with shortcut states extracted at
even-input transitions. All eight candidates matched. The CLI's direct
`verify --path-records` route was also exercised on a tiny N16 CPU run with
cutoff 16; all 665 exported candidates matched that oracle.

These are captured candidates, not a claim that any of them is a global path
record. See [the artifact format and coverage](../../docs/PATH_RECORDS.md).
