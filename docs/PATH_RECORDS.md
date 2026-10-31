# Exporting trajectory metrics for captured candidates

`cv` can replay captured overflow starts on the CPU and export their complete
trajectory metrics. The input is the verification report's `flagged` list.
This export does not search for additional candidates or establish a complete
list of global path records.

## Commands

Export from an existing verification report without repeating verification:

```sh
cv export-paths --report verify.json --out paths.json
```

Or request the same artifact during a new verification:

```sh
cv verify --n 55 --gpu --cutoff-bits 97 --json verify.json --path-records paths.json
```

The export itself needs no GPU. The verification report and path artifact must
use different files. Every captured start must have a successful wide recheck;
the exporter then replays each distinct start all the way to 1. An empty
`flagged` list produces an artifact with no candidates.
The example lowers the cutoff to capture high excursions that would remain
below the default 120-bit threshold.

## Metrics and representation

The replay includes the start at step 0 and stops on the first arrival at 1.
It uses checked unsigned 256-bit arithmetic. Exceeding that range, including
in an unhalved `3*n+1` intermediate, or failing to reach 1 within 10,000,000
shortcut steps causes an error instead of a successful artifact.

Two maps are measured separately:

* **Shortcut:** odd `n` becomes `(3*n+1)/2`; even `n` becomes `n/2`.
* **Standard:** odd `n` becomes `3*n+1`; even `n` becomes `n/2`.

| Candidate field | Meaning |
|---|---|
| `start` | Original positive starting value |
| `occurrences` | Number of occurrences in the source flag list |
| `shortcut_maximum`, `standard_maximum` | Largest value on the complete trajectory under each map, including the start |
| `shortcut_peak_step`, `standard_peak_step` | First step attaining that maximum, in the corresponding map's units |
| `total_shortcut_steps`, `total_standard_steps` | Transitions from the start to the first arrival at 1 |
| `first_descent_step` | First **shortcut** step below the original start; `null` for start 1 |

For start 1, both maxima are 1 and all step counts are zero. Replay continues
after first descent for other starts: for example, 9 descends below its start
before reaching its eventual standard maximum of 52.

Starts and maxima are decimal strings, preserving integers beyond JavaScript's
exact numeric range. Candidates appear in ascending **numeric** order, with
duplicates replayed once and counted in `occurrences`. No record ranking is
assigned.

## Scope and provenance

The artifact declares `selection: "overflow_flags_only"`,
`global_path_records_complete: false`, and `all_input_flags_replayed: true`.
It also includes a schema version, source report path when supplied, source
verification settings and checksum, total flag occurrences, and the number
of unique candidates.

New verification reports include `cutoff_bits`, `tail_oee`, and
`verification_scope` (`full_range` or `selected_work_units`). Missing fields
in older reports remain `null` in the artifact; the exporter does not infer
their cutoff or coverage. A source run covering the whole range still does
not make the exported candidates a complete path-record census.

CPU fallback selects endpoint classes outside the configured GPU bitvector
range. It does not identify path-record candidates, and its ordinary cases
are not added to the export. Overflow starts from either backend are captured
when the verifier reaches its arithmetic cutoff. The default 120-bit cutoff
applies to the internal shortcut iterate plus one, rather than a separate
recording threshold.

For scale, [David Barina's path-record table](https://pcbarina.fit.vut.cz/path-records.htm)
lists `10709980568908647`, below `2^55`, with shortcut maximum
`175294593968539094415936960141122`. Its standard maximum is
`350589187937078188831873920282244`, as checked by the replay tests. These
108-bit and 109-bit peaks are below the default cutoff; that cutoff does not
capture the known path records below `2^55`. An empty export at that bound
therefore does not mean there are no path records.

Convergence pruning also has different requirements from record discovery.
For example, 7 reaches a standard maximum of 52, then merges into the path
from 3: `T^7(7) = 5 = T(3)`. The OEE sieve can discard it at that merge even
though its peak exceeds 3's maximum of 16. A complete record search needs
peak accounting and pruning that preserves record candidates. Stopping below
the start can be used with suitable accounting, but the verifier's current
merge and lookahead certificates establish convergence, not completeness of
the exported peaks.

The replay implementation and arithmetic checks are in
[`src/path_records.rs`](../src/path_records.rs).
An [example artifact](../results/path_records/README.md) replays eight captured
starts from the saved N72 benchmark without repeating verification.
The [completed N56 run](../results/path_records/n56/README.md) uses cutoff 97
and independently matches all 2,236 exported trajectories, including published
records #73–77.
