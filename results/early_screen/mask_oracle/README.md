# Independent early-screen mask oracle

The [saved result](result.json) passed 128 GPU launches and checked every bit
of 111,680 exported mask words: 3,573,760 completion decisions in total.

[`oracle.cu`](oracle.cu) compares the returned masks with an independent scalar
CPU calculation using wrapped bit indices, full-width starting values modulo 9,
and direct OEE tests on each candidate's iterate. It also compares returned
affine32 records exactly, including CPU reconstruction of the terminal-parent
and compact-chain expansions.

Coverage includes A8/A10, M62/M64, shared and split bitvector families, OEE
states 0/1/2 with the filter enabled and disabled, 128/256/512-thread blocks,
partial blocks, bitvector wrap boundaries and low-64-bit carry in the initial
iterate. The test verifies rejection of slot reuse/preparation while an export
is in flight, an ordinary wait on a screen-export slot, and a second screen wait
after consumption.

The input states and resident masks are synthetic. This isolates mask indexing,
filter combination, affine transport and slot ownership; integrated tests still
check the mathematical bitvectors and complete verification counts/checksums.

Reproduce from the repository root:

```bash
/usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_89 \
  results/early_screen/mask_oracle/oracle.cu -o /tmp/collatz_screen_mask_oracle
/tmp/collatz_screen_mask_oracle
```

No binary is stored here. [metadata.json](metadata.json) records source and
executed-binary hashes, compiler version, commands and coverage. The saved
harness changes only its first include to the equivalent relative repository
path; the executed harness hash is retained separately.
