# SWEEP

SWEEP is a CUDA prefix-scan engine: one single-pass decoupled-lookback scan
that runs over an arbitrary monoid, so any loop whose step can be written as a
small combinable object with an associative combine rule — a running total, a
linear recurrence, a tridiagonal elimination, a parser state, a Viterbi path —
becomes a parallel scan instead of a sequential dependency chain. Around that
engine sit a zoo of those operators (with associativity proofs), four
applications written the ordinary sequential way and then as scans (SSM
recurrence, tridiagonal solver, CSV indexer, max-plus Viterbi), and the part
that decides whether any of it is usable in practice: measurement studies of
what the reformulation costs in floating-point accuracy and in run-to-run
reproducibility, plus a fingerprint-driven selector that picks a schedule.
The target is an ISPASS 2027 paper with a public artifact.

- **[SWEEP_MASTER_PLAN.md](SWEEP_MASTER_PLAN.md)** — the authoritative
  step-by-step (Steps 6–27), each with its Verify gate. It is the plan of
  record: if code and plan disagree, the plan wins.
- **[CLAUDE.md](CLAUDE.md)** — working rules for this repo: environment
  detection, the non-negotiable invariants (ordering contract, descriptor
  discipline, numerics, warp shuffles), testing and benchmark rules, and
  session discipline.
- **[docs/monoid_zoo.md](docs/monoid_zoo.md)** — the operator catalogue.
- **[results/PROGRESS.md](results/PROGRESS.md)** — current step, last green
  gate, device, open issues. Read it first each session.

## Status

Repository skeleton only (Setup 3). No CUDA source has been written, compiled,
or run yet.

## Build

```sh
make bin/<name> ARCH=sm_89     # builds from tests/|bench/|apps/|study/<name>.cu
make test_cpu                  # host-only; the only target that builds without CUDA
```

`ARCH`: `sm_60` P100 · `sm_75` T4 · `sm_80` A100 · `sm_86` 3090 · `sm_89` 4090
(default) · `sm_120` 5090 (needs CUDA ≥ 12.8).

## Non-goals

No autograd/backward pass. No multi-GPU. No general regex engine (the CSV FSM is 3 states by design; Viterbi is fixed-K). No pivoting tridiagonal solves — diagonally dominant systems only, stated plainly. No tensor-core/matmul reformulation of the scan (Mamba-2's fork, acknowledged). Determinism claims are per-device at fixed launch configuration. No cross-host performance comparisons — every A-vs-B number comes from one session on one machine. Every number in the write-up links to its CSV, its provenance line, and its commit.
