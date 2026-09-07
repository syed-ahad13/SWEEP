# CLAUDE.md — SWEEP

GPU prefix-scan engine (single-pass decoupled lookback over arbitrary monoids) + four applications (SSM recurrence, tridiagonal solver, CSV indexer, max-plus Viterbi) + numerics/determinism studies + fingerprint-driven schedule selector. Target: ISPASS 2027 paper + public artifact.

**`SWEEP_MASTER_PLAN.md` is the authoritative step-by-step (Steps 6–27) with per-step Verify gates. Follow it; do not improvise scope. If code and plan disagree, the plan wins — ask before deviating.**

## Environment detection — do this FIRST, every session

Three environments run this repo. Check with `command -v nvcc && nvidia-smi` before acting:

| Environment | Detect | Allowed actions |
|---|---|---|
| Mac (no GPU) | no `nvcc` | Edit code, run `study/plots.py`, CPU-only logic in `tests/common.hpp`, docs. **Never** run `make bin/*` or anything CUDA |
| Vast pod (root container, CUDA 12.8) | `nvcc` present, `nvidia-smi` shows RTX/Tesla | Full build/test/bench. Disposable: anything unpushed dies with the instance |
| Kaggle notebook | `/kaggle` exists | Same as pod but commands need `!` prefix in cells |

Device → `ARCH` (always pass explicitly to make):

```
RTX 4090 = sm_89 (daily)   T4 = sm_75   P100 = sm_60
A100 = sm_80   RTX 5090 = sm_120 (needs CUDA >= 12.8)   RTX 3090 = sm_86
```

## Commands

```bash
make bin/<name> ARCH=sm_89            # builds tests/|bench/|apps/|study/<name>.cu
./bin/<name>
compute-sanitizer --tool racecheck ./bin/<name>
compute-sanitizer --tool memcheck  ./bin/<name>
bash scripts/provenance.sh            # provenance CSV suffix for benchmark rows
python3 study/plots.py                # Mac only, against committed results/*.csv
```

**New-pod gates (before any real work, in order):** `nvcc --version` (12.8) → `nvidia-smi -q -d POWER` (limit near spec) → `make bin/bw_probe ARCH=<dev> && ./bin/bw_probe` must land inside that device's band in `results/machines.json`. Any failure → stop and tell the user to destroy the host and take the next offer.

## Non-negotiable invariants (violations compile fine and produce wrong results)

1. **Ordering contract.** `combine(earlier, later)` — first argument is earlier in the array. For transformation monoids (MatModP, Möbius, FsmCsv, MaxPlus) that means mathematical composition later∘earlier — matrix product **B·A, not A·B**. The lookback walk **prepends**: `excl = combine(found, excl)`. Never swap arguments anywhere; `tests/test_ordering.cu` (MatModP through every code path) is the test-of-record — run it after touching any combine call site, and rerun the mutation check if you change the prepend line.
2. **Descriptor discipline.** `cudaMemsetAsync` the descriptor array AND the ticket counter to 0 before **every** launch. Logical block id comes from `atomicAdd(ticket, 1)` — never use `blockIdx.x` for anything a block waits on (deadlock). In the generic `Desc<T>`: `aggregate` and `inclusive` are each written **exactly once**; status goes through `cuda::atomic_ref` with `release` on write / `acquire` on read. Never weaken these to `relaxed` — except inside the one labeled negative-control test. The packed 64-bit fp32 path legitimately uses `relaxed` (single-word atomicity); do not "harmonize" the two paths.
3. **Numerics are load-bearing.** Never compile `include/sweep/numerics.cuh` or `study/*` with `-use_fast_math`. Never "simplify" `two_sum`, `two_prod`, `df_add`, `df_mul` — the algebraically-redundant-looking operations ARE the algorithm (they capture rounding error exactly). After touching them, the e≠0 exactness unit tests must still pass, and a `cuobjdump -sass` spot-check should show the correction terms survived.
4. **Warp shuffles.** Full mask `0xffffffffu` always; inactive/tail elements are padded with `M::identity()`, never masked-out lanes; a shuffle never sits inside a divergent branch (guard the *use* of the result, not the shuffle).
5. **Generic block scan launches need dynamic shared memory:** `kernel<<<grid, 256, 256 * sizeof(typename M::T)>>>(…)`. blockDim is 256 (multiple of 32, ≤1024) unless a plan step says otherwise.
6. **Don't "fix" the nondeterminism** of plain lookback (float monoids, >1 distinct hash across runs) — it is the study's subject. Deterministic behavior belongs only to: chained mode, two-pass, integer monoids. An **integer** monoid producing >1 hash is a real race → Step 14, sanitizers, stop everything else.

## Testing rules

- Exact monoids (AddU64, MatModP, FsmCsv, Segmented<exact>) → **bit-exact** vs the CPU oracle. Float monoids → compare vs the **fp64 oracle with a recorded ULP tolerance**; never bit-exact asserts on floats (they flap by design).
- Always fuzz the full `test_sizes()` set (0, 1, 31, 32, 33, …, 2^24) — single-block sizes never exercise the lookback prepend.
- After ANY edit to `scan.cuh` or `descriptor.cuh`: rerun `test_generic`, `test_ordering`, the 500-launch hang test, and both sanitizers before proceeding. A hang means `blockIdx.x` leaked in or a memset was skipped.
- Never advance past a red Verify gate. Each green gate = one commit.

## Benchmark rules

- `time_kernel_ms` (5 warmup, 50 reps, **median**). Contenders interleaved A/B/A/B in the same session on the same machine — never compare numbers across hosts, never compare hashes across GPUs and call it nondeterminism.
- Every benchmark CSV row appends the `provenance.sh` fields (gpu, driver, clocks, temp, power, host, commit, timestamp). Study configurations use **≥20 seeds**.
- Byte counts include our own descriptor memsets. Report GB/s and % of the *measured* bandwidth from `machines.json` — never % of datasheet.
- Determinism study logs **hashes + lookback-depth signatures only** — never persist the 10,000 raw output arrays.

## Git & session discipline

- Commit message pattern: `step N: <what> [device]`, after every green gate. `results/*.csv` and plots are committed; `bin/` is not.
- `GH_TOKEN` is an env var only — never written into files, templates, or commits.
- Long runs live inside tmux; run Claude Code itself inside tmux on pods (`tmux new -s claude`), so the session survives SSH drops.
- Before a pod session ends: `git push` must succeed, then remind the user: **Destroy the instance (not stop), verify the meter reads $0.**

## Style

- C++17 CUDA. Each file includes only what it strictly needs; never `<bits/stdc++.h>`.
- `include/sweep/` is the engine (headers); `tests/ bench/ apps/ study/` are leaf `.cu` files with `main()`.
- No new dependencies. CUB appears only as a baseline in `bench/` and `study/`; never change the CUB/CCCL version mid-study.
- Correctness first, performance passes as separate later commits. Readable kernels beat clever ones.

## Session-state protocol

At session start: read `results/PROGRESS.md` (create if missing: current step, last green gate, open issues, current device). Update it before the session ends. It is the handoff between sessions, machines, and Claude Code instances.

## Scope guard — do not add

No autograd/backward pass. No multi-GPU. No pivoting in the tridiagonal solver. No general regex (CSV FSM stays 3 states; Viterbi stays fixed-K). No tensor-core/matmul scan reformulation. No schedules beyond lookback / chained / two-pass unless the plan's Phase 2 says so. No removal of ticket assignment or per-launch memsets as an "optimization." No volumes or persistent storage on pods — git is the only persistence.
