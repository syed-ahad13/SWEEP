# SWEEP — Master Plan v4 (Claude Code edition)

**Supersedes v3.** Scope is identical — engine, four applications, both studies, phase diagram, adaptive scheduler, six GPUs, paper. Two things changed: every implementation step now ships as a **Claude Code prompt** instead of finished code, and the v3 plan has been **audited**; the findings are listed first so you know what moved and why. Dated: Thursday, September 3, 2026. The calendar is re-anchored to a Monday, September 7 start.

## What the audit found in v3 (and what v4 does about it)

| # | Finding | Severity | Fix in v4 |
|---|---|---|---|
| 1 | **CSV delimiter rule was wrong.** v3 marked a delimiter as real when "mode-before == FIELD". A comma right after a *closing* quote (`"abc",`) has mode-before = QSEEN, so v3 would have **missed every delimiter following a quoted field**. | Correctness bug | Rule is now: byte is DELIM **and** mode-before ≠ QUOTED (Step 19) |
| 2 | **Bit-exact variant of the packed lookback used u64 values.** A 64-bit value cannot share a 64-bit word with a 2-bit status — the packed path physically can't carry it. | Design bug | Packed-path exact variant uses **u32** (32-bit wraparound add, exact, associative); AddU64 arrives with the generic descriptor in Step 14 |
| 3 | **CUB `run_to_run` baseline needs a newer CUB than the CUDA 12.8 image ships.** The determinism API is CUDA-13.4-era CCCL. | Would fail at build | Makefile gains `CCCL_DIR`; Step 22 pins a CCCL git tag and overrides include paths; version pinned once, never changed mid-study |
| 4 | **Max-plus composition text was misleading.** Code was right, prose said "same B·A convention"; with from→to (row) matrices the correct composition is earlier ⊗ later. Also "−∞ + −∞ gives NaN" is false (it gives −∞); NaN needs a +∞. | Clarity | Both corrected in Step 24 |
| 5 | Host wrappers never stated the **n = 0** case (grid of zero blocks must skip the launch). | Edge case | In every wrapper spec |
| 6 | `provenance.sh` GPU names ("NVIDIA GeForce RTX 4090") didn't match `machines.json` keys ("RTX_4090"). | Tooling gap | Normalization mapping specified in Setup 3 and Step 6 |
| 7 | Kaggle images may not have `nvcc` on `PATH`. | Friction | `export PATH=/usr/local/cuda/bin:$PATH` in Ritual B |
| 8 | `df_add` was the fast ("sloppy") double-float add, labeled as ~2× precision without qualification. | Precision claim | Both fast and accurate variants implemented; study reports which is used |
| 9 | Negative-control test assumed the platform *will* expose weak-memory behavior under `relaxed`. Not guaranteed on modern GPUs. | Overclaim | Prompt records either outcome; release/acquire stays because the PTX memory model permits the reordering regardless |
| 10 | Calendar started Aug 24; it is Sep 3. | Schedule | Re-anchored (Part IX); the Phase-2-into-submission fold is now a dated decision, not an assumption |
| 11 | Paper claim C3 said "first measurement" flatly. | Reviewer risk | "First, to our knowledge" |
| 12 | Minor: missing `<cstdlib>` in v3 code; `blockIdx.x` legitimately used in two-pass kernels (no waiting) — noted so nobody "fixes" it. | Minor | Encoded in prompts |

## How to use this plan with Claude Code (Ritual E, read now)

Each step below has a **Claude Code prompt** in a text box. The prompt is written so that *you* understand what is about to happen (its opening "Context" paragraph is the plain-language explanation) and so that Claude Code has every detail it needs. The workflow per step:

1. Be on the right machine (Mac for CPU-only steps; the rented pod for anything CUDA — the prompt says which). On a pod, first start tmux, then start Claude Code *inside* tmux: `tmux new -s claude` → `claude`. A dropped Wi-Fi then pauses nothing.
2. Paste the prompt. Claude Code reads `CLAUDE.md` automatically (it must be at the repo root) and works. **Ask it to explain its plan before it writes code** if anything is unclear to you — that's how you learn the step.
3. When it finishes, run the **Your Verify** checks yourself. Claude Code proposes; the gate is yours. Never accept "tests pass" without seeing the test output.
4. If Verify is green: confirm the commit exists (`git log -1`), confirm `results/PROGRESS.md` was updated, `git push`.
5. If red: paste the failing output back to Claude Code with "Fix this; do not change the test's tolerance or the contract." Repeat.

---

## Part 0 — The mental model

You work with **three computers and one website**, each with one job. **Your MacBook** is the cockpit: editing, plots, the paper, and terminals that remote-control the other machines; it has no NVIDIA GPU and never runs CUDA; it holds your secrets (SSH key, GitHub token). **A rented Vast.ai instance** is the disposable muscle: clicking RENT starts a *container* — a sealed Linux box on someone's datacenter machine, booted from the Docker image you chose (`nvidia/cuda:12.8.1-devel-ubuntu22.04`, i.e., Ubuntu with the CUDA compiler preinstalled); you control it over **SSH** (your keyboard, their computer); it is cattle, not a pet — create, use for hours, destroy; tomorrow's is a different physical machine that starts blank. **Kaggle notebooks** are free second and third GPUs (T4, P100) driven by `!`-prefixed commands in cells. **GitHub** is the *only permanent home anything has*.

**The prime rule:** anything not pushed to GitHub when an instance is destroyed is gone forever. Every ritual ends with `git push`. The corollary: because git is the persistence layer, instances are freely destroyable — which is what makes the economics work.

**The money model** (learned the hard way already):

| Instance state | You pay | Who has the GPU |
|---|---|---|
| **Running** | Full rate (~$0.31/hr), connected or not, GPU busy or idle | You, exclusively; on-demand cannot be taken from you |
| **Stopped** | Storage only — but the GPU is *released*: another renter can take the card, and your instance may be unable to restart, stranding data while billing disk | Possibly someone else — the trap state |
| **Destroyed** | Nothing | Nobody; clean |

No middle state in this project: an instance is running-and-in-use or destroyed. Never stopped.

**Why tmux is non-negotiable.** A program you start over SSH is a child of that connection; if Wi-Fi blips or the Mac sleeps, the connection dies and *kills the job*. `tmux` owns your terminal session server-side: if SSH drops, the session (and any running benchmark, or Claude Code itself) keeps going; you reconnect and reattach. Rule: **anything longer than two minutes — including Claude Code — runs inside tmux.**

**A typical day.** Rent (or reuse today's) instance → SSH in → `tmux` → clone → work through the day's steps with Claude Code → CSVs land in `results/` → `git push` → exit → Destroy → meter reads $0. Kaggle days: same shape in cells. The Mac holds a clone for editing and for `plots.py`.

---

## Part I — Vocabulary (skim now, return as needed)

**Instance / container / VM.** The rented isolated Linux environment. Technically a container sharing the host's GPU driver, booted from a **Docker image** (a filesystem snapshot: OS + tools). The `-devel` suffix guarantees the CUDA *compiler* is inside, not just runtime libraries.

**SSH.** Encrypted remote terminal. Authentication by **key pair**: private key stays on the Mac; public key is pasted into websites; possession of the private key is your password.

**PAT (personal access token).** A revocable, scope-limited password substitute so a rented machine can push to one repo without your real credentials.

**tmux.** Persistent server-side terminal sessions (above).

**Claude Code.** An AI agent running in your terminal that reads `CLAUDE.md`, edits files, runs commands, and commits. It proposes; you verify.

**Kernel.** A function run on the GPU (`__global__`), launched from CPU code, executed by thousands of threads.

**Thread, block, grid.** Threads group into **blocks** (we use 256); all blocks of one launch form the **grid**. Threads inside a block synchronize cheaply (`__syncthreads()`); blocks cannot easily wait on each other — the limitation that makes the core algorithm interesting.

**Warp and lane.** Hardware runs threads in fixed teams of 32 (**warps**); your position 0–31 is your **lane**. Lanes swap register values directly with **shuffle** instructions.

**Global vs shared memory.** Global: the GPU's big DRAM, visible to all blocks, hundreds of cycles away. Shared: a small per-block scratchpad inside each GPU core, ~100× closer.

**Coalescing.** A warp reading 32 *neighboring* addresses gets one wide transaction.

**Bandwidth-bound, roofline.** Speed limited by byte movement, not math. Scan's limit is `bytes_moved / bandwidth`, judged against a bandwidth *you measure* (Step 6), never a datasheet.

**Occupancy.** Warps resident per GPU core; more residency hides memory latency.

**Atomic.** An indivisible hardware operation (`atomicAdd`).

**Race condition.** Two threads touch the same memory, one writes, no ordering.

**Release/acquire.** Hardware may show your writes to others *out of order*. Release store = "publish: everything before this is visible to whoever sees it." Acquire load = "subscribe: seeing it guarantees seeing what preceded it." How blocks hand multi-word values across; wrong = once-per-100k bugs.

**Scan.** Inclusive: `[x0, x0⊕x1, x0⊕x1⊕x2, …]`. Exclusive: shifted by one.

**Monoid.** Value type + associative combine + identity. Associativity — `(a⊕b)⊕c = a⊕(b⊕c)` — is the only property tree parallelism needs. Commutativity is not required; several of ours lack it.

**Semiring, max-plus.** Two operations where "multiply" distributes over "add." Max-plus: add = `max`, multiply = `+`; its matrix products make Viterbi decoding scannable (Step 24).

**Oracle.** Slow, obviously-correct CPU reference; GPU disagreement means the GPU is wrong until proven otherwise.

**Fuzzing.** Volumes of random inputs plus adversarial sizes.

**Litmus test.** A tiny concurrent program run under stress, watching for a forbidden outcome (Alglave et al., ASPLOS 2015, caught real GPUs reordering in code shaped like ours).

**ULP.** Unit in the last place — the gap between adjacent floats; the honest unit for float error.

**Provenance.** Machine, driver, clocks, commit — attached to every measurement. Mandatory on marketplace GPUs.

**Devices and compiler flags** (`ARCH=` passed to `make`):

| Device | Where | `ARCH=` | Memory | Role |
|---|---|---|---|---|
| RTX 4090 24GB | Vast ~$0.31/hr (your KR host) | `sm_89` | GDDR6X ~1008 GB/s | Daily driver |
| Tesla T4 | Kaggle, free | `sm_75` | ~320 GB/s | Legacy point |
| Tesla P100 | Kaggle, free | `sm_60` | ~732 GB/s | Legacy point |
| A100 80GB | Vast ~$0.93/hr, ~4h in Nov | `sm_80` | HBM2e ~1935 GB/s | HBM contrast |
| RTX 5090 32GB | Vast ~$0.41/hr, ~4h in Nov | `sm_120` | GDDR7 ~1792 GB/s | Newest point |
| RTX 3090 24GB | Vast ~$0.15/hr, ~2h in Nov | `sm_86` | GDDR6X ~936 GB/s | Within-Ampere control |

---
## Part II — Day 0: one-time setup (Mac, ~90 minutes, $0)

### Setup 1 — SSH key pair

**What/why.** One key pair authenticates you to every instance you'll ever rent. Private half stays on the Mac; public half is pasted into Vast once.

**Do (Mac Terminal):** `ssh-keygen -t ed25519 -C "ahad-vast"` (Enter for default path, passphrase optional) → `cat ~/.ssh/id_ed25519.pub` → copy the line → Vast console → Account → SSH Keys → Add → save. Keys are injected at instance *creation*; for an already-running instance use the **key icon** on its card.

**Done when.** The key is listed under Account → SSH Keys; connecting to a fresh instance needs no password.

### Setup 2 — GitHub repo + fine-grained token

**Do.** GitHub → New repository → `sweep`, private → Settings → Developer settings → Personal access tokens → Fine-grained → Generate: Repository access = only `sweep`; Permissions = Contents: Read and write; 90-day expiry. Copy once into a locked note on the Mac. It will appear inside clone URLs on rented machines — acceptable *because* scoped and expiring; never put it in the Vast template or in any file.

**Done when.** `git clone https://<TOKEN>@github.com/<you>/sweep.git` works on the Mac. Put `CLAUDE.md` and this file at the repo root.

### Setup 3 — Repository skeleton (Claude Code, on the Mac)

**What we're doing.** Creating the folder structure, the build file (`Makefile`), the helper scripts, and the reference bandwidth table — all of it plain files, no CUDA compiled (the Mac can't). Claude Code does the typing; you check the result.

**Claude Code prompt:**

```text
Context: We are starting a research project called SWEEP — a GPU library that computes "prefix scans" (running totals under an arbitrary associative combine rule) with the decoupled-lookback algorithm, plus applications and measurement studies leading to a paper. This task creates the repository skeleton on my Mac. IMPORTANT: this Mac has no NVIDIA GPU and no nvcc — create files only, do not try to compile anything. The repo is already initialized with a GitHub remote, and CLAUDE.md is at the root: read it first.

Create these directories: include/sweep, tests, bench, apps, study, scripts, results, docs, bin.

Create these files:

1) Makefile
- Variable ARCH ?= sm_89, with a comment table: RTX 4090 = sm_89 (daily), Tesla T4 = sm_75, Tesla P100 = sm_60, A100 = sm_80, RTX 5090 = sm_120 (requires CUDA >= 12.8), RTX 3090 = sm_86.
- Variable CCCL_DIR ?= (empty). If non-empty, add -I$(CCCL_DIR)/cub -I$(CCCL_DIR)/thrust -I$(CCCL_DIR)/libcudacxx/include BEFORE other includes, so a pinned newer CUB/CCCL checkout overrides the toolkit's headers (needed in Step 22 for CUB's determinism API).
- NVCC = nvcc -O3 -std=c++17 -arch=$(ARCH) -lineinfo -Iinclude $(CCCL_INC). Never add -use_fast_math anywhere in this Makefile; add a comment saying why (the numerics library depends on exact rounding behavior).
- Pattern rules so that `make bin/<name> ARCH=...` builds bin/<name> from whichever of tests/<name>.cu, bench/<name>.cu, apps/<name>.cu, study/<name>.cu exists (order-only prerequisite on the bin directory; a rule creating bin with mkdir -p).
- Target test_cpu: builds tests/test_common.cpp with $(CXX) -std=c++17 -O2 into bin/test_common (this is the only target that must work on the Mac).
- Target clean removing bin.

2) scripts/provenance.sh — prints ONE CSV line: gpu_key,driver,sm_clock,mem_clock,temp,power_limit,host,commit,timestamp_utc. Get the fields from `nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.mem,temperature.gpu,power.limit --format=csv,noheader`. gpu_key must be a normalized name: strip the words NVIDIA and GeForce, replace spaces with underscores, and map known names: "Tesla T4"->Tesla_T4, "Tesla P100-PCIE-16GB"->Tesla_P100, any name containing "A100" and "80GB"->A100_80GB, "RTX 4090"->RTX_4090, "RTX 5090"->RTX_5090, "RTX 3090"->RTX_3090. host = hostname; commit = git rev-parse --short HEAD; timestamp = date -u +%FT%TZ.

3) scripts/pod_init.sh — expects GH_TOKEN in the environment; clones https://$GH_TOKEN@github.com/<you>/sweep.git, sets git user.name "Ahad" and user.email "<you>@users.noreply.github.com", then prints nvcc --version and nvidia-smi. Use set -e.

4) results/machines.json — JSON object mapping gpu_key to {"expected_gbps": [lo, hi]}: RTX_4090 [850, 950], Tesla_T4 [220, 270], Tesla_P100 [500, 580], A100_80GB [1500, 1800], RTX_5090 [1450, 1700], RTX_3090 [800, 900]. These are the achieved-copy-bandwidth bands a healthy card must land in.

5) results/PROGRESS.md — template with headings: Current step, Last green gate, Device used, Open issues, Next action. Fill: Current step = "Setup 3 complete", Next action = "Step 6 on the 4090".

6) docs/monoid_zoo.md — title plus empty sections: AddU64, MatModP, Segmented<M>, AffineF32, MobiusF32, FsmCsv, MaxPlus<K>, each with sub-headings "Definition", "Associativity proof", "Identity".

7) .gitignore — bin/, *.o, __pycache__/, .DS_Store, results/*.tmp.

8) README.md — one paragraph describing SWEEP, pointers to SWEEP_MASTER_PLAN.md and CLAUDE.md, and a "Non-goals" section copied verbatim from the plan's Appendix C.

Do not write CUDA source yet. Commit with message "setup 3: repo skeleton, Makefile, scripts, machines.json" and show me `git log -1 --stat`.
```

**Your Verify.** `make test_cpu` fails only because `tests/test_common.cpp` doesn't exist yet (that's Step 7) — the Makefile itself parses. `bash scripts/provenance.sh` on the Mac fails gracefully (no nvidia-smi) — fine. Tree matches. Pushed.

### Setup 4 — Kaggle

Account → verify phone (required for GPUs). A session = New Notebook → Session options → Accelerator → GPU T4 ×1 or P100. ~30 GPU-hours/week; idles out after ~20 minutes — commit early.

### Setup 5 — Vast template (already built; verify once)

Template `sweep-cuda1281-devel-ssh`, private: image `nvidia/cuda` tag `12.8.1-devel-ubuntu22.04`; Interactive shell + SSH, direct; ports and environment empty; on-start `apt-get update -qq && apt-get install -y -qq git tmux`; disk 25 GB; **no volume**. The CLI preview must read exactly `--image nvidia/cuda:12.8.1-devel-ubuntu22.04 --onstart-cmd '…' --disk 25 --ssh --direct` — you already caught the space-around-the-colon failure this way. Wallet ~$20; at $0 balance running instances are terminated.

---

## Part III — The session rituals (memorize)

### Ritual A — a Vast work session, end to end

1. **Get an instance.** Search with the template → offer passes the recipe (Verified, reliability ≥99%, on-demand, Max CUDA ≥12.8, sane $/TB internet, non-CN preferred for git speed) → RENT.
2. **Connect.** Card → **>_Connect** → copy the shown `ssh -p <PORT> root@<IP>` (drop any `-L 8080:…` suffix) → paste into a Mac terminal → first contact asks `(yes/no)` → `yes`. You're now a root shell inside the container. `Permission denied (publickey)` = instance predates your key → key icon on the card, or re-rent.
3. **tmux, always.** `tmux new -s work`. Detach: `Ctrl-B` then `D`. Resume after reconnect: `tmux attach -t work`. List: `tmux ls`.
4. **Bootstrap.** `export GH_TOKEN=<token>` then `bash <(curl -fsSL https://raw.githubusercontent.com/<you>/sweep/main/scripts/pod_init.sh)` — or paste the script's three lines. (Private repo makes the raw URL need the token too; pasting is simpler.)
5. **Gates** (any *new* host, first five minutes; Step 6 builds the script): `bash scripts/gate.sh sm_89` → PASS required. FAIL → destroy, next offer; you lost ~$0.01.
6. **Claude Code.** `tmux new -s claude` (a second session) → `claude` → paste the day's step prompt (Ritual E).
7. **Leave.** `git status` clean & pushed → `exit` (tmux) → `exit` (ssh) → console → **trash → Destroy** → meter $0. Stopped is not an exit state.

**If SSH drops:** reconnect, `tmux attach -t work` (or `-t claude`). Nothing was lost.

### Ritual B — a Kaggle session (T4/P100)

```python
!export PATH=/usr/local/cuda/bin:$PATH; nvidia-smi; nvcc --version   # nvcc may be off PATH on Kaggle
!git clone https://<TOKEN>@github.com/<you>/sweep.git
%cd sweep
!export PATH=/usr/local/cuda/bin:$PATH; bash scripts/gate.sh sm_75      # sm_60 on P100
# work cells: !export PATH=/usr/local/cuda/bin:$PATH; make bin/<x> ARCH=sm_75 && ./bin/<x>
!git config user.email "<you>@users.noreply.github.com" && git config user.name "Ahad"
!git add -A && git commit -m "kaggle T4: <what>" && git push
```

Edit on the Mac → push → `!git pull` in the notebook; `%%writefile path` exists for hotfixes. Kaggle Secrets can hold the token.

### Ritual C — VS Code Remote (recommended from Day 2)

VS Code → Extensions → Remote - SSH → `Cmd+Shift+P` → *Remote-SSH: Add New SSH Host* → paste the Connect command → connect → Open Folder `/root/sweep`. Real editor on files living on the pod; the integrated terminal is on the pod (still start tmux for long runs). New instance = new IP/port = add again (10 seconds). (Sublime has no built-in terminal — the Terminus package adds one — but it can't edit remote files natively; VS Code Remote is the path.)

### Ritual D — host-acceptance gate

`scripts/gate.sh <ARCH>`: nvcc version ✓ → power limit near spec ✓ → bandwidth probe inside the `machines.json` band ✓ → appends a provenance line to `results/machines_log.csv`. Then time one `git push` round-trip. Any ✗ → destroy, next offer. Never trust a host's label; trust its measurement.

### Ritual E — running a step with Claude Code

See "How to use this plan with Claude Code" at the top: right machine → tmux → `claude` → paste prompt → read its plan → let it work → run **Your Verify** yourself → check commit + PROGRESS.md → push. Red → paste failure back with "fix without changing the test tolerance or the ordering contract."

---
## Part IV — Week 1: the scan engine for plain numbers (Days 1–7)

Strategy: build scan in four rings — one warp (registers), one block (shared memory), many blocks the easy slow way (two passes), many blocks the hard fast way (single-pass decoupled lookback). Each ring is verified before the next uses it. Every step: Goal → plain-language "what we're doing" → Claude Code prompt → **Your Verify** (the human gate) → what to watch for in Claude Code's output. Never advance past a red gate.

### Step 6 — Timing harness, bandwidth probe, host gate (pod, `sm_89`)

**Goal.** Measure this 4090's *achieved* bandwidth — the denominator of every performance claim for four months — and turn the measurement into a reusable host-acceptance test.

**What we're doing, plainly.** GPU kernels launch asynchronously: the CPU's `start = now(); launch(); end = now()` measures only how long it took to *queue* the work. So we timestamp on the GPU itself with CUDA events, take many repetitions, and report the median (clocks wobble). The probe kernel is the simplest possible: copy 256 MB from one buffer to another with the widest loads. Its speed is the physical ceiling ("roofline") — anything we build later is judged as a percentage of it. And because rented hardware can be throttled or misconfigured, the same probe becomes a gate: a card measuring far below its known band gets rejected.

**Claude Code prompt:**

```text
Context: This is the first GPU task of SWEEP. Read CLAUDE.md first. Environment: a Vast.ai pod with an RTX 4090, CUDA 12.8 devel image; confirm with `nvcc --version` (12.8) and `nvidia-smi` before building; ARCH=sm_89. We need (a) a correct GPU timing helper and (b) a bandwidth probe whose result becomes our roofline denominator and our host-acceptance gate.

1) include/sweep/timing.cuh (header-only; includes only <cuda_runtime.h>, <cstdio>, <cstdlib>, <algorithm>, <vector>):
- Macro CUDA_CHECK(x): if the cudaError_t is not cudaSuccess, print cudaGetErrorString plus __FILE__:__LINE__ to stderr and exit(1).
- template <class F> float time_kernel_ms(F fn, int warmup = 5, int reps = 50): call fn() warmup times untimed; then reps times: cudaEventRecord(start); fn(); cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime. Sort and return the MEDIAN in ms (median, not mean — robust to clock wobble on shared hosts). Destroy the events.

2) bench/bw_probe.cu:
- N = 1u << 26 floats (256 MB) — large enough that caches cannot hide DRAM speed. cudaMalloc two buffers of N floats.
- Kernel copy4(const float4* in, float4* out, size_t n4): each thread copies exactly one float4 (16 bytes — the widest single load; adjacent threads hit adjacent addresses so the warp's accesses coalesce into full transactions). Grid = ceil(n4 / 256), block = 256.
- Time it with time_kernel_ms. Print exactly one line "achieved bandwidth: <X> GB/s" where bytes = 2 * N * sizeof(float) (N read + N written). Also print the device name via cudaGetDeviceProperties on a separate line "device: <name>". Free memory, return 0.

3) scripts/gate.sh <ARCH> (default sm_89): (a) print nvcc version; (b) print power limit: nvidia-smi -q -d POWER | grep -i "power limit" | head -n 2; (c) make bin/bw_probe ARCH=<ARCH> && ./bin/bw_probe, capturing the GB/s number; (d) compute gpu_key the same way scripts/provenance.sh does (call provenance.sh and take field 1); (e) read [lo, hi] for that key from results/machines.json with python3 -c (fall back to jq if python3 is absent); (f) print PASS if lo <= X <= hi else FAIL, append "<provenance line>,<X>,<PASS|FAIL>" to results/machines_log.csv, exit 0 on PASS and 1 on FAIL. If the key is missing from machines.json, print "UNKNOWN DEVICE" and exit 2 so I can add a band manually.

Run gate.sh here and report the exact GB/s. Expected for a healthy 4090: 850–950 (the Vast listing's own measurement was 901). If it is below band, STOP and tell me — do not tune anything to make it pass.

Do not use -use_fast_math. Keep the kernel trivially simple. Commit: "step 6: timing harness, bandwidth probe, host gate [4090]". Update results/PROGRESS.md (Current step 6 done, device RTX_4090, next Step 7). Show me git log -1 --stat and the gate output.
```

**Your Verify.** `bash scripts/gate.sh sm_89` prints PASS with ~850–950 GB/s and a new row in `results/machines_log.csv`. Commit pushed.

**Watch for.** Claude Code using `clock()`/`chrono` instead of events; using mean instead of median; a probe size under ~100 MB (cache-inflated numbers); "fixing" a FAIL by widening the band.

### Step 7 — The oracle and the test harness (Mac or pod)

**Goal.** A slow, obviously-correct CPU scan and the checking helpers every later test calls.

**What we're doing, plainly.** Before any parallel trick, we write the boring for-loop version — the *oracle*. It defines "correct." Integer results from the GPU must match it bit for bit. Float results legitimately differ in rounding depending on the order things were added, so for floats we compare against a *64-bit* oracle and measure the gap in ULPs (how many representable floats apart). We also fix a list of test sizes that deliberately hits every boundary: empty, one element, 31/32/33 (warp edges), 255/256/257 (block edges), tile edges, and something big.

**Claude Code prompt:**

```text
Context: SWEEP needs a CPU oracle and shared test helpers before any GPU scan code. Read CLAUDE.md. This step must compile on my Mac with clang++ (no CUDA) and also under nvcc later; it may run on either machine.

Create tests/common.hpp (header-only, C++17, includes only <vector>, <cmath>, <cstdint>, <cstring>, <cstdio>, <cstdlib>, <random>, <algorithm>):
- template <class T, class Op> std::vector<T> cpu_inclusive_scan(const std::vector<T>& x, Op op): out[0] = x[0]; out[i] = op(out[i-1], x[i]). The op takes (earlier, later) in that order — this is THE project-wide ordering contract; write it as a comment above the function.
- inline std::vector<int> test_sizes(): {0, 1, 2, 31, 32, 33, 255, 256, 257, 1023, 1024, 1025, 65535, 65536, 65537, (1<<20)+7, 1<<24}.
- inline int64_t ulp_diff(float a, float b): copy bits into int32 via memcpy; for negative values map i -> INT32_MIN - i so the integer order matches float order (this also maps -0.0 onto 0); return |ia - ib| as int64. Provide a double overload using int64.
- Seeded generators using std::mt19937_64: fill_uniform(std::vector<float>&, float lo, float hi, uint64_t seed), fill_u32(std::vector<uint32_t>&, seed), fill_u64(std::vector<uint64_t>&, seed).
- Reporting: a global int g_failures; report_exact(const char* name, bool ok) and report_ulp(const char* name, int64_t max_ulp, int64_t tolerance) print "PASS <name>" or "FAIL <name> ..." and bump g_failures on failure; macro FINISH() that prints a summary and returns g_failures ? 1 : 0 from main.

Create tests/test_common.cpp: cpu_inclusive_scan on {1,2,3,4} with + equals {1,3,6,10}; ulp_diff(1.0f, nextafterf(1.0f, 2.0f)) == 1; ulp_diff(-0.0f, 0.0f) == 0; ulp_diff(1.0f, -1.0f) is large and positive; a 10^6-element float scan's cpu result compared against a double-precision cpu scan reports max ULP (just print it).

Build with `make test_cpu` and run bin/test_common. Commit: "step 7: cpu oracle, test sizes, ulp helpers". Update PROGRESS.md. Show me the test output.
```

**Your Verify.** `make test_cpu && ./bin/test_common` prints all PASS. The summary line shows zero failures.

**Watch for.** A `ulp_diff` that forgets the negative-value remap (it would report absurd distances across zero).

### Step 8 — Warp scan: 32 threads, zero memory (pod)

**Goal.** Inclusive scan across one warp using only register shuffles.

**What we're doing, plainly.** A warp's 32 threads can hand values to each other through registers with a "shuffle" instruction. For offsets d = 1, 2, 4, 8, 16, every lane fetches the value from the lane d below it and adds it if such a lane exists. After five rounds each lane holds its running total. This is Kogge–Stone (1973) — the carry-lookahead adder inside every CPU, running on a GPU.

**Claude Code prompt:**

```text
Context: SWEEP Step 8. Read CLAUDE.md. Pod, RTX 4090, ARCH=sm_89. We implement the smallest ring of the scan engine: an inclusive scan across one warp (32 lanes) using __shfl_up_sync, Kogge–Stone style.

Create include/sweep/scan.cuh with:
- __device__ __forceinline__ float warp_inclusive_scan(float x): for d = 1, 2, 4, 8, 16: y = __shfl_up_sync(0xffffffffu, x, d); if (lane >= d) x += y; where lane = threadIdx.x & 31. Return x.
- An overload for unsigned int (same structure, wraparound add — exact arithmetic for bit-exact tests).
RULES that must hold: the shuffle is executed by all 32 lanes unconditionally with the FULL mask 0xffffffff; only the ADD is conditional; the shuffle must never sit inside a divergent branch. Callers pad missing elements with the identity (0) rather than leaving lanes out — write this rule as a comment.

Create tests/test_warp.cu (uses tests/common.hpp and include/sweep/timing.cuh's CUDA_CHECK): launch 1 block of 32 threads. Tests: (a) all ones -> 1..32 exactly; (b) 1000 random float arrays of 32 compared to a double-precision CPU scan — report the maximum ULP over all runs (record the number; a single warp combines in a fixed order so expect a small constant, likely <= 2); (c) 1000 random unsigned int arrays — bit-exact vs the CPU oracle. Use report_exact / report_ulp / FINISH.

Build: make bin/test_warp ARCH=sm_89; run; then run compute-sanitizer --tool memcheck ./bin/test_warp. Commit: "step 8: warp inclusive scan + tests [4090]". Update PROGRESS.md. Show test and sanitizer output.
```

**Your Verify.** All PASS; memcheck reports 0 errors.

**Watch for.** A partial shuffle mask, or the shuffle placed inside the `if` (undefined behavior that may happen to work here and explode on another GPU).

### Step 9 — Block scan: 256 threads via shared memory (pod)

**Goal.** Inclusive scan across one block.

**What we're doing, plainly.** Eight warps each scan themselves; their eight totals are scanned by warp 0 through a tiny shared-memory array; each thread then adds the total of all *previous* warps to its own value. Two levels of the same idea — the grid-level algorithm will have this exact shape.

**Claude Code prompt:**

```text
Context: SWEEP Step 9. Read CLAUDE.md. Pod, ARCH=sm_89. Add a block-level inclusive scan to include/sweep/scan.cuh built on the Step-8 warp scan.

__device__ float block_inclusive_scan(float x) (and an unsigned int overload):
- Precondition (comment it): blockDim.x is a multiple of 32 and <= 1024; the project uses 256.
- __shared__ float warp_sums[32]; lane = threadIdx.x & 31; wid = threadIdx.x >> 5.
- v = warp_inclusive_scan(x); if lane == 31, warp_sums[wid] = v; __syncthreads().
- Warp 0 only: nwarps = (blockDim.x + 31) / 32; w = (lane < nwarps) ? warp_sums[lane] : 0; w = warp_inclusive_scan(w); warp_sums[lane] = w. Then __syncthreads() — this second barrier is the classically forgotten one (warp 0 writes, other warps read).
- Return (wid > 0 ? warp_sums[wid - 1] : 0) + v.

Create tests/test_block.cu: 1 block × 256 threads; (a) 1000 random float arrays vs double CPU oracle, report max ULP; (b) an array with exactly 33 nonzero elements followed by zeros, checking positions 0..32 and that the remaining positions equal the running total; (c) unsigned int bit-exact fuzz. Build, run, then run compute-sanitizer --tool racecheck ./bin/test_block — shared-memory races are exactly what racecheck finds; it must report zero hazards.

Commit: "step 9: block inclusive scan + tests [4090]". Update PROGRESS.md.
```

**Your Verify.** All PASS; racecheck: 0 hazards.

**Watch for.** Missing second `__syncthreads()` (racecheck will catch it — that's why it's in the gate).

### Step 10 — Multi-block, the honest slow way: two-pass reduce-then-scan (pod)

**Goal.** A correct scan for any n; the baseline the single-pass engine must beat; later the deterministic "fixed tree" schedule for the study.

**What we're doing, plainly.** With many blocks, block b needs the total of blocks 0..b−1, but blocks can't wait on each other safely in a normal launch. Two passes sidestep waiting entirely: pass A — each block sums its tile and writes one number; a tiny second kernel scans those numbers; pass B — each block re-scans its tile and adds its offset. Correct and simple, but the input is read *twice*: ~3n bytes of traffic against the ideal 2n, a 1.5× ceiling. That ceiling is the entire motivation for Step 11.

**Claude Code prompt:**

```text
Context: SWEEP Step 10. Read CLAUDE.md. Pod, ARCH=sm_89. Implement the classical two-pass multi-block scan in include/sweep/scan.cuh, float and unsigned int versions, plus a host wrapper.

Kernels (tile = blockDim.x = 256 elements; here using blockIdx.x is FINE because nothing waits on another block — add that comment so nobody "fixes" it later):
- reduce_tiles(const T* in, T* partials, int n): tile sum via block_inclusive_scan (the last thread's inclusive value); out-of-range elements contribute the identity 0; last thread writes partials[blockIdx.x].
- scan_partials(T* partials, int nblocks): ONE block of 256 threads loops over the partials in chunks of blockDim: s = block_inclusive_scan(x) + carry; write back; the last thread stores the chunk's final value into a __shared__ carry slot; __syncthreads(); carry = slot; __syncthreads(). Must be correct when nblocks > blockDim (n = 2^24 gives 65,536 partials).
- scan_tiles(const T* in, T* out, const T* partials, int n): re-scan the tile; offset = (blockIdx.x > 0) ? partials[blockIdx.x - 1] : 0 (EXCLUSIVE offset — off by one here is the classic bug); out = offset + local for in-range elements.
Host wrapper template <class T> void scan_two_pass(const T* d_in, T* d_out, int n, T* d_partials, cudaStream_t s = 0): if n == 0 return without launching (a zero-block grid is a launch error); grid = ceil(n / 256); launch the three kernels in order on the stream.

Tests (tests/test_two_pass.cu): every size in test_sizes(): float vs double CPU oracle (report max ULP per size and the overall max; record the overall number in results/notes_ulp.md), unsigned int bit-exact. Benchmark (bench/bench_two_pass.cu): n = 2^24, report ms, GB/s using bytes = 3 * n * 4 (read, read, write) plus partial traffic, and the percentage of the measured bandwidth from the latest RTX_4090 row of results/machines_log.csv. Expect roughly 55–70% of measured bandwidth.

Sanitizers: memcheck + racecheck on the test binary. Commit: "step 10: two-pass reduce-then-scan + tests + bench [4090]". Update PROGRESS.md; report the ULP number and the GB/s.
```

**Your Verify.** All sizes PASS (including 0 and 2^24); the GB/s is in the 55–70% band of your Step-6 number.

**Watch for.** A `scan_partials` that only handles ≤256 partials; an inclusive offset used where exclusive is required.

### Step 11 — Decoupled lookback: the single-pass engine (pod)

**Goal.** One pass over memory, ~2n traffic, any n. The hardest and most important step.

**What we're doing, plainly.** Merrill & Garland's 2016 algorithm — the one inside NVIDIA's CUB. Every block scans its tile *immediately*. To learn "what came before me," it consults a small **descriptor** table in global memory with one slot per block. A slot says one of three things: INVALID (that block hasn't published yet), AGG ("here is my tile's total"), or PFX ("here is the total of everything up to and including me"). Block b walks left from slot b−1: wait until the slot isn't INVALID; if it says AGG, take the value and keep walking; if it says PFX, take it and stop — everything further left is already inside it. Then b publishes its own PFX so blocks to its right stop early. Consuming AGGs instead of waiting for a single predecessor's final answer is what keeps blocks from serializing — redundant reads "decouple" local work from global propagation.

Two traps: **(1) Deadlock.** The GPU doesn't promise block 4 starts before block 5. If block 5 occupies the last free slot spinning on block 4, which is still queued behind it, everything hangs forever. Fix: ignore `blockIdx.x`; each block's thread 0 takes a **ticket** from a global counter, and the block uses the ticket as its position. Ticket order = actual start order, so any block a block waits on has already started and will publish. **(2) Torn reads.** Status and value must be seen together. For a 32-bit value: pack 2 status bits + 32 value bits into one 64-bit word, so every publish — including the later AGG→PFX upgrade that *changes the value* — rewrites the whole word atomically. (Bigger values can't do this; Step 14 builds the general protocol.)

**Claude Code prompt:**

```text
Context: SWEEP Step 11 — the core algorithm. Read CLAUDE.md invariants 2 and 4 first. Pod, RTX 4090 (128 SMs), ARCH=sm_89. Implement single-pass decoupled-lookback inclusive scan for float and for unsigned int (u32 — NOT u64: a 64-bit value cannot share one 64-bit word with a 2-bit status).

include/sweep/descriptor.cuh (packed path):
- namespace sweep; constants ST_INVALID=0, ST_AGG=1, ST_PFX=2 (unsigned long long).
- pack(status, float v) = (status << 62) | (unsigned long long)__float_as_uint(v); status_of(d) = d >> 62; value_of(d) = __uint_as_float((unsigned)(d & 0xffffffffull)). Provide the u32 analogues (value stored directly in the low 32 bits). A memset-to-zero descriptor therefore reads as INVALID with value 0 — say so in a comment.

include/sweep/scan.cuh — kernel scan_lookback_f32(const float* in, float* out, int n, unsigned long long* desc, int* ticket) and the u32 twin (template on T if you prefer, with T-specific pack/unpack):
1. __shared__ int sbid; thread 0: sbid = atomicAdd(ticket, 1); __syncthreads(); bid = sbid. This ticket is the block's LOGICAL id; tile index = bid * blockDim.x; blockIdx.x is never used for anything another block waits on (deadlock trap — comment it).
2. x = (i < n) ? in[i] : identity; local = block_inclusive_scan(x).
3. __shared__ agg; the last thread stores its inclusive value; __syncthreads().
4. Thread 0 publishes IMMEDIATELY, before looking back: bid == 0 ? pack(ST_PFX, agg) : pack(ST_AGG, agg), via cuda::atomic_ref<unsigned long long, cuda::thread_scope_device>(desc[bid]).store(..., cuda::memory_order_relaxed). Relaxed is sufficient in THIS packed path because status and value are one atomic word and no other memory must be ordered with it — write that justification as a comment, and note Step 14's generic path needs release/acquire.
5. Thread 0 lookback: excl = identity; for p = bid-1 down to 0: spin: d = atomic_ref(desc[p]).load(relaxed) until status_of(d) != ST_INVALID; excl = combine(value_of(d), excl) — PREPEND: the found value is EARLIER than what we hold, so it goes on the left; for float this is value + excl; write it in that order even though float addition is commutative, because later monoids are not; if status_of(d) == ST_PFX break.
6. Thread 0 publishes its own PFX: store(pack(ST_PFX, excl + agg), relaxed). Store excl in __shared__; __syncthreads(); every in-range thread writes out[i] = excl + local.
Host wrapper template <class T> struct LookbackScanner { allocate desc for max_blocks and one int ticket at construction; void run(const T* in, T* out, int n, cudaStream_t s = 0): if n == 0 return; grid = ceil(n/256); cudaMemsetAsync(desc, 0, grid*8, s); cudaMemsetAsync(ticket, 0, 4, s); launch; } — both memsets before EVERY launch, same stream, non-negotiable (stale PFX entries from a previous run give silently wrong answers). Comment that a phase-counter trick could remove the memsets and that it is a stretch goal only.

Tests (tests/test_lookback.cu): (a) every size in test_sizes(), float vs double oracle (max ULP recorded) and u32 bit-exact; n = 2^24 gives 65,536 blocks on 128 SMs — the regime where the deadlock trap bites if the ticket were wrong; (b) hang test: 500 back-to-back launches at n = 2^22 must complete; run the binary under `timeout 180` in the Makefile-free command line so a hang becomes a visible failure; (c) sanitizers: compute-sanitizer --tool memcheck and --tool racecheck both clean.

If any test hangs: the causes are (in order) blockIdx.x used where the ticket should be, or a skipped memset. Do not add sleeps or retries to "fix" a hang.

Commit: "step 11: decoupled lookback scan f32/u32 + tests [4090]". Update PROGRESS.md. Show me: test output, the hang-test timing, and both sanitizer summaries.
```

**Your Verify.** All sizes PASS; 500 launches complete well within the timeout; memcheck and racecheck report zero. Read the kernel yourself once and find the ticket, the immediate publish, and the prepend line — you should be able to point at all three.

**Watch for.** Any `blockIdx.x` used in the lookback path; a missing memset; the prepend written as `excl + value` "because it's the same for floats."

### Step 12 — Benchmark vs CUB; close the gap; meet the nondeterminism (pod)

**Goal.** Within ~10% of CUB's `DeviceScan`; first contact with the phenomenon Week 4 studies.

**What we're doing, plainly.** One element per thread pays the descriptor protocol for every 256 elements. The standard fix: each thread loads *four* consecutive floats as one 16-byte `float4`, adds them up serially in registers, the block scans the per-thread totals, and each thread reconstructs its four running values locally. Tiles grow 4×, the protocol cost amortizes 4×, memory traffic stays 2n. Then we compare against NVIDIA's own implementation — the honest yardstick. Finally we run the scan 10,000 times on identical input and hash each output: we expect to see *more than one distinct result*, and that is not a bug — how far each block's lookback walks depends on scheduling timing, different walks group the same additions differently, and floating-point addition rounds differently under different grouping. NVIDIA's docs say the same; Week 4 makes this a study.

**Claude Code prompt:**

```text
Context: SWEEP Step 12. Read CLAUDE.md (invariants 4, 6; benchmark rules). Pod, ARCH=sm_89. Three tasks: an ITEMS=4 lookback variant, a benchmark against CUB, and the first determinism harness.

1) ITEMS-per-thread lookback (scan.cuh): template parameter ITEMS (implement 4; keep 1 working). Tile = ITEMS * 256 elements. Thread t loads its ITEMS consecutive elements starting at tile_base + ITEMS*t as one float4 when the whole float4 is in range, else element-by-element with identity padding for the tail. Serial prefix over its items in registers: p[0]=x[0], p[k]=p[k-1]+x[k]. Block-scan the per-thread totals p[ITEMS-1] with block_inclusive_scan; thread's exclusive offset = inclusive - total. Descriptor protocol exactly as Step 11 (ticket, immediate publish, prepend, memsets). Output out[tile_base + ITEMS*t + k] = block_excl + thread_excl + p[k]; store as float4 when fully in range. Grid = ceil(n / (ITEMS*256)). Keep the u32 twin so bit-exact fuzz still runs. All Step-11 tests must pass for ITEMS=4 too.

2) bench/bench_scan.cu: contenders = two-pass, lookback ITEMS=1, lookback ITEMS=4, CUB (cub::DeviceScan::InclusiveSum: first call with nullptr temp storage to get temp_bytes, cudaMalloc, then time the real call). Sizes n = 2^20, 2^22, 2^24, 2^26. For each size run the contenders INTERLEAVED (A B C D A B C D ...) rather than all of one then all of the next — shared hosts drift. Byte accounting: ours = 2*n*4 plus our memsets (grid*8 + 4) — count them honestly; CUB = 2*n*4. Output CSV results/bench_scan_<gpu_key>.csv with columns: contender,n,ms,gbps,pct_of_measured_bw,<the 9 provenance fields from scripts/provenance.sh>. pct uses the latest RTX_4090 gbps from results/machines_log.csv (read it in the harness; fail loudly if absent).

3) study/determinism.cu (v1): one fixed-seed random float array, n = 2^24; run lookback ITEMS=4 K = 10,000 times on identical input; after each run compute a 64-bit FNV-1a hash over the output bytes ON THE HOST (copy back) or with a simple device reduction — host is fine for v1; count distinct hashes; print "distinct outputs: <k> of 10000". Also run the u32 variant the same way and print its distinct count (must be exactly 1 — the control that proves the float variance is rounding order, not a race). Print a one-paragraph explanation of WHY the float count exceeds 1 (lookback stopping point depends on scheduling → different parenthesization → different rounding).

Exit criteria: ITEMS=4 >= 90% of CUB's GB/s at n = 2^24 and >= 80% of the measured bandwidth; two-pass visibly slower; float determinism > 1 distinct hash; u32 exactly 1. If ITEMS=4 misses 90% of CUB, profile-by-reasoning first: check occupancy with cudaOccupancyMaxActiveBlocksPerMultiprocessor, check that loads are float4, check the lookback isn't doing extra atomic traffic; report what you find before changing the algorithm. Do NOT "fix" the float nondeterminism.

Commit: "step 12: ITEMS=4 lookback, CUB benchmark, determinism v1 [4090]". Update PROGRESS.md. Show me the CSV rows for n=2^24 and both distinct-hash counts.
```

**Your Verify.** CSV rows exist with provenance columns; ITEMS=4 within the bands; the two hash counts are (>1, 1). Look at `results/bench_scan_RTX_4090.csv` yourself.

**Watch for.** Benchmarks run all-A-then-all-B; CUB timed with temp allocation inside the timed region; a "helpful" change that makes the float scan deterministic (that destroys the study's subject).

---
## Part V — Week 2: from float-add to any monoid (Days 8–12)

Week 1's engine is welded to `float` and `+`. Week 2 makes the value type and the combine rule template parameters. That forces two real problems: handing a *multi-word* value between blocks safely, and getting left/right order right when order matters.

### Step 13 — The monoid interface; the engine templated over it (pod)

**Goal.** `scan<M>` for any monoid `M`; two exact monoids proving it bit for bit.

**What we're doing, plainly.** A "monoid" in code is a struct with a value type, an `identity()`, and `combine(earlier, later)`. The word *earlier* is a contract: the first argument is always the one that sits earlier in the array. We test with two monoids whose arithmetic is exact (so no float tolerance can hide bugs): 64-bit integer addition, and 2×2 matrices of integers modulo a prime — matrix multiplication is associative but **not commutative**, which arms the trap Step 15 springs. Because warp shuffles only move 32-bit words, the generic block scan goes through shared memory instead.

**Claude Code prompt:**

```text
Context: SWEEP Step 13. Read CLAUDE.md invariant 1 (ordering contract) and 5. Pod, ARCH=sm_89. Generalize the engine from float/+ to any monoid.

include/sweep/monoids.cuh:
- Interface (document at top): a monoid M has `using T = ...` (trivially copyable POD), `__host__ __device__ static T identity()`, `__host__ __device__ static T combine(T earlier, T later)`. CONTRACT: `earlier` is the element that sits earlier in the array. For monoids whose elements are transformations (matrices acting on column vectors, state tables, functions), "earlier applied first, then later" is the composition later∘earlier, i.e. the matrix product B*A when A is earlier and B is later. Put this explanation in a comment block; it is the single most important comment in the repo.
- struct AddU64: T = unsigned long long; identity 0; combine = earlier + later (wraparound is still associative).
- struct MatModP: P = 1000000007; T { unsigned long long m[4]; } row-major m00 m01 m10 m11; identity {1,0,0,1}; combine(a, b) returns b*a mod P (a = earlier, b = later): r00 = b00*a00 + b01*a10; r01 = b00*a01 + b01*a11; r10 = b10*a00 + b11*a10; r11 = b10*a01 + b11*a11, each % P (entries < 2^30 so products fit in 64 bits before the sum; reduce after the sum). Exact and NON-commutative on purpose.
- struct AddF32 wrapping the Week-1 float path (identity 0.f, combine earlier + later) so the templated API covers floats too.

include/sweep/scan.cuh:
- template <class M> __device__ typename M::T block_scan_generic(typename M::T v): Hillis–Steele in dynamic shared memory: extern __shared__ unsigned char smem_raw[]; T* s = (T*)smem_raw; s[t] = v; __syncthreads(); for d = 1, 2, 4, ... < blockDim: if (t >= d) earlier = s[t-d]; __syncthreads(); if (t >= d) s[t] = M::combine(earlier, s[t]); __syncthreads(); return s[t]. Two barriers per round — the read must fully complete before any write (racecheck will confirm). Launch sites must pass 256*sizeof(T) bytes of dynamic shared memory: kernel<<<grid, 256, 256*sizeof(typename M::T)>>>. Keep the Week-1 float/u32 shuffle path as a specialization (faster); it and the generic path must agree.
- template <class M> generic two-pass multi-block scan (reduce_tiles<M>, scan_partials<M>, scan_tiles<M>) with partials of type T — this is the multi-block path for arbitrary monoids until Step 14 delivers the generic descriptor. Host API: template <class M> void scan_two_pass(const T*, T*, int n, T* partials, stream); n == 0 returns early.
- A single entry point template <class M> void sweep::scan(const T* in, T* out, int n, Workspace& ws, stream) that dispatches to the best available path for M (two-pass generic for now; Step 14 switches to lookback).

tests/test_generic.cu: for every size in test_sizes(): AddU64 and MatModP through block_scan_generic (single block) and through the generic two-pass (multi-block), compared BIT-EXACT to cpu_inclusive_scan with the same combine(earlier, later). The oracle lambda must call M::combine(prev, x) in that order — verify by inspection and say so in the output header. Also AddF32 through the generic path vs the float specialization: results must match to <= 2 ULP (same tree shape) — record.

Sanitizers: racecheck on test_generic (the two-barrier discipline). Commit: "step 13: monoid interface, AddU64/MatModP, generic block scan + two-pass [4090]". Update PROGRESS.md; show test output.
```

**Your Verify.** Bit-exact PASS for both exact monoids at every size; racecheck clean. Open `monoids.cuh` and read the contract comment — you should be able to explain B·A vs A·B to yourself.

**Watch for.** `a*b` instead of `b*a` in MatModP; a single barrier in Hillis–Steele; launches without dynamic shared memory (illegal address at runtime).

### Step 14 — Big-payload descriptor: release/acquire done right (pod)

**Goal.** The inter-block handoff for values too big to fit in one atomic word — with a stress test that would catch a wrong protocol.

**What we're doing, plainly.** A MatModP element is 32 bytes; no 32-byte atomic exists on any of our GPUs. So the status flag and the value now live in *separate* memory words, and the hardware is allowed to make another block see them in the wrong order — status first, stale value second — unless we say otherwise. Two ingredients make it safe. First, **each value slot is written exactly once**: the descriptor holds *two* value slots, `aggregate` and `inclusive`; the AGG→PFX upgrade writes the second slot rather than overwriting the first, so a reader who saw AGG and is reading `aggregate` can never have it change underneath them. Second, **release/acquire**: the writer stores the value, then stores the status with *release* ("everything before this is visible to whoever sees it"); the reader loads the status with *acquire* ("if I see it, I see what preceded it"). Then we stress-test it the way GPU memory-model researchers do: a tiny two-block program repeated 100,000 times with randomized delays, hunting for a torn value.

**Claude Code prompt:**

```text
Context: SWEEP Step 14. Read CLAUDE.md invariant 2 carefully. Pod, ARCH=sm_89. Build the generic (multi-word) descriptor protocol, the generic lookback kernel, and a litmus stress test.

include/sweep/descriptor.cuh (generic part):
- enum { GST_INVALID = 0, GST_AGG = 1, GST_PFX = 2 }.
- template <class T> struct Desc { T aggregate; T inclusive; int status; }. status is a plain int so cudaMemsetAsync(0) resets it; ALL accesses go through cuda::atomic_ref<int, cuda::thread_scope_device>. aggregate is written exactly once (before status -> AGG); inclusive is written exactly once (before status -> PFX). Never overwrite a slot.
- template <class T> __device__ void publish(Desc<T>* d, const T& v, int st): plain-store v into the slot named by st (aggregate for AGG, inclusive for PFX), then atomic_ref(status).store(st, cuda::memory_order_release).
- template <class T> __device__ int spin_read(Desc<T>* d, T& out): loop atomic_ref(status).load(cuda::memory_order_acquire) until != GST_INVALID; then out = (st == GST_AGG) ? d->aggregate : d->inclusive; return st. Comment: acquire pairs with the writer's release, so the slot read after it is guaranteed to observe the write that preceded the release.
- Comment explaining why a reader may consume an AGG and keep walking even if that block upgrades to PFX a microsecond later: allowed; it only costs redundant reads and changes the grouping — the grouping variation is what the Week-4 study measures.

include/sweep/scan.cuh: template <class M, int ITEMS> __global__ void scan_lookback(const T* in, T* out, int n, Desc<T>* desc, int* ticket): Step-11 skeleton with the ticket, immediate publish (bid==0 -> PFX else AGG) via publish(), block_scan_generic<M> (dynamic shared memory), the lookback walk via spin_read with PREPEND excl = M::combine(found, excl), break on PFX, then publish own PFX = M::combine(excl, agg), then out = M::combine(excl, local) per element. Host: LookbackScanner<M> memsets desc (grid * sizeof(Desc<T>)) and ticket before EVERY launch; n == 0 early return. Route sweep::scan<M> to this path; keep two-pass available under an explicit name.

tests/test_generic_lookback.cu: AddU64 and MatModP bit-exact at every size; 500 back-to-back launches at n = 2^22 under `timeout 240`; compute-sanitizer memcheck and racecheck clean.

tests/litmus.cu (message-passing stress test in the style of Alglave et al., ASPLOS 2015): two blocks. Block A: write a 32-byte pattern derived from the iteration number into a Desc<Pattern>.aggregate, spin a random number of clock64() cycles (0–2000), then publish(AGG). Block B: spin_read, then verify every byte of the pattern matches the iteration; count torn observations. Run 100 launches × 1000 iterations each (reset the descriptor between iterations with a device-side store of INVALID plus a __threadfence, or memset between launches). Print torn count — required: 0. Add a compile flag -DSWEEP_NEGATIVE_CONTROL that swaps release/acquire for relaxed in publish/spin_read ONLY (nothing else). Build both variants; run the negative control for 10× more iterations. Report the torn count for both. Note in the output: if the negative control shows torn reads, the platform exposed the weak behavior; if it shows zero, the platform did not expose it in this run — the release/acquire protocol stays either way because the PTX memory model permits the reordering.

Commit: "step 14: generic release/acquire descriptor, generic lookback, litmus + negative control [4090]". Update PROGRESS.md. Show test output, sanitizer summaries, and both litmus counts.
```

**Your Verify.** Exact monoids PASS at all sizes through lookback; hang test completes; sanitizers clean; litmus (strong) = 0 torn. Note the negative-control result in `results/notes_litmus.md` — either outcome is data.

**Watch for.** A single value slot that gets overwritten on upgrade; `relaxed` in the production build; forgetting the `Desc` memset.

### Step 15 — Springing the noncommutativity trap (pod)

**Goal.** Proof, by test, that ordering is right everywhere.

**What we're doing, plainly.** With float addition, swapping the two arguments of `combine` anywhere changes nothing — so a swapped argument can hide for months. With matrices it changes everything. There are exactly four places order can silently flip: the block scan's combine, the lookback prepend, the per-thread serial fold in the ITEMS>1 path, and the final "block exclusive ∘ local" combine. We run the noncommutative matrix monoid through every code path, then deliberately *break* the prepend and confirm the test catches it — a test you've never seen fail is a test you can't trust.

**Claude Code prompt:**

```text
Context: SWEEP Step 15. Read CLAUDE.md invariant 1. Pod, ARCH=sm_89. Audit and test the ordering contract everywhere.

1) Audit: find every call to any monoid's combine in include/sweep/. There should be exactly four semantic sites: (a) block scan combine(s[t-d], s[t]); (b) lookback prepend combine(found, excl); (c) per-thread serial fold in the ITEMS>1 path combine(running, x[k]); (d) final combine(excl, local) — plus the two-pass equivalents. Add the comment marker `// ORDER-CONTRACT: (earlier, later)` at each. List them in your reply with file:line.

2) tests/test_ordering.cu: MatModP through EVERY code path — generic block scan alone, two-pass generic, lookback ITEMS=1, lookback ITEMS=4 (generic ITEMS path must exist for non-float T; if it doesn't yet, implement it now: serial fold in registers over ITEMS elements of type T loaded element-wise) — at every size in test_sizes(), bit-exact vs the oracle. Also include a random-sized-sequence torture: 200 random n in [1, 2^20].

3) Mutation check (do this, report it, then UNDO it): temporarily edit the lookback prepend to combine(excl, found); rebuild; run test_ordering; it must FAIL for MatModP at multi-block sizes (single-block sizes may pass — that is expected and is exactly why test_sizes includes large n). Then `git checkout` the file to restore it, rebuild, rerun, confirm PASS. Do not commit the mutated version. Report which sizes failed under mutation.

Commit: "step 15: ordering audit + noncommutative test-of-record [4090]". Update PROGRESS.md.
```

**Your Verify.** Green normally; red under the mutation at multi-block sizes; `git diff` shows the mutation is gone.

**Watch for.** The mutation "accidentally" passing — that means the test isn't covering that path; fix the test, not the claim.

### Step 16 — Segmented scan: many scans in one launch (pod)

**Goal.** Thousands of independent sequences scanned in one array with one kernel.

**What we're doing, plainly.** Week 3's SSM app needs one scan *per sequence* over a batch. Launching a kernel per sequence would waste the GPU. The classical trick makes segmentation itself a monoid: pair every value with a flag ("this position starts a new segment"), and define the combine so that a later flagged element simply ignores everything before it. The engine doesn't change at all — that is the payoff of Week 2.

**Claude Code prompt:**

```text
Context: SWEEP Step 16. Pod, ARCH=sm_89. Add a segmented-scan adapter as a monoid wrapper and prove it associative.

include/sweep/monoids.cuh: template <class M> struct Segmented { struct T { typename M::T v; int flag; }; identity = {M::identity(), 0}; combine(earlier, later) = { later.flag ? later.v : M::combine(earlier.v, later.v), earlier.flag | later.flag }; }.

docs/monoid_zoo.md: write the associativity proof of Segmented<M>: for x, y, z with flags fx, fy, fz, compute (x∘y)∘z and x∘(y∘z) in all cases (fz=1: both give (z.v, 1); fz=0, fy=1: both give (combine(y.v, z.v), 1); fz=0, fy=0: both reduce to M's associativity for the value and OR-associativity for the flag). Six lines; also record identity and note that commutativity is not required or claimed.

tests/test_segmented.cu: Segmented<AddU64> and Segmented<MatModP> through lookback and two-pass at all sizes, bit-exact vs a CPU reference that restarts at flags. Flag layouts: (a) every position flagged (all segments length 1); (b) no flags; (c) random flags with probability 1/64; (d) ADVERSARIAL: flags placed exactly at multiples of the tile size (256 for ITEMS=1, 1024 for ITEMS=4) and at tile boundaries ±1 — segment-reset logic meets block-handoff logic exactly there; oversample these.

Commit: "step 16: segmented scan adapter + proof + boundary tests [4090]". Update PROGRESS.md.
```

**Your Verify.** All four layouts PASS bit-exact for both monoids; the proof in `docs/monoid_zoo.md` reads correctly to you.

**Watch for.** The flag OR missing (segments would silently merge across blocks).

---

## Part VI — Week 3: three applications (Days 13–18)

Each app is a translation exercise: find the small object that represents one loop step, check that combining two steps yields the same kind of object (closure), hand it to the engine. Each ends with a roofline benchmark and, where one exists, a production-library comparison. All three ship; if a day overruns, the CSV app *moves into the writing window* — nothing is cut.

### Step 17 — App 1: the SSM recurrence (the Mamba connection) (pod)

**Goal.** `h[t] = a[t]·h[t-1] + b[t]` — the core recurrence of Mamba-class models — parallel, batched, measured.

**What we're doing, plainly.** One step of that recurrence is the function f(x) = a·x + b, fully described by the pair (a, b). Doing f then g gives g(f(x)) = (ga·fa)·x + (ga·fb + gb) — still a pair. So pairs compose into pairs (closure), composition is associative, and the identity is (1, 0). The scan therefore computes, at every position, the *single function equal to all steps so far composed*; applying it to the initial state gives every h[t] at once. This is exactly the operator S5- and Mamba-class models feed to an associative scan; Mamba-2 later moved to matrix-multiply form to use tensor cores — a design fork worth a paragraph in the paper.

**Claude Code prompt:**

```text
Context: SWEEP Step 17, first application. Read CLAUDE.md. Pod, ARCH=sm_89. Implement the batched affine (SSM) recurrence on the engine and a rigorous validation harness that Week 4 will reuse.

include/sweep/monoids.cuh: struct AffineF32 { struct T { float a, b; }; identity {1.f, 0.f}; combine(f, g) with f earlier, g later = { g.a * f.a, fmaf(g.a, f.b, g.b) } } — this is g∘f: apply f first. Note the FMA gives one rounding for the b term.

apps/ssm.cu:
- Inputs: B sequences × L steps (defaults B = 1024, L = 16384 so B*L = 2^24), per-step (a_t, b_t) as float pairs generated on the host with a seed; regime for a_t: uniform in (0.9, 1.0) by default (command-line selectable: also (0.99, 1.01) and a signed regime); b_t uniform in (-1, 1); initial state h0 per sequence.
- Flatten to one array of Segmented<AffineF32>::T with flag=1 at each sequence start; run sweep::scan; then an elementwise kernel applying each prefix pair to its sequence's h0: h_t = A_t * h0 + B_t.
- Validation: sequential fp64 CPU loop per sequence; metrics: max and median relative error over all positions, and a ULP histogram with buckets {0, 1, 2-3, 4-7, 8-15, 16-63, 64-255, 256+}. Also test a_t = 0 at random positions (state resets are legal input) and L = 1 sequences.
- Throughput: bytes = read 8 bytes per pair + write 8 per prefix (+ apply pass 4-byte writes); report GB/s and % of measured bandwidth; CSV results/ssm_<gpu_key>.csv with provenance.
- Print how error grows with L by running L in {2^10, 2^12, 2^14} at fixed B*L — do NOT attempt to reduce the error; record it (Week 4's opening plot).

Exit criteria: bit-exact agreement of the flag layout with Step 16's tests; relative error recorded (expect growth with L); throughput >= 70% of the roofline for the 8-in/8-out traffic at 2^24. Commit: "step 17: SSM affine recurrence app + validation [4090]". Update PROGRESS.md; show the error table and GB/s.
```

**Your Verify.** Error table printed with L-dependence; throughput ≥70% of roofline; a=0 resets pass.

**Watch for.** Scanning *states* instead of *functions* (a wrong but plausible design); combine written as f∘g.

### Step 18 — App 2: tridiagonal solver via 2×2 matrices (pod)

**Goal.** Solve a·x[i-1] + b·x[i] + c·x[i+1] = d (splines, 1-D diffusion, ADI stencils) with three scans.

**What we're doing, plainly.** The textbook sequential method (Thomas) eliminates forward: c'ᵢ = cᵢ / (bᵢ − aᵢ·c'ᵢ₋₁). That is not affine in c'ᵢ₋₁ — it's a ratio of two affine things, a "Möbius" map t ↦ (α·t+β)/(γ·t+δ). Represent each map by the 2×2 matrix [[α,β],[γ,δ]]; composing maps is exactly multiplying matrices (two lines to check). So one matrix scan produces every c'. Given c', the d' recurrence is plain affine (Step 17's monoid), and back-substitution is affine in reverse. Products of many float matrices overflow, so after each combine we divide all four entries by the largest magnitude — legal because a Möbius map doesn't change when its matrix is scaled. We only promise correctness for *diagonally dominant* systems (|b| > |a|+|c|), the standard condition under which elimination without row-swapping is stable.

**Claude Code prompt:**

```text
Context: SWEEP Step 18. Pod, ARCH=sm_89. Tridiagonal solve via three scans, validated against fp64 Thomas and benchmarked against cuSPARSE.

include/sweep/monoids.cuh: struct MobiusF32 { struct T { float m[4]; } // [[m0,m1],[m2,m3]] representing t -> (m0*t + m1)/(m2*t + m3); identity {1,0,0,1}; combine(a earlier, b later) = normalize(b*a) with the same row-major product as MatModP; normalize divides all four entries by max(|m0|,|m1|,|m2|,|m3|) (skip if that max is 0 — degenerate; assert in debug). __device__ float apply(T, float t). }

apps/tridiag.cu — solve for x given a[i], b[i], c[i], d[i], i = 0..n-1 (a[0] = 0, c[n-1] = 0):
- Scan 1 (Möbius): element_i = {0, c_i, -a_i, b_i} (forced a_0 = 0 makes the first map constant so the initial t is irrelevant: use 0). c'_i = apply(prefix_i, 0).
- Elementwise: denom_i = b_i - a_i * c'_{i-1} (c'_{-1} = 0).
- Scan 2 (AffineF32): element_i = { -a_i / denom_i, d_i / denom_i } -> prefix applied to initial 0 gives d'_i.
- Scan 3 (AffineF32 on the REVERSED index j = n-1-i): element = { -c'_i, d'_i }; initial state 0; prefix applied gives x_i (x_{n-1} = d'_{n-1} falls out since c'_{n-1} = 0). Implement the reversal as an index mapping inside a small gather/scatter kernel pair; fuzz the mapping separately with a permutation test.
- Generator: diagonally dominant systems with dominance parameter delta: |b_i| = (|a_i| + |c_i|) * (1 + delta), random signs, delta in {1.0, 0.5, 0.1, 0.02}; n in {2^16, 2^20, 2^22}.
- Reference: CPU Thomas algorithm in fp64. Report max relative error per (n, delta) as a table into results/tridiag_<gpu_key>.csv with provenance.
- Baseline: cuSPARSE gtsv2 (cusparseSgtsv2_bufferSizeExt then cusparseSgtsv2 — check the exact symbol names in the cuSPARSE headers of this toolkit; if they differ, use the nopivot variant as second comparison and tell me). Time both with time_kernel_ms; report the ratio.
- README section: this solver does NO pivoting; diagonally dominant systems only; ill-conditioned systems are out of scope by design.

Exit criteria: relative error <= 1e-5 for delta >= 0.5 at n = 2^22 vs fp64 Thomas; error-vs-delta degradation recorded (a finding, not a failure); performance within ~2x of gtsv2 (report honestly either way). Commit: "step 18: tridiagonal via Möbius+affine scans, Thomas validation, gtsv2 baseline [4090]". Update PROGRESS.md.
```

**Your Verify.** Error table shows ≤1e-5 at δ ≥ 0.5 and visible degradation at δ = 0.02; a gtsv2 ratio is printed.

**Watch for.** Normalizing by a *signed* value (flips the map); off-by-one in the reversed scan; `a*b` orientation in Möbius combine.

### Step 19 — App 3: CSV structural indexing at GB/s (pod)

**Goal.** Every *real* field and row boundary in RFC-4180 CSV — where commas and newlines inside quotes don't count — at a healthy fraction of memory bandwidth.

**What we're doing, plainly.** "Am I inside quotes?" is a tiny state machine with three modes: FIELD (outside quotes), QUOTED (inside), QSEEN (just saw a quote while inside — it's either the closing quote or the first half of an escaped `""`). One byte's effect is a *function* from mode to mode — a 3-entry table — and composing two bytes' functions is three table lookups. Tables compose associatively (very non-commutatively), and three modes × 2 bits fit in one byte. Scan the tables, apply each prefix to the start mode, and every byte knows its mode — the sequential dependency is gone. **The audit fix:** a delimiter is real when its mode-before is *not QUOTED* — the v3 rule "mode-before == FIELD" missed every delimiter that follows a closing quote (mode-before = QSEEN). This is established art (Mytkowicz et al., ASPLOS 2014; ParPaRaw; cuDF); it is a *demo of the engine*, labeled as such.

**Claude Code prompt:**

```text
Context: SWEEP Step 19. Pod, ARCH=sm_89. Build a GPU CSV structural indexer on the scan engine, validated against a sequential CPU parser. IMPORTANT CORRECTNESS RULE: a delimiter byte is a real structural delimiter iff the parser mode BEFORE that byte is not QUOTED (FIELD or QSEEN both count). Do not use "mode == FIELD".

include/sweep/monoids.cuh: struct FsmCsv { using T = unsigned; modes FIELD=0, QUOTED=1, QSEEN=2 packed 2 bits each: get(f, s) = (f >> (2*s)) & 3; identity = 0 | (1<<2) | (2<<4) (each mode maps to itself); combine(earlier, later): r_s = get(later, get(earlier, s)) for s in 0..2 (apply earlier first). }

apps/csv_index.cu pipeline:
1. Byte classes via a 256-entry __constant__ LUT: QUOTE ('"'), DELIM (',' and '\n'), OTHER (everything else, including '\r' — CRLF handling: rows end at '\n'; '\r' is OTHER; document this).
2. Per byte, its transition table by class: FIELD: QUOTE->QUOTED, DELIM->FIELD, OTHER->FIELD. QUOTED: QUOTE->QSEEN, DELIM->QUOTED, OTHER->QUOTED. QSEEN: QUOTE->QUOTED (escaped ""), DELIM->FIELD, OTHER->FIELD. Encode as a T per byte.
3. Inclusive scan of the T array with sweep::scan<FsmCsv>.
4. mode_before[i] = get(prefix[i-1], FIELD) with prefix[-1] = identity.
5. mark[i] = 1 iff class[i] == DELIM and mode_before[i] != QUOTED.
6. Stream compaction: exclusive scan of marks (use the u32 lookback path) gives each marked byte its output slot; scatter the byte offsets into a dense array; also emit a separate row-end array for '\n' marks.
Fuse steps 1-2 into one kernel and 4-6 into one kernel where convenient; keep a clear unfused version too for validation.

Validation: a sequential CPU reference parser implementing the same three-mode machine (plain loop), returning the same two offset arrays. Adversarial generator: fields with escaped quotes (""), quoted fields containing commas and newlines, quotes placed exactly at multiples of the tile size and ±1, a single quoted field spanning >10 tiles, empty fields, empty lines. Also validate on one large file: synthesize a 2–4 GB CSV with the generator (write to /tmp, never commit) — zero mismatches in both arrays. Throughput: total pipeline GB/s over input bytes, plus per-stage breakdown; CSV results/csv_<gpu_key>.csv with provenance. Tens of GB/s is the right ballpark on a 4090 given ~5 bytes moved per input byte; report what you get.

Exit criteria: zero mismatches on adversarial and large inputs; throughput reported with stage breakdown. Commit: "step 19: CSV structural indexer (FSM scan + compaction) [4090]". Update PROGRESS.md.
```

**Your Verify.** Zero mismatches; the test set explicitly includes `"abc",` style closing-quote-then-comma cases and they pass; GB/s printed.

**Watch for.** The wrong delimiter rule creeping back in; `get(earlier, get(later, s))` (composition order flipped); CRLF undocumented.

---
## Part VII — Week 4: the studies (Days 19–24)

The engine and apps make the repo good; the studies make it distinct. Two questions plus one map, all verified open at this exact spot. **(Q1)** How does floating-point error in scans over nontrivial operators grow with sequence length and input regime, and what do cheap fixes buy? **(Q2)** How large is decoupled lookback's run-to-run nondeterminism, what does it correlate with, and what does determinism cost — per operator? **(Map)** At what state size and operator cost does single-pass lookback stop beating simpler schedules?

Context you will cite: CUB documents scan nondeterminism (and once shipped, then retracted, a determinism guarantee — CUB 1.16.0 changelog, issue #432). In the CUDA 13.4-era CCCL docs, DeviceScan accepts a determinism requirement — but `run_to_run` for floating point is supported **only for `cuda::std::plus`**; every other operator/type combination is *rejected at compile time*. The vendor's deterministic scan covers exactly the trivial operator and declares everything this project cares about unsupported. Adjacent work: FP-non-associativity variability in PyTorch ops (Shanmugavelu et al., arXiv 2408.05148); deterministic-attention cost (DASH, ~38% for FlashAttention-3 backward); ScanWeaver's numerics is a single bounded-vs-exponential paragraph with instability listed as an open limitation. Condition-resolved error curves + scheduling-variance measurement + a deterministic mode for *arbitrary* operators — the compile-time-rejected case — is the seam.

### Step 20 — Two better number representations (pod)

**Goal.** Three interchangeable arithmetic backends for the affine monoid — plain fp32, compensated fp32, signed-log — so Steps 21–22 can race them.

**What we're doing, plainly.** When floats are added, the rounding error thrown away is itself an exactly representable float, and two classic gadgets recover it: **TwoSum** (six additions; returns the rounded sum *and* the exact leftover) and **TwoProdFMA** (a fused multiply-add computes a·b − p with one rounding, which *is* the exact product error). Carrying every number as (main, leftover) and propagating leftovers is "double-float" arithmetic: about twice fp32's precision for a few extra operations. The other backend stores sign and log|x|: multiplication becomes addition of logs (immune to overflow — attractive because affine scans multiply long chains of a's), but addition needs log-sum-exp and *subtracting* nearly equal magnitudes is catastrophic. Great products, fragile sums — a finding to demonstrate, not hide. One trap: a compiler under "fast-math" will happily simplify `(a − (s − t)) + (b − t)` to zero. Never compile these files with `-use_fast_math`.

**Claude Code prompt:**

```text
Context: SWEEP Step 20. Read CLAUDE.md invariant 3 (numerics are load-bearing). Pod, ARCH=sm_89. Implement compensated (double-float) and signed-log arithmetic and wrap them as affine monoids. These formulas are exact algorithms; do not simplify or reorder them.

include/sweep/numerics.cuh (all __host__ __device__ inline; no fast-math anywhere):
- float2 two_sum(float a, float b): s = a + b; t = s - a; e = (a - (s - t)) + (b - t); return {s, e}. Property: s + e == a + b exactly (Knuth). The apparently redundant operations ARE the algorithm.
- float2 two_prod(float a, float b): p = a * b; e = fmaf(a, b, -p); return {p, e}. Property: p + e == a * b exactly.
- Double-float values as float2 {hi, lo}. df_add_fast(x, y): s = two_sum(x.hi, y.hi); s.lo += x.lo + y.lo; return two_sum(s.hi, s.lo) (renormalize). df_add_accurate(x, y): s = two_sum(x.hi, y.hi); t = two_sum(x.lo, y.lo); s.lo += t.hi; s = quick renormalize (two_sum(s.hi, s.lo)); s.lo += t.lo; return two_sum(s.hi, s.lo). df_mul(x, y): p = two_prod(x.hi, y.hi); p.lo = fmaf(x.hi, y.lo, fmaf(x.lo, y.hi, p.lo)); return two_sum(p.hi, p.lo). Document that fast is Dekker-style "sloppy" add and accurate is the standard error-free-transformation version; the study reports which one each result used.
- struct SLog { float l; int s; } meaning value = s * exp(l), s in {-1, 0, +1}. slog_mul: if either s == 0 return zero; {a.l + b.l, a.s * b.s}. slog_add: zero handling; hi = the larger-l operand, lo the other; d = lo.l - hi.l (<= 0); same signs: {hi.l + log1pf(expf(d)), hi.s}; opposite signs: if d == 0 return zero (exact cancellation); else {hi.l + log1pf(-expf(d)), hi.s} — comment: near-cancellation (d -> 0 from below) is where precision dies; this is a measured finding.

include/sweep/monoids.cuh: AffineDF (T = {float2 a, b}; combine(f, g) = { df_mul(g.a, f.a), df_add_<variant>(df_mul(g.a, f.b), g.b) }; template parameter selects fast/accurate add) and AffineSLog (T = {SLog a, b}; combine analogous with slog ops). Both plug into Segmented<> and sweep::scan unchanged.

tests/test_numerics.cu: (a) exactness: for 10^6 random pairs check in double that two_sum's s + e == a + b exactly and two_prod's p + e == a * b exactly (compute the double-precision truth; the pair must reproduce it bit-for-bit when converted); (b) a canary: two_sum(1e8f, 1.0f).e must be nonzero — if it is zero the compiler folded the algorithm (fast-math or reassociation) and the test FAILS loudly; (c) df chains: 10^4-length random affine chains via AffineDF vs fp64 reference — relative error should track ~1e-13..1e-14 on tame inputs; (d) slog vs fp64 including sign flips and near-cancellation cases (record the error blow-up rather than asserting on it). Build with the project Makefile (no fast-math). Also run `cuobjdump -sass bin/test_numerics | grep -c -E "FADD|FFMA"` and paste the count into results/notes_numerics.md as a rough sanity that the arithmetic was not folded away.

Commit: "step 20: compensated + signed-log numerics, AffineDF/AffineSLog [4090]". Update PROGRESS.md.
```

**Your Verify.** Exactness tests PASS; the canary `e ≠ 0` passes; df chain error ~1e-13.

**Watch for.** Any "simplification" of `two_sum`; a `-use_fast_math` sneaking into the build line.

### Step 21 — The error study (Q1) (pod + Mac for plots)

**Goal.** Error-vs-length curves per input regime per representation — the plot that carries the paper.

**What we're doing, plainly.** How wrong the scan gets depends on the a's: their running products amplify early rounding errors into later positions (that amplification is what "conditioning" means here). So we sweep four input *regimes* that dial conditioning from benign to hostile, cross them with the four arithmetic backends and a range of lengths, repeat with ≥20 random seeds, and compare against a 64-bit sequential reference. We also compute per-sequence "condition proxies" so error can be plotted against *measured* conditioning, not just regime labels.

**Claude Code prompt:**

```text
Context: SWEEP Step 21 — the error study. Pod for the sweep (ARCH=sm_89), Mac for plots. Read CLAUDE.md benchmark rules (>= 20 seeds).

study/error_study.cu:
- Regimes for a_t: R1 decay: uniform (0.9, 1.0); R2 near-neutral: uniform (0.99, 1.01) (the Mamba-like regime); R3 log-normal: exp(N(0, 0.05)) clipped so the running product stays within fp32 range — record the clip; R4 signed: uniform (-1.0, 1.0). b_t uniform (-1, 1).
- Representations: fp32 (AffineF32), df32-fast, df32-accurate (AffineDF variants), slog (AffineSLog), fp64-on-GPU (an AffineF64 monoid, the ceiling).
- Lengths L in {2^10, 2^12, 2^14, 2^16, 2^18, 2^20, 2^22, 2^24}; batch so that each run has at least 2^20 elements; seeds 0..19 (>= 20).
- Reference: sequential CPU double loop per sequence (this is the oracle; document that Step 21b checks the oracle itself).
- Metrics per (regime, representation, L, seed): max relative error, median relative error, ULP histogram (same buckets as Step 17), and two condition proxies: sum of |log a_t| and max over t of |prod_{s<=t} a_s| (computed in double on the CPU).
- Output CSV results/error_study_<gpu_key>.csv with one row per (regime, representation, L, seed) plus provenance fields. Also print throughput per representation at L = 2^20 (the mitigation cost: df and slog vs fp32 — a headline number).

study/plots.py (runs on the Mac; matplotlib + pandas; reads all results/error_study_*.csv): Figure F2 — 4 panels (one per regime), x = L (log), y = max relative error (log), one line per representation, median over seeds with an IQR band; Figure F3 — scatter of max relative error vs the sum-|log a| proxy, colored by representation. Save PNGs to results/figures/. Print a short text summary of what the plots show.

tests/ref_longdouble.cpp (Mac, clang++): recompute the reference for a few R2 and R4 sequences in long double and report its disagreement with the double reference — a reference can be wrong too; record the number.

Exit criteria: CSV complete (no missing cells), plots render, throughput costs recorded. Whatever the curves show, report what you see; do not tune representations to make a nicer plot. Commit: "step 21: error study sweep + plots [4090]". Update PROGRESS.md.
```

**Your Verify.** `results/figures/F2_*.png` and `F3_*.png` exist and are legible; the CSV has 4 × 5 × 8 × 20 = 3,200 rows for the 4090; throughput costs for df/slog printed.

**Watch for.** Single-seed runs; regimes that overflow to inf (measuring inf-handling instead of error).

### Step 22 — The determinism study (Q2), the deterministic mode, the phase diagram (pod)

**Goal.** Quantify lookback's run-to-run variation; explain it mechanically; ship a deterministic mode for arbitrary operators and price it *per operator*; map where lookback stops winning.

**What we're doing, plainly.** Mechanism first: a block's lookback stops at the first PFX it meets; *where* that happens depends on scheduling timing; each stopping pattern is a different way of grouping the same additions; rounding makes different groupings different bits. We instrument exactly that — record how far each block walked (its "signature") every run — and test the prediction "same signatures ⇒ same bits." Then two levers that buy determinism: more serial work per thread (fewer cross-thread combines to vary), and "chained mode," where a block waits only for its immediate predecessor's *final* prefix — one fixed left-to-right grouping, bit-stable at a fixed launch configuration, at the cost of serializing propagation. NVIDIA's own deterministic scan supports float-plus only; ours supports any operator — so we measure what that costs for each. Same harness, one more experiment: synthetic monoids with tunable size and cost, to map where single-pass lookback stops beating the simpler schedules.

**Claude Code prompt:**

```text
Context: SWEEP Step 22 — determinism study + deterministic mode + phase diagram. Read CLAUDE.md invariant 6 (never "fix" plain lookback's nondeterminism) and the benchmark rules. Pod, ARCH=sm_89.

A) Chained mode (scan.cuh): template flag or separate kernel scan_chained<M>: identical to lookback except the walk only ever inspects desc[bid-1] and spins until its status == PFX (never consumes AGG). Cross-block grouping is then strictly left-to-right — a fixed parenthesization at a fixed launch configuration. Same memsets, tickets, prepend.

B) Depth-signature instrumentation: an optional int* depth (preallocated, grid-sized) that thread 0 writes with the number of descriptors it consumed during its walk. Zero overhead when nullptr. No printf in kernels.

C) Pinned CUB with the determinism API: git clone https://github.com/NVIDIA/cccl into /opt/cccl (outside the repo), check out the newest release tag whose documentation includes DeviceScan's determinism environment (search the docs/headers for the determinism requirement on DeviceScan; the API shape is cuda::execution::require(cuda::execution::determinism::run_to_run) passed as an environment argument — verify the exact spelling in that tag's headers and use what the headers say). Record the tag in results/notes_cccl.md. Build with make CCCL_DIR=/opt/cccl so the pinned headers override the toolkit's. If the toolkit's CUDA 12.8 nvcc cannot compile that tag, step back one release and record it. Never change this tag again for the rest of the project (mid-study version drift invalidates comparisons).

D) study/determinism.cu (v2): variants = lookback ITEMS in {1, 4, 16}, chained, two-pass, CUB InclusiveSum default, CUB InclusiveSum with run_to_run (float plus only — the one case the vendor supports; note that requesting run_to_run with any custom operator is rejected at compile time by CUB and quote the compile error into notes). Operator classes = float add (AddF32), AffineF32, a float 2x2 matrix monoid (MatF32, unnormalized, on inputs scaled to avoid overflow), AffineSLog. Regimes = Step 21's R1 and R4 for the affine/matrix operators; uniform (-1,1) for float add. n in {2^20, 2^24}. For each cell: K = 10,000 runs on identical input; per run: hash (64-bit FNV-1a) of the output, per-position ULP spread tracked at 16 fixed sample positions, depth signature (hash of the depth array); throughput. Metrics per cell: distinct output hashes, distinct signatures, whether identical signature always implied identical output (the mechanism prediction), max ULP spread, GB/s. Output results/determinism_<gpu_key>.csv with provenance.
Controls: AddU64 and MatModP through every variant must show exactly 1 distinct hash (a variance here is a race, not rounding — stop and debug via Step 14).

E) study/phase_diagram.cu: synthetic monoid template Payload<BYTES, KREP>: T = unsigned int data[BYTES/4]; combine = elementwise 32-bit wraparound add (associative, exact) evaluated KREP times with the result of all but the last evaluation discarded through asm volatile("" :: "r"(x)) barriers so the compiler cannot remove the repeated work (results identical for any KREP -> bit-exact testable; cost scales with KREP). Grid: BYTES in {8, 16, 32, 64, 128} x KREP in {1, 4, 16, 64} x schedule in {lookback ITEMS=1, chained, two-pass} x n in {2^20, 2^22, 2^24}. Verify bit-exact vs oracle for one cell per BYTES. Output results/phase_<gpu_key>.csv (ms, gbps, provenance). VRAM: 128 B x 2^24 in+out = 4.3 GB — fine on 24 GB; do not exceed 2^24 here.

F) study/plots.py additions (Mac): F4 distinct-hash and max-ULP-spread per variant (bars, per operator); F5 the signature=>bits check (fraction of runs where identical signature gave identical bits — must be 1.0); F6 throughput vs determinism per operator (x = variant, y = GB/s, annotated with distinct-hash count); F7 phase-diagram heatmap (which schedule wins each (BYTES, KREP) cell, per n).

Exit criteria: plain lookback float > 1 hash; chained and two-pass exactly 1 on this device; the CUB run_to_run baseline runs and is deterministic; controls all 1; deterministic-mode slowdown reported as a % per operator; phase heatmap rendered. Commit: "step 22: determinism study, chained mode, CUB run_to_run baseline, phase diagram [4090]". Update PROGRESS.md.
```

**Your Verify.** `notes_cccl.md` names the pinned tag; the CSV has the control rows at exactly 1; F4–F7 exist; the per-operator determinism-cost numbers are in the summary.

**Watch for.** Anyone changing the CCCL tag later; instrumentation that perturbs timing (printf); mixing hashes across devices.

### Step 23 — Legacy replication on Kaggle + report skeleton (Kaggle ×2, then Mac)

**Goal.** The full battery on T4 and P100; the paper's skeleton standing.

**What we're doing, plainly.** Two free Kaggle sessions rerun everything on two older architectures. Shapes should replicate and constants should move; a conclusion that *flips* between architectures is a highlighted finding, not noise. Then the report skeleton, so that Week 5's writing starts from headings and figure slots rather than a blank page.

**Claude Code prompt (run on Kaggle in a cell-driven session, or on the pod to prepare the script first):**

```text
Context: SWEEP Step 23. Two tasks. Read CLAUDE.md.

1) scripts/battery.sh <ARCH> <gpu_key>: runs, in order, gate.sh (must PASS or abort), test_generic_lookback, test_ordering, bench_scan, the Step-21 error study at a reduced grid (L in {2^12, 2^16, 2^20}, seeds 0..19 — still >= 20), the Step-22 determinism study at n = 2^20 with K = 3,000 runs, and the phase diagram at n = 2^22 only; every CSV is written with this device's gpu_key in its name and provenance rows; the script prints a completion summary and exits nonzero on any failure. Add `export PATH=/usr/local/cuda/bin:$PATH` at the top for Kaggle. On P100 (sm_60, pre-Volta) confirm nothing spins within a warp on other lanes (our spins are cross-block only) — grep for intra-warp spin patterns and report none.

2) docs/report.md skeleton with these sections and one-line intents, plus figure placeholders F1–F8: 1 Problem & idea (algebraic reformulation vs scan schedule; operator zoo; the three monoid-zoo proofs); 2 Engine (decoupled lookback; ticket liveness argument; release/acquire protocol; litmus evidence; benchmark vs CUB and vs measured rooflines, all devices); 3 Applications (SSM, tridiagonal, CSV, Viterbi; each: translation to a monoid, validation, throughput, honest baseline, explicit prior-art label); 4 Studies (Q1 error curves; Q2 determinism + deterministic-mode cost per operator; phase diagram; selector); 5 Related work (list in Appendix F of the plan); 6 Threats to validity (marketplace hosts and clock variance; absence of hardware counters — mitigation: event timing + measured rooflines + occupancy API; per-device determinism claims; reference-oracle validity); 7 Non-goals. Do not write prose yet beyond the intents.

Then run battery.sh on this Kaggle GPU (ARCH from nvidia-smi: T4 -> sm_75, P100 -> sm_60), commit the CSVs: "step 23: battery on <gpu_key>". Report which conclusions from the 4090 replicate and which constants moved.
```

**Your Verify.** Two new sets of CSVs (T4, P100) committed; `docs/report.md` has all sections and F1–F8 placeholders; the replication summary reads sensibly.

**Watch for.** Kaggle idling out mid-battery (run inside the 20-minute activity window — keep the tab active, or split the battery into two cells).

---
## Part VIII — Phase 2 (Nov 9 – Dec 7): fourth workload, adaptive scheduler, sweep days

Phase 2 runs *before* the December submission if the Nov 28 checkpoint says so (Part IX). It upgrades the paper from characterization to characterization + selection.

### Step 24 — App 4: max-plus Viterbi (a genuinely different algebra) (pod)

**Goal.** HMM best-path decoding via scans over the **max-plus (tropical) semiring** — algebraic diversity the first three apps lack. (Recursive IIR filtering was considered and rejected: it is the same affine/small-matrix monoid as Step 17, and Zhai & Paris 2026 just occupied it with a specialized system — now a citation.)

**What we're doing, plainly.** Viterbi's recurrence — best score of being in state j at time t = maxᵢ(score[i, t−1] + w_t[i→j]) — is matrix-vector "multiplication" where add is `max` and multiply is `+`. K×K matrices of "best score from i to j" compose associatively (max-plus is a semiring: + distributes over max), so per-step matrices scan exactly like our other transformation monoids. **Audit clarification:** MatModP acts on *column* vectors, so "earlier then later" is the product B·A; Viterbi matrices are from→to (row convention), so the same "earlier then later" is written earlier ⊗ later — both mean *apply earlier first*. Also: −∞ + (−∞) = −∞ in IEEE arithmetic (fine); NaN appears only if a +∞ sneaks in, e.g., log(∞) or an overflow — keep log-scores finite. The numeric personality is new: `max` is exact, `+` drifts — a third profile for Q1/Q2.

**Claude Code prompt:**

```text
Context: SWEEP Step 24, fourth application. Pod, ARCH=sm_89. Read CLAUDE.md invariant 1 — and read this orientation note: this monoid's matrices are from->to (row convention), so combine(earlier, later) computes r[i][j] = max_k (earlier[i][k] + later[k][j]) — "earlier then later" written as earlier ⊗ later. That is the SAME semantic contract as MatModP's B*A (apply earlier first), just in row convention. Document this in monoid_zoo.md with a two-line proof that max-plus matrix product is associative (+ distributes over max).

include/sweep/monoids.cuh: template <int K> struct MaxPlus { struct T { float m[K*K]; } // m[i*K+j] = best score i -> j; identity: 0 on the diagonal, -INFINITY elsewhere; combine(a, b): r[i*K+j] = max over k of (a[i*K+k] + b[k*K+j]) with fmaxf. }. Keep all scores finite except the identity's -INFINITY; assert no +INFINITY is ever generated (a debug check in the tests).

apps/viterbi.cu:
- Synthetic HMM with K in {4, 8, 16}: random log-transition matrix (rows log-normalized), random emission log-probs, observation sequence of length L up to 2^20.
- Per-step matrix w_t[i][j] = logA[i][j] + logB[j][obs_t]. Forward best scores at t = init_vec ⊗ (prefix_t) where prefix_t is the inclusive scan of the step matrices (row-vector times matrix in max-plus).
- Path recovery: CPU backward pass using stored per-step matrices and the forward scores (argmax with smallest-index tie-break). A reverse GPU scan is a stretch, not required.
- Reference: CPU Viterbi in double with the same smallest-index tie-break; compare the optimal path SCORE (exact match required in float given identical operation order is impossible, so compare within 1e-4 relative and report), and the path itself (must match wherever the reference's best score is separated from the runner-up by more than the observed float drift; report the fraction of ambiguous positions).
- Metrics: per-step drift vs double along the path; throughput vs roofline (K=16 is 1 KB per element — this is a heavy-payload workload; add it to the phase diagram as a real data point); register spills check via -Xptxas -v (report spill bytes per K).

Exit criteria: path scores agree within tolerance for all K; drift table recorded; spill report attached. Commit: "step 24: max-plus Viterbi app [4090]". Update PROGRESS.md.
```

**Your Verify.** Score agreement for K = 4, 8, 16; the orientation note appears in `monoid_zoo.md`; spill report present.

**Watch for.** `later ⊗ earlier` (reversed); +∞ generation; NaN-propagating comparisons.

### Step 25 — The fingerprint-driven adaptive scheduler (pod, then Mac)

**Goal.** Given (monoid, n, device), *predict* the best schedule — validated against exhaustive search.

**What we're doing, plainly.** Your GTL methodology transplanted: measure a device's fingerprints once (achieved bandwidth, how fast each combine runs, how many blocks fit per core), describe the workload by numbers (payload size, combine cost, n), and fit a small cost model per schedule; the selector picks the schedule with the smallest predicted time. Then test it honestly: on grid cells the model never saw, compare its pick against the true best found by trying everything.

**Claude Code prompt:**

```text
Context: SWEEP Step 25 — adaptive schedule selection. Pod for probes/validation (ARCH=sm_89), Mac for fitting/plots. Read CLAUDE.md.

study/probe_suite.cu: measures per device, once: (1) achieved bandwidth (reuse bw_probe); (2) combine throughput per monoid (AddF32, AffineF32, MobiusF32, MatModP, MaxPlus<8>, MaxPlus<16>, Payload<32,1>, Payload<128,16>): a kernel doing 10^6 dependent combines from registers per thread, report ns per combine; (3) occupancy per schedule kernel via cudaOccupancyMaxActiveBlocksPerMultiprocessor; (4) descriptor round-trip latency: time for a block to observe a status flip from another block (a two-block ping-pong, median). Output results/fingerprint_<gpu_key>.json.

study/selector.py (Mac): cost model per schedule s for workload (payload bytes p, combine ns c, n): predicted_time = max(bytes(s, p, n) / BW, depth(s) * c * n_combines(s, n), overhead(s, grid(n)) * latency) with 2–3 free constants per schedule fitted by least squares on a CALIBRATION grid (the phase-diagram CSV rows plus the app benchmarks) for each device. Selector = argmin over schedules. Then VALIDATION on a HELD-OUT grid (cells never used for fitting: different n values and payload/cost combinations), on every device with data: for each cell, compare the selector's choice vs the oracle-best (min measured time); metrics: fraction of cells within 10% of oracle, mean and worst regret. Print a table per device and save Figure F8 (selector regret distribution, one panel per device) to results/figures/.

Exit criteria: within 10% of oracle on >= 80% of held-out cells per device; failures analyzed in results/notes_selector.md — where the model breaks is a finding, not something to hide. Commit: "step 25: fingerprints + adaptive selector + validation". Update PROGRESS.md.
```

**Your Verify.** `fingerprint_RTX_4090.json` exists; F8 rendered; the regret table shows the ≥80% criterion met or an honest analysis of why not.

### Step 26 — Sweep days: A100, RTX 5090, 3090 control (Vast, three short rentals)

**Goal.** The full battery on the rented flagships; the six-device matrix complete.

**What we're doing, plainly.** Three short rentals: an A100 (same Ampere generation as the 3090 but HBM memory — the GDDR-vs-HBM contrast), an RTX 5090 (the newest architecture; the image is already CUDA-12.8-ready for it), and a two-hour 3090 as the within-Ampere control. Each is one sitting on one pinned machine, gated twice as hard — a bad host here would poison a headline device.

**Claude Code prompt (on each rented device):**

```text
Context: SWEEP Step 26 sweep day on <device>. Read CLAUDE.md. Before anything: bash scripts/gate.sh <ARCH> must PASS (A100 = sm_80, expected 1500–1800 GB/s; RTX 5090 = sm_120, expected 1450–1700; RTX 3090 = sm_86, expected 800–900). If FAIL, stop and tell me to destroy the host. Then run bash scripts/battery.sh <ARCH> <gpu_key> in full (not the reduced Kaggle grid: use the full Step-21/22 grids but K = 5,000 runs for determinism to fit the rental window), then study/probe_suite for the selector. Commit all CSVs and JSON: "step 26: battery on <gpu_key>". Report: gate number, any test failure, and the three headline numbers for this device (ITEMS=4 % of CUB, deterministic-mode cost for AffineF32, phase-diagram crossover).
```

**Your Verify.** `machines_log.csv` has PASS rows for all six devices; every figure regenerates with six lines/panels from the Mac.

### Step 27 — Fold into the paper and submit (Mac)

**Claude Code prompt:**

```text
Context: SWEEP Step 27 — integration. Mac. Regenerate every figure F1–F8 from results/ via study/plots.py with all devices present. Write scripts/reproduce_all.sh that, given a device's ARCH and gpu_key, reruns the battery and regenerates figures (this is the artifact's one-command reproduce). Update README.md: artifact instructions, device matrix, non-goals, and a "numbers to CSV to commit" traceability table listing, for every figure, the CSV files and the commit hashes that produced them. Draft docs/report.md prose for sections 2 and 4 from the notes files (notes_ulp, notes_litmus, notes_numerics, notes_cccl, notes_selector) — cite plan Appendix F. Do not invent numbers: every figure statement must point at a CSV cell. Commit: "step 27: figures, artifact reproduce script, report draft".
```

---

## Part IX — Paper track and re-anchored calendar

**Claims.** C1: open generalized-lookback engine with verified ordering protocol (artifact, not novelty claim). C2: condition-resolved error characterization across operator classes and representations, with measured mitigation costs. C3: *to our knowledge* the first scheduling-variance measurement of decoupled lookback (depth signatures ⇒ bits) plus a deterministic scan mode for arbitrary operators — precisely the case CUB's determinism API rejects at compile time — priced per operator. C4: the state-size × operator-cost × schedule phase diagram. C5: fingerprint-driven schedule selection within 10% of oracle on ≥80% of a held-out grid across ≥3 architectures.

**Figures.** F1 roofline-vs-CUB bars (all devices); F2 error vs L (4 regimes × 5 representations); F3 error vs condition proxy; F4 distinct-hash + ULP spread per schedule; F5 signature⇒bits check; F6 determinism cost per operator; F7 phase-diagram heatmap; F8 selector regret.

**Venues.** Primary **ISPASS 2027** (measurement/characterization venue; 2026 ran abstract Dec 8 / full Dec 15, 9-page IEEE, arXiv-tolerant; expect the 2027 CFP ~October — verify the day it posts). Same-week alternate: ICS 2027 (abstract ~Dec 4 / paper ~Dec 11) — decide by Nov 25. IPDPS 2027 (Oct 2/9) falls inside the build window — no. Fallbacks after a ~Feb reject: EuroMLSys 2027 (~Feb, workshop length), IPDPS 2027 workshops (~Jan–Feb), Correctness@SC'27 (~Aug). **arXiv preprint + public artifact ship in early November regardless** — the job-search deliverable, decoupled from any committee.

**Reviewer-proofing (built in).** ≥20 seeds + medians + IQR on every plot; six devices, four architectures, three memory technologies; CUB baselines including `run_to_run` float-plus; one-command artifact; threats-to-validity naming marketplace hosts, absent hardware counters, per-device determinism claims, and reference-oracle validity; faculty co-author (SEECS supervisor) looped in by mid-September.

**Calendar (re-anchored Sep 3; Mon–Sat working days).** The two-week slip from the original plan consumed the buffer; Phase 2's inclusion in the December submission is now a dated decision.

| Dates | Work | Gate |
|---|---|---|
| Sep 3–5 | Day 0 (Part II) on the Mac; Step 6 on the running 4090 | Gate PASS recorded |
| Sep 7 – Oct 3 | Steps 6–23 (24 working days; Appendix B). Evenings Wks 1–2: related-work + background. Wk 3: methodology | Step-23 exits green |
| Oct 4–10 | Gate week: does determinism-cost-per-operator and the phase diagram show strong structure? Strong → ISPASS clock. Flat → arXiv + Phase-2 strengthening; never venue-shop a thin result | Go/no-go |
| Oct 11–31 | Full 9-page draft; CSV app lands here if Week 3 overran | Draft done |
| Nov 1–7 | Internal review (supervisor, Maram, Mohsin); revise | Reviewed |
| Nov 8–10 | **arXiv + public repo live** | Preprint shipped |
| Nov 9 – Dec 7 | Phase 2: Step 24 (Nov 9–14), Step 25 (Nov 16–28), Step 26 sweep days (Nov 30 – Dec 2), Step 27 integration (Dec 3–7) | C5 validated |
| **Nov 28** | **Checkpoint:** is C5 validated? Yes → fold Phase 2 into the December submission. No → submit characterization-only (C1–C4) in December; Phase 2 becomes the camera-ready extension or the next venue's paper. Either way, no scope is dropped — only its submission sequencing moves | Decision logged |
| ~Dec 7 / 14 | ISPASS 2027 abstract + full (TBC vs actual CFP; ICS decision by Nov 25) | Submitted |
| ~Feb 2027 | Notification → camera-ready, or 2-week revision into the fallback ladder | Placed |

---

## Appendix A — When things go wrong

| Symptom | Likely cause | Fix / where |
|---|---|---|
| `Permission denied (publickey)` | Instance predates your key | Key icon on the card; or re-rent |
| `REMOTE HOST IDENTIFICATION HAS CHANGED` | New instance reused a memorized IP:port | `ssh-keygen -R '[<IP>]:<PORT>'`, reconnect |
| SSH dropped mid-run | Harmless if inside tmux | `tmux attach -t work` / `-t claude` |
| `tmux: no sessions` | You never started one; the job died with the pipe | Ritual A rule 3; rerun inside tmux |
| Claude Code "finished" but tests weren't shown | It summarized instead of running | Ask for the raw test output; never accept a summary |
| Claude Code changed a tolerance or swapped combine arguments | Fixing symptoms | Revert; paste: "fix without changing the test tolerance or the ordering contract" |
| `git push` → 403 | PAT expired/wrong scope | New fine-grained token, Contents R/W on `sweep` |
| Kernel hangs | `blockIdx.x` where the ticket belongs; descriptor memset forgotten | Step 11 |
| Wrong only when grid is large | Prepend flipped; tile-boundary handoff | Steps 15, 16 |
| Wrong *sometimes* (1 in 10³–10⁵) | Missing release/acquire; torn multi-word read | Step 14 + litmus |
| Integer monoid gives >1 hash | Real data race (never rounding) | Step 14; sanitizers |
| Illegal address / launch failure | OOB at padded tail; dynamic shared size missing at launch | Steps 11, 13 |
| Float tests flap | Bit-exact assert on a float monoid — wrong test design | Step 7 |
| Great on 4090, mediocre on P100 | `ARCH` mismatch; roofline denominator not re-measured | Steps 6, 23 |
| Compensated arithmetic "does nothing" | Fast-math deleted the error term | Step 20 canary; check SASS |
| CUB `run_to_run` won't compile | Toolkit CUB too old, or custom operator requested | Step 22: pinned CCCL_DIR; float plus only |
| `nvcc: command not found` on Kaggle | Off PATH | `export PATH=/usr/local/cuda/bin:$PATH` |
| bw_probe below band | Bad marketplace host | Ritual D: destroy, next offer |
| Instance won't boot / host offline | Marketplace tax | Destroy; verify meter; next offer |
| CSV indexer misses delimiters after quoted fields | v3's wrong rule resurrected | Step 19: mode-before ≠ QUOTED |
| NaN in Viterbi | A +∞ leaked into log-scores | Step 24 |

## Appendix B — Day map (build weeks, Mon–Sat)

| Date | Day | Steps | Milestone |
|---|---|---|---|
| Sep 3–5 | 0 | Setup 1–5, Step 6 | Keys, token, repo, Kaggle, template; 4090 bandwidth recorded |
| Sep 7–8 | 1–2 | 7–8 | Oracle + warp scan green |
| Sep 9 | 3 | 9 | Block scan green, racecheck clean |
| Sep 10 | 4 | 10 | Two-pass correct + benchmarked |
| Sep 11–12 | 5–6 | 11 | Lookback correct, no hangs, sanitizer-clean |
| Sep 14 | 7 | 12 | ≥90% of CUB; nondeterminism logged |
| Sep 15–16 | 8–9 | 13 | Generic engine + exact monoids, bit-exact |
| Sep 17–18 | 10–11 | 14 | Big-payload protocol + litmus |
| Sep 19 | 12 | 15–16 | Ordering trap survived; segmented scan green |
| Sep 21–22 | 13–14 | 17 | SSM app validated + benchmarked |
| Sep 23–24 | 15–16 | 18 | Tridiagonal vs Thomas + gtsv2 |
| Sep 25–26 | 17–18 | 19 | CSV indexer at GB/s |
| Sep 28 | 19 | 20 | df32 + signed-log verified |
| Sep 29–30 | 20–21 | 21 | Error-study plots |
| Oct 1–2 | 22–23 | 22 | Determinism + deterministic mode + phase diagram |
| Oct 3 | 24 | 23 | T4/P100 batteries + report skeleton |

## Appendix C — Non-goals (README verbatim)

No autograd/backward pass. No multi-GPU. No general regex engine (the CSV FSM is 3 states by design; Viterbi is fixed-K). No pivoting tridiagonal solves — diagonally dominant systems only, stated plainly. No tensor-core/matmul reformulation of the scan (Mamba-2's fork, acknowledged). Determinism claims are per-device at fixed launch configuration. No cross-host performance comparisons — every A-vs-B number comes from one session on one machine. Every number in the write-up links to its CSV, its provenance line, and its commit.

## Appendix D — Budget

| Item | Est. |
|---|---|
| Vast 4090 daily (~110 h × ~$0.31) | ~$34 |
| Vast A100 sweep (~4 h × ~$0.93) | ~$4 |
| Vast 5090 sweep (~4 h × ~$0.41) | ~$2 |
| Vast 3090 control (~2 h × ~$0.15) | ~$0.5 |
| Optional RunPod Secure finals (~6 h) | ~$4 |
| Bandwidth + disk contingency | ~$3 |
| Kaggle T4 + P100 | $0 |
| **Total** | **~$48** |

## Appendix E — Command cheat sheet (print this)

```bash
# --- connect & survive ---
ssh -p <PORT> root@<IP>              # from the card's Connect button
tmux new -s work                     # durable session (ALWAYS)
tmux new -s claude && claude         # Claude Code, also inside tmux
# Ctrl-B then D = detach       tmux attach -t work        tmux ls
# --- bootstrap (each fresh instance) ---
export GH_TOKEN=<token>
git clone https://$GH_TOKEN@github.com/<you>/sweep.git && cd sweep
git config user.name "Ahad" && git config user.email "<you>@users.noreply.github.com"
# --- gate ---
bash scripts/gate.sh sm_89           # PASS required; FAIL -> destroy host
# --- build & run ---
make bin/<target> ARCH=sm_89 && ./bin/<target>
compute-sanitizer --tool racecheck ./bin/<target>
compute-sanitizer --tool memcheck  ./bin/<target>
make bin/<target> ARCH=sm_89 CCCL_DIR=/opt/cccl   # Step 22 onward for CUB baselines
# --- leave (NON-NEGOTIABLE ORDER) ---
git add -A && git commit -m "step N: <what> [device]" && git push
exit; exit                            # tmux, then ssh
# console -> trash icon -> Destroy -> meter shows $0
```

## Appendix F — References (status-verified this cycle)

Kogge–Stone 1973; Blelloch 1990; Sengupta et al., Scan Primitives for GPU Computing; Merrill & Garland, NVR-2016-002; Smith, Decoupled Fallback, SPAA 2025; Sorensen–Evrard–Donaldson, CONCUR 2018; Alglave et al., ASPLOS 2015; CUB DeviceScan docs + 1.16.0/#432 history + CUDA 13.4-era determinism environment (scan `run_to_run` = float `plus` only; all other operators rejected at compile time); ScanWeaver arXiv 2606.00601 (compiler/MLIR, Blelloch-only schedule, one-paragraph numerics, instability listed open); COREY arXiv 2604.10597; Shanmugavelu et al. arXiv 2408.05148; DASH (deterministic FlashAttention); Mytkowicz et al. ASPLOS 2014; ParPaRaw PVLDB 2020; Zhang–Cohen–Owens 2010 (GPU tridiagonal); Zhai & Paris arXiv 2607.23763 + 2607.14054 (cascaded IIR via lookback + cost model); S5 (Smith et al.) / Mamba (Gu & Dao); Dekker / Ogita–Rump–Oishi (verify exact citations before camera-ready).
