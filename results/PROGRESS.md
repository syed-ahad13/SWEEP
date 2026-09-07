# PROGRESS

Session-state handoff between machines and Claude Code instances. Read this
first thing every session; update it before the session ends.

## Current step

Setup 3 complete — repository skeleton, Makefile, scripts, machines.json.
No CUDA source written yet.

## Last green gate

None yet. (Setup 3 has no Verify gate beyond "the Makefile parses and the tree
matches"; the numbered gates start at Step 6.)

## Device used

Mac, host only — no NVIDIA GPU, no nvcc. Nothing in this repo has been
compiled or run.

## Open issues

- `make test_cpu` cannot succeed until `tests/test_common.cpp` exists (Step 7).
- `bash scripts/provenance.sh` exits 1 on the Mac (no nvidia-smi). Expected.
- No host has passed the bandwidth acceptance gate yet; `results/machines.json`
  bands are unverified against a real card.

## Next action

Step 6 on the 4090.
