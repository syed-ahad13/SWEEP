#!/usr/bin/env bash
# First command on a freshly rented pod.
#
#   export GH_TOKEN=<fine-grained token>   # env var only; never written to a file
#   bash pod_init.sh && cd /workspace/SWEEP
#
# GH_TOKEN is read from the environment by a credential helper at each git
# operation. It never enters a remote URL, .git/config, or argv.
set -euo pipefail

REPO_OWNER="syed-ahad13"
REPO_NAME="SWEEP"
WORK_ROOT="${WORK_ROOT:-/workspace}"
GIT_USER_NAME="Ahad"
GIT_USER_EMAIL="abdulahad17100@gmail.com"

die() { echo "pod_init.sh: $*" >&2; exit 1; }
[ -n "${GH_TOKEN:-}" ] || die "GH_TOKEN is not set. export it first (it is never stored on disk)."
[ -d "$WORK_ROOT" ] || WORK_ROOT="$HOME"
REPO_DIR="$WORK_ROOT/$REPO_NAME"
[ -n "${TMUX:-}" ] || echo "pod_init.sh: WARNING - not inside tmux; an SSH drop kills whatever is running."

export GIT_TERMINAL_PROMPT=0
git config --global credential."https://github.com".helper \
  '!f() { test "$1" = get && printf "username=x-access-token\npassword=%s\n" "$GH_TOKEN"; }; f'

if [ -d "$REPO_DIR/.git" ]; then
  echo "pod_init.sh: $REPO_DIR already cloned; fetching."
  git -C "$REPO_DIR" fetch --quiet --prune
else
  git clone --quiet "https://github.com/${REPO_OWNER}/${REPO_NAME}.git" "$REPO_DIR"
fi

cd "$REPO_DIR"
git config user.name  "$GIT_USER_NAME"
git config user.email "$GIT_USER_EMAIL"

FAIL=0

echo
echo "=== toolkit ==="
if command -v nvcc >/dev/null 2>&1; then
  nvcc --version | sed -n '/release/p'
else
  echo "nvcc: MISSING - runtime-only image. Destroy and rent a *-devel image."
  FAIL=1
fi

echo
echo "=== card ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,compute_cap,driver_version,power.limit,power.default_limit \
    --format=csv
  PL="$(nvidia-smi --query-gpu=power.limit         --format=csv,noheader,nounits | head -1 | tr -d ' ')"
  PD="$(nvidia-smi --query-gpu=power.default_limit --format=csv,noheader,nounits | head -1 | tr -d ' ')"
  if awk -v a="$PL" -v b="$PD" 'BEGIN{exit !(b>0 && a >= 0.95*b)}'; then
    echo "power: ${PL}W of ${PD}W default - OK"
  else
    echo "power: ${PL}W of ${PD}W default - host has capped this card; every bandwidth number will be low."
    FAIL=1
  fi
else
  echo "nvidia-smi: MISSING - not a GPU host."
  FAIL=1
fi

echo
git status -sb
echo "Repo at $REPO_DIR on $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD)"
echo "Run:  cd $REPO_DIR"
echo "Remaining gate:"
echo "  make bin/bw_probe ARCH=<sm_XX> && ./bin/bw_probe  -> inside the band in results/machines.json"
echo "Any gate red: destroy this instance and take the next offer."

[ "$FAIL" -eq 0 ] || die "host gate FAILED - destroy this instance."