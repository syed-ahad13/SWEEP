#!/usr/bin/env bash
# Host gate. Run by the template's on-start script; safe to re-run by hand:
#     bash /workspace/SWEEP/scripts/pod_init.sh
# Writes its own verdict to /root/init.status, so a manual re-run refreshes
# the login banner instead of leaving a stale FAIL on screen.
set -uo pipefail

REPO_OWNER="syed-ahad13"
REPO_NAME="SWEEP"
GIT_USER_NAME="Ahad"
GIT_USER_EMAIL="abdulahad17100@gmail.com"
WORK_ROOT="${WORK_ROOT:-/workspace}"
MIN_CUDA_MAJ=12
MIN_CUDA_MIN=8

STATUS=/root/init.status
[ -d "$WORK_ROOT" ] || WORK_ROOT="$HOME"
REPO_DIR="$WORK_ROOT/$REPO_NAME"
export GIT_TERMINAL_PROMPT=0

FAIL=0; WARN=0
ok()   { echo "[ok]   $*"; }
warn() { WARN=$((WARN+1)); echo "[warn] $*"; }
bad()  { FAIL=$((FAIL+1)); echo "[FAIL] $*"; }
isnum(){ [ -n "${1:-}" ] && printf '%s' "$1" | grep -Eq '^[0-9]+([.][0-9]+)?$'; }
verdict() {
  if [ "$FAIL" -eq 0 ]; then
    echo "GATE PASS ($WARN warn) - $(date -u +%FT%TZ)" | tee "$STATUS"
    exit 0
  fi
  echo "GATE FAIL ($FAIL fail, $WARN warn) - destroy this instance (see /root/init.log)" | tee "$STATUS"
  exit 1
}

# --- repo -------------------------------------------------------------------
if [ -d "$REPO_DIR/.git" ]; then
  git -C "$REPO_DIR" fetch --quiet --prune 2>/dev/null || warn "git fetch failed (token expired?)"
elif [ -n "${GH_TOKEN:-}" ]; then
  git clone --quiet "https://github.com/$REPO_OWNER/$REPO_NAME.git" "$REPO_DIR" \
    && ok "repo cloned" || bad "clone failed - check the token's Contents permission"
else
  bad "no repo and no GH_TOKEN"
fi
if cd "$REPO_DIR" 2>/dev/null; then
  git config user.name "$GIT_USER_NAME"
  git config user.email "$GIT_USER_EMAIL"
fi

# --- gate 1: toolkit --------------------------------------------------------
echo
echo "=== toolkit ==="
NVCC="$(command -v nvcc || true)"
[ -n "$NVCC" ] || [ ! -x /usr/local/cuda/bin/nvcc ] || NVCC=/usr/local/cuda/bin/nvcc
if [ -z "$NVCC" ]; then
  bad "nvcc not found on PATH or at /usr/local/cuda/bin - runtime-only image?"
else
  REL=$("$NVCC" --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9]*\)\.\([0-9][0-9]*\).*/\1 \2/p')
  # shellcheck disable=SC2086
  set -- $REL
  MAJ="${1:-0}"; MIN="${2:-0}"
  if [ $((MAJ * 100 + MIN)) -ge $((MIN_CUDA_MAJ * 100 + MIN_CUDA_MIN)) ]; then
    ok "nvcc $MAJ.$MIN at $NVCC"
  else
    bad "nvcc $MAJ.$MIN is older than $MIN_CUDA_MAJ.$MIN_CUDA_MIN"
  fi
fi

# --- gate 2: the card -------------------------------------------------------
echo
echo "=== card ==="
if ! command -v nvidia-smi >/dev/null 2>&1; then
  bad "nvidia-smi missing - not a GPU host"
else
  nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total,power.limit,power.default_limit \
    --format=csv 2>/dev/null | sed 's/^/  /'
  q() { nvidia-smi --query-gpu="$1" --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '; }
  PL=$(q power.limit); PD=$(q power.default_limit); USED=$(q memory.used); CC=$(q compute_cap)

  if ! isnum "$PL" || ! isnum "$PD" || [ "${PD%%.*}" -eq 0 ] 2>/dev/null; then
    warn "power limit unreadable (PL='$PL' PD='$PD') - cannot verify the card is uncapped"
  elif awk -v a="$PL" -v b="$PD" 'BEGIN{exit !(a >= 0.95*b)}'; then
    ok "power ${PL}W of ${PD}W"
  else
    bad "power ${PL}W of ${PD}W - host has capped this card"
  fi

  if ! isnum "$USED"; then
    warn "memory.used unreadable"
  elif awk -v u="$USED" 'BEGIN{exit !(u < 512)}'; then
    ok "vram ${USED} MiB in use"
  else
    bad "vram ${USED} MiB already allocated - shared card or leaked context"
  fi

  [ "$CC" = "8.9" ] && ok "compute_cap 8.9 (Ada)" || warn "compute_cap $CC - not the RTX 4090 you gated for"
fi

# --- next -------------------------------------------------------------------
echo
if [ -d "$REPO_DIR/.git" ]; then
  git -C "$REPO_DIR" status -sb | head -3
  echo "repo: $REPO_DIR @ $(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null)"
fi
verdict
