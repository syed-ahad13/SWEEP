#!/usr/bin/env bash
# First command on a freshly rented pod.
#
#   export GH_TOKEN=<fine-grained token>   # env var only; never written to a file
#   bash pod_init.sh                       # or: curl it, or paste it
#
# Clones the repo, sets the commit identity, then prints the two things that
# decide whether this host is usable at all: the CUDA version and the card.
#
# GH_TOKEN lives in the environment and in the clone URL only. It must never be
# committed, echoed into a file, or put into the Vast template.
set -euo pipefail

REPO_OWNER="syed-ahad13"
REPO_NAME="SWEEP"
GIT_USER_NAME="Ahad"
GIT_USER_EMAIL="syed-ahad13@users.noreply.github.com"

if [ -z "${GH_TOKEN:-}" ]; then
  echo "pod_init.sh: GH_TOKEN is not set. export it first (it is never stored on disk)." >&2
  exit 1
fi

if [ -d "$REPO_NAME/.git" ]; then
  echo "pod_init.sh: $REPO_NAME already cloned here; skipping clone."
else
  git clone "https://${GH_TOKEN}@github.com/${REPO_OWNER}/${REPO_NAME}.git"
fi

cd "$REPO_NAME"
git config user.name  "$GIT_USER_NAME"
git config user.email "$GIT_USER_EMAIL"

echo
echo "=== nvcc --version ==="
nvcc --version

echo
echo "=== nvidia-smi ==="
nvidia-smi

echo
echo "Repo ready at $(pwd) on $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD)"
echo "Next gates, in order:"
echo "  1. nvcc --version reports 12.8"
echo "  2. nvidia-smi -q -d POWER  -> power limit near this card's spec"
echo "  3. make bin/bw_probe ARCH=<sm_XX> && ./bin/bw_probe  -> inside the band in results/machines.json"
echo "Any gate red: destroy this instance and take the next offer."
