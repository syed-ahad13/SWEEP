#!/usr/bin/env bash
# The one place that knows what SWEEP needs installed on a pod.
# Called by the template's on-start script after the repo is cloned.
#
# RULE: anything you apt-get or pip install by hand on a pod gets added here
# and pushed BEFORE you destroy that pod. Otherwise the next pod is a different
# machine than the one your last measurement came from.
set -u
export DEBIAN_FRONTEND=noninteractive

APT=(
  tmux
  build-essential
  cmake
  ninja-build
  jq
  ripgrep
  # cuda-nsight-compute-12-8   # uncomment when you start the profiling steps
)

apt-get install -y -qq --no-install-recommends "${APT[@]}" || exit 1

# PIP=( numpy pandas )
# [ ${#PIP[@]} -eq 0 ] || pip3 install --quiet --break-system-packages "${PIP[@]}"

echo "deps.sh: ${#APT[@]} apt packages ok"
