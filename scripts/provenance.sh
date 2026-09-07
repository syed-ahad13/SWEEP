#!/usr/bin/env bash
# Prints ONE CSV line describing the machine a measurement was taken on.
#
#   gpu_key,driver,sm_clock,mem_clock,temp,power_limit,host,commit,timestamp_utc
#
# Every benchmark CSV row appends these fields. Usage:
#
#   export SWEEP_PROV="$(bash scripts/provenance.sh)"
#
# On a machine with no nvidia-smi (the Mac) this exits non-zero with a message
# on stderr and prints nothing on stdout — that is the expected failure.
set -euo pipefail

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "provenance.sh: no nvidia-smi on this machine (host-only environment); no provenance line." >&2
  exit 1
fi

QUERY=$(nvidia-smi \
  --query-gpu=name,driver_version,clocks.sm,clocks.mem,temperature.gpu,power.limit \
  --format=csv,noheader) || {
  echo "provenance.sh: nvidia-smi query failed." >&2
  exit 1
}

# Multi-GPU hosts: we only ever use device 0.
QUERY=$(printf '%s\n' "$QUERY" | head -n 1)

trim() { printf '%s' "$1" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'; }

# Strip the unit suffix nvidia-smi appends ("2520 MHz", "450.00 W") so the
# fields land in the CSV as bare numbers.
strip_unit() { printf '%s' "$1" | sed -e 's/[[:space:]]*\(MHz\|W\|%\)$//'; }

IFS=',' read -r RAW_NAME RAW_DRIVER RAW_SMCLK RAW_MEMCLK RAW_TEMP RAW_PWR <<< "$QUERY"

NAME=$(trim "$RAW_NAME")
DRIVER=$(trim "$RAW_DRIVER")
SM_CLOCK=$(strip_unit "$(trim "$RAW_SMCLK")")
MEM_CLOCK=$(strip_unit "$(trim "$RAW_MEMCLK")")
TEMP=$(strip_unit "$(trim "$RAW_TEMP")")
POWER_LIMIT=$(strip_unit "$(trim "$RAW_PWR")")

# gpu_key: the join key into results/machines.json. Known cards are mapped by
# hand so that marketing variations in the reported name cannot split one
# device into two keys.
gpu_key() {
  local n="$1"
  case "$n" in
    *A100*80GB*)     echo "A100_80GB"  ; return ;;
    *A100*80*GB*)    echo "A100_80GB"  ; return ;;
    *"Tesla T4"*|*"T4"*)   echo "Tesla_T4"   ; return ;;
    *P100*)          echo "Tesla_P100" ; return ;;
    *"RTX 4090"*)    echo "RTX_4090"   ; return ;;
    *"RTX 5090"*)    echo "RTX_5090"   ; return ;;
    *"RTX 3090"*)    echo "RTX_3090"   ; return ;;
  esac
  # Unknown card: strip the vendor words, collapse whitespace to underscores.
  # An unmapped key has no band in machines.json, which is exactly the signal
  # the acceptance gate needs.
  printf '%s' "$n" \
    | sed -e 's/NVIDIA//g' -e 's/GeForce//g' \
          -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' \
          -e 's/[[:space:]][[:space:]]*/_/g'
}

GPU_KEY=$(gpu_key "$NAME")
HOST=$(hostname)
COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "nogit")
TIMESTAMP=$(date -u +%FT%TZ)

printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
  "$GPU_KEY" "$DRIVER" "$SM_CLOCK" "$MEM_CLOCK" "$TEMP" \
  "$POWER_LIMIT" "$HOST" "$COMMIT" "$TIMESTAMP"
