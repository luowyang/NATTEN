#!/usr/bin/env bash
# pick_idle_gpu.sh -- print the index of one genuinely idle GPU on this
# machine, or fail. Used by the self-hosted GPU workflows (wheel.yml,
# extended.yml) to choose CUDA_VISIBLE_DEVICES before building/testing,
# so a CI job never collides with whatever else is running on this dev
# machine's other GPUs.
#
# "Idle" here means, per `nvidia-smi`: near-zero memory used AND no
# compute-apps process listed for that GPU. Both conditions are checked
# (not just memory) because a process can hold the device with 0 MiB
# reported momentarily, and because a stray few MiB of driver/context
# overhead should not by itself disqualify a GPU.
#
# NOTE on nvidia-smi trustworthiness on this cluster: the *GPU name/model
# string* nvidia-smi reports is known to be spoofed by the virtualization
# layer here (see AGENTS.md-adjacent tribal knowledge; architecture
# identity should be confirmed via torch.cuda.get_device_capability(),
# not nvidia-smi's name field). That does not extend to the per-GPU
# memory/process telemetry this script reads (memory.used,
# --query-compute-apps) -- there is no alternative source for "is
# another process using this GPU right now" short of NVML/nvidia-smi
# itself, and this script's own idle-selection logic doubles as a
# behavioral check: if it selected a GPU another job was actually using,
# that job's own tests would fail loudly, not silently.
#
# USAGE: pick_idle_gpu.sh [--max-used-mib N]
#   Prints the winning GPU index (a single integer) to stdout on
#   success. On failure (no idle GPU, or nvidia-smi unusable), prints a
#   diagnostic table to stderr and exits 1.
set -euo pipefail

MAX_USED_MIB=32
if [[ "${1:-}" == "--max-used-mib" ]]; then
  MAX_USED_MIB="${2:?--max-used-mib requires a value}"
fi

command -v nvidia-smi >/dev/null 2>&1 || {
  echo "ERROR: nvidia-smi not found on PATH." >&2
  exit 1
}

# uuid -> 1 for every GPU with a live compute process attached.
declare -A BUSY_UUID
while IFS=',' read -r uuid _rest; do
  uuid="$(echo "$uuid" | xargs)"
  [[ -n "$uuid" ]] && BUSY_UUID["$uuid"]=1
done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null || true)

WINNER=""
{
  echo "index,uuid,memory.used_mib,busy_by_compute_app,verdict"
  while IFS=',' read -r idx uuid mem_used; do
    idx="$(echo "$idx" | xargs)"
    uuid="$(echo "$uuid" | xargs)"
    mem_used="$(echo "$mem_used" | xargs)"
    busy="no"
    [[ -n "${BUSY_UUID[$uuid]:-}" ]] && busy="yes"
    verdict="busy"
    if [[ "$busy" == "no" && "$mem_used" -le "$MAX_USED_MIB" ]]; then
      verdict="idle"
      [[ -z "$WINNER" ]] && WINNER="$idx"
    fi
    echo "$idx,$uuid,$mem_used,$busy,$verdict"
  done < <(nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits)
} >&2

if [[ -z "$WINNER" ]]; then
  echo "ERROR: no idle GPU found (threshold: <= ${MAX_USED_MIB} MiB used AND no compute-apps entry). See the table above." >&2
  exit 1
fi

echo "$WINNER"
