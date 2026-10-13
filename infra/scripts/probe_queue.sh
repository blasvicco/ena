#!/bin/bash
# Launch timing probes one after another as the vCPU quota allows: each "TYPE AGENT MINUTES" waits, retrying every
# RETRY_SECONDS, while the quota is full (VcpuLimitExceeded), then launches as probe.sh does. Run it in the background:
#   nohup bash /home/scripts/probe_queue.sh "c8g.xlarge PPO 45" "c8g.xlarge SAC 60" > /tmp/probe_queue.log 2>&1 &
set -uo pipefail
source "$(dirname "$0")/common.sh"
RETRY_SECONDS=${RETRY_SECONDS:-300}

for PROBE in "$@"; do
  read -r TYPE AGENT MINUTES <<< "${PROBE}"
  NAME="probe-${TYPE}-${AGENT}"
  ARGS="--num-experiments 1 --replicate 1 --variants ${AGENT}"
  TAGS="Tags=[{Key=Name,Value=${NAME}},{Key=Project,Value=${STACK}},{Key=Args,Value=${ARGS}},{Key=MaxMinutes,Value=${MINUTES}}]"
  while true; do
    RESULT=$(launch_machine 0 "${TAGS}" "${TYPE}" 2>/dev/null)
    CODE=$?
    if [ "${CODE}" -eq 0 ]; then
      echo "$(date '+%H:%M:%S') ${NAME}: launched ${RESULT% *} for at most ${MINUTES} min"
      break
    elif [ "${CODE}" -eq 3 ] || [ "${CODE}" -eq 1 ]; then
      sleep "${RETRY_SECONDS}"
    else
      echo "$(date '+%H:%M:%S') ${NAME}: launch failed, skipped"
      break
    fi
  done
done
echo "$(date '+%H:%M:%S') all probes launched"
