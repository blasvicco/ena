#!/bin/bash
# Launch one on-demand machine per replicate from the stack's launch template, tagged Name=exp_k and
# Project=<stack>: launch.sh 1 (the reproduction check: replicate 1 must equal the local validation) or launch.sh 2-20.
# A replicate already running, or complete in S3, is skipped.
set -euo pipefail
source "$(dirname "$0")/common.sh"

RANGE=${1:?usage: launch.sh K or K-L}
FIRST=${RANGE%-*}
LAST=${RANGE#*-}
RUNNING=$(machines pending,running | awk '{print $1}')

for K in $(seq "${FIRST}" "${LAST}"); do
  NAME="exp_${K}"
  if grep -qx "${NAME}" <<< "${RUNNING}"; then
    echo "${NAME}: already running, skipped"
    continue
  fi
  if complete "${K}"; then
    echo "${NAME}: complete in S3, skipped"
    continue
  fi
  # Spread the machines over the zones; a zone without capacity passes the machine on to the next one
  if RESULT=$(launch_machine "${K}" "Tags=[{Key=Name,Value=${NAME}},{Key=Project,Value=${STACK}}]"); then
    echo "${NAME}: launched ${RESULT% *} in ${RESULT#* }"
  elif [ $? -eq 1 ]; then
    echo "${NAME}: no zone has capacity now; run launch.sh ${K} again later"
  else
    echo "${NAME}: launch failed"
    exit 1
  fi
done
