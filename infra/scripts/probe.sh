#!/bin/bash
# Timing probe: one machine of TYPE trains AGENT alone (the study's replicate 1, same seeds) for at most MINUTES and
# terminates itself. Its pace (progress.csv) and logs go to s3://<bucket>/probes/probe-<TYPE>-<AGENT>/, where
# probe_report.sh reads them. Probes never touch the study's exp_k/ results.
#   probe.sh c8g.xlarge SAC 50      (AGENT: ENA-Combined, PPO or SAC)
set -euo pipefail
source "$(dirname "$0")/common.sh"

TYPE=${1:?usage: probe.sh TYPE AGENT MINUTES}
AGENT=${2:?usage: probe.sh TYPE AGENT MINUTES}
MINUTES=${3:?usage: probe.sh TYPE AGENT MINUTES}
NAME="probe-${TYPE}-${AGENT}"
ARGS="--num-experiments 1 --replicate 1 --variants ${AGENT}"
TAGS="Tags=[{Key=Name,Value=${NAME}},{Key=Project,Value=${STACK}},{Key=Args,Value=${ARGS}},{Key=MaxMinutes,Value=${MINUTES}}]"

if RESULT=$(launch_machine 0 "${TAGS}" "${TYPE}"); then
  echo "${NAME}: launched ${RESULT% *} in ${RESULT#* } for at most ${MINUTES} min"
else
  echo "${NAME}: not launched"
  exit 1
fi
