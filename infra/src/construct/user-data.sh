#!/bin/bash
# Boot script of a study machine (the launch template's user data; construct/ec2.ts fills the __PLACEHOLDERS__).
# Runs this machine's replicate (its Name tag, exp_k) or, for a timing probe, its own study arguments; keeps its S3
# prefix current, records its pace (progress.csv), and powers off whatever happens, which terminates the machine.
set -uo pipefail

BUCKET='__BUCKET__'
IMAGE='__IMAGE__'
MAX_HOURS='__MAX_HOURS__'
REGION='__REGION__'
REPLICATES='__REPLICATES__'
SYNC_MINUTES='__SYNC_MINUTES__'
RESULTS=/results

# 1. Who am I: the Name tag, from the instance metadata (IMDSv2, tags exposed by the launch template). A study
# machine is Name=exp_k (launch.sh); a timing probe (probe.sh) also carries Args (its study arguments) and MaxMinutes
# (its time cap), and keeps everything under s3://<bucket>/probes/<Name>/.
TOKEN=$(curl -sX PUT http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 21600')
imds() { curl -sf -H "X-aws-ec2-metadata-token: ${TOKEN}" "http://169.254.169.254/latest/meta-data/$1"; }
NAME=$(imds tags/instance/Name)
ARGS=$(imds tags/instance/Args || true)
MAX_MINUTES=$(imds tags/instance/MaxMinutes || echo $((MAX_HOURS * 60)))
if [ -n "${ARGS}" ]; then
  OUT=${RESULTS}
  DEST="s3://${BUCKET}/probes/${NAME}/"
  STUDY_ARGS=${ARGS}
else
  K=${NAME#exp_}
  OUT=${RESULTS}/${NAME}
  DEST="s3://${BUCKET}/${NAME}/"
  STUDY_ARGS="--num-experiments ${REPLICATES} --replicate ${K}"
fi
mkdir -p "${OUT}"

# 2. Whatever happens next: copy what exists to S3, then power off (terminates the machine)
finish() {
  aws s3 sync "${OUT}" "${DEST}" --region "${REGION}" --only-show-errors
  aws s3 cp /var/log/cloud-init-output.log "${DEST}boot.log" --region "${REGION}" --only-show-errors
  shutdown -h now
}
trap finish EXIT
if [ -z "${ARGS}" ]; then
  case "${K}" in
    ''|*[!0-9]*) echo "Name tag '${NAME}' is not exp_<k>"; exit 1 ;;
  esac
fi

# 3. Docker and the study image; record exactly which image ran
dnf install -y docker
systemctl start docker
aws ecr get-login-password --region "${REGION}" | docker login --username AWS --password-stdin "${IMAGE%%/*}"
docker pull "${IMAGE}"
docker inspect --format '{{index .RepoDigests 0}} {{index .Config.Labels "org.opencontainers.image.revision"}}' "${IMAGE}" > "${OUT}/image.txt"

# 4. Keep S3 current while the study runs (a lost machine keeps everything up to its last sync), and record its pace:
# every minute, the time and the env-step reached by the agent training now (the last column of its newest
# live_training CSV), so probe_report.sh can fit each agent's steps per second
( while sleep $((SYNC_MINUTES * 60)); do
  aws s3 sync "${OUT}" "${DEST}" --region "${REGION}" --only-show-errors
done ) &
( while sleep 60; do
  FILES=$(find "${RESULTS}" -name 'live_training_*.csv')
  [ -n "${FILES}" ] || continue
  # shellcheck disable=SC2086 # one argument per file (no spaces in the study's file names)
  LIVE=$(ls -t ${FILES} | head -1)
  if [ -n "${LIVE}" ]; then
    AGENT=$(basename "${LIVE}" .csv)
    echo "$(date +%s),${AGENT#live_training_},$(tail -1 "${LIVE}" | awk -F, '{print $NF}')" >> "${OUT}/progress.csv"
  fi
done ) &

# 5. The study, capped at MAX_MINUTES. --init runs a tiny init as the container's process 1, so the study (Python) is
# not process 1 and dies on the cap's SIGTERM (process 1 ignores signals it does not handle); --kill-after makes sure.
# shellcheck disable=SC2086 # STUDY_ARGS holds several arguments
timeout --kill-after=2m "${MAX_MINUTES}m" docker run --rm --init -v "${RESULTS}:${RESULTS}" "${IMAGE}" \
  ${STUDY_ARGS} --output-dir "${RESULTS}" > "${OUT}/run.log" 2>&1
CODE=$?
echo "exit code ${CODE}" >> "${OUT}/run.log"
# A cap that fired (124, or 137 after the hard kill) must not leave the container running: stop whatever is left
docker ps -q | xargs -r docker kill
