#!/bin/bash
# Boot script of a study machine (the launch template's user data; construct/ec2.ts fills the __PLACEHOLDERS__).
# Runs this machine's replicate (its Name tag, exp_k), keeps s3://<bucket>/exp_k/ current, and powers off whatever
# happens, which terminates the machine.
set -uo pipefail

BUCKET='__BUCKET__'
IMAGE='__IMAGE__'
MAX_HOURS='__MAX_HOURS__'
REGION='__REGION__'
REPLICATES='__REPLICATES__'
SYNC_MINUTES='__SYNC_MINUTES__'
RESULTS=/results

# 1. Who am I: the Name tag (exp_k), from the instance metadata (IMDSv2, tags exposed by the launch template)
TOKEN=$(curl -sX PUT http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 21600')
NAME=$(curl -s -H "X-aws-ec2-metadata-token: ${TOKEN}" http://169.254.169.254/latest/meta-data/tags/instance/Name)
K=${NAME#exp_}
OUT=${RESULTS}/${NAME}
mkdir -p "${OUT}"

# 2. Whatever happens next: copy what exists to S3, then power off (terminates the machine)
finish() {
  aws s3 sync "${OUT}" "s3://${BUCKET}/${NAME}/" --region "${REGION}" --only-show-errors
  aws s3 cp /var/log/cloud-init-output.log "s3://${BUCKET}/${NAME}/boot.log" --region "${REGION}" --only-show-errors
  shutdown -h now
}
trap finish EXIT
case "${K}" in
  ''|*[!0-9]*) echo "Name tag '${NAME}' is not exp_<k>"; exit 1 ;;
esac

# 3. Docker and the study image; record exactly which image ran
dnf install -y docker
systemctl start docker
aws ecr get-login-password --region "${REGION}" | docker login --username AWS --password-stdin "${IMAGE%%/*}"
docker pull "${IMAGE}"
docker inspect --format '{{index .RepoDigests 0}} {{index .Config.Labels "org.opencontainers.image.revision"}}' "${IMAGE}" > "${OUT}/image.txt"

# 4. Keep S3 current while the replicate runs (a lost machine keeps everything up to its last sync)
( while sleep $((SYNC_MINUTES * 60)); do
  aws s3 sync "${OUT}" "s3://${BUCKET}/${NAME}/" --region "${REGION}" --only-show-errors
done ) &

# 5. The replicate, capped at MAX_HOURS
timeout "${MAX_HOURS}h" docker run --rm -v "${RESULTS}:${RESULTS}" "${IMAGE}" \
  --num-experiments "${REPLICATES}" --replicate "${K}" --output-dir "${RESULTS}" > "${OUT}/run.log" 2>&1
echo "exit code $?" >> "${OUT}/run.log"
