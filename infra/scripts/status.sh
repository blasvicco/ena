#!/bin/bash
# The study machines and their state, then every replicate in S3: complete or not, and the last line of its run log
# (the runner's timestamped status, e.g. "Training SAC (ETA 2h 27m)").
set -euo pipefail
source "$(dirname "$0")/common.sh"
BUCKET=$(output BucketName)

echo "Machines (Name, id, state, launched):"
machines pending,running,shutting-down,stopping,stopped | sort -t_ -k2 -n
echo
echo "Replicates in s3://${BUCKET}/:"
for PREFIX in $(aws s3 ls "s3://${BUCKET}/" --region "${REGION}" | awk '/PRE exp_/{print $2}' | sort -t_ -k2 -n); do
  NAME=${PREFIX%/}
  LAST=$(aws s3 cp "s3://${BUCKET}/${NAME}/run.log" - --region "${REGION}" 2>/dev/null | tr '\r' '\n' | grep -v '^[[:space:]]*$' | tail -1 || true)
  STATE=$(complete "${NAME#exp_}" && echo complete || echo 'in progress')
  echo "${NAME}  ${STATE}  ${LAST}"
done
