#!/bin/bash
# Terminate every study machine still alive (tag Project=<stack>) and wait until they are gone; run it before
# cdk destroy, which cannot delete the network while machines use it.
set -euo pipefail
source "$(dirname "$0")/common.sh"

IDS=$(machines pending,running,stopping,stopped | awk '{print $2}' | xargs)
if [ -z "${IDS}" ]; then
  echo "No study machine is running"
  exit 0
fi
# shellcheck disable=SC2086 # one argument per instance id
aws ec2 terminate-instances --region "${REGION}" --instance-ids ${IDS} --output text > /dev/null
# shellcheck disable=SC2086
aws ec2 wait instance-terminated --region "${REGION}" --instance-ids ${IDS}
echo "Terminated: ${IDS}"
