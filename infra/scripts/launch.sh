#!/bin/bash
# Launch one on-demand machine per replicate from the stack's launch template, tagged Name=exp_k and
# Project=<stack>: launch.sh 1 (the reproduction check: replicate 1 must equal the local validation) or launch.sh 2-20.
# A replicate already running, or complete in S3, is skipped.
set -euo pipefail
source "$(dirname "$0")/common.sh"

RANGE=${1:?usage: launch.sh K or K-L}
FIRST=${RANGE%-*}
LAST=${RANGE#*-}
TEMPLATE=$(output LaunchTemplateId)
IFS=',' read -ra SUBNETS <<< "$(output SubnetIds)"
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
  # Spread the machines over the public subnets (one per availability zone)
  SUBNET=${SUBNETS[$((K % ${#SUBNETS[@]}))]}
  TAGS="Tags=[{Key=Name,Value=${NAME}},{Key=Project,Value=${STACK}}]"
  ID=$(aws ec2 run-instances --region "${REGION}" --count 1 \
    --launch-template "LaunchTemplateId=${TEMPLATE},Version=\$Latest" --subnet-id "${SUBNET}" \
    --tag-specifications "ResourceType=instance,${TAGS}" "ResourceType=volume,${TAGS}" \
    --query 'Instances[0].InstanceId' --output text)
  echo "${NAME}: launched ${ID} in ${SUBNET}"
done
