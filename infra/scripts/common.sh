#!/bin/bash
# Shared settings and helpers of the cloud-run scripts (sourced). They run inside the toolbox container:
#   docker compose exec ena_infra bash /home/scripts/<script>.sh ...
# The stack's name and region come from the CDK project's config (mounted at /home/app), so the scripts follow the
# stack wherever it is deployed; STACK and REGION in the environment override them.
QOS=/home/app/qos/qos.json
STACK=${STACK:-$(jq -r '.stack.name' "${QOS}")}
REGION=${REGION:-$(jq -r '.aws.region' "${QOS}")}

# A stack output by key (LaunchTemplateId, SubnetIds, BucketName, RepositoryUri)
output() {
  aws cloudformation describe-stacks --stack-name "${STACK}" --region "${REGION}" \
    --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue" --output text
}

# The study machines (tag Project=<stack>, set by launch.sh) in the given states: Name, id, state, launch time
machines() {
  aws ec2 describe-instances --region "${REGION}" \
    --filters "Name=tag:Project,Values=${STACK}" "Name=instance-state-name,Values=${1:-pending,running,stopping,stopped}" \
    --query 'Reservations[].Instances[].[Tags[?Key==`Name`]|[0].Value, InstanceId, State.Name, LaunchTime]' --output text
}

# Whether replicate K's results in S3 are complete (its outputs.txt holds the test results)
complete() {
  aws s3 cp "s3://$(output BucketName)/exp_$1/outputs.txt" - --region "${REGION}" 2>/dev/null | grep -q 'Test Results ---'
}

# Run one machine from the stack's launch template, starting in subnet number START (mod the subnet count) and passing
# on to the next zone when a zone has no capacity for the instance type (InsufficientInstanceCapacity). Prints the
# instance id and subnet on success; returns 1 when no zone has capacity, 3 when the account's vCPU quota is full
# (VcpuLimitExceeded: retry once running machines end), 2 on any other error.
#   launch_machine START TAG_SPEC [INSTANCE_TYPE]   (TAG_SPEC: Tags=[{Key=Name,Value=...},...])
launch_machine() {
  local start=$1 tags=$2 type=${3:-} template subnets offset subnet result
  template=$(output LaunchTemplateId)
  IFS=',' read -ra subnets <<< "$(output SubnetIds)"
  for offset in $(seq 0 $((${#subnets[@]} - 1))); do
    subnet=${subnets[$(((start + offset) % ${#subnets[@]}))]}
    if result=$(aws ec2 run-instances --region "${REGION}" --count 1 \
      --launch-template "LaunchTemplateId=${template},Version=\$Latest" --subnet-id "${subnet}" \
      ${type:+--instance-type "${type}"} \
      --tag-specifications "ResourceType=instance,${tags}" "ResourceType=volume,${tags}" \
      --query 'Instances[0].InstanceId' --output text 2>&1); then
      echo "${result} ${subnet}"
      return 0
    fi
    if grep -q 'VcpuLimitExceeded' <<< "${result}"; then
      echo "the vCPU quota is full" >&2
      return 3
    fi
    if ! grep -q 'InsufficientInstanceCapacity' <<< "${result}"; then
      echo "${result}" >&2
      return 2
    fi
    echo "no capacity in ${subnet}, trying the next zone" >&2
  done
  return 1
}
