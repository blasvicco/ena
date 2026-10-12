#!/bin/bash
# Shared settings and helpers of the cloud-run scripts (sourced). They run inside the toolbox container:
#   docker compose exec ena_infra bash /home/scripts/<script>.sh ...
STACK=${STACK:-ena-experiment}
REGION=${REGION:-eu-south-2}

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
