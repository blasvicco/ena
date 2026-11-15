#!/bin/bash
# Copy every replicate's results from S3 into app/outputs/half_cheetah_outputs/ (mounted at /home/outputs), where
# app/src/tools/paper_results.py reads the study.
set -euo pipefail
source "$(dirname "$0")/common.sh"
aws s3 sync "s3://$(output BucketName)/" /home/outputs/ --region "${REGION}"
echo "Downloaded to app/outputs/half_cheetah_outputs/"
