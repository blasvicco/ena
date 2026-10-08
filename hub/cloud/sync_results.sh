#!/usr/bin/env bash
# Mirror a study's output directory to object storage every few minutes, so a lost machine loses no finished replicate.
#
# Usage (on the cloud host, outside the container, with the AWS CLI configured):
#   nohup hub/cloud/sync_results.sh app/half_cheetah_outputs s3://<bucket>/half_cheetah_outputs 120 > sync.log 2>&1 &
#
# Upload-only and never deletes anything remote. Run it once more by hand after the study ends.
# Other providers: replace the aws line with `gsutil -m rsync -r "$SRC" "$DST"` or `rclone copy "$SRC" "$DST"`.
set -u
SRC=${1:?local output dir}
DST=${2:?remote destination, e.g. s3://bucket/prefix}
EVERY=${3:-120}

while true; do
	aws s3 sync "$SRC" "$DST" --only-show-errors || echo "$(date -u +%FT%TZ) sync failed; retrying in ${EVERY}s" >&2
	sleep "$EVERY"
done
