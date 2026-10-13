#!/bin/bash
# The timing probes' measured pace, and the time and cost of one study replicate per instance type.
# Each agent's rate is a least-squares fit of env-steps against time (progress.csv, one row a minute) over its rows
# past WARMUP steps (SAC's first 10,000 are random and fast); a phase is the study's 3 x 1M env-steps at that rate.
# ENA's training time is the measured one when its probe finished training (outputs.txt), else its fitted rate.
#   probe_report.sh [OVERHEAD_MINUTES]   (boot, image pull and the frozen test; default 20)
set -euo pipefail
source "$(dirname "$0")/common.sh"
BUCKET=$(output BucketName)
OVERHEAD=${1:-20}
STEPS=3000000
WARMUP=20000
WORK=$(mktemp -d)
aws s3 sync "s3://${BUCKET}/probes/" "${WORK}/" --region "${REGION}" --only-show-errors --exclude '*' \
  --include '*/progress.csv' --include '*/outputs.txt' --include '*/exp_1/outputs.txt'

printf '%-30s %-14s %10s %10s %12s\n' 'probe' 'agent' 'rows' 'steps/s' 'phase (h)'
: > "${WORK}/phases.txt"
for DIR in "${WORK}"/probe-*; do
  [ -f "${DIR}/progress.csv" ] || continue
  PROBE=$(basename "${DIR}")
  TYPE=${PROBE#probe-}
  TYPE=${TYPE%%-*}  # the instance type has no hyphen; the agent may (ENA-Combined)
  for AGENT in $(cut -d, -f2 "${DIR}/progress.csv" | sort -u); do
    read -r ROWS RATE < <(awk -F, -v agent="${AGENT}" -v warm="${WARMUP}" '
      $2 == agent && $3 > warm { n++; x += $1; y += $3; xx += $1 * $1; xy += $1 * $3 }
      END { slope = (n > 1) ? (n * xy - x * y) / (n * xx - x * x) : 0; printf "%d %.2f\n", n, slope }' "${DIR}/progress.csv")
    HOURS=$(awk -v rate="${RATE}" -v steps="${STEPS}" 'BEGIN { printf "%.2f", (rate > 0 ? steps / rate / 3600 : 0) }')
    # ENA's measured training time, when its probe got that far
    MEASURED=$(cat "${DIR}"/outputs.txt "${DIR}"/exp_1/outputs.txt 2>/dev/null | awk -v agent="${AGENT}" '$1 == agent ":" { printf "%.2f", $2 / 3600 }' | head -1 || true)
    if [ -n "${MEASURED}" ]; then
      HOURS=${MEASURED}
      RATE="measured"
    fi
    printf '%-30s %-14s %10s %10s %12s\n' "${PROBE}" "${AGENT}" "${ROWS}" "${RATE}" "${HOURS}"
    echo "${TYPE} ${AGENT} ${HOURS}" >> "${WORK}/phases.txt"
  done
done

echo
echo "One replicate per instance type (ENA + PPO + SAC phases, plus ${OVERHEAD} min overhead), 20 replicates on demand:"
printf '%-14s %8s %8s %8s %10s %10s %12s\n' 'type' 'ENA (h)' 'PPO (h)' 'SAC (h)' 'total (h)' '$/hour' '20 x cost ($)'
for TYPE in $(cut -d' ' -f1 "${WORK}/phases.txt" | sort -u); do
  ENA=$(awk -v t="${TYPE}" '$1 == t && $2 == "ENA-Combined" { print $3 }' "${WORK}/phases.txt" | tail -1)
  PPO=$(awk -v t="${TYPE}" '$1 == t && $2 == "PPO" { print $3 }' "${WORK}/phases.txt" | tail -1)
  SAC=$(awk -v t="${TYPE}" '$1 == t && $2 == "SAC" { print $3 }' "${WORK}/phases.txt" | tail -1)
  PRICE=$(aws pricing get-products --region us-east-1 --service-code AmazonEC2 --filters \
    "Type=TERM_MATCH,Field=instanceType,Value=${TYPE}" "Type=TERM_MATCH,Field=regionCode,Value=${REGION}" \
    'Type=TERM_MATCH,Field=operatingSystem,Value=Linux' 'Type=TERM_MATCH,Field=tenancy,Value=Shared' \
    'Type=TERM_MATCH,Field=preInstalledSw,Value=NA' 'Type=TERM_MATCH,Field=capacitystatus,Value=Used' \
    --query 'PriceList[0]' --output text | jq -r '.terms.OnDemand[].priceDimensions[].pricePerUnit.USD')
  awk -v t="${TYPE}" -v e="${ENA:-0}" -v p="${PPO:-0}" -v s="${SAC:-0}" -v o="${OVERHEAD}" -v price="${PRICE}" 'BEGIN {
    total = e + p + s + o / 60
    printf "%-14s %8.2f %8.2f %8.2f %10.2f %10.4f %12.0f\n", t, e, p, s, total, price, 20 * total * price }'
done
rm -rf "${WORK}"
