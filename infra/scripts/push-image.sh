#!/bin/bash
# Build the study image (hub/cloud/Dockerfile) for ARM and push it to the stack's ECR repository, tagged with the
# commit it was built from and 'latest'. Runs on the host (it needs Docker); AWS calls go through the toolbox.
# Refuses uncommitted changes to the image's inputs unless --allow-dirty (the image must match a known commit).
#   infra/scripts/push-image.sh [--allow-dirty]
set -euo pipefail
cd "$(dirname "$0")/../.."

INPUTS=(app/src app/requirements.txt app/cloud_manifest.txt hub/cloud)
DIRTY=$(git status --porcelain -- "${INPUTS[@]}")
if [ -n "${DIRTY}" ] && [ "${1:-}" != "--allow-dirty" ]; then
  echo "Uncommitted changes to the image's inputs; commit first (or pass --allow-dirty):"
  echo "${DIRTY}"
  exit 1
fi
COMMIT="$(git rev-parse --short=12 HEAD)$([ -n "${DIRTY}" ] && echo '-dirty' || true)"

TOOLBOX=(docker compose exec -T ena_infra bash -c)
REPOSITORY=$("${TOOLBOX[@]}" 'source /home/scripts/common.sh && output RepositoryUri')
REGISTRY=${REPOSITORY%%/*}

# 1. Build for the machines' architecture (Graviton, arm64)
docker build --platform linux/arm64 -f hub/cloud/Dockerfile --build-arg "GIT_COMMIT=${COMMIT}" \
  -t "${REPOSITORY}:${COMMIT}" -t "${REPOSITORY}:latest" .

# 2. Log in (password from the toolbox's credentials) and push both tags
"${TOOLBOX[@]}" 'source /home/scripts/common.sh && aws ecr get-login-password --region "${REGION}"' \
  | docker login --username AWS --password-stdin "${REGISTRY}"
docker push "${REPOSITORY}:${COMMIT}"
docker push "${REPOSITORY}:latest"
echo "Pushed ${REPOSITORY}:${COMMIT} (and :latest)"
