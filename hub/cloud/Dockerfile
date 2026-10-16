# The HalfCheetah study's cloud image (ECR): the runtime requirements and only the source files the study imports
# (app/cloud_manifest.txt). Build from the repository root for ARM (the study's machines are Graviton):
#   docker build --platform linux/arm64 -f hub/cloud/Dockerfile --build-arg GIT_COMMIT=$(git rev-parse HEAD) .

# 1. Source stage: pick the manifest's files out of app/src, so the final image never holds the rest
FROM python:3.11-slim AS source
COPY ./app/src /src
COPY ./app/cloud_manifest.txt /tmp/cloud_manifest.txt
RUN mkdir /export \
    && cd /src \
    && grep -vE '^(#|$)' /tmp/cloud_manifest.txt | xargs -I{} cp -r --parents {} /export \
    && find /export -name '__pycache__' -prune -exec rm -rf {} + \
    && find /export -name '*.md' -delete

# 2. Runtime stage: the base the local experiments ran on, with the pinned requirements minus the dev-only tools
FROM python:3.11-slim AS ena_cloud
ARG GIT_COMMIT=unknown
LABEL org.opencontainers.image.revision=$GIT_COMMIT

# MuJoCo's and matplotlib's shared libraries (no rendering, no compilers or editors)
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# One source of truth for versions (app/requirements.txt), so physics, random streams and numerics match the local runs
COPY ./app/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && grep -vE '^(black-with-tabs|pylint|pytest|Shimmy)==' /tmp/requirements.txt > /tmp/runtime.txt \
    && pip install --no-cache-dir -r /tmp/runtime.txt \
    && rm /tmp/requirements.txt /tmp/runtime.txt

WORKDIR /home/app
COPY --from=source /export ./src
RUN echo "$GIT_COMMIT" > /home/app/VERSION

# python src/half_cheetah.py --replicate K --output-dir /results (the machine's user data passes the arguments)
ENTRYPOINT ["python", "src/half_cheetah.py"]
