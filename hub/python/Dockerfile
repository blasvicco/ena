# Use slim-bullseye or slim-bookworm for a small footprint with glibc support
FROM python:3.11-slim AS ena_experiment

# Define package variables
# Note: build-base/gcc equivalent in Debian is build-essential
ENV ESSENTIAL_PACKAGES="python3-dev" \
    UTILITY_PACKAGES="plocate vim" \
    BUILD_PACKAGES="build-essential gcc" \
    OPENCV_PACKAGES="libglib2.0-0 libgl1 libxcb1"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    $ESSENTIAL_PACKAGES \
    $UTILITY_PACKAGES \
    $BUILD_PACKAGES \
    $OPENCV_PACKAGES \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Python requirements
COPY ./app/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Set working directory
WORKDIR /home/app
