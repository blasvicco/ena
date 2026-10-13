# Toolbox for the cloud run's infrastructure (infra/): Node 20 for the CDK, the AWS CLI and jq for the scripts.
FROM node:20-alpine3.20

ENV ESSENTIAL_PACKAGES="aws-cli bash jq"

RUN apk update && \
    apk --no-cache --progress add $ESSENTIAL_PACKAGES

WORKDIR /home/app

# Copy package files into the workdir
COPY infra/src/package.json ./

# Install locally (creates /home/app/node_modules inside the image)
RUN npm install -g npm@11.10.1 && \
    npm install
