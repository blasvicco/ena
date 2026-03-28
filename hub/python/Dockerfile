FROM python:3.12-alpine AS ena_experiment

ENV ESSENTIAL_PACKAGES="python3-dev" \
  UTILITY_PACKAGE="mlocate vim" \
  BUILD_PACKAGES="build-base gcc"

RUN apk update \
    && apk --no-cache add $ESSENTIAL_PACKAGES $UTILITY_PACKAGE $BUILD_PACKAGES

RUN pip install --upgrade pip

COPY ./app/requirements.txt /tmp/requirements.txt
RUN cd /tmp \
    && pip install -r requirements.txt
