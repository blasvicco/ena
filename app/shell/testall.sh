#!/bin/bash

pytest \
-n auto \
--cov=core \
--cov=drf_api \
--cov=web_socket \
--cov-report=term-missing \
--cov-fail-under=100 \
--cov-report=term \
--dist=loadfile
