#!/bin/bash

black --line-length 150 "$@" && pylint "$@"
