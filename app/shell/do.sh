#!/bin/bash

black --line-length 150 "$1" && pylint "$1"
