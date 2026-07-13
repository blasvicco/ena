#!/bin/bash

source=$(echo "$1" | sed 's|/tests/|/|' | sed 's|_test\.py|.py|')

while [ ! -f "$source" ]; do
	# Also try <path>/main.py for package-style modules
	main="${source%.py}/main.py"
	if [ -f "$main" ]; then
		source="$main"
		break
	fi
	parent=$(dirname "$source")
	if [ "$parent" = "." ] || [ "$parent" = "$source" ]; then
		echo "Error: could not find source file for $1" >&2
		exit 1
	fi
	source="${parent}.py"
done

module=$(echo "$source" | sed 's|/|.|g' | sed 's|\.py$||')
pytest "$1" --cov="$module" --cov-report=term-missing
