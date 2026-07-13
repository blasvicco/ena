#!/bin/bash

find . -type f | grep '.py$' | while read fname; do
	./shell/do.sh "$fname"
done
