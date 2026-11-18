#!/bin/bash
set -euo pipefail

RES_DIR="${1:-./res}"
mkdir -p "$RES_DIR"

for color in red green blue black white_0 white_1 smth; do
    output="$RES_DIR/q1q2r_${color}.dat"
    find "$RES_DIR" -maxdepth 1 -type f \
        -name "q1q2r_${color}_*.txt" -exec cat {} + > "$output"
done
