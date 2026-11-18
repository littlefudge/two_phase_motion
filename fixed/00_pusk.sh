#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
qsub -d "$SCRIPT_DIR" "$SCRIPT_DIR/01_run.sh"
