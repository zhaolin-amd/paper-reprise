#!/usr/bin/env bash
# Single runnable entrypoint for the TurboQuant from-scratch reproduction.
# Invoked by paper-reprise as `bash impl/run_eval.sh <claim_id>` (cwd = run root), or
# `bash impl/run_eval.sh --smoke` for the tiny self-test. Computes & prints the metric.
set -euo pipefail
cd "$(dirname "$0")"        # run from impl/ so the local modules import cleanly
exec python eval.py "$@"
