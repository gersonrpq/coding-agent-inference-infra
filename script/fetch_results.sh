#!/usr/bin/env bash
# Copies the run results from the server to this repo (metrics/runs/). Needs LAMBDA and LAMBDA_SSH_KEY.
set -euo pipefail
mkdir -p metrics/runs
rsync -avz -e "ssh -i $LAMBDA_SSH_KEY" "$LAMBDA:~/final_project/metrics/runs/" metrics/runs/
