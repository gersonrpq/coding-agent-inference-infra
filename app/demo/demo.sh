#!/usr/bin/env bash
# DEMO: pi builds a todo list app, and every call goes through the cluster (guard -> tenant -> admission -> placement -> SGLang).
#
#   app/demo/demo.sh                 # open the tunnel if needed, run pi on app/demo/prompt.md in app/todo-app/, show what the cluster did
#   DEMO_CLEAN=1 app/demo/demo.sh    # first delete what an earlier run generated in app/todo-app/
#   DEMO_PROMPT="..." app/demo/demo.sh
#   app/demo/demo.sh --interactive   # open pi interactively in app/todo-app/ instead (you type the task)
#
# Needs: pi installed (https://github.com/badlogic/pi-mono), .env with LITELLM_MASTER_KEY, LAMBDA and LAMBDA_SSH_KEY.
# While it runs, open Grafana (http://localhost:3000, dashboard "03 - Gateway & Admission") to watch the places, the placement and the sheds.
# Do not run it during a load test: it adds an uncontrolled session to the measurement.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../.." && pwd)"
workdir="$root/app/todo-app"
set -a; source "$root/.env"; set +a
: "${LITELLM_MASTER_KEY:?set LITELLM_MASTER_KEY in .env}"

echo "== 1. tunnel to the cluster"
if curl -s -m 3 -o /dev/null -f "http://localhost:4000/health/liveliness"; then
  echo "gateway already reachable on localhost:4000 (an existing tunnel); not opening another"
else
  "$here/tunnel.sh" status >/dev/null 2>&1 || "$here/tunnel.sh" up
fi
curl -s -m 5 -o /dev/null -w "gateway health: HTTP %{http_code}\n" "http://localhost:4000/health/liveliness"

echo "== 2. before"
python3 "$here/gateway_stats.py" snapshot /tmp/demo-before.json

mkdir -p "$workdir"
if [[ "${DEMO_CLEAN:-0}" == "1" ]]; then
  find "$workdir" -mindepth 1 -not -name .gitkeep -delete
fi

echo "== 3. pi works in ${workdir#$root/}"
cd "$workdir"
start=$(date +%s)
if [[ "${1:-}" == "--interactive" ]]; then
  "$root/app/run.sh"
else
  prompt="${DEMO_PROMPT:-$(cat "$here/prompt.md")}"
  "$root/app/run.sh" -p "$prompt"
fi
echo "pi finished in $(( $(date +%s) - start )) s"

echo "== 4. what pi built"
ls -l "$workdir" | awk 'NR>1 {print "  " $5 " bytes  " $9}'
for f in index.html style.css app.js; do [[ -s "$workdir/$f" ]] && echo "  ok: $f ($(wc -l < "$workdir/$f") lines)" || echo "  missing: $f"; done

echo "== 5. what the cluster did"
python3 "$here/gateway_stats.py" diff /tmp/demo-before.json

echo
echo "Try the app:  cd app/todo-app && python3 -m http.server 8000   ->   http://localhost:8000"
