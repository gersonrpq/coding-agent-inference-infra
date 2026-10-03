#!/usr/bin/env bash
# Opens (or closes) the tunnel from this machine to the cluster, so that pi, Grafana and Prometheus are on localhost.
#
#   app/demo/tunnel.sh up        # starts the server-side port-forwards if needed, then ssh -L 4000 (gateway), 3000 (Grafana), 9090 (Prometheus)
#   app/demo/tunnel.sh status
#   app/demo/tunnel.sh down
#
# Reads LAMBDA (user@host) and LAMBDA_SSH_KEY from .env; nothing secret is printed. The server side is
# `setup/start_forward.sh` (kubectl port-forward to the LiteLLM, Grafana and Prometheus services); the forwards stick to one pod,
# so this script restarts them after a LiteLLM restart.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
set -a; source "$root/.env"; set +a
: "${LAMBDA:?set LAMBDA in .env}"; : "${LAMBDA_SSH_KEY:?set LAMBDA_SSH_KEY in .env}"
sock="${TMPDIR:-/tmp}/coding-agent-tunnel.sock"
ssh_opts=(-i "$LAMBDA_SSH_KEY" -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o StrictHostKeyChecking=accept-new)

alive() { ssh "${ssh_opts[@]}" -S "$sock" -O check "$LAMBDA" 2>/dev/null; }

case "${1:-status}" in
  up)
    if alive; then echo "tunnel already up"; exit 0; fi
    echo "starting the server-side port-forwards ..."
    ssh "${ssh_opts[@]}" "$LAMBDA" 'cd ~/final_project && export KUBECONFIG=/etc/rancher/k3s/k3s.yaml && bash setup/stop_forward.sh >/dev/null 2>&1; WITH_ENGINES=0 setsid nohup bash setup/start_forward.sh > ~/forward.log 2>&1 < /dev/null & sleep 6'
    ssh "${ssh_opts[@]}" -M -S "$sock" -fN -L 4000:127.0.0.1:4000 -L 3000:127.0.0.1:3000 -L 9090:127.0.0.1:9090 "$LAMBDA"
    echo "tunnel up: gateway http://localhost:4000  Grafana http://localhost:3000  Prometheus http://localhost:9090"
    ;;
  down)
    alive && ssh "${ssh_opts[@]}" -S "$sock" -O exit "$LAMBDA" 2>/dev/null && echo "tunnel closed" || echo "tunnel was not up"
    ;;
  status)
    if alive; then echo "tunnel up"; else echo "tunnel down"; exit 1; fi
    ;;
  *) echo "usage: tunnel.sh up|down|status" >&2; exit 2 ;;
esac
