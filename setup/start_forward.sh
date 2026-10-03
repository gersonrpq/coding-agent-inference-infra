#!/bin/bash
# Background port-forwards to the cluster services; PIDs go to ~/.k8s_forwards.pids
# (stop them with stop_forward.sh). Forwards stick to one pod: after a LiteLLM or
# Prometheus restart, run stop_forward.sh and start_forward.sh again.
#   WITH_ENGINES=1 also forwards the SGLang workers (debugging only: clients must go through LiteLLM).
PID_FILE="$HOME/.k8s_forwards.pids"
> "$PID_FILE"

forward() {  # forward <service> <local port> <remote port>
  kubectl port-forward -n gpu-serving "svc/$1" "$2:$3" > /dev/null 2>&1 &
  echo $! >> "$PID_FILE"
  echo "✔ $1 -> localhost:$2 (PID: $!)"
}

forward grafana 3000 3000
forward prometheus 9090 9090
forward litellm 4000 4000

if [[ "${WITH_ENGINES:-0}" == 1 ]]; then
  forward sglang-worker-0 30000 30000
  forward sglang-worker-1 30001 30001
fi

echo "Port-forwards running in the background; stop them with stop_forward.sh"
