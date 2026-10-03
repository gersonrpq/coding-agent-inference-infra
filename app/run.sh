#!/usr/bin/env bash
# The application: pi (a coding agent) talking ONLY to our serving path (LiteLLM -> guard -> admission -> SGLang).
#
#   app/run.sh                              # interactive pi
#   app/run.sh "add a test for foo"         # interactive, with a first message
#   app/run.sh -p "explain this repo"       # one-shot, non-interactive
#   app/run.sh --thinking high              # any pi option; the client chooses the thinking level
#
# Needs a tunnel from this machine to the gateway on localhost:4000 (you open it, for example with
# `ssh -L 4000:127.0.0.1:4000 <server>` plus `setup/start_forward.sh` on the server) and LITELLM_MASTER_KEY in the
# environment or in .env (LITELLM_KEY_PI_DEMO, created by script/make_keys.py). pi uses its OWN config directory (app/pi/agent), so your ~/.pi is not touched: that matters
# because pi's global default provider may be an external API, which would bypass the gateway.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(dirname "$here")"

if [[ -z "${LITELLM_KEY_PI_DEMO:-}" && -f "$root/.env" ]]; then
  set -a; source "$root/.env"; set +a
fi
# pi uses its own virtual key (script/make_keys.py creates it); the master key stays for administration.
if [[ -z "${LITELLM_KEY_PI_DEMO:-}" ]]; then
  : "${LITELLM_MASTER_KEY:?set LITELLM_KEY_PI_DEMO (python3 script/make_keys.py) or, failing that, LITELLM_MASTER_KEY}"
  echo "warning: no LITELLM_KEY_PI_DEMO; using the master key. Run: python3 script/make_keys.py" >&2
  export LITELLM_KEY_PI_DEMO="$LITELLM_MASTER_KEY"
fi
export LITELLM_KEY_PI_DEMO

url="${GATEWAY_URL:-http://localhost:4000}"
if ! curl -s -m 5 -o /dev/null -w '%{http_code}' "$url/health/liveliness" | grep -q 200; then
  echo "The gateway does not answer at $url. Open the tunnel first (see the header of this script)." >&2
  exit 1
fi

export PI_CODING_AGENT_DIR="$here/pi/agent"
export PI_CODING_AGENT_SESSION_DIR="${PI_CODING_AGENT_SESSION_DIR:-$here/.sessions}"
export PI_SKIP_VERSION_CHECK=1
exec pi --provider litellm --model qwen-coding-local "$@"
