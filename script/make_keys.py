#!/usr/bin/env python3
"""Creates the LiteLLM virtual keys of the project (one per client or tenant) and saves them in .env.

    python3 script/make_keys.py                 # the default set below, through http://localhost:4000 (the tunnel) or GATEWAY_URL
    python3 script/make_keys.py --rotate        # delete and recreate the keys that exist
    python3 script/make_keys.py --only pi-demo

Needs PostgreSQL behind LiteLLM (keys live in the database) and the master key (LITELLM_MASTER_KEY in the environment or .env).
The master key is for administration only: clients (pi, the load generator, the tenants) use their own key, which carries its own
limits. Keys are appended to .env as `export LITELLM_KEY_<ALIAS>=...`; they are never printed. Standard library only.
"""
import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV = os.path.join(ROOT, ".env")
MODELS = ["qwen-coding-local"]      # the public alias; the gateway rewrites it to a worker alias after authentication

# alias -> limits. max_parallel_requests is the per-key concurrency cap (429 beyond it); tpm/rpm are token/request rates per minute.
KEYS = {
    "pi-demo":     {"max_parallel_requests": 4},          # the real client (app/run.sh, app/demo)
    "loadgen":     {},                                    # the synthetic load: no per-key limit, the fleet cap applies
    "tenant-acme": {"max_parallel_requests": 3},          # tenants of the tenant probe: acme is limited,
    "tenant-beta": {"max_parallel_requests": 3},          # beta is a second tenant that must not be affected
}


def load_env() -> dict:
    values = {}
    if os.path.exists(ENV):
        for line in open(ENV):
            m = re.match(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)", line)
            if m:
                values[m.group(1)] = m.group(2).strip().strip("'\"")
    return values


def call(url: str, master: str, path: str, body: dict, missing_ok: bool = False) -> dict:
    req = urllib.request.Request(url + path, json.dumps(body).encode(),
                                 {"Authorization": "Bearer " + master, "Content-Type": "application/json"})
    try:
        return json.load(urllib.request.urlopen(req, timeout=30))
    except urllib.error.HTTPError as e:
        if missing_ok and e.code == 404:   # nothing to delete (a new database): not an error
            return {}
        raise SystemExit(f"{path} -> HTTP {e.code}: {e.read().decode()[:300]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", action="append", help="create only this alias (repeatable)")
    ap.add_argument("--rotate", action="store_true", help="delete and recreate keys that already exist")
    args = ap.parse_args()
    env = load_env()
    master = os.environ.get("LITELLM_MASTER_KEY") or env.get("LITELLM_MASTER_KEY")
    if not master:
        sys.exit("LITELLM_MASTER_KEY is not set (environment or .env)")
    url = os.environ.get("GATEWAY_URL", "http://localhost:4000")

    created = []
    for alias, limits in KEYS.items():
        if args.only and alias not in args.only:
            continue
        var = "LITELLM_KEY_" + re.sub(r"[^A-Za-z0-9]", "_", alias).upper()
        if var in env and not args.rotate:
            print(f"  {alias}: already in .env ({var}), skipped")
            continue
        if args.rotate:
            call(url, master, "/key/delete", {"key_aliases": [alias]}, missing_ok=True)
        key = call(url, master, "/key/generate", {"key_alias": alias, "models": MODELS, **limits})["key"]
        with open(ENV, "a") as f:
            f.write(f"\nexport {var}={key}\n")
        created.append(alias)
        print(f"  {alias}: created, saved to .env as {var} ({', '.join(f'{k}={v}' for k, v in limits.items()) or 'no per-key limits'})")
    if not created:
        print("nothing created")


if __name__ == "__main__":
    main()
