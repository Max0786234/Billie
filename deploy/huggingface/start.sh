#!/bin/sh
# Start the embedding server, wait until it answers, then serve Billie on the Space port.
set -e
ollama serve >/tmp/ollama.log 2>&1 &
for i in $(seq 1 60); do
  ollama list >/dev/null 2>&1 && break
  sleep 1
done
# --proxy-headers: the Space proxy forwards the visitor IP, used for per-visitor rate limits.
exec uvicorn billie.api.app:create_app --factory --host 0.0.0.0 --port 7860 \
  --proxy-headers --forwarded-allow-ips '*'
