#!/usr/bin/env bash
# Überträgt den aktuellen Commit auf den Pi und baut/startet den Container neu.
# Nutzung: ./deploy.sh [user@host]
set -euo pipefail
HOST="${1:-joshua@raspberrypi5}"
cd "$(dirname "$0")"
git archive --format=tar HEAD | ssh "$HOST" 'mkdir -p ~/SaveBill && tar -x -C ~/SaveBill'
ssh "$HOST" 'cd ~/SaveBill && docker compose up -d --build && docker image prune -f >/dev/null'
ssh "$HOST" 'curl -fsS http://localhost:8080/api/health && echo'
