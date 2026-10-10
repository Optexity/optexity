#!/usr/bin/env bash
# Run ONE workflow file through a locally running Optexity server and time it.
#
# What it does, in plain English:
#   1. Copies the workflow file you give it to /tmp/optexity_current.json.
#      (The server was started with OPTEXITY_TEST_AUTOMATION pointing there,
#      so whatever is in that file is what runs.)
#   2. Sends one task to the server.
#   3. Waits until the server says no task is running, then prints how many
#      seconds it took.
#
# Usage:  samples/memory/run_scenario.sh <workflow.json> <endpoint_name> [server_port]
# Start the server first, e.g.:
#   OPTEXITY_RECORD_ACTIONS=true OPTEXITY_TEST_AUTOMATION=/tmp/optexity_current.json \
#   ENV_PATH=~/optexity/.env optexity inference --port 9000 --child_process_id 0
set -euo pipefail

file="$1"
endpoint="$2"
port="${3:-9000}"

cp "$file" /tmp/optexity_current.json
start=$(date +%s)

curl -s -X POST "localhost:${port}/inference" -H 'Content-Type: application/json' \
  -d "{\"endpoint_name\":\"${endpoint}\",\"input_parameters\":{\"stock_ticker\":[\"AAPL\"]}}"
echo

sleep 5  # give the server time to pick the task up
for _ in $(seq 1 120); do
  if curl -s "localhost:${port}/health" | grep -q '"task_running":false'; then
    break
  fi
  sleep 3
done
echo "elapsed_seconds=$(( $(date +%s) - start ))"
