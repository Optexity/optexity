#!/usr/bin/env bash
# Usage: ./chrome-proxy.sh <port> <ssh-host>
# Example: ./chrome-proxy.sh 9007 dev-inference-api
set -euo pipefail

PORT="${1:-}"; HOST="${2:-}"
[[ -n "$PORT" && -n "$HOST" ]] || { echo "Usage: $0 <port> <ssh-host>"; exit 1; }

EXT_ID="pbaganbicadeoacahamnbgohafchgakp"
HOME_URL="${HOME_URL:-https://www.google.com}"
BASE_DIR="${BASE_DIR:-$HOME/.chrome-optexity}"
PROFILE_DIR="$BASE_DIR/profile"
BROWSER_DIR="$BASE_DIR/browser"
EXT_DIR="$BASE_DIR/extension"

mkdir -p "$PROFILE_DIR" "$BROWSER_DIR" "$EXT_DIR"

# Cleanup on Ctrl+C, kill, or normal exit
CHROME_PID=""; TUNNEL_PID=""
cleanup() {
  trap - INT TERM EXIT
  [[ -n "$CHROME_PID" ]] && kill "$CHROME_PID" 2>/dev/null
  [[ -n "$TUNNEL_PID" ]] && kill "$TUNNEL_PID" 2>/dev/null
  wait 2>/dev/null || true
  echo "Closed."
}
trap cleanup INT TERM EXIT

# Chrome for Testing (supports --load-extension)
if [[ -z "${CHROME:-}" ]]; then
  command -v npx >/dev/null || { echo "Need node/npx to install Chrome for Testing"; exit 1; }
  CHROME="$(npx --yes @puppeteer/browsers install chrome@stable --path "$BROWSER_DIR" | tail -n1 | sed 's/^[^ ]* //')"
fi
[[ -x "$CHROME" ]] || { echo "Chrome not found at: $CHROME"; exit 1; }

# Extension: download CRX once and unpack (delete $EXT_DIR to update)
if [[ ! -f "$EXT_DIR/manifest.json" ]]; then
  echo "Downloading extension"
  CRX="$BASE_DIR/ext.crx"
  curl -fsSL -o "$CRX" \
    "https://clients2.google.com/service/update2/crx?response=redirect&prodversion=130.0&acceptformat=crx2,crx3&x=id%3D${EXT_ID}%26installsource%3Dondemand%26uc"
  unzip -q -o "$CRX" -d "$EXT_DIR" 2>/dev/null || true   # CRX header makes unzip warn
  [[ -f "$EXT_DIR/manifest.json" ]] || { echo "Extension unpack failed"; exit 1; }
fi

port_open() { nc -z 127.0.0.1 "$PORT" >/dev/null 2>&1; }
proxy_works() {
  curl -s -o /dev/null --max-time 8 --socks5-hostname "127.0.0.1:$PORT" https://www.google.com/generate_204
}
kill_port() {
  local pids
  pids="$(lsof -ti tcp:"$PORT" -sTCP:LISTEN 2>/dev/null || true)"
  [[ -n "$pids" ]] && { echo "Killing stale listener on $PORT: $pids"; kill $pids; sleep 1; }
  return 0
}

# Tunnel: reuse if healthy, otherwise kill and restart
if port_open && proxy_works; then
  echo "Tunnel on $PORT is healthy, reusing it"
else
  port_open && kill_port
  echo "Starting tunnel to $HOST on port $PORT"
  ssh -f -N -D "127.0.0.1:$PORT" \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    -o ConnectTimeout=10 \
    "$HOST"
  for _ in {1..10}; do port_open && break; sleep 0.5; done
  port_open && proxy_works || { echo "Tunnel failed to come up"; exit 1; }
  TUNNEL_PID="$(lsof -ti tcp:"$PORT" -sTCP:LISTEN | head -n1)"
fi

# Pin extension, open only the home page on start, never restore old tabs
python3 - "$PROFILE_DIR" "$EXT_DIR" "$HOME_URL" <<'PY'
import sys, os, json, hashlib, base64
profile, ext, home = sys.argv[1], os.path.realpath(sys.argv[2]), sys.argv[3]
m = json.load(open(os.path.join(ext, "manifest.json")))
src = base64.b64decode(m["key"]) if "key" in m else ext.encode()
ext_id = hashlib.sha256(src).hexdigest()[:32].translate(str.maketrans("0123456789abcdef", "abcdefghijklmnop"))
d = os.path.join(profile, "Default"); os.makedirs(d, exist_ok=True)
p = os.path.join(d, "Preferences")
try: prefs = json.load(open(p))
except Exception: prefs = {}
pins = prefs.setdefault("extensions", {}).setdefault("pinned_extensions", [])
if ext_id not in pins: pins.append(ext_id)
s = prefs.setdefault("session", {})
s["restore_on_startup"] = 4
s["startup_urls"] = [home]
pr = prefs.setdefault("profile", {})
pr["exit_type"] = "Normal"; pr["exited_cleanly"] = True
json.dump(prefs, open(p, "w"))
PY

# Screen size (points) for a normal window that fills the screen
read -r _ _ W H < <(osascript -e 'tell application "Finder" to get bounds of window of desktop' | tr -d ',')

# Wipe saved sessions so old tabs can never be restored
rm -rf "$PROFILE_DIR/Default/Sessions" \
       "$PROFILE_DIR/Default/Current Session" \
       "$PROFILE_DIR/Default/Current Tabs" \
       "$PROFILE_DIR/Default/Last Session" \
       "$PROFILE_DIR/Default/Last Tabs"

"$CHROME" \
  --user-data-dir="$PROFILE_DIR" \
  --load-extension="$EXT_DIR" \
  --proxy-server="socks5://127.0.0.1:$PORT" \
  --window-position=0,0 \
  --window-size="$W,$H" \
  --disable-session-crashed-bubble \
  --no-first-run \
  --no-default-browser-check &
CHROME_PID=$!
wait "$CHROME_PID"