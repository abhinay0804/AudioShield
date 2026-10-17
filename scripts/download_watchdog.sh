#!/usr/bin/env bash
# Watchdog for scripts/download_asvspoof.sh: if data/LA.zip stops growing for 3 minutes while curl is running,
# kill curl so the downloader loop resumes it. Exits once the downloader is gone.
# Run detached:  setsid nohup bash scripts/download_watchdog.sh > data/watchdog.log 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
last=-1
while pgrep -f download_asvspoof.sh > /dev/null; do
    sleep 180
    now=$(stat -c %s data/LA.zip 2>/dev/null || echo 0)
    if [ "$now" -eq "$last" ] && pgrep -f "curl -L --fail -C -" > /dev/null; then
        echo "[$(date '+%F %T')] stalled at $now bytes; killing curl to force resume"
        pkill -f "curl -L --fail -C -"
    fi
    last=$now
done
echo "[$(date '+%F %T')] downloader finished; watchdog exiting"
