#!/usr/bin/env bash
# Robust, resumable download of the official ASVspoof 2019 LA dataset (Edinburgh DataShare, DOI 10.7488/ds/2555).
# Uses the direct bitstream URL (the /download redirect breaks wget resume). Loops until the full size is on disk.
# Run detached:  setsid nohup bash scripts/download_asvspoof.sh > data/download.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
URL="https://datashare.ed.ac.uk/server/api/core/bitstreams/a9f87c35-f055-4015-80e2-2fdff0d46269/content"
OUT="data/LA.zip"
EXPECTED=7640952520

size() { stat -c %s "$OUT" 2>/dev/null || echo 0; }

while [ "$(size)" -lt "$EXPECTED" ]; do
    echo "[$(date '+%F %T')] resuming at $(size) / $EXPECTED bytes"
    curl -L --fail -C - --retry 5 --retry-delay 15 --speed-limit 1024 --speed-time 120 \
         -o "$OUT" "$URL" || echo "[$(date '+%F %T')] curl exited $?; retrying in 20s"
    sleep 20
done

echo "[$(date '+%F %T')] download complete: $(size) bytes"
if unzip -tq "$OUT" > /dev/null; then
    echo "[$(date '+%F %T')] zip integrity OK, extracting to data/"
    unzip -q -n "$OUT" -d data/ && echo "[$(date '+%F %T')] EXTRACTED" && ls data/LA
else
    echo "[$(date '+%F %T')] ZIP INTEGRITY CHECK FAILED"
    exit 1
fi
