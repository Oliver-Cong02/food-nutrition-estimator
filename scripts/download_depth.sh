#!/usr/bin/env bash
# Parallel download of depth_raw.png for every dish in available_dish_ids.txt.
# Uses public HTTPS (no auth needed) + xargs -P for concurrency.
# Idempotent: skips files that already exist.
set -euo pipefail

DISH_LIST="${1:-data/sample/available_dish_ids.txt}"
IMG_ROOT="${2:-data/sample/imagery}"
PARALLEL="${3:-16}"
BASE_URL="https://storage.googleapis.com/nutrition5k_dataset/nutrition5k_dataset/imagery/realsense_overhead"

if [ ! -f "$DISH_LIST" ]; then
  echo "Missing $DISH_LIST" >&2
  exit 1
fi

n_total=$(wc -l < "$DISH_LIST")
echo "Downloading depth_raw.png for $n_total dishes with $PARALLEL parallel curl workers..."
start_ts=$(date +%s)

awk -v root="$IMG_ROOT" -v base="$BASE_URL" '{
  src = base "/" $1 "/depth_raw.png"
  dst = root "/" $1 "/depth_raw.png"
  print src "\t" dst
}' "$DISH_LIST" \
  | xargs -P "$PARALLEL" -I LINE bash -c '
      IFS=$'"'"'\t'"'"' read -r src dst <<< "LINE"
      [ -f "$dst" ] && [ -s "$dst" ] && exit 0
      mkdir -p "$(dirname "$dst")"
      curl -sSL --retry 3 --retry-delay 2 -o "$dst" "$src" || { echo "FAIL $dst" >&2; rm -f "$dst"; exit 0; }
    '

end_ts=$(date +%s)
echo "Done in $((end_ts - start_ts))s. Verify with scripts/verify_depth.py"
