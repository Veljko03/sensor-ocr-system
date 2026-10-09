#!/bin/sh
set -eu
preview_project_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
preview_parent_dir=$(dirname -- "$preview_project_dir")
cd "$preview_project_dir"
exec "$preview_parent_dir/sensor-ocr-system/env/bin/python" -m tools.live_preview \
  --host 0.0.0.0 --port 5000 --root "$preview_parent_dir/sensor-ocr-data/live-preview"
