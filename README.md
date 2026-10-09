# Sensor OCR system — Raspberry Pi / Festo

ESP32 camera devices upload JPEG photographs to FastAPI `POST /upload`. The dashboard shows the latest original image and a numerical value only after valid OCR and temporal confirmation. A failed or unconfirmed frame has a NULL measurement, is stored with `ocr_ok=0`, and shows `-`. An actual confirmed `000` is `0.00`. Only valid measurements enter the chart.

## Status of this development branch

This is an isolated engineering workspace, not a new production deployment. `main.py` and `real_meter_ocr.py` preserve the current Raspberry Pi application as checkpoint commit `24cf496`. The separate `ocr_candidate.py` correctly reads 73/73 readable photographs and rejects all 161 negatives in the fully human-reviewed development/regression dataset. The baseline reads 65/73 correctly and accepts 16 wrong values on the same reviewed labels. Independent validation/test captures remain necessary; these development results do not establish robustness across camera angles or distances. See [research evidence and remaining gates](docs/OCR_RESEARCH.md).

The original project remains in `~/sensor-ocr-system`. Development is in `~/sensor-ocr-dev`. Private datasets live in `~/sensor-ocr-data/robustness-v1`, outside Git. Nothing here should modify production `merenja.db` during research.

52 automated tests cover labeling/capture, OCR structure, ordered frame confirmation, bounded JPEG ingestion, atomic latest-image replacement, the dataset-only HTTP receiver and the live OCR preview. Production main.py still uses its preserved legacy implementation. The temporary receiver and preview can use port 5000 only while the normal app is off; neither writes production measurement history. Stop the running temporary server before starting another application on the same port.

## Simple live OCR test

Run one command on the Raspberry Pi:

```bash
bash ~/sensor-ocr-dev/start_preview.sh
```

Open `http://RPI_IP:5000/` on the same network, replacing `RPI_IP` with the address returned by `hostname -I`, and switch on the existing ESP camera. The page automatically displays the current original photograph and candidate OCR value. Two distinct frames with the same valid reading confirm the number, including `0.00`. Rejected, malformed, duplicate or unconfirmed frames show `-`; after 15 seconds without a new frame the value also disappears. This does not collect a new dataset or modify the production SQLite database. The process stops with Ctrl+C and must be started again after a reboot. No SSH tunnel or labeling page is needed for this preview.

The preview was checked with 52 passing tests and actual candidate OCR on reviewed historical photographs: two frames confirmed `0.00` and `1.61`; a subsequent negative photograph cleared each value. These integration checks are not independent accuracy validation. The existing private dataset currently has 250 images, with 234 labeled and 16 newly captured images pending human labels.

## Current application and OCR

`main.py` contains FastAPI endpoints, SQLite integration and dashboard HTML/JavaScript. It imports `real_meter_ocr.read_real_meter_image` for `esp_1`; other real devices are unconfigured and reject OCR. The actual esp_1 Festo range is 0.00–9.99, exactly three digits with two fixed decimal places. Other ESP devices must be configured and verified separately.

This is a rule-based pipeline: blue-color segmentation → morphology → geometric contour filtering → normalized screen warp → three fixed digit slots → binary masks → seven-segment occupancy decoding → confidence gate. It is not a trained ML model. Geometric display presence and digit correctness are separate requirements. A high decoder score alone does not establish display presence.

ESP uploads normally occur about every five seconds. A filename containing `esp_1` or `esp_2` selects the legacy camera ID; an optional `?esp=esp_1` query also exists. Legacy unknown-ID fallback is a known weakness. The existing two-consecutive-frame confirmation is per-device and resets on rejection, but has no duplicate, age or sequence verification. Do not run multiple server workers; confirmation state is process-local.

## Run the existing production server

Use the established environment; these tools do not install packages or alter the OS:

```bash
cd ~/sensor-ocr-system
source env/bin/activate
python3 -m uvicorn main:app --host 0.0.0.0 --port 5000 --reload
```

`--reload` is the user's existing development workflow. Continuous-operation deployment should eventually use a reviewed service and one worker, after explicit approval. No boot service or OS settings have been changed. Importing `main.py` initializes its database, so do not import it casually in algorithm experiments.

The existing `requirements.txt` describes web dependencies but is incomplete for computer vision. The inspected environment already provides OpenCV 4.12.0, NumPy 2.2.6, pytesseract and Pillow, including system packages. A fresh installation must provide those dependencies as well; a Raspberry Pi native OpenCV/NumPy installation is different from a pip wheel. Dependency lock/rebuild verification remains pending. No paid APIs or new ML runtime are required by the tools.

## Collect and label photographs

See [collection and labeling guide](docs/DATASET_WORKFLOW.md). Start a loopback-only labeler from the development folder using the existing environment:

```bash
cd ~/sensor-ocr-dev
~/sensor-ocr-system/env/bin/python -m tools.label_app \
  --dataset ~/sensor-ocr-data/robustness-v1 \
  --source ~/sensor-ocr-system/pictures/latest_original_esp_1.jpg \
  --port 8001
```

Open an SSH tunnel on the Windows laptop (replace `RPI_HOST` with the actual SSH hostname/IP):

```powershell
ssh -N -L 127.0.0.1:8801:127.0.0.1:8001 RPI_USER@RPI_HOST
```

Browse `http://127.0.0.1:8801`. Use the existing configured SSH key or agent. Never publish the labeler on `0.0.0.0`. Ground truth comes only from human entry; OCR predictions are not written as labels.

## Tests and evaluations

```bash
cd ~/sensor-ocr-dev
~/sensor-ocr-system/env/bin/python -m unittest discover -s tests -v
~/sensor-ocr-system/env/bin/python -m tools.dataset \
  --dataset ~/sensor-ocr-data/robustness-v1 validate
~/sensor-ocr-system/env/bin/python -m tools.evaluate \
  --legacy ~/sensor-ocr-system/dataset/real_meter \
  --output benchmark-results/legacy-baseline.json
```

The evaluator refuses to overwrite an experiment file. Use a new result name for each meaningful change. To evaluate a new partition, export its manifest next to its images:

```bash
~/sensor-ocr-system/env/bin/python -m tools.dataset \
  --dataset ~/sensor-ocr-data/robustness-v1 export \
  --output ~/sensor-ocr-data/robustness-v1/manifest-v1.csv
~/sensor-ocr-system/env/bin/python -m tools.evaluate \
  --manifest ~/sensor-ocr-data/robustness-v1/manifest-v1.csv \
  --split development --output benchmark-results/development-v1.json
```

Use separate validation groups for candidate selection. Final held-out tests require `--split test --final-test`, fully reviewed labels, and a frozen candidate selected without inspecting/tuning on test results. The tool checks exact duplicate hashes and group/partition consistency. It cannot prove that two different JPEGs are independent scenes; capture design must prevent near-identical scenes leaking across partitions.

Reports include exact full-reading matches, false acceptance rates for absent and unreadable images, numerical error among accepted frames, digit accuracy (with rejected frames counted as failures), conditions/groups, latency, process RSS, CPU time, code/dataset hashes, and missing-label coverage. These are frame-level metrics; temporal/upload integration metrics remain a separate evaluation phase. No timing results imply a trained classifier probability or a mathematical guarantee of zero future false positives.

## Storage, troubleshooting and recovery

The current application overwrites each device's latest original image; historical timestamped files are not automatically purged. Latest writes are currently non-atomic. The separate collector watches changed stable JPEG files and may miss a frame overwritten between polls. It does not promise to capture every upload. It has explicit start/stop, count and disk limits, and exact-hash deduplication. It never deletes existing dataset images or production history to free space.

If capture remains at zero: confirm that the production server is running, ESP uploads are arriving, the configured source file changes, and the disk quota is available. The initial old latest image is intentionally ignored. If a source is being rewritten or a JPEG is incomplete, the tool retries rather than inventing a frame. If the laptop URL fails, check that the SSH tunnel is running and port 8801 is free. An interrupted labeling service preserves committed labels; restart the service and refresh the page. A stale label revision is rejected to prevent a second browser tab from silently overwriting a newer annotation.

The recovery checkpoint is in `~/sensor-ocr-recovery/20261009T121834Z`: source/dataset archive, Git bundle, original hashes and uncommitted main patch. The archive excludes the production DB and runtime images; those remain untouched in the original project. Git history remains intact. The first development commit preserves current main/OCR. A rollback of later development code can use a new worktree at `24cf496`; never reset the active dirty production worktree or rewrite shared history. Deployment/rollback of a running service requires a reviewed plan and explicit approval.

This development release preserves Git history and keeps useful historical scripts and experiments. No files were permanently removed, so destructive cleanup approval is not required for publishing this branch. Photographs, ground-truth databases, logs, environments and credentials are excluded from new commits. The release checks are recorded in [the release report](docs/RELEASE_CHECKS.md). Main-branch integration and replacement of the production application remain pending independent-data validation and deployment review.
