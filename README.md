# Sensor OCR system — Raspberry Pi / Festo

ESP32 camera devices upload JPEG photographs to FastAPI `POST /upload`. The dashboard shows the latest original image and a numerical value only after valid OCR and temporal confirmation. A failed or unconfirmed frame has a NULL measurement, is stored with `ocr_ok=0`, and shows `-`. An actual confirmed `000` is `0.00`. Only valid measurements enter the chart.

## Status of this development branch

`main.py` now uses the structurally verified OCR in `ocr_candidate.py`, ordered frame confirmation and bounded atomic JPEG ingestion. It works with the existing SQLite measurement schema and dashboard. The previous application remains recoverable at checkpoint commit `24cf496` and the private deployment backup. On the reviewed development/regression dataset, the new OCR reads 73/73 readable photographs and rejects all 161 negatives; the baseline reads 65/73 correctly and accepts 16 wrong values. Independent validation/test captures remain necessary; these development results do not establish robustness across camera angles or distances. See [research evidence and remaining gates](docs/OCR_RESEARCH.md).

The main application is in `~/sensor-ocr-system`; development is in `~/sensor-ocr-dev`. Private datasets live in `~/sensor-ocr-data/robustness-v1`, outside Git. Integration tests use temporary databases, including a copy of the existing history. Normal server operation stores measurements in the existing `merenja.db`.

63 automated tests cover labeling/capture, OCR structure, ordered frame confirmation, bounded JPEG ingestion, atomic latest-image replacement, the HTTP receivers and FastAPI/SQLite integration. Real uvicorn HTTP checks additionally confirm zero/nonzero readings, rejected images, restart behavior and preservation of historical records. The temporary receiver and preview can use port 5000 only while the main app is off. Start one server on this port.

## Optional standalone OCR preview

Run one command on the Raspberry Pi:

```bash
bash ~/sensor-ocr-dev/start_preview.sh
```

Open `http://RPI_IP:5000/` on the same network, replacing `RPI_IP` with the address returned by `hostname -I`, and switch on the existing ESP camera. The page automatically displays the current original photograph and candidate OCR value. Two distinct frames with the same valid reading confirm the number, including `0.00`. Rejected, malformed, duplicate or unconfirmed frames show `-`; after 15 seconds without a new frame the value also disappears. This does not collect a new dataset or modify the production SQLite database. The process stops with Ctrl+C and must be started again after a reboot. No SSH tunnel or labeling page is needed for this preview.

The preview was checked with 52 passing tests and actual candidate OCR on reviewed historical photographs: two frames confirmed `0.00` and `1.61`; a subsequent negative photograph cleared each value. These integration checks are not independent accuracy validation. The existing private dataset currently has 250 images, with 234 labeled and 16 newly captured images pending human labels.

## Current application and OCR

`main.py` contains FastAPI endpoints, SQLite integration and dashboard HTML/JavaScript. It calls `ocr_candidate.read_real_meter_image` for `esp_1`; other real devices are unconfigured and reject OCR. The actual esp_1 Festo range is 0.00–9.99, exactly three digits with two fixed decimal places. Other ESP devices must be configured and verified separately. The main server rejects the unsafe geometry-only experimental mode at startup.

This is a rule-based pipeline: blue-color segmentation → quadrilateral candidates → normalized perspective warp → verified blue background and dynamic digit-row localization → local binary masks and bounded slant correction → structural seven-segment decoding → ambiguity/confidence gates. It is not a trained ML model. Geometric display presence and digit correctness are separate requirements. A high decoder score alone does not establish display presence.

ESP uploads normally occur about every five seconds. A bounded `esp_1`/`esp_2` filename identifier or explicit verified `?esp=esp_1` query selects the device. Unknown IDs are rejected. Two distinct, ordered, consistently decoded frames confirm a measurement; rejected, changed, duplicated, late or too-close frames reset confirmation. State is separate per camera and starts empty after restart. Do not run multiple workers; confirmation state is process-local. Receive time and JPEG hashes do not authenticate camera capture freshness.

## Run the main server

Use the established environment; these tools do not install packages or alter the OS:

```bash
cd ~/sensor-ocr-system
source env/bin/activate
python3 -m uvicorn main:app --host 0.0.0.0 --port 5000 --reload
```

Open `http://RPI_IP:5000/`, where `RPI_IP` is returned by `hostname -I`. The main dashboard now shows original photographs, confirmed values and valid measurement history using the new OCR. No separate preview command is needed. `--reload` is the existing development workflow. Continuous-operation deployment should eventually use a reviewed service and one worker. No boot service or OS settings have been changed. Importing `main.py` initializes its database; tests must set `SENSOR_OCR_DATA_DIR` to a temporary directory before importing or starting the app.

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

The current application atomically overwrites each device's latest original JPEG; historical timestamped files are not automatically purged. Multipart ingress is limited before parsing, JPEG bytes/dimensions are verified, free disk space is checked, and processing is serialized per device. Failed writes preserve the previous original and clear the current value. The dashboard fetches a frame matched to its metadata; failures show `-`, and a value expires after 15 seconds without a new image. Charts use only valid DB records. The separate collector watches changed stable JPEG files and may miss a frame overwritten between polls. It has explicit start/stop, count and disk limits, and exact-hash deduplication. It never deletes existing datasets or production history to free space. Measurement DB growth still requires a separately agreed retention policy.

If capture remains at zero: confirm that the production server is running, ESP uploads are arriving, the configured source file changes, and the disk quota is available. The initial old latest image is intentionally ignored. If a source is being rewritten or a JPEG is incomplete, the tool retries rather than inventing a frame. If the laptop URL fails, check that the SSH tunnel is running and port 8801 is free. An interrupted labeling service preserves committed labels; restart the service and refresh the page. A stale label revision is rejected to prevent a second browser tab from silently overwriting a newer annotation.

The recovery checkpoint is in `~/sensor-ocr-recovery/20261009T121834Z`: source/dataset archive, Git bundle, original hashes and uncommitted main patch. The archive excludes the production DB and runtime images; those remain untouched in the original project. Git history remains intact. The first development commit preserves current main/OCR. A rollback of later development code can use a new worktree at `24cf496`; never reset the active dirty production worktree or rewrite shared history. Deployment/rollback of a running service requires a reviewed plan and explicit approval.

This release preserves Git history and keeps useful historical scripts and experiments. No files were permanently removed. Photographs, ground-truth databases, logs, environments and credentials are excluded from new commits. Checks are recorded in [the release report](docs/RELEASE_CHECKS.md) and [the main-server integration report](docs/MAIN_SERVER.md). Installation in the existing Raspberry Pi application folder is separate from merging the GitHub branch: no merge into Git main is performed. Independent accuracy validation remains pending.
