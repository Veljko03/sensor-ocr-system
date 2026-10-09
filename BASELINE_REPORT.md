# Baseline audit — 9 October 2026

Inspected `/home/veljko03/sensor-ocr-system` on `rpi` as `veljko03`. The source hashes and measurement history identify this as the current project. No application was listening on port 5000 during inspection; no server was stopped or restarted. Initial inspection and benchmark did not import `main.py`, because importing it initializes the production database.

## Architecture and working behavior

`main.py` provides FastAPI, a dashboard embedded in an HTML/JavaScript string, `/upload`, and SQLite persistence. JPEG uploads reach `pictures/latest_original_esp_1.jpg`. Only `esp_1` calls the active `real_meter_ocr.read_real_meter_image`; `esp_2` fails closed because its real device is unconfigured. The active rule-based OCR combines HSV and blue dominance masks, morphology, contour filtering (aspect >= 0.84), a minimum-area rotated rectangle warped to 500×500, three fixed relative digit slots, binary masks, seven-segment occupancy decoding, and confidence >= 0.45. There is no trained model.

Two identical candidate values in succession are required per device. Rejected frames reset the sequence. Unconfirmed/rejected records contain NULL values and `ocr_ok=0`; the chart includes only valid values. The current card uses the most recent frame's status, so ordinary OCR failure does not reuse the previous accepted value. `0.00` is preserved by explicit None checks. Server restarts initialize empty confirmation state.

## Newly reproduced benchmark

Run directly on the Raspberry Pi, OpenCV 4.12.0, NumPy 2.2.6, Python 3.13.5, one OpenCV thread; confidence unchanged at 0.45.

| Metric | Result |
|---|---:|
| Original labeled photographs | 111 |
| Readable photographs | 39 |
| No-display photographs | 71 |
| Display/unreadable or other negative | 1 |
| Correct readings accepted | 38 / 39 (97.44%) |
| Wrong numerical readings accepted | 0 |
| Readable photographs rejected | 1 |
| Original negatives accepted | 0 / 72 |
| Historical isolated digit crops decoded correctly | 116 / 117 (99.15%) |
| Mean / p95 / maximum OCR latency | 172 / 239 / 265 ms |
| Peak process RSS (including OpenCV) | 56.6 MiB |
| CPU time for full-frame and crop run | 24.50 seconds |

The rejected readable frame is `20260910_131325_esp_1.jpg` (human label 6.62); its third slot proposes 4 at confidence 0.3445. Lowering the threshold would accept a wrong value.

These are frame-level regression results, not independent held-out results or temporal performance. The original photographs come from one short capture session on 10 September. No human-labeled photograph of an actual 0.00 reading exists in the 39 positives. Digit crops derive from these same images and are not independent validation data.

The claimed separate 25-image ceiling benchmark has no identified ground-truth manifest or executable benchmark in the project. `ceiling_test.log` records 50 uploads and earlier false candidates including 8.88 and 4.48. Some October 7 originals survive in `pictures`, but their exact classification and association with the claimed 25-image set require verification. Therefore 0/97 negatives is not a newly verified result. The 72-image original-negative result is verified. A finite benchmark does not guarantee zero future false positives.

OCR SHA-256: `2ef85ada4fe345a87cb62e64ca63f4c2c9aa9f9c71b92527a18642d9c3d052cb`.
Active main SHA-256: `b76a349d3084e5950afbcd705cf3ca94d018329e1f5a73260b7d5c9878fb3969`.

## Weaknesses and operational risks

- Minimum-area rectangles correct rotation but do not recover true trapezoidal LCD corners under perspective. The aspect gate excludes some legitimate oblique views; loosening it alone removes the existing negative protection.
- Morphology uses fixed pixel kernels; contour ranking selects only the largest/highest-area candidate; three slot locations remain fixed after normalization. Decoder confidence is a heuristic score, not a calibrated probability of display presence.
- Confirmation has no frame identity, age, sequence number, gap limit, or duplicate prevention. Re-uploading identical bytes can confirm a candidate. Two consistent false display detections can also confirm a wrong measurement.
- Single-worker uploads currently block the event loop through synchronous OCR/SQLite. There is no explicit synchronization for threaded/multiworker use, and memory confirmation state is not shared between workers. Starting multiple workers is unsafe.
- Latest image writes are non-atomic. A static-image request or capture tool can observe a partially written file. There are no explicit body/decompressed-pixel limits or pre-OCR upload validation. Unknown explicit device IDs can fall back to esp_1.
- Latest image names vary by extension; old timestamped originals/debug files remain (246 files, about 5 MiB). SQLite failure rows and verbose per-frame logs grow without a configured limit. Automatic deletion must not target production history or datasets.
- Dashboard depends on remotely loaded Chart.js; if it fails, chart construction can prevent the refresh script from running. Session filtering uses second-resolution timestamps instead of an exact database boundary. The delete endpoint is destructive and unauthenticated.
- Legacy OCR functions in main and OCR helpers are not called by the current path, but need a documented reference audit before cleanup. `main_before_real_meter.py` and `main_before_real_ocr_integration.py` are byte-identical. Other backups preserve distinct historical behavior; none has been deleted.
- Requirements omit active OpenCV, NumPy and pytesseract dependencies. The old labeling tool binds all interfaces, rewrites CSV non-atomically, accepts out-of-range values, and has no distinction/resume safeguards for new capture sessions. No automated test suite or benchmark runner exists.

## Data, Git and resources

`dataset/real_meter/images`: 111 originals, 2.31 MB; CSV has no missing files or duplicate filenames. `dataset/real_meter/digits`: 117 crops. Dataset/debug assets total 3.57 MB. Existing labels are preserved.

Repository: `git@github.com:Veljko03/sensor-ocr-system.git`, verified public through GitHub API; default branch `main`. Current HEAD and origin/main are `27a5748`. Only five files are tracked. `main.py` has uncommitted changes; `real_meter_ocr.py`, dataset, labeler, historical logs and five backups are untracked. These changes must be preserved before development. Author identity is configured. Push credentials have not been verified; no push or merge has occurred.

Disk: 29 GiB filesystem, approximately 22 GiB free (21% used). RAM: 424 MiB total, approximately 252 MiB available; 423 MiB swap, 74 MiB used at audit. Heavy models would be inappropriate without measured justification.

Production database was opened read-only: 978 records, with 471 failed/59 valid esp_1 and 324 failed/124 valid esp_2 records. No failed row contains a non-NULL measurement. The latest recorded upload is 9 October at 13:51:40 (RPi clock).

## Engineering sequence

1. Create a private recovery archive and Git history bundle; create an isolated development worktree preserving the current uncommitted application code as a checkpoint.
2. Add reproducible frame/digit evaluation, versioned manifests, group-based splits, and honest coverage/acceptance reports. Keep legacy data as regression/development data; reserve new capture sessions before tuning.
3. Build a separate bounded capture tool and loopback labeling app with immediate transactional saves, explicit zero/unreadable/absent/skip labels, metadata, zoom, keyboard controls and export. Import remaining historical originals as unlabeled only. Exact image hashes prevent duplicate records and split leakage.
4. Pause for actual camera movements and human ground truth. Include diverse nonzero values, actual 0.00, angle/distance/lighting variations and challenging negatives. Do not invent labels.
5. Evaluate bounded detector/geometry/slot/decoder candidates using development and validation groups, keeping held-out sessions untouched. Compare frame and temporal metrics, latency, memory and safety regressions.
6. Test upload safety, atomic storage and temporal identity safeguards against isolated databases. Keep the production version available; deploying/restarting it requires explicit approval.
7. Prepare cleanup manifest for approval, security review, operating/rollback documentation, meaningful commits, and a non-force push only after the requested validation and approvals. No merge into main without approval.
