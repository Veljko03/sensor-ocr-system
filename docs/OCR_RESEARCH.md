# OCR research checkpoint — 9 October 2026

This branch preserves production main/OCR and adds an isolated candidate. It has
not passed an independent held-out acceptance test and must not be deployed yet.

## Human ground truth and reproducible comparison

234 photographs are fully reviewed: 73 readable, including 33 real 0.00 labels,
and 161 images marked unreadable without a separate display-presence assertion.
One label, `20261007_135221_esp_1.jpg`, was explicitly corrected by the human to
0.00 after a visual review in chat. The prior SQLite checkpoint and frozen CSV
remain available privately. No OCR prediction became a label automatically.

Dataset SHA-256: `0cb495c0931b672bcab0a7a8e57a0cd23d9f7bec7f242a031d21cfdbc4c952ad`.

| Partition | Readable / negative | Baseline correct | Baseline wrong accepts | Candidate correct | Candidate wrong accepts |
|---|---:|---:|---:|---:|---:|
| Historical regression | 39 / 72 | 38 | 0 | 39 | 0 |
| New development images | 34 / 89 | 27 | 16 | 34 | 0 |
| Total | 73 / 161 | 65 | 16 | 73 | 0 |

On the reviewed development labels, the baseline's 16 errors comprise seven
wrong readable values and nine acceptances of negative images. Baseline has
one rejected readable historical image; candidate rejects none in this dataset.
Full-frame digit exactness is 100% for the candidate on these 73 readable frames.

Candidate mean latency is 216 ms on development and 198 ms on historical data;
p95 is 341 / 333 ms. Peak process RSS is 63,096 / 61,864 KiB. OpenCV uses one
thread on the Raspberry Pi 3 A+. Runs used OpenCV 4.12.0, NumPy 2.2.6,
Pillow 12.2.0 and Python 3.13.5. Raw JSON results stay private because they
include image paths and per-image diagnostics. The committed summary omits
photographs and private filesystem paths.

Candidate file SHA-256: `eb39db6f554d90c55d73fd24e8d9e04115397bc38e4104949fbe8850db811010`.
Baseline dependency SHA-256: `2ef85ada4fe345a87cb62e64ca63f4c2c9aa9f9c71b92527a18642d9c3d052cb`.

## Hypotheses and resulting candidate

The first ablation changed geometry without validating display structure. It
regressed: on the pre-review labels it obtained 20/33 development positives
with 42 total wrong accepts, and 27/39 historical positives with 12 wrong
accepts. Its source is preserved in `experiments/quad_v1.py` for reproduction.
Never use this ablation as the application OCR.

The selected development candidate, `ocr_candidate.py`, detects up to three
blue quadrilateral regions, corrects their perspective, checks predominantly
blue background, locates the digit row vertically, and splits its projection.
Disconnected segments can use the verified Festo LCD pitch, with boundaries
adjusted to nearby projection valleys. Local thresholds avoid contamination
from variable lighting across digits; small affine hypotheses normalize the
slanted LCD font. Exact seven-segment patterns, empty interior regions, shape
checks and ambiguity rejection precede a numerical result. The three digits
are validated as esp_1's 0.00–9.99 reading. This remains rule-based OCR.

The less restrictive geometry is justified only together with these structural
checks; the unsafe geometry-only result demonstrates why lowering an aspect
threshold alone is not acceptable. Filled light stripes are not digits.

```bash
cd ~/sensor-ocr-dev
~/sensor-ocr-system/env/bin/python -m tools.evaluate \
  --manifest ~/sensor-ocr-data/robustness-v1/manifest-completed-20261009T154851Z.csv \
  --split development --ocr ocr_candidate.py --output benchmark-results/new-candidate-run.json
OCR_CANDIDATE_MODE=quad ~/sensor-ocr-system/env/bin/python -m tools.evaluate \
  --manifest ~/sensor-ocr-data/robustness-v1/manifest-completed-20261009T152402Z.csv \
  --split development --ocr experiments/quad_v1.py --output benchmark-results/new-quad-ablation.json
```

Choose a new result filename. The evaluator checks dataset and code hashes and
records algorithm mode and baseline dependency hash for the current candidate.

## Confirmation and ingestion primitives

`frame_confirmation.py` verifies ordered, per-device, distinct JPEG hashes with
bounded inter-frame spacing, rejects delayed processing, resets on rejected or
changed readings, and retains only bounded process-local replay history. Zero
is a valid value. It cannot authenticate camera freshness: ingress sequence
and receive time do not prove a newly captured photograph. A verified camera
capture sequence/time protocol remains necessary for stronger replay protection.

`image_ingest.py` validates configured camera identities, reads bytes in bounded
chunks, verifies JPEG dimensions and completeness, checks free disk space,
and atomically replaces only the latest original. The existing latest survives
failed replacements. A future FastAPI integration also needs a multipart body
limit before framework spooling, serial processing/order enforcement, isolated
database tests, and verification of dashboard and history behavior. These
primitives are tested but are not yet connected to production main.py.

46 automated tests cover tools, human-label revision conflicts, zero, invalid
values, concurrent duplicates, stale results, independent device state, JPEG
limits, failed atomic writes, false digit structures, and the dataset-only
HTTP receiver. These unit/HTTP tests do not replace end-to-end application
validation or real independent images.

## Safe controlled collection

`tools.capture_receiver` is a separate temporary JPEG receiver. It imports no
production app, stores no numerical measurement or database history, and keeps
one original for esp_1. The labeler's collector saves selected sessions with
its existing explicit start/stop and count/disk quotas. During collection the
normal app must remain off when both would use the same port. Never interrupt
a running production service automatically. Stop the temporary receiver before
starting the normal app on its port. No boot service or OS settings are changed.

```bash
cd ~/sensor-ocr-dev
~/sensor-ocr-system/env/bin/python -m tools.capture_receiver \
  --host 0.0.0.0 --port 5000 --root ~/sensor-ocr-data/receiver-latest
```

Point the loopback-only labeler at
`~/sensor-ocr-data/receiver-latest/latest_original_esp_1.jpg` and keep using its
SSH tunnel. Capture is disabled until the human explicitly starts a session.

## Remaining acceptance gates

All current images were available during development. Near-identical frames
and an original recovered from an old experiment prevent interpreting this
as independent evidence. Angle, distance and lighting are unknown in this
dataset. New separate validation sessions and a later untouched test set are
required, including different actual values and blue-object hard negatives.
Negative labels need an explicit human display-presence review before reporting
no-display false acceptance separately. Do not tune on final test results.

Production integration, dashboard/temporary database verification, deployment
and rollback review, cleanup manifest approval, security checks and final
non-force GitHub push remain pending. No finite test guarantees future zero
false positives.
