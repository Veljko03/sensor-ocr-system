# Development release checks — 9 October 2026

This release delivers the standalone live OCR preview, candidate algorithm,
private dataset capture/labeling tools, tests and reproducible research reports.
It does not replace the production FastAPI application or its measurement DB.

- All 52 unit and HTTP tests passed on Raspberry Pi 3 A+, including resource-warning checks.
- The candidate benchmark was rerun against the frozen human-reviewed dataset: 73/73 readable photographs correctly decoded, all 161 negatives rejected, zero wrong accepted readings. Dataset SHA-256: `0cb495c0931b672bcab0a7a8e57a0cd23d9f7bec7f242a031d21cfdbc4c952ad`.
- Actual candidate processing through the preview confirmed `0.00` and `1.61` from two distinct reviewed frames each. A subsequent negative frame cleared each value.
- Live ESP uploads and browser photo refresh were checked. The observed ceiling scene showed `-`, without reusing an earlier number.
- Production main/OCR/database hashes remained unchanged.
- New commits were scanned for private keys, common token patterns, credential assignments and prohibited data paths; no findings. Photographs, SQLite datasets, runtime logs and virtual environments remain private and outside new commits.
- Useful historical implementations and the unsuccessful geometry-only experiment remain preserved for rollback and reproduction. No permanent deletions, history rewrite, force push or merge into main were performed.

These are development/regression results, not independent held-out accuracy.
The private collection has 250 photographs, of which 234 are labeled and 16
await human labels. Angle, distance and lighting coverage has not been verified.
Production acceptance still requires independent sessions, confirmed labels and
application/database integration tests. Finite tests do not guarantee that
future frames cannot produce a wrong reading.

For operation use `bash ~/sensor-ocr-dev/start_preview.sh`, then open
`http://RPI_IP:5000/`. Start only one server on this port. The original application
in `~/sensor-ocr-system` remains available; stop the preview before starting it.
