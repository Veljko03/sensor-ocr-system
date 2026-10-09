# Main-server OCR integration — 9 October 2026

The main FastAPI application now calls the verified dynamic structural OCR for
esp_1. The usual `main:app` entry point, port 5000, measurement database and
dashboard remain the normal workflow:

```bash
cd ~/sensor-ocr-system
source env/bin/activate
python3 -m uvicorn main:app --host 0.0.0.0 --port 5000 --reload
```

Only one server should use this port. No background server or boot service is
installed by this integration. Stop the foreground process with Ctrl+C.

The original JPEG is decoded from immutable ingress bytes, atomically saved as
the latest original, and displayed through a sequence-matched image endpoint.
OCR runs outside the event loop under per-device ordering. Confirmation requires
two distinct ordered frames, validates values including zero, and resets on
rejection, change, duplicates, failed writes or delayed processing. Values expire
after 15 seconds without a new image. Only confirmed values have `ocr_ok=1` in
SQLite; other processed frames are stored with NULL and are excluded from charts.
The unsafe geometry-only experimental OCR mode cannot start the main server.

63 tests pass, including actual FastAPI routes with temporary SQLite databases,
concurrent duplicates, older uploads finishing later, image and DB write errors,
body limits, zero, invalid values, expiry, original image matching and restart.
A separate real uvicorn HTTP probe uses reviewed JPEGs and a private copy of the
existing history: it confirms `0.00` and `1.61`, clears the value after a negative
image, checks new DB rows and verifies that all historical rows remain intact.
Restart empties current state while preserving measurement history.

This integration does not establish independent held-out OCR accuracy. Camera
capture freshness is not authenticated by ingress time or JPEG hashes. Separate
sessions with human-reviewed labels are still necessary for broad angle,
distance and lighting claims. The second device remains unconfigured for OCR.
The existing database continues to grow with normal records; no automatic
history deletion or photograph collection is introduced.

Before replacing files, the stopped application source and a consistent SQLite
backup are preserved privately in `~/sensor-ocr-recovery/main-server-*`.
To roll back while the server is stopped, restore `source/main.py` from that
checkpoint to `~/sensor-ocr-system/main.py`. Do not restore an old database over
newer measurements merely to roll back OCR code; the DB schema is unchanged.
New dependency files can remain in place, unused by the previous main.py.
