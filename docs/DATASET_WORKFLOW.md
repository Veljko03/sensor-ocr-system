# Controlled ESP dataset and human labeling

## Existing data

The new private store is `~/sensor-ocr-data/robustness-v1`. Historical import copies 111 original labeled photographs and their human CSV labels, then copies 123 remaining unique historical originals (including the full original recovered from an experiment folder) as **unlabeled**. It does not classify the remaining photographs from logs or OCR predictions. Repeated import is idempotent and does not overwrite human corrections. Original datasets and `pictures` files stay unchanged.

Images have collision-resistant UUID filenames, SHA-256 content identities, device IDs, session/group IDs, partition, scenario, angle, distance, lighting, source filename and source filesystem timestamp. `captured_utc` is the time the collector/importer recorded the snapshot, not a verified camera exposure timestamp. Historical original filenames preserve their original date/time strings; their timezone is not inferred. Imported historical sessions have unknown camera conditions, and cannot establish coverage for new viewing angles/distances.

The manifest SQLite database is separate from production `merenja.db`. Annotation updates are transactional with revision checks. Its CSV export preserves structured metadata and numeric values as explicit two-decimal text. `readable` requires 0.00–9.99; `unreadable`, `absent` and `skip` contain no numerical value. Skip remains unfinished work, not a benchmark negative.

## Labeling controls

The default view shows every original, including the preserved historical labels. There are two labeling actions: enter a value (6.62 or 662, including 0.00 or 000) and press Save/Enter, or press Unreadable. Saving advances to the next photograph. Previous/Next navigate without changing a label. Zoom enlarges the full original. Select the unfinished filter when revisiting only pending work. Progress is committed immediately and resumed after restart.

The simple Unreadable action records a non-numeric negative with human_simple origin and unspecified display presence. It does not assert whether a Festo display is present. Evaluation reports these separately as unreadable_or_absent; they cannot be presented as verified no-display hard negatives. Existing absent/unreadable distinctions in historical and previously entered labels remain preserved. Any final no-display benchmark still needs human-verified presence labels.

Capture controls are on the separate /capture page, reached through the small collection link. The label screen stays focused on the photograph and two actions. CSV export remains available. Recorded timestamps refer to collection/import time; original filenames retain the historical capture strings.

## New controlled collection

Capture is **off on every startup**. Open the separate collection page and enter a scenario, angle, distance and lighting. A group is a meaningful shared recording/scene that must remain in one data partition. Leaving the group blank assigns a new capture-session group. Reusing a group in a different partition is rejected. Use the same group for closely related repeated scenes rather than splitting almost identical frames between development and test.

Specify the maximum number of new unique pictures for that session (default 200) and total dataset disk budget (default 64 MiB, includes imported data and SQLite). The tool also reserves at least 64 MiB free disk. It stops when the count/quota limit is reached or storage fails. Limits are per session for image count and per dataset for bytes, not a promise to limit all system storage. No production or labeled files are automatically deleted.

Press **Pokreni snimanje** explicitly. The production server must receive ESP uploads at its unchanged port 5000. This sidecar reads only `pictures/latest_original_esp_1.jpg`; it does not modify the ESP firmware, intercept the uploader, restart the application, or change production database records. Its 0.5-second polling interval can miss overwritten frames. The first preexisting image is ignored. Stable reads, complete JPEG decoding, pixel/byte limits and unique content hashes reduce incomplete/duplicate captures; byte-identical camera frames are deduplicated and cannot supply additional independent examples.

Stop before changing scenario metadata, then start another controlled session. Choose development, validation and final-test groups **before** OCR candidate tuning. Keep final-test images out of candidate inspection and threshold selection. Human labeling is allowed for all groups, without OCR suggestions. Export/version a completed manifest before experiments.

## Hardware collection checklist

Collect separate sessions for straight view, left/right tilt, above/below, close/medium/far distances, centered/offset display, normal/dim/bright lighting, mild glare, blur, partial obstruction and complete absence. Include walls, ceilings, unrelated blue rectangles and difficult digit combinations. Record multiple actual nonzero measurements and actual 000; do not infer pressure or fabricate ground truth from the OCR. Ensure readable and negative cases are represented across meaningful independent sessions. The old 39 positives do not contain a human-labeled 0.00 photograph.

An initial manageable collection is 15–25 varied pictures per scenario, followed by human labeling and a coverage audit. Counts alone do not establish adequacy or the 95% target. Determine final group sizes/coverage before validation, and collect the next missing conditions if the target is unsupported. This is a collection plan, not a fabricated dataset or reported result.

When human labeling is complete, tell the engineering chat. The next phase validates image hashes, manifest values and group coverage, then evaluates detector/perspective/slot candidates on development and validation data. Final acceptance targets remain zero wrong accepted values and zero no-display false accepts on the designated independent test sets, with >=95% correct acceptance on adequately represented readable conditions. These targets have not yet been demonstrated on new data.
