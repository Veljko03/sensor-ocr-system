"""Explicit, bounded snapshot collection from a production latest JPEG.

This watcher reads production images only. It may miss frames overwritten
between polls; it never claims to observe each incoming upload.
"""
from pathlib import Path
import threading
import time
from .dataset_store import LimitError, MAX_IMAGE_BYTES


class Capture:
    def __init__(self, store, source):
        self.store = store
        self.source = Path(source).resolve()
        self.lock = threading.RLock()
        self.active = False
        self.state = {'active':False, 'saved':0, 'duplicates':0, 'errors':0, 'reason':'Nije pokrenuto.'}
        self.last_signature = None
        self.session_id = None
        self.thread = None
        self.stop_event = threading.Event()

    def start(self, group_id='', split='development', scenario='straight_on', angle='straight',
              distance='medium', lighting='normal', max_images=200, max_bytes=64*1024**2):
        max_images, max_bytes = int(max_images), int(max_bytes)
        if not 1 <= max_images <= 5000 or not 2*1024**2 <= max_bytes <= 2*1024**3:
            raise ValueError('Nevažeće ograničenje snimanja.')
        if split == 'legacy':
            raise ValueError('New captures cannot enter legacy regression data')
        with self.lock:
            if self.active:
                raise ValueError('Najpre zaustavi trenutnu sesiju.')
            self.session_id = self.store.create_session(group_id=group_id or None, split=split,
                scenario=scenario, angle=angle, distance=distance, lighting=lighting)
            self.max_images, self.max_bytes = max_images, max_bytes
            self.last_signature = self._signature()  # Never treat the initial stale image as a new upload.
            self.active = True
            self.state = {'active':True,'saved':0,'duplicates':0,'errors':0,'reason':'Snimanje novih slika.',
                          'session_id':self.session_id,'max_images':max_images,'max_bytes':max_bytes}
        return self.status()

    def stop(self, reason='Ručno zaustavljeno.'):
        with self.lock:
            self.active = False
            self.state.update(active=False, reason=reason)
        return self.status()

    def status(self):
        with self.lock:
            return dict(self.state)

    def _signature(self):
        try:
            s = self.source.stat()
            return s.st_ino, s.st_mtime_ns, s.st_size
        except FileNotFoundError:
            return None

    def poll(self):
        with self.lock:
            if not self.active:
                return
            before = self._signature()
            if before is None or before == self.last_signature:
                return
            if before[2] > MAX_IMAGE_BYTES:
                self.last_signature = before
                self.state['errors'] += 1
                self.state['reason'] = 'Slika je veća od dozvoljene veličine.'
                return
            try:
                with self.source.open('rb') as file:
                    data = file.read(MAX_IMAGE_BYTES+1)
                if before != self._signature():
                    return  # Production writer is active; try again on the next poll.
                _, created = self.store.ingest(data, self.session_id, 'esp_1', self.source.name, before[1],
                                                self.max_images, self.max_bytes)
                self.last_signature = before
                self.state['saved' if created else 'duplicates'] += 1
                if self.state['saved'] >= self.max_images:
                    self.stop('Dostignut maksimalni broj slika.')
            except LimitError as exc:
                self.stop(str(exc))
            except (ValueError, OSError) as exc:
                self.state['errors'] += 1
                self.state['reason'] = str(exc)

    def run(self):
        while not self.stop_event.wait(0.5):
            try:
                self.poll()
            except Exception as exc:
                self.stop('Snimanje zaustavljeno zbog greške: ' + str(exc))

    def launch(self):
        if self.thread is None:
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()

    def close(self):
        self.stop()
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=5)
