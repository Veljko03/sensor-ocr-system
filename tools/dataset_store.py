"""Private, transactional ground truth store. No production database access."""
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from contextlib import contextmanager

SCHEMA_VERSION = 1
CLASSES = {'readable', 'unreadable', 'absent', 'skip'}
SPLITS = {'legacy', 'development', 'validation', 'test'}
MAX_IMAGE_BYTES = 2 * 1024 * 1024
MAX_PIXELS = 4_000_000


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds')


def normalize_value(text):
    text = str(text).strip().replace(',', '.')
    if re.fullmatch(r'[0-9]{3}', text):
        return text[0] + '.' + text[1:]
    if not re.fullmatch(r'[0-9]\.[0-9]{2}', text):
        raise ValueError('Unesi 0.00–9.99 sa dve decimale ili tačno tri cifre.')
    return text


def verify_jpeg(data):
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise ValueError('JPEG je prazan ili veći od 2 MiB.')
    if not data.startswith(b'\xff\xd8') or not data.endswith(b'\xff\xd9'):
        raise ValueError('Nepotpun ili nevažeći JPEG.')
    from PIL import Image
    with Image.open(io.BytesIO(data)) as image:
        if image.format != 'JPEG' or image.width * image.height > MAX_PIXELS:
            raise ValueError('Nepodržan format ili previše piksela.')
        image.load()
        return image.width, image.height


class ConflictError(ValueError):
    pass


class LimitError(ValueError):
    pass


class Store:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.images = self.root / 'images'
        self.images.mkdir(exist_ok=True, mode=0o700)
        self.db = self.root / 'manifest.sqlite3'
        with self.connect() as conn:
            conn.executescript('''
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                INSERT OR IGNORE INTO metadata VALUES ('schema_version', '1');
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY, group_id TEXT NOT NULL, split TEXT NOT NULL,
                    scenario TEXT NOT NULL, angle TEXT NOT NULL, distance TEXT NOT NULL,
                    lighting TEXT NOT NULL, created_utc TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS images (
                    id TEXT PRIMARY KEY, filename TEXT UNIQUE NOT NULL,
                    sha256 TEXT UNIQUE NOT NULL, bytes INTEGER NOT NULL,
                    width INTEGER NOT NULL, height INTEGER NOT NULL,
                    source_filename TEXT NOT NULL, device_id TEXT NOT NULL,
                    captured_utc TEXT NOT NULL, source_mtime_ns INTEGER,
                    session_id TEXT NOT NULL REFERENCES sessions(id)
                );
                CREATE TABLE IF NOT EXISTS labels (
                    image_id TEXT PRIMARY KEY REFERENCES images(id),
                    class TEXT NOT NULL CHECK(class IN ('readable','unreadable','absent','skip')),
                    value TEXT, note TEXT NOT NULL DEFAULT '', origin TEXT NOT NULL,
                    updated_utc TEXT NOT NULL, revision INTEGER NOT NULL DEFAULT 1,
                    CHECK((class='readable' AND value IS NOT NULL) OR (class!='readable' AND value IS NULL))
                );
            ''')
            version = conn.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()[0]
            if version != str(SCHEMA_VERSION):
                raise ValueError('Unsupported dataset schema version')

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.db, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA foreign_keys=ON')
        conn.execute('PRAGMA synchronous=FULL')
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def create_session(self, session_id=None, group_id=None, split='development', scenario='unknown',
                       angle='unknown', distance='unknown', lighting='unknown'):
        if split not in SPLITS:
            raise ValueError('Invalid split')
        session_id = session_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:10]
        group_id = group_id or session_id
        fields = [session_id, group_id, scenario, angle, distance, lighting]
        if any(not isinstance(f, str) or not f.strip() or len(f) > 120 for f in fields):
            raise ValueError('Metadata must be nonempty text, at most 120 characters')
        with self.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            existing = conn.execute('SELECT DISTINCT split FROM sessions WHERE group_id=?', (group_id,)).fetchall()
            if any(r['split'] != split for r in existing):
                raise ValueError('Ista grupa/snimanje ne sme biti u različitim skupovima.')
            conn.execute('INSERT INTO sessions VALUES (?,?,?,?,?,?,?,?)',
                         (session_id, group_id, split, scenario, angle, distance, lighting, utc_now()))
        return session_id

    def ingest(self, data, session_id, device_id, source_filename, source_mtime_ns=None,
               max_images=300, max_bytes=64*1024*1024, min_free_bytes=64*1024*1024):
        if device_id != 'esp_1':
            raise ValueError('Za ovaj Festo skup konfigurisan je samo esp_1.')
        if not 1 <= max_images <= 5000 or not MAX_IMAGE_BYTES <= max_bytes <= 2*1024**3:
            raise ValueError('Invalid dataset limits')
        width, height = verify_jpeg(data)
        digest = hashlib.sha256(data).hexdigest()
        image_id = uuid.uuid4().hex
        filename = image_id + '.jpg'
        path = self.images / filename
        with self.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            existing = conn.execute('SELECT id FROM images WHERE sha256=?', (digest,)).fetchone()
            if existing:
                return existing['id'], False
            if conn.execute('SELECT id FROM sessions WHERE id=?', (session_id,)).fetchone() is None:
                raise ValueError('Unknown session')
            count = conn.execute('SELECT COUNT(*) FROM images WHERE session_id=?', (session_id,)).fetchone()[0]
            if count >= max_images:
                raise LimitError('Dostignut maksimalni broj slika u sesiji.')
            # Count all tool-owned files, including SQLite/WAL and temporary overhead.
            disk_bytes = sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file())
            reserve = len(data) + 64*1024
            if disk_bytes + reserve > max_bytes:
                raise LimitError('Dostignuto ograničenje prostora za dataset.')
            if shutil.disk_usage(self.root).free < min_free_bytes + reserve:
                raise LimitError('Nedovoljno slobodnog prostora; snimanje zaustavljeno.')
            owned_file = False
            try:
                with path.open('xb') as handle:
                    owned_file = True
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
                conn.execute('INSERT INTO images VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                             (image_id, filename, digest, len(data), width, height, Path(source_filename).name,
                              device_id, utc_now(), source_mtime_ns, session_id))
            except Exception:
                # Only roll back the new tool-owned file, never preexisting source/data.
                if owned_file and path.exists():
                    path.unlink()
                raise
        return image_id, True

    def label(self, image_id, classification, value='', note='', revision=0, origin='human'):
        if classification not in CLASSES:
            raise ValueError('Nepoznata oznaka.')
        value = normalize_value(value) if classification == 'readable' else None
        if len(note) > 1000:
            raise ValueError('Napomena je predugačka.')
        with self.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            if conn.execute('SELECT id FROM images WHERE id=?', (image_id,)).fetchone() is None:
                raise ValueError('Unknown image')
            previous = conn.execute('SELECT revision FROM labels WHERE image_id=?', (image_id,)).fetchone()
            actual = previous['revision'] if previous else 0
            if actual != int(revision):
                raise ConflictError('Oznaka je promenjena u drugom prozoru. Osveži sliku.')
            conn.execute('''INSERT INTO labels VALUES (?,?,?,?,?,?,?)
                ON CONFLICT(image_id) DO UPDATE SET class=excluded.class,value=excluded.value,
                note=excluded.note,origin=excluded.origin,updated_utc=excluded.updated_utc,revision=excluded.revision''',
                         (image_id, classification, value, note, origin, utc_now(), actual+1))

    def rows(self):
        with self.connect() as conn:
            return [dict(r) for r in conn.execute('''SELECT i.*,s.group_id,s.split,s.scenario,s.angle,
                s.distance,s.lighting,l.class,l.value,l.note,l.origin AS label_origin,l.revision
                FROM images i JOIN sessions s ON s.id=i.session_id LEFT JOIN labels l ON l.image_id=i.id
                ORDER BY i.captured_utc,i.id''')]

    def image_path(self, image_id):
        if not re.fullmatch('[a-f0-9]{32}', image_id):
            raise ValueError('Invalid image ID')
        with self.connect() as conn:
            row = conn.execute('SELECT filename FROM images WHERE id=?', (image_id,)).fetchone()
        if not row:
            raise ValueError('Unknown image')
        path = (self.images / row['filename']).resolve()
        if path.parent != self.images:
            raise ValueError('Image path escaped dataset')
        return path

    def summary(self):
        rows = self.rows()
        completed = sum(r['class'] in {'readable', 'unreadable', 'absent'} for r in rows)
        return {'total':len(rows), 'completed':completed, 'remaining':len(rows)-completed,
                'skipped':sum(r['class']=='skip' for r in rows),
                'bytes':sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file())}

    def export_csv(self):
        fields = ['id','filename','sha256','device_id','captured_utc','source_filename','source_mtime_ns',
                  'session_id','group_id','split','scenario','angle','distance','lighting',
                  'class','value','note','label_origin','revision']
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for row in self.rows():
            row['filename'] = 'images/' + row['filename']
            writer.writerow(row)
        return output.getvalue()

    def version(self):
        return hashlib.sha256(self.export_csv().encode()).hexdigest()


def import_legacy(store, project):
    project = Path(project).resolve()
    manifest = project / 'dataset/real_meter/labels.csv'
    with manifest.open(newline='', encoding='utf-8') as file:
        labels = list(csv.DictReader(file))
    session_id = 'legacy_20260910'
    with store.connect() as conn:
        exists = conn.execute('SELECT id FROM sessions WHERE id=?', (session_id,)).fetchone()
    if not exists:
        store.create_session(session_id, session_id, split='legacy', scenario='historical_labeled')
    known = set()
    imported = {'historical_labeled':0, 'historical_unlabeled':0, 'duplicates':0}
    for row in labels:
        filename = row['filename']
        if Path(filename).name != filename:
            raise ValueError('Unsafe legacy filename')
        path = project / 'dataset/real_meter/images' / filename
        data = path.read_bytes()
        known.add(hashlib.sha256(data).hexdigest())
        image_id, created = store.ingest(data, session_id, 'esp_1', filename, path.stat().st_mtime_ns,
                                        max_images=5000, max_bytes=256*1024**2)
        if created:
            classification = 'readable' if row['readable']=='1' else ('absent' if row['note']=='no_meter' else 'unreadable')
            store.label(image_id, classification, row['display_text'], row['note'], origin='legacy_human_csv')
            imported['historical_labeled'] += 1
    for path in sorted((project / 'pictures').glob('*.jpg')):
        if not re.fullmatch(r'\d{8}_\d{6}_esp_1\.jpg|latest_original_esp_1\.jpg', path.name):
            continue
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() in known:
            imported['duplicates'] += 1
            continue
        date = path.name[:8] if path.name[:8].isdigit() else 'latest_snapshot'
        session_id = 'historical_unclassified_' + date
        with store.connect() as conn:
            exists = conn.execute('SELECT id FROM sessions WHERE id=?', (session_id,)).fetchone()
        if not exists:
            store.create_session(session_id, session_id, split='development', scenario='historical_unclassified')
        _, created = store.ingest(data, session_id, 'esp_1', path.name, path.stat().st_mtime_ns,
                                  max_images=5000, max_bytes=256*1024**2)
        imported['historical_unlabeled'] += int(created)
    return imported
