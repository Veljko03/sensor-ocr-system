import asyncio
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image
from image_ingest import identify_device,read_bounded_upload,replace_latest
from tools.dataset_store import MAX_IMAGE_BYTES


def jpeg(color):
    stream=io.BytesIO();Image.new('RGB',(60,40),color).save(stream,format='JPEG');return stream.getvalue()


class FakeUpload:
    def __init__(self,data):self.stream=io.BytesIO(data);self.closed=False;self.max_read=0
    async def read(self,count):self.max_read=max(self.max_read,count);return self.stream.read(count)
    async def close(self):self.closed=True;self.stream.close()


class IngestTests(unittest.TestCase):
    def test_device_identity_and_no_unknown_fallback(self):
        self.assertEqual(identify_device('strujomer_esp_1.jpg'),'esp_1')
        self.assertEqual(identify_device('anything.jpg','esp_2'),'esp_2')
        for name,explicit in [('unknown.jpg',None),('esp_10.jpg',None),('esp_1_esp_2.jpg',None),('esp_1.jpg','esp_3')]:
            with self.subTest(name=name),self.assertRaises(ValueError):identify_device(name,explicit)

    def test_upload_chunk_bound_hash_and_close(self):
        data=jpeg('blue');upload=FakeUpload(data)
        returned,digest=asyncio.run(read_bounded_upload(upload))
        self.assertEqual(data,returned);self.assertEqual(digest,hashlib.sha256(data).hexdigest())
        self.assertTrue(upload.closed);self.assertLessEqual(upload.max_read,64*1024)

    def test_oversized_invalid_truncated_and_png_refused(self):
        stream=io.BytesIO();Image.new('RGB',(20,20)).save(stream,format='PNG')
        for data in (b'',b'a'*(MAX_IMAGE_BYTES+1),jpeg('blue')[:-2],stream.getvalue()):
            upload=FakeUpload(data)
            with self.subTest(size=len(data)),self.assertRaises(ValueError):asyncio.run(read_bounded_upload(upload))
            self.assertTrue(upload.closed)

    def test_latest_is_original_and_old_frame_survives_failed_replace(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);first=jpeg('blue');second=jpeg('red')
            target=replace_latest(root,'esp_1',first)
            with patch('image_ingest.os.replace',side_effect=OSError('test disk failure')):
                with self.assertRaises(OSError):replace_latest(root,'esp_1',second)
            self.assertEqual(target.read_bytes(),first)
            self.assertEqual(list(root.iterdir()),[target])
            replace_latest(root,'esp_1',second);self.assertEqual(target.read_bytes(),second)

    def test_no_existing_file_is_deleted_on_invalid_upload_or_low_disk(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);evidence=root/'evidence.jpg';evidence.write_bytes(jpeg('green'))
            with self.assertRaises(ValueError):replace_latest(root,'esp_1',b'broken')
            with patch('image_ingest.shutil.disk_usage',return_value=type('Usage',(),{'free':0})()):
                with self.assertRaises(OSError):replace_latest(root,'esp_1',jpeg('blue'))
            self.assertEqual(list(root.iterdir()),[evidence])


if __name__=='__main__':unittest.main()
