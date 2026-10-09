import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from urllib.request import Request,urlopen
from urllib.error import HTTPError
from PIL import Image
from frame_confirmation import FrameConfirmation
from tools.live_preview import Preview


def jpeg(color):
    stream=io.BytesIO();Image.new('RGB',(60,40),color).save(stream,format='JPEG');return stream.getvalue()


def multipart(data):return b'--live\r\nContent-Disposition: form-data; name="file"; filename="strujomer_esp_1.jpg"\r\nContent-Type: image/jpeg\r\n\r\n'+data+b'\r\n--live--\r\n'


class PreviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.now=0.
        self.server=Preview(('127.0.0.1',0),self.tmp.name,ocr=lambda image,**kwargs:(0.,{}),clock=lambda:self.now)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.url='http://127.0.0.1:'+str(self.server.server_port)

    def tearDown(self):self.server.shutdown();self.server.server_close();self.thread.join();self.tmp.cleanup()

    def read(self,path):
        with urlopen(self.url+path,timeout=3) as response:return response.read()

    def post(self,data):
        with urlopen(Request(self.url+'/upload',data=multipart(data),headers={'Content-Type':'multipart/form-data; boundary=live'}),timeout=3) as response:return json.load(response)

    def test_zero_confirmed_reject_resets_and_original_preserved(self):
        first=jpeg('blue');second=jpeg('red');third=jpeg('green')
        self.server.process(first,1,0.);self.assertIsNone(self.server.snapshot()[0]['value'])
        self.now=5.;self.server.process(second,2,5.);self.assertEqual(self.server.snapshot()[0]['value'],0.)
        self.server.ocr=lambda image,**kwargs:(None,{})
        self.now=10.;self.server.process(third,3,10.)
        state,data=self.server.snapshot();self.assertIsNone(state['value']);self.assertEqual(data,third)
        self.assertEqual(self.read('/frame.jpg?frame=3'),third)
        self.assertFalse(list(Path(self.tmp.name).glob('*.db')))

    def test_current_photo_and_empty_value_while_processing(self):
        data=jpeg('blue');entered=threading.Event();release=threading.Event()
        def slow(image,**kwargs):entered.set();release.wait(2);return 0.,{}
        self.server.ocr=slow
        worker=threading.Thread(target=lambda:self.server.process(data,1,0.));worker.start()
        try:
            self.assertTrue(entered.wait(1))
            state=json.loads(self.read('/api/live'));self.assertEqual(state['status'],'processing');self.assertIsNone(state['value'])
            self.assertEqual(self.read('/frame.jpg?frame=1'),data)
        finally:release.set();worker.join()

    def test_missing_camera_frames_hide_expired_value(self):
        self.server.process(jpeg('blue'),1,0.)
        self.now=5.;self.server.process(jpeg('red'),2,5.)
        timestamp=self.server.snapshot()[0]['updated_at']
        with patch('tools.live_preview.time.time',return_value=timestamp+16):
            state=json.loads(self.read('/api/live'))
        self.assertIsNone(state['value'])
        self.assertEqual(state['status'],'waiting_new_frame')

    def test_stale_frame_url_refused_and_restart_has_no_old_value(self):
        self.server.process(jpeg('blue'),1,0.)
        self.now=5.;self.server.process(jpeg('red'),2,5.)
        with self.assertRaises(HTTPError) as error:self.read('/frame.jpg?frame=1')
        self.assertEqual(error.exception.code,409)
        self.server.server_close()
        replacement=Preview(('127.0.0.1',0),self.tmp.name,ocr=lambda image,**kwargs:(0.,{}))
        try:self.assertEqual(replacement.snapshot(),({'frame':0,'value':None,'status':'waiting','received':0,'latency_ms':None,'updated_at':None},None))
        finally:replacement.server_close()

    def test_duplicate_and_invalid_upload_do_not_reuse_confirmed_value(self):
        self.server.process(jpeg('blue'),1,0.);self.now=5.;self.server.process(jpeg('red'),2,5.)
        self.assertEqual(self.server.snapshot()[0]['value'],0.)
        self.now=10.;self.server.process(jpeg('red'),3,10.)
        self.assertEqual(self.server.snapshot()[0]['status'],'duplicate_frame');self.assertIsNone(self.server.snapshot()[0]['value'])
        with self.assertRaises(HTTPError):self.post(b'not JPEG')
        self.assertEqual(self.server.snapshot()[0]['status'],'invalid_image')
        self.now=15.;self.server.process(jpeg('green'),4,15.)
        self.assertFalse(self.server.snapshot()[0]['value'] is not None)

    def test_http_upload_health_page_and_nonfinite_ocr_fail_closed(self):
        self.server.clock=time.monotonic
        self.post(jpeg('blue'))
        self.assertEqual(json.loads(self.read('/health'))['mode'],'ocr_live_preview')
        self.assertIn('OCR uživo'.encode(),self.read('/'))
        self.assertEqual(self.read('/frame.jpg?frame=1'),jpeg('blue'))
        self.server.ocr=lambda image,**kwargs:(float('nan'),{})
        self.post(jpeg('red'))
        state=json.loads(self.read('/api/live'));self.assertIsNone(state['value']);self.assertEqual(state['status'],'invalid_value')


if __name__=='__main__':unittest.main()
