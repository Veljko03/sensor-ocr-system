import csv
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from PIL import Image
from tools.dataset_store import Store, ConflictError, LimitError, normalize_value, import_legacy
from tools.dataset import validate
from tools.capture import Capture
from tools.label_app import Server
from tools.evaluate import summarize


def jpeg(color):
    image=Image.new('RGB',(60,40),color)
    output=io.BytesIO();image.save(output,format='JPEG');return output.getvalue()


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.store=Store(self.root/'staging')
        self.session=self.store.create_session(group_id='group-a')

    def tearDown(self):
        self.tmp.cleanup()

    def add(self,color='blue',session=None,**limits):
        return self.store.ingest(jpeg(color),session or self.session,'esp_1','original.jpg',**limits)[0]

    def test_values_zero_and_device_range(self):
        for value,expected in [('000','0.00'),('0.00','0.00'),('999','9.99'),('6,62','6.62')]:
            self.assertEqual(normalize_value(value),expected)
        for value in ['10.00','1000','-1.00','NaN','inf','1.2','00.00','1e2','']:
            with self.assertRaises(ValueError): normalize_value(value)

    def test_negative_label_is_not_zero(self):
        image=self.add();self.store.label(image,'absent','0.00')
        row=self.store.rows()[0];self.assertIsNone(row['value']);self.assertEqual(row['class'],'absent')

    def test_edit_is_transactional_and_stale_edit_refused(self):
        image=self.add();self.store.label(image,'readable','0.00')
        with self.assertRaises(ConflictError): self.store.label(image,'readable','6.62',revision=0)
        self.assertEqual(self.store.rows()[0]['value'],'0.00')
        self.store.label(image,'readable','6.62',revision=1)
        self.assertEqual(self.store.rows()[0]['revision'],2)

    def test_resume_and_duplicate_ingest(self):
        image=self.add();self.store.label(image,'skip',note='review')
        other=Store(self.store.root)
        image_again,created=other.ingest(jpeg('blue'),self.session,'esp_1','duplicate.jpg')
        self.assertEqual(image,image_again);self.assertFalse(created)
        self.assertEqual(other.summary()['total'],1);self.assertEqual(other.summary()['skipped'],1)

    def test_invalid_jpeg_rejected_without_record(self):
        for data in [b'',b'garbage',jpeg('blue')[:-3],b'\xff\xd8junk\xff\xd9']:
            with self.assertRaises((ValueError,OSError)):
                self.store.ingest(data,self.session,'esp_1','bad.jpg')
        self.assertEqual(self.store.summary()['total'],0)

    def test_other_device_not_assumed_festo(self):
        with self.assertRaises(ValueError):self.store.ingest(jpeg('blue'),self.session,'esp_2','image.jpg')

    def test_session_count_disk_and_free_space_limits(self):
        self.add(max_images=1)
        with self.assertRaises(LimitError):self.add('red',max_images=1)
        with patch('tools.dataset_store.shutil.disk_usage') as disk:
            disk.return_value.free=0
            with self.assertRaises(LimitError):self.add('green')
        # Allocate a file in the test store only; existing data must never be deleted to satisfy quota.
        reserve=self.store.root/'preserved.bin';reserve.write_bytes(b'x'*(2*1024**2))
        with self.assertRaises(LimitError):self.add('yellow',max_bytes=2*1024**2)
        self.assertTrue(reserve.exists());self.assertEqual(self.store.summary()['total'],1)

    def test_concurrent_duplicates_one_record(self):
        barrier=threading.Barrier(4);results=[];errors=[]
        def ingest():
            try:
                barrier.wait();results.append(self.store.ingest(jpeg('red'),self.session,'esp_1','same.jpg'))
            except Exception as exc:errors.append(exc)
        threads=[threading.Thread(target=ingest) for _ in range(4)]
        for t in threads:t.start()
        for t in threads:t.join()
        self.assertFalse(errors);self.assertEqual(sum(r[1] for r in results),1)
        self.assertEqual(self.store.summary()['total'],1)

    def test_group_cannot_cross_partitions(self):
        with self.assertRaises(ValueError):self.store.create_session(group_id='group-a',split='test')

    def test_image_path_no_traversal(self):
        for path in ['../manifest.sqlite3','a'*32,'/etc/passwd']:
            with self.assertRaises(ValueError):self.store.image_path(path)

    def test_export_and_checksum_validation(self):
        image=self.add();self.store.label(image,'readable','0.00')
        data=list(csv.DictReader(io.StringIO(self.store.export_csv())))
        self.assertEqual(data[0]['value'],'0.00');self.assertTrue(validate(self.store)['valid'])
        self.store.image_path(image).write_bytes(jpeg('red'))
        self.assertFalse(validate(self.store)['valid'])

    def test_historical_import_preserves_sources_and_human_edits(self):
        project=self.root/'project';images=project/'dataset/real_meter/images';images.mkdir(parents=True)
        pictures=project/'pictures';pictures.mkdir()
        rows=[('20260910_130740_esp_1.jpg','0.00','1','','blue'),
              ('20260910_130745_esp_1.jpg','','0','no_meter','red')]
        for name,value,readable,note,color in rows:(images/name).write_bytes(jpeg(color))
        manifest=images.parent/'labels.csv'
        with manifest.open('w',newline='') as file:
            writer=csv.writer(file);writer.writerow(['filename','display_text','readable','note'])
            writer.writerows(r[:4] for r in rows)
        original=manifest.read_bytes()
        (pictures/'20261007_134003_esp_1.jpg').write_bytes(jpeg('green'))
        result=import_legacy(self.store,project)
        self.assertEqual(result['historical_labeled'],2);self.assertEqual(result['historical_unlabeled'],1)
        record=next(r for r in self.store.rows() if r['class']=='readable')
        self.store.label(record['id'],'readable','6.62',revision=1)
        import_legacy(self.store,project)
        self.assertEqual(manifest.read_bytes(),original)
        self.assertEqual(next(r for r in self.store.rows() if r['id']==record['id'])['value'],'6.62')
        self.assertEqual(self.store.summary()['total'],3)

    def test_watcher_requires_start_and_new_frame_and_stops_at_limit(self):
        source=self.root/'latest.jpg';source.write_bytes(jpeg('blue'))
        capture=Capture(self.store,source);capture.poll();self.assertEqual(self.store.summary()['total'],0)
        capture.start(max_images=1);capture.poll();self.assertEqual(self.store.summary()['total'],0)
        time.sleep(.01);source.write_bytes(jpeg('red'));capture.poll()
        self.assertEqual(self.store.summary()['total'],1);self.assertFalse(capture.status()['active'])
        source.write_bytes(jpeg('green'));capture.poll();self.assertEqual(self.store.summary()['total'],1)
        self.assertEqual(source.read_bytes(),jpeg('green'))

    def test_watcher_rejects_partial_jpeg_then_recovers(self):
        source=self.root/'latest.jpg';capture=Capture(self.store,source);capture.start()
        source.write_bytes(jpeg('blue')[:60]);capture.poll();self.assertEqual(self.store.summary()['total'],0)
        source.write_bytes(jpeg('red'));capture.poll();self.assertEqual(self.store.summary()['total'],1)
        capture.stop()

    def test_metrics_wrong_zero_is_false_accept_not_success(self):
        report=summarize([{'class':'absent','value':None,'prediction':'0.00'},
                          {'class':'readable','value':'0.00','prediction':'0.00'},
                          {'class':'readable','value':'6.62','prediction':None}])
        self.assertEqual(report['incorrectly_accepted'],1)
        self.assertEqual(report['correctly_accepted'],1)
        self.assertEqual(report['false_acceptance_rate_no_display'],1)
        self.assertEqual(report['correct_acceptance_rate'],.5)


class HttpTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.store=Store(self.root/'data');session=self.store.create_session()
        self.image=self.store.ingest(jpeg('blue'),session,'esp_1','synthetic.jpg')[0]
        self.capture=Capture(self.store,self.root/'source.jpg')
        self.server=Server(('127.0.0.1',0),self.store,self.capture)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.base='http://127.0.0.1:'+str(self.server.server_port)

    def tearDown(self):
        self.server.shutdown();self.server.server_close();self.thread.join();self.tmp.cleanup()

    def request(self,path,data=None,token=True,headers=None):
        hdr={'Content-Type':'application/json'}
        if token:hdr['X-CSRF-Token']=self.server.token
        hdr.update(headers or {})
        request=Request(self.base+path,data=json.dumps(data).encode() if data is not None else None,headers=hdr)
        return urlopen(request,timeout=5)

    def test_label_save_zero_edit_and_negative_via_http(self):
        self.request('/api/label',{'id':self.image,'class':'readable','value':'0.00'}).close()
        self.assertEqual(self.store.rows()[0]['value'],'0.00')
        self.request('/api/label',{'id':self.image,'class':'unreadable','revision':1}).close()
        self.assertIsNone(self.store.rows()[0]['value'])
        with self.assertRaises(HTTPError) as error:
            self.request('/api/label',{'id':self.image,'class':'readable','value':'9.99','revision':1})
        self.assertEqual(error.exception.code,409)

    def test_no_token_cross_origin_and_invalid_host_refused(self):
        for kwargs in [{'token':False},{'headers':{'Origin':'http://example.com'}},{'headers':{'Host':'example.com'}}]:
            with self.assertRaises(HTTPError) as error:
                self.request('/api/capture/start',{},**kwargs)
            self.assertEqual(error.exception.code,403)
        self.assertFalse(self.capture.active)

    def test_invalid_numeric_value_and_arbitrary_capture_source_refused(self):
        for path,data in [('/api/label',{'id':self.image,'class':'readable','value':'10.00'}),
                          ('/api/capture/start',{'source':'/etc/passwd'})]:
            with self.assertRaises(HTTPError) as error:self.request(path,data)
            self.assertEqual(error.exception.code,400)
        self.assertIsNone(self.store.rows()[0]['class'])

    def test_page_images_counts_export_and_loopback(self):
        with self.request('/') as response:self.assertIn(b'Alt +',response.read())
        with self.request('/image/'+self.image) as response:self.assertEqual(response.read(),jpeg('blue'))
        with self.request('/api/status') as response:self.assertEqual(json.load(response)['dataset']['total'],1)
        with self.request('/export.csv') as response:self.assertIn(b'group_id',response.read())
        with self.assertRaises(ValueError):Server(('0.0.0.0',0),self.store,self.capture)


if __name__=='__main__':
    unittest.main()
