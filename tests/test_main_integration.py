"""Real FastAPI routes and SQLite, isolated from production paths."""
import asyncio
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import threading
import time
import unittest
from contextlib import closing
from unittest.mock import patch
from PIL import Image


def jpeg(color):
    output=io.BytesIO();Image.new('RGB',(60,40),color).save(output,format='JPEG')
    return output.getvalue()


def multipart(data,filename='strujomer_esp_1.jpg'):
    return b'--main\r\nContent-Disposition: form-data; name="file"; filename="'+filename.encode()+b'"\r\nContent-Type: image/jpeg\r\n\r\n'+data+b'\r\n--main--\r\n'


async def request(app,path,body=b'',headers=None,chunks=None,method=None):
    clean,_,query=path.partition('?')
    scope={'type':'http','asgi':{'version':'3.0'},'http_version':'1.1',
           'method':method or ('POST' if body or chunks else 'GET'),'scheme':'http',
           'path':clean,'raw_path':clean.encode(),'query_string':query.encode(),
           'root_path':'','headers':headers or [],'client':('127.0.0.1',1000),
           'server':('127.0.0.1',5000)}
    messages=[];packets=list(chunks or [body]);index=0
    async def receive():
        nonlocal index
        if index>=len(packets):return {'type':'http.disconnect'}
        chunk=packets[index];index+=1
        return {'type':'http.request','body':chunk,'more_body':index<len(packets)}
    async def send(message):messages.append(message)
    await app(scope,receive,send)
    start=next(m for m in messages if m['type']=='http.response.start')
    payload=b''.join(m.get('body',b'') for m in messages if m['type']=='http.response.body')
    return start['status'],payload


class MainIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        spec=importlib.util.spec_from_file_location('isolated_main',Path(__file__).resolve().parents[1]/'main.py')
        self.main=importlib.util.module_from_spec(spec)
        with patch.dict(os.environ,{'SENSOR_OCR_DATA_DIR':str(self.root)}):spec.loader.exec_module(self.main)
        self.clock=100.
        self.main.OCR_CONFIRMATION.min_interval=.001
        self.main.ocr_reading=lambda data,device:{'value':0. if device=='esp_1' else None,'raw':'test','engine':'test','ok':device=='esp_1'}

    def tearDown(self):self.tmp.cleanup()

    def send(self,data,filename='strujomer_esp_1.jpg',path='/upload'):
        body=multipart(data,filename)
        with patch.object(self.main.time,'monotonic',return_value=self.clock):
            status,payload=asyncio.run(request(self.main.app,path,body,[(b'content-type',b'multipart/form-data; boundary=main'),(b'content-length',str(len(body)).encode())]))
        return status,json.loads(payload)

    def api(self):
        with patch.object(self.main.time,'monotonic',return_value=self.clock):
            status,body=asyncio.run(request(self.main.app,'/api/podaci'))
        self.assertEqual(status,200);return json.loads(body)

    def records(self):
        with closing(sqlite3.connect(self.main.DB_PATH)) as conn:return conn.execute('SELECT vrednost,ocr_ok,original_slika FROM merenja ORDER BY id').fetchall()

    def test_zero_confirmation_original_and_failure_history(self):
        self.assertEqual(self.send(jpeg('blue'))[0],200)
        self.assertIsNone(self.api()['trenutno']['sat_1'])
        self.clock=105.;status,result=self.send(jpeg('red'))
        self.assertEqual(result['ocitano'],0.);self.assertEqual(self.api()['trenutno']['sat_1'],0.)
        self.assertEqual((self.root/'pictures/latest_original_esp_1.jpg').read_bytes(),jpeg('red'))
        self.main.ocr_reading=lambda *args:{'value':None,'raw':'reject','ok':False}
        self.clock=110.;self.send(jpeg('green'))
        api=self.api();self.assertIsNone(api['trenutno']['sat_1'])
        self.assertEqual(len(api['istorija']['sat_1']),1)
        self.assertEqual(self.records(),[(None,0,'latest_original_esp_1.jpg'),(0.,1,'latest_original_esp_1.jpg'),(None,0,'latest_original_esp_1.jpg')])

    def test_changed_nonfinite_and_duplicate_values_fail_closed(self):
        self.send(jpeg('blue'));self.clock=105.;self.send(jpeg('red'))
        self.clock=110.;self.send(jpeg('red'));self.assertIsNone(self.api()['trenutno']['sat_1'])
        for index,value in enumerate((1.61,float('nan'),float('inf'),True,10.,-1.)):
            self.main.ocr_reading=lambda *args,v=value:{'value':v,'ok':True,'raw':'test'}
            self.clock+=5;self.send(jpeg((index*30,80,100)))
            self.assertIsNone(self.api()['trenutno']['sat_1'])
        self.assertEqual(sum(ok for value,ok,_ in self.records()),1)

    def test_invalid_identity_or_jpeg_preserves_last_original(self):
        self.send(jpeg('blue'));self.clock=105.;self.send(jpeg('red'))
        for filename,path in [('unknown.jpg','/upload'),('esp_12.jpg','/upload'),('strujomer_esp_1.jpg','/upload?esp=unknown')]:
            self.assertEqual(self.send(jpeg('green'),filename,path)[0],400)
        self.clock=110.;self.assertEqual(self.send(b'not a jpeg')[0],400)
        self.assertIsNone(self.api()['trenutno']['sat_1'])
        self.assertEqual((self.root/'pictures/latest_original_esp_1.jpg').read_bytes(),jpeg('red'))
        self.clock=115.;self.send(jpeg('green'));self.assertIsNone(self.api()['trenutno']['sat_1'])

    def test_esp2_is_unconfigured_and_does_not_confirm_esp1(self):
        self.send(jpeg('blue'));self.clock=105.;self.send(jpeg('red'),'strujomer_esp_2.jpg')
        self.assertIsNone(self.api()['trenutno']['sat_2'])
        self.clock=110.;self.send(jpeg('green'));self.assertEqual(self.api()['trenutno']['sat_1'],0.)

    def test_expiry_restart_and_frame_race(self):
        self.send(jpeg('blue'));self.clock=105.;self.send(jpeg('red'))
        status,data=asyncio.run(request(self.main.app,'/api/frame/esp_1?sequence=2'))
        self.assertEqual((status,data),(200,jpeg('red')))
        self.assertEqual(asyncio.run(request(self.main.app,'/api/frame/esp_1?sequence=1'))[0],409)
        self.clock=121.;self.assertIsNone(self.api()['trenutno']['sat_1'])
        self.main.LIVE_FRAMES.clear();self.assertIsNone(self.api()['trenutno']['sat_1'])
        self.assertEqual(len(self.api()['istorija']['sat_1']),1)

    def test_declared_and_chunked_body_limits(self):
        from upload_limits import MAX_UPLOAD_BODY
        oversized=b'x'*(MAX_UPLOAD_BODY+1)
        status,_=asyncio.run(request(self.main.app,'/upload',b'x',[(b'content-length',str(len(oversized)).encode())]))
        self.assertEqual(status,413)
        status,_=asyncio.run(request(self.main.app,'/upload',headers=[(b'content-type',b'multipart/form-data; boundary=main')],chunks=[multipart(b'x')[:90],oversized]))
        self.assertEqual(status,413);self.assertEqual(self.records(),[])

    def test_oversized_body_clears_active_confirmation(self):
        self.send(jpeg('blue'));self.clock=105.;self.send(jpeg('red'))
        self.assertEqual(self.api()['trenutno']['sat_1'],0.)
        status,_=asyncio.run(request(self.main.app,'/upload',b'x',[(b'content-length',b'99999999')]))
        self.assertEqual(status,413);self.assertIsNone(self.api()['trenutno']['sat_1'])
        self.clock=110.;self.send(jpeg('green'));self.assertIsNone(self.api()['trenutno']['sat_1'])

    def test_explicit_delete_clears_current_values_only_in_isolated_test_database(self):
        self.send(jpeg('blue'));self.clock=105.;self.send(jpeg('red'))
        status,_=asyncio.run(request(self.main.app,'/api/obrisi',method='DELETE'))
        self.assertEqual(status,200);self.assertEqual(self.records(),[])
        self.assertIsNone(self.api()['trenutno']['sat_1'])
        self.assertEqual(self.api()['slike']['esp_1']['sequence'],0)

    def test_storage_and_database_failures_never_show_a_valid_number(self):
        self.send(jpeg('blue'));self.clock=105.;self.send(jpeg('red'))
        self.clock=110.
        with patch.object(self.main,'replace_latest',side_effect=OSError('test disk error')):
            self.assertEqual(self.send(jpeg('green'))[0],503)
        self.assertIsNone(self.api()['trenutno']['sat_1'])
        self.assertEqual((self.root/'pictures/latest_original_esp_1.jpg').read_bytes(),jpeg('red'))
        self.clock=115.;self.send(jpeg('green'))
        self.clock=120.
        with patch.object(self.main,'insert_measurement',side_effect=sqlite3.OperationalError('test DB error')):
            self.assertEqual(self.send(jpeg('yellow'))[0],503)
        self.assertIsNone(self.api()['trenutno']['sat_1'])
        self.assertEqual(sum(row[1] for row in self.records()),1)

    def test_processing_clears_value_and_duplicates_cannot_double_confirm(self):
        data=jpeg('blue');entered=threading.Event();release=threading.Event()
        def slow(*args):entered.set();release.wait(3);return {'value':0.,'raw':'test','ok':True}
        self.main.ocr_reading=slow
        async def concurrent():
            body=multipart(data);headers=[(b'content-type',b'multipart/form-data; boundary=main')]
            first=asyncio.create_task(request(self.main.app,'/upload',body,headers))
            await asyncio.to_thread(entered.wait,2)
            self.assertTrue(entered.is_set())
            status,payload=await request(self.main.app,'/api/podaci')
            self.assertIsNone(json.loads(payload)['trenutno']['sat_1'])
            second=asyncio.create_task(request(self.main.app,'/upload',body,headers))
            release.set();await asyncio.gather(first,second)
        asyncio.run(concurrent());self.assertEqual([r[1] for r in self.records()],[0,0])

    def test_late_older_ingress_cannot_overwrite_newer_frame(self):
        original=self.main.read_bounded_upload
        async def scenario():
            first_entered=asyncio.Event();release=asyncio.Event()
            async def delayed(upload):
                data,digest=await original(upload)
                if data==jpeg('blue'):first_entered.set();await release.wait()
                return data,digest
            self.main.read_bounded_upload=delayed
            headers=[(b'content-type',b'multipart/form-data; boundary=main')]
            first=asyncio.create_task(request(self.main.app,'/upload',multipart(jpeg('blue')),headers))
            await first_entered.wait()
            newer=await request(self.main.app,'/upload',multipart(jpeg('red')),headers)
            release.set();older=await first
            self.assertEqual(newer[0],200);self.assertEqual(older[0],409)
        asyncio.run(scenario())
        self.assertEqual((self.root/'pictures/latest_original_esp_1.jpg').read_bytes(),jpeg('red'))
        self.assertEqual(len(self.records()),1)


if __name__=='__main__':unittest.main()
