import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.request import Request,urlopen
from urllib.error import HTTPError
from PIL import Image
from tools.capture_receiver import Receiver,MAX_BODY


def multipart(data,filename='strujomer_esp_1.jpg'):
    return b'--capture\r\nContent-Disposition: form-data; name="file"; filename="'+filename.encode()+b'"\r\nContent-Type: image/jpeg\r\n\r\n'+data+b'\r\n--capture--\r\n'


class ReceiverTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.server=Receiver(('127.0.0.1',0),self.root/'private_latest')
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.url='http://127.0.0.1:'+str(self.server.server_port)
        stream=io.BytesIO();Image.new('RGB',(60,40),'blue').save(stream,format='JPEG');self.data=stream.getvalue()

    def tearDown(self):self.server.shutdown();self.server.server_close();self.thread.join();self.tmp.cleanup()

    def upload(self,body,path='/upload',**headers):
        request=Request(self.url+path,data=body,headers={'Content-Type':'multipart/form-data; boundary=capture',**headers})
        return urlopen(request,timeout=3)

    def test_real_jpeg_stored_exactly_without_database_or_history(self):
        for _ in range(2):
            with self.upload(multipart(self.data)) as response:
                result=json.load(response);self.assertIsNone(result['ocitano'])
        self.assertEqual((self.root/'private_latest/latest_original_esp_1.jpg').read_bytes(),self.data)
        self.assertEqual(len(list(self.root.rglob('*'))),2)
        with urlopen(self.url+'/health') as response:self.assertEqual(json.load(response)['received'],2)

    def test_invalid_jpeg_or_unknown_device_preserves_previous_original(self):
        with self.upload(multipart(self.data)):pass
        for body,path in [(multipart(b'broken'),'/upload'),(multipart(self.data),'/upload?esp=esp_9'),(multipart(self.data,'esp_2.jpg'),'/upload')]:
            with self.subTest(path=path),self.assertRaises(HTTPError) as error:self.upload(body,path)
            self.assertEqual(error.exception.code,400)
        self.assertEqual((self.root/'private_latest/latest_original_esp_1.jpg').read_bytes(),self.data)

    def test_excessive_declared_body_rejected_before_reading(self):
        with self.assertRaises(HTTPError) as error:self.upload(b'x',**{'Content-Length':str(MAX_BODY+1)})
        self.assertEqual(error.exception.code,413);self.assertEqual(self.server.received,0)


if __name__=='__main__':unittest.main()
