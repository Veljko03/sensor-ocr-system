"""Temporary dataset-only ESP receiver. Never imports main or accesses its DB.

Use on an explicitly chosen free port; stop it before starting the normal app
on that port. Only the latest original JPEG is retained here; the separate
controlled collector saves selected sessions with quotas and human labels.
"""
import argparse
from email.parser import BytesParser
from email.policy import default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit,parse_qs
from image_ingest import identify_device,replace_latest
from tools.dataset_store import MAX_IMAGE_BYTES

MAX_BODY=MAX_IMAGE_BYTES+128*1024


def parse_upload(content_type,body,explicit=None):
    if len(body)>MAX_BODY:raise ValueError('Body exceeds upload limit')
    if not content_type.lower().startswith('multipart/form-data') or '\r' in content_type or '\n' in content_type:
        raise ValueError('Expected multipart upload')
    message=BytesParser(policy=default).parsebytes(('Content-Type: '+content_type+'\r\nMIME-Version: 1.0\r\n\r\n').encode()+body)
    if not message.is_multipart():raise ValueError('Invalid multipart body')
    parts=list(message.iter_parts())
    files=[p for p in parts if p.get_param('name',header='content-disposition')=='file' and p.get_filename()]
    if len(files)!=1 or len(parts)>4:raise ValueError('Expected exactly one file')
    file=files[0];device=identify_device(file.get_filename(),explicit)
    if device!='esp_1':raise ValueError('Only esp_1 capture is configured')
    data=file.get_payload(decode=True)
    if not isinstance(data,bytes):raise ValueError('Invalid file payload')
    return device,data


class Receiver(ThreadingHTTPServer):
    daemon_threads=True
    def __init__(self,address,root,handler=None,on_frame=None,on_rejected=None):
        self.root=Path(root).resolve();self.root.mkdir(parents=True,exist_ok=True,mode=0o700)
        self.write_lock=threading.Lock();self.last=None;self.received=0
        self.on_frame=on_frame
        self.on_rejected=on_rejected
        super().__init__(address,handler or Handler)


class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def setup(self):
        super().setup();self.connection.settimeout(5)

    def respond(self,status,data):
        encoded=json.dumps(data).encode()
        self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(encoded)))
        self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(encoded)

    def do_GET(self):
        if urlsplit(self.path).path!='/health':return self.respond(404,{'error':'Unknown path'})
        self.respond(200,{'mode':'dataset_capture_only','received':self.server.received,'production_db_used':False})

    def reject_upload(self,status,message):
        if self.server.on_rejected is not None:
            with self.server.write_lock:self.server.on_rejected()
        self.respond(status,{'error':message})

    def do_POST(self):
        received_at=time.monotonic()
        parsed=urlsplit(self.path)
        if parsed.path!='/upload':return self.respond(404,{'error':'Unknown path'})
        if self.headers.get('Transfer-Encoding'):
            self.close_connection=True;return self.reject_upload(400,'Content-Length required')
        try:size=int(self.headers.get('Content-Length','0'))
        except ValueError:size=0
        if not 0<size<=MAX_BODY:
            self.close_connection=True;return self.reject_upload(413,'Invalid or excessive body size')
        try:
            body=self.rfile.read(size)
            if len(body)!=size:raise ValueError('Incomplete upload body')
            query=parse_qs(parsed.query)
            if set(query)-{'esp'} or len(query.get('esp',[]))>1:raise ValueError('Invalid device query')
            device,data=parse_upload(self.headers.get('Content-Type',''),body,query.get('esp',[None])[0])
            with self.server.write_lock:
                latest=replace_latest(self.server.root,device,data)
                self.server.last=latest;self.server.received+=1
                if self.server.on_frame is not None:
                    self.server.on_frame(data,self.server.received,received_at)
            self.respond(200,{'status':getattr(self,'upload_status','primljeno_za_dataset'),'esp_id':device,'slika':latest.name,'ocitano':None})
        except (ValueError,TypeError):self.reject_upload(400,'Invalid esp_1 JPEG upload')
        except OSError:self.reject_upload(503,'Image storage unavailable')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,help='Private latest-image directory, outside production')
    parser.add_argument('--host',default='127.0.0.1');parser.add_argument('--port',type=int,default=5001)
    args=parser.parse_args();server=Receiver((args.host,args.port),args.root)
    print(json.dumps({'mode':'dataset_capture_only','address':server.server_address}),flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()


if __name__=='__main__':main()
