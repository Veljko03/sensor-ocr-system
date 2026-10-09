"""Live development OCR preview, with no production database or dataset writes."""
import argparse
import hashlib
import json
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit,parse_qs

from frame_confirmation import FrameConfirmation
from .capture_receiver import Receiver,Handler


class Preview(Receiver):
    def __init__(self,address,root,ocr=None,clock=time.monotonic):
        self.clock=clock
        self.confirmation=FrameConfirmation()
        self.state_lock=threading.Lock()
        self.image=None
        self.state={'frame':0,'value':None,'status':'waiting','received':0,'latency_ms':None,'updated_at':None}
        if ocr is None:
            import cv2
            from ocr_candidate import read_real_meter_image
            cv2.setNumThreads(1)
            ocr=read_real_meter_image
        self.ocr=ocr
        super().__init__(address,root,handler=PreviewHandler,on_frame=self.process,on_rejected=self.invalidate)

    def invalidate(self):
        self.confirmation.invalidate('esp_1')
        with self.state_lock:self.state.update(value=None,status='invalid_image',updated_at=time.time())

    def process(self,data,sequence,received_at):
        # Receiver serializes writes and processing; immutable bytes are decoded
        # directly, never by rereading a mutable latest-image path.
        with self.state_lock:
            self.image=data
            self.state={'frame':sequence,'value':None,'status':'processing','received':sequence,'latency_ms':None,'updated_at':time.time()}
        started=self.clock()
        try:
            import cv2
            import numpy as np
            image=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
            value,_=self.ocr(image,confidence_threshold=.45)
            if value is not None:value=float(value)
        except Exception:
            value=None
        finished=self.clock()
        decision=self.confirmation.observe('esp_1',value,hashlib.sha256(data).hexdigest(),sequence,received_at,finished)
        with self.state_lock:
            self.state.update(value=decision.value,status=decision.reason,latency_ms=round((finished-started)*1000,1))

    def snapshot(self):
        with self.state_lock:state,image=dict(self.state),self.image
        if state['updated_at'] is not None and time.time()-state['updated_at']>15:
            state.update(value=None,status='waiting_new_frame')
        return state,image


class PreviewHandler(Handler):
    upload_status='primljeno_za_ocr_probu'
    def send_bytes(self,status,data,content_type):
        self.send_response(status)
        self.send_header('Content-Type',content_type);self.send_header('Content-Length',str(len(data)))
        self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
        self.end_headers();self.wfile.write(data)

    def do_GET(self):
        parsed=urlsplit(self.path)
        if parsed.path=='/':
            return self.send_bytes(200,(Path(__file__).parent/'live_preview.html').read_bytes(),'text/html; charset=utf-8')
        if parsed.path=='/api/live':
            state,_=self.server.snapshot();return self.respond(200,state)
        if parsed.path=='/frame.jpg':
            state,data=self.server.snapshot()
            try:requested=int(parse_qs(parsed.query).get('frame',['-1'])[0])
            except ValueError:return self.respond(400,{'error':'Invalid frame'})
            if data is None:return self.respond(404,{'error':'No image yet'})
            if requested!=state['frame']:return self.respond(409,{'error':'Frame changed; refresh'})
            return self.send_bytes(200,data,'image/jpeg')
        if parsed.path=='/health':
            state,_=self.server.snapshot()
            return self.respond(200,{'mode':'ocr_live_preview','received':state['received'],'production_db_used':False})
        return self.respond(404,{'error':'Unknown path'})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host',default='127.0.0.1');parser.add_argument('--port',type=int,default=5001)
    parser.add_argument('--root',required=True)
    args=parser.parse_args();server=Preview((args.host,args.port),args.root)
    print(json.dumps({'mode':'ocr_live_preview','address':server.server_address,'production_db_used':False}),flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()


if __name__=='__main__':main()
