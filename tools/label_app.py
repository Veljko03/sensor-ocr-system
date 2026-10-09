"""Loopback-only, human-ground-truth labeler with controlled capture."""
import argparse
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
from urllib.parse import parse_qs, urlparse
from .dataset_store import Store, ConflictError, import_legacy
from .capture import Capture


class Server(ThreadingHTTPServer):
    daemon_threads = True
    def __init__(self, address, store, capture):
        if address[0] not in {'127.0.0.1', 'localhost'}:
            raise ValueError('Labeler must bind loopback. Use an SSH tunnel.')
        self.store, self.capture = store, capture
        self.token = secrets.token_urlsafe(32)
        self.page = (Path(__file__).parent / 'label_ui.html').read_bytes()
        super().__init__(address, Handler)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send(self, data, status=200, content_type='application/json; charset=utf-8', attachment=None):
        if not isinstance(data, bytes):
            data = json.dumps(data, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('X-Frame-Options', 'DENY')
        self.send_header('Referrer-Policy', 'no-referrer')
        if attachment:
            self.send_header('Content-Disposition', 'attachment; filename="'+attachment+'"')
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if urlparse('http://' + self.headers.get('Host','')).hostname not in {'127.0.0.1','localhost'}:
            self.send({'error':'Invalid local host'},403)
            return
        path = urlparse(self.path).path
        try:
            if path == '/':
                self.send(self.server.page, content_type='text/html; charset=utf-8')
            elif path == '/api/status':
                self.send({'dataset':self.server.store.summary(), 'capture':self.server.capture.status(),
                           'csrf':self.server.token, 'schema_version':1})
            elif path == '/api/items':
                self.send({'items':self.server.store.rows()})
            elif path.startswith('/image/'):
                self.send(self.server.store.image_path(path[7:]).read_bytes(), content_type='image/jpeg')
            elif path == '/export.csv':
                self.send(self.server.store.export_csv().encode('utf-8'), content_type='text/csv; charset=utf-8',
                          attachment='festo_manifest.csv')
            else:
                self.send({'error':'Not found'},404)
        except (ValueError, FileNotFoundError):
            self.send({'error':'Image not found'},404)

    def do_POST(self):
        try:
            if urlparse('http://' + self.headers.get('Host','')).hostname not in {'127.0.0.1','localhost'}:
                self.send({'error':'Invalid local host'},403)
                return
            token = self.headers.get('X-CSRF-Token', '')
            if not hmac.compare_digest(token, self.server.token):
                self.send({'error':'Invalid session token'},403)
                return
            origin = self.headers.get('Origin')
            if origin and urlparse(origin).netloc != self.headers.get('Host'):
                self.send({'error':'Cross-origin request refused'},403)
                return
            length = int(self.headers.get('Content-Length','0'))
            if not 0 < length <= 16384:
                raise ValueError('Request size must be 1–16384 bytes')
            data = json.loads(self.rfile.read(length))
            if not isinstance(data,dict):
                raise ValueError('Expected JSON object')
            if self.path == '/api/label':
                self.server.store.label(data['id'],data['class'],data.get('value',''),
                                        data.get('note',''),data.get('revision',0))
                self.send({'saved':True})
            elif self.path == '/api/capture/start':
                allowed={'group_id','split','scenario','angle','distance','lighting','max_images','max_bytes'}
                if set(data)-allowed:
                    raise ValueError('Unknown capture option')
                self.send(self.server.capture.start(**data))
            elif self.path == '/api/capture/stop':
                self.send(self.server.capture.stop())
            else:
                self.send({'error':'Not found'},404)
        except ConflictError as exc:
            self.send({'error':str(exc)},409)
        except (KeyError, ValueError, TypeError) as exc:
            self.send({'error':str(exc)},400)
        except Exception:
            self.send({'error':'Čuvanje nije uspelo; postojeće oznake su sačuvane. Proveri prostor i dnevnik.'},500)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--source',required=True,help='Read-only path to production latest_original_esp_1.jpg')
    parser.add_argument('--port',type=int,default=8001)
    parser.add_argument('--host',choices=['127.0.0.1'],default='127.0.0.1')
    parser.add_argument('--import-legacy',metavar='PROJECT')
    args=parser.parse_args()
    store=Store(args.dataset)
    if args.import_legacy:
        print(json.dumps(import_legacy(store,args.import_legacy)),flush=True)
    capture=Capture(store,args.source)
    server=Server((args.host,args.port),store,capture)
    capture.launch()
    print('Labeler listening on http://127.0.0.1:'+str(args.port)+'; capture is OFF.',flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        capture.close()
        server.server_close()


if __name__=='__main__':
    main()
