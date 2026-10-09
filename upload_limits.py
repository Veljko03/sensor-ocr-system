"""Bound the multipart request before Starlette can spool a large upload."""
from fastapi.responses import JSONResponse

MAX_UPLOAD_BODY = 2 * 1024 * 1024 + 128 * 1024


class UploadBodyLimit:
    def __init__(self, app, limit=MAX_UPLOAD_BODY, on_rejected=None):
        self.app=app;self.limit=limit;self.on_rejected=on_rejected

    async def reject(self,scope,receive,send,message):
        if self.on_rejected is not None:await self.on_rejected(scope)
        await JSONResponse({'detail':message},status_code=413)(scope,receive,send)

    async def __call__(self, scope, receive, send):
        if scope['type']!='http' or scope['method']!='POST' or scope['path']!='/upload':
            return await self.app(scope,receive,send)
        headers=dict(scope.get('headers',[]))
        declared=headers.get(b'content-length')
        if declared is not None:
            try:valid=0<int(declared)<=self.limit
            except ValueError:valid=False
            if not valid:
                return await self.reject(scope,receive,send,'Invalid or excessive upload size')
        # Finish bounded ingress before multipart parsing, so rejecting a
        # chunked body cannot leave partly opened spool files behind.
        body=bytearray()
        while True:
            message=await receive()
            if message['type']=='http.disconnect':return
            chunk=message.get('body',b'')
            if len(body)+len(chunk)>self.limit:
                return await self.reject(scope,receive,send,'Upload exceeds body limit')
            body.extend(chunk)
            if not message.get('more_body',False):break
        delivered=False
        async def bounded_receive():
            nonlocal delivered
            if not delivered:
                delivered=True
                return {'type':'http.request','body':bytes(body),'more_body':False}
            return await receive()
        await self.app(scope,bounded_receive,send)
