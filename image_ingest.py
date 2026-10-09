"""Isolated upload primitives; no production paths or retention deletion."""
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
from tools.dataset_store import MAX_IMAGE_BYTES, verify_jpeg


DEVICES={'esp_1','esp_2'}


def identify_device(filename, explicit=None):
    if explicit is not None:
        if explicit not in DEVICES:raise ValueError('Unknown explicit device')
        return explicit
    import re
    name=Path((filename or '').replace('\\','/')).name.lower()
    found=set(re.findall(r'(?<![a-z0-9])(?:esp|sat)_([12])(?![a-z0-9])',name))
    if len(found)!=1:raise ValueError('Filename must identify exactly one configured device')
    return 'esp_'+found.pop()


async def read_bounded_upload(upload):
    content=bytearray()
    try:
        while True:
            chunk=await upload.read(min(64*1024,MAX_IMAGE_BYTES+1-len(content)))
            if not chunk:break
            content.extend(chunk)
            if len(content)>MAX_IMAGE_BYTES:raise ValueError('Image exceeds byte limit')
        data=bytes(content)
        verify_jpeg(data)
        return data,hashlib.sha256(data).hexdigest()
    finally:
        await upload.close()


def replace_latest(root,device,data,min_free_bytes=64*1024*1024):
    if device not in DEVICES:raise ValueError('Unknown device')
    verify_jpeg(data)
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(root).free<len(data)+min_free_bytes:raise OSError('Insufficient free disk space')
    target=root/('latest_original_'+device+'.jpg')
    fd,name=tempfile.mkstemp(prefix='.incoming-',suffix='.jpg',dir=root)
    temporary=Path(name)
    try:
        with os.fdopen(fd,'wb') as file:
            file.write(data);file.flush();os.fsync(file.fileno())
        os.replace(temporary,target)
        return target
    finally:
        temporary.unlink(missing_ok=True)
