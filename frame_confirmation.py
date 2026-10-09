"""Bounded, per-device confirmation for ordered image ingress.

Caller supplies increasing ingress sequence and monotonic receive time, not OCR
completion time. Distinct JPEG hashes and receive times are replay safeguards;
they cannot authenticate camera capture freshness without a camera protocol.
"""
from collections import deque
from dataclasses import dataclass
import math
import re
import threading


@dataclass(frozen=True)
class Decision:
    confirmed: bool
    value: float | None
    count: int
    reason: str


class FrameConfirmation:
    def __init__(self, devices=('esp_1',), required=2, min_interval=1., max_interval=15., max_age=10., replay_window=64):
        if required<2 or not 0<min_interval<max_interval or max_age<=0 or replay_window<required:
            raise ValueError('Invalid confirmation configuration')
        self.devices=frozenset(devices)
        self.required=required;self.min_interval=min_interval;self.max_interval=max_interval
        self.max_age=max_age;self.replay_window=replay_window
        self._chain={};self._latest={};self._seen={};self._lock=threading.Lock()

    def invalidate(self,device):
        with self._lock:self._chain.pop(device,None)

    def observe(self, device, value, frame_sha256, sequence, received_at, now):
        with self._lock:
            if device not in self.devices:return Decision(False,None,0,'unsupported_device')
            if not isinstance(sequence,int) or isinstance(sequence,bool) or sequence<1:
                self._chain.pop(device,None);return Decision(False,None,0,'invalid_sequence')
            previous_sequence=self._latest.get(device,0)
            if sequence<=previous_sequence:return Decision(False,None,0,'out_of_order')
            self._latest[device]=sequence
            if not all(isinstance(t,(float,int)) and math.isfinite(t) for t in (received_at,now)) or not 0<=now-received_at<=self.max_age:
                self._chain.pop(device,None);return Decision(False,None,0,'stale_processing')
            if not isinstance(frame_sha256,str) or re.fullmatch('[0-9a-f]{64}',frame_sha256) is None:
                self._chain.pop(device,None);return Decision(False,None,0,'invalid_fingerprint')
            seen=self._seen.setdefault(device,deque(maxlen=self.replay_window))
            if frame_sha256 in seen:
                self._chain.pop(device,None);return Decision(False,None,0,'duplicate_frame')
            seen.append(frame_sha256)
            if value is None:
                self._chain.pop(device,None);return Decision(False,None,0,'ocr_rejected')
            if isinstance(value,bool) or not isinstance(value,(float,int)) or not math.isfinite(value) or not 0<=value<=9.99 or abs(value*100-round(value*100))>1e-6:
                self._chain.pop(device,None);return Decision(False,None,0,'invalid_value')
            cents=round(value*100)
            previous=self._chain.get(device)
            if previous is not None and received_at-previous[2]<self.min_interval:
                self._chain.pop(device,None);return Decision(False,None,0,'frames_too_close')
            consecutive=previous is not None and previous[0]==cents and self.min_interval<=received_at-previous[2]<=self.max_interval and sequence==previous[3]+1
            count=min(self.required,previous[1]+1) if consecutive else 1
            self._chain[device]=(cents,count,received_at,sequence)
            confirmed=count>=self.required
            return Decision(confirmed,cents/100 if confirmed else None,count,'confirmed' if confirmed else 'awaiting_confirmation')
