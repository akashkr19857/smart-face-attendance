"""Threaded frame grabber for USB webcams, RTSP/HTTP streams, GStreamer pipelines and video files.

Only the newest frame is kept, so a slow consumer never builds up lag. Streams reconnect
automatically with exponential backoff.
"""
import os
import threading
import time

# Must be set before the first FFMPEG capture is opened: TCP transport, 5 s socket timeout.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|stimeout;5000000|timeout;5000000")
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")  # much faster MSMF webcam open on Windows

import cv2
from loguru import logger


class Camera:
    def __init__(self, cam_id, source, width=1280, height=720, fps=30):
        self.id = cam_id
        if isinstance(source, str):
            source = source.strip()
            source = int(source) if source.isdigit() else os.path.expanduser(source)  # "~/video.mp4" works too
        self.source = source
        self.width, self.height, self.fps_target = width, height, fps
        self.connected = False
        self.input_fps = 0.0
        self.paused = False      # video files only: hold the current frame
        self.position = 0.0      # video files: seconds played / total length
        self.duration = 0.0
        self._rewind = False

        self._frame = None
        self._seq = 0
        self._ts = 0.0
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"cam-{cam_id}", daemon=True)

    @property
    def kind(self):
        s = self.source
        if isinstance(s, int):
            return "usb"
        if "!" in s:
            return "gstreamer"
        if s.lower().startswith(("rtsp://", "rtmp://", "http://", "https://")):
            return "stream"
        return "file"

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        self._thread.join(timeout=3)

    def pause(self, paused=True):
        self.paused = bool(paused) and self.kind == "file"

    def rewind(self):
        """Video files: start again from the first frame."""
        self._rewind = True
        self.paused = False

    def latest(self):
        """Returns (seq, frame, timestamp) of the newest frame, frame is None until connected."""
        with self._cond:
            return self._seq, self._frame, self._ts

    def wait(self, last_seq, timeout=1.0):
        """Blocks until a frame newer than last_seq arrives. Returns (seq, frame, ts) or None."""
        with self._cond:
            if self._seq == last_seq:
                self._cond.wait(timeout)
            if self._seq == last_seq or self._frame is None:
                return None
            return self._seq, self._frame, self._ts

    def _open(self):
        kind = self.kind
        if kind == "usb":
            # Windows: MSMF delivers 720p at full rate (DirectShow often falls back to ~10 fps YUY2)
            for api in ((cv2.CAP_MSMF, cv2.CAP_DSHOW) if os.name == "nt" else (cv2.CAP_V4L2, cv2.CAP_ANY)):
                cap = cv2.VideoCapture(self.source, api)
                if cap.isOpened():
                    break
                cap.release()
            # MJPG lets most webcams deliver 720p/1080p at full frame rate
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps_target)
            if api == cv2.CAP_MSMF:
                # MSMF opens many webcams with a fixed short exposure (almost black image); turn auto-exposure on
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        elif kind == "gstreamer":
            cap = cv2.VideoCapture(self.source, cv2.CAP_GSTREAMER)
        elif kind == "stream":
            cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        else:
            cap = cv2.VideoCapture(self.source)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            cap.release()
            return None
        return cap

    def _publish(self, frame):
        with self._cond:
            self._frame = frame
            self._seq += 1
            self._ts = time.time()
            self._cond.notify_all()

    def _run(self):
        backoff = 1.0
        while not self._stop.is_set():
            cap = self._open()
            if cap is None:
                logger.warning("[{}] cannot open source {} - retrying in {:.0f}s", self.id, self._safe_source(), backoff)
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 30.0)
                continue

            backoff = 1.0
            self.connected = True
            is_file = self.kind == "file"
            frame_period = 1.0 / (cap.get(cv2.CAP_PROP_FPS) or 25.0) if is_file else 0.0
            if is_file:
                self.duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) * frame_period
            logger.info("[{}] connected: {} ({}x{})", self.id, self._safe_source(),
                        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

            count, t_count, failures = 0, time.time(), 0
            while not self._stop.is_set():
                if is_file and self._rewind:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self._rewind = False
                if self.paused:
                    self.input_fps = 0.0
                    self._stop.wait(0.1)
                    continue
                t_read = time.time()
                ok, frame = cap.read()
                if not ok or frame is None:
                    if is_file and failures == 0:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop test videos
                        failures += 1
                        continue
                    break
                failures = 0
                self._publish(frame)
                if is_file:
                    self.position = cap.get(cv2.CAP_PROP_POS_FRAMES) * frame_period

                count += 1
                now = time.time()
                if now - t_count >= 1.0:
                    self.input_fps = count / (now - t_count)
                    count, t_count = 0, now
                if frame_period:
                    self._stop.wait(max(0.0, frame_period - (time.time() - t_read)))

            cap.release()
            self.connected = False
            self.input_fps = 0.0
            if not self._stop.is_set():
                logger.warning("[{}] stream lost - reconnecting", self.id)
                self._stop.wait(1.0)

    def _safe_source(self):
        """Source string with RTSP credentials masked for logs."""
        s = str(self.source)
        if "@" in s and "://" in s:
            scheme, rest = s.split("://", 1)
            return f"{scheme}://***@{rest.split('@', 1)[1]}"
        return s
