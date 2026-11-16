"""Test videos uploaded from the dashboard. Uploads arrive in chunks (tunnels such as Cloudflare limit one
request to 100 MB), are checked with OpenCV and stored in data/videos. Each video can become a test camera."""
import os
import re
import threading

import cv2
from loguru import logger

VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm", ".mpg", ".mpeg", ".ts", ".wmv")


def safe_name(name):
    base, ext = os.path.splitext(os.path.basename(str(name)))
    base = re.sub(r"[^A-Za-z0-9_\-]+", "_", base).strip("_")[:60] or "video"
    return base + ext.lower()


def probe(path):
    """(ok, info) - the file must open and deliver a frame."""
    cap = cv2.VideoCapture(path)
    try:
        ok, frame = cap.read()
        if not ok or frame is None:
            return False, {}
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        return True, {"width": frame.shape[1], "height": frame.shape[0], "fps": round(fps, 1),
                      "duration": round(cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps, 1)}
    finally:
        cap.release()


class VideoLibrary:
    def __init__(self, folder, max_mb=2048):
        self.folder = folder
        self.max_bytes = int(max_mb) * 1024 * 1024
        self._lock = threading.Lock()
        os.makedirs(folder, exist_ok=True)
        for f in os.listdir(folder):  # unfinished uploads from a previous run
            if f.endswith(".part"):
                os.remove(os.path.join(folder, f))

    def path(self, name):
        name = os.path.basename(name)
        p = os.path.join(self.folder, name).replace("\\", "/")
        if not name or not os.path.isfile(p):
            raise KeyError("Video not found")
        return p

    def list(self):
        out = []
        for f in sorted(os.listdir(self.folder)):
            p = os.path.join(self.folder, f)
            if f.lower().endswith(VIDEO_EXT) and os.path.isfile(p):
                out.append({"name": f, "path": p.replace("\\", "/"), "size_mb": round(os.path.getsize(p) / 2**20, 1)})
        return out

    async def receive_chunk(self, name, upload_id, offset, stream):
        """Appends one chunk (request body stream) to <upload_id>.part. Returns the new size."""
        if not os.path.splitext(name)[1].lower() in VIDEO_EXT:
            raise ValueError("Not a supported video file (" + ", ".join(VIDEO_EXT) + ")")
        if not re.fullmatch(r"[A-Za-z0-9]{8,40}", upload_id or ""):
            raise ValueError("Invalid upload id")
        part = os.path.join(self.folder, upload_id + ".part")
        size = os.path.getsize(part) if os.path.exists(part) else 0
        if offset != size:
            raise ValueError(f"Upload out of order (server has {size} bytes)")
        with open(part, "ab") as f:
            async for chunk in stream:
                size += len(chunk)
                if size > self.max_bytes:
                    f.close()
                    os.remove(part)
                    raise ValueError(f"Video is larger than {self.max_bytes // 2**20} MB")
                f.write(chunk)
        return size

    def finish(self, name, upload_id):
        """Checks the finished upload and gives it its final name. Returns (path, info)."""
        part = os.path.join(self.folder, upload_id + ".part")
        if not os.path.exists(part):
            raise ValueError("Upload not found - please upload again")
        ok, info = probe(part)
        if not ok:
            os.remove(part)
            raise ValueError("This file cannot be read as a video (try MP4 / H.264)")
        with self._lock:
            final = safe_name(name)
            base, ext = os.path.splitext(final)
            n = 1
            while os.path.exists(os.path.join(self.folder, final)):
                n += 1
                final = f"{base}_{n}{ext}"
            dst = os.path.join(self.folder, final).replace("\\", "/")
            os.replace(part, dst)
        logger.info("Test video uploaded: {} ({}x{}, {} s)", final, info["width"], info["height"], info["duration"])
        return dst, info

    def delete(self, name):
        p = self.path(name)
        os.remove(p)
        logger.info("Test video deleted: {}", name)

    def clear(self):
        n = 0
        for f in os.listdir(self.folder):
            try:
                os.remove(os.path.join(self.folder, f))
                n += 1
            except OSError:
                pass
        return n
