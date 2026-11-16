"""Per-camera worker: frame -> detect (ROI only) -> track -> batched recognition + liveness -> verdict.

Each camera runs in its own thread and always works on the newest frame, so there is no backlog.
Run modes per camera:  setup (preview only, no attendance) | live (full pipeline) | off (camera closed).
CameraManager creates, updates and removes cameras at runtime (dashboard), persisted in the database.
"""
import os
import re
import threading
import time
from datetime import datetime

import cv2
import numpy as np
from loguru import logger

from core.camera import Camera
from core.engine import align_face, boost_if_dark, face_quality
from core.tracker import PENDING, SPOOF, VERIFIED, FaceTracker

GREEN, WHITE, ORANGE, RED, CYAN, GREY = (60, 200, 60), (240, 240, 240), (0, 165, 255), (40, 40, 230), (230, 200, 40), (150, 150, 150)


DEFAULT_ROI = [[0.03, 0.03], [0.97, 0.03], [0.97, 0.97], [0.03, 0.97]]
CAMERA_KEYS = ("source", "width", "height", "fps")


class GateWorker(threading.Thread):
    def __init__(self, row, engine, registry, attendance, cfg):
        super().__init__(name=f"gate-{row['cam_id']}", daemon=True)
        self.row = dict(row)
        self.cam_id = row["cam_id"]
        self.engine, self.registry, self.attendance = engine, registry, attendance
        # shared config dicts: dashboard tuning changes them in place and every frame reads the current values
        self.icfg, self.rcfg, self.qcfg, self.lcfg = cfg["inference"], cfg["recognition"], cfg["quality"], cfg["liveness"]
        self.tcfg = cfg.get("tracking", {})
        self.tracker = self._new_tracker()
        self.annotate = cfg.get("display", {}).get("annotate", True)
        rec = cfg.get("recording", {})
        self.record_dir = rec.get("dir", "outputs") if rec.get("enabled") else None
        self.record_fps = rec.get("fps", 25)
        self._writer = None

        self.camera = None
        self._restart = False
        self.fps = 0.0
        self.infer_ms = 0.0
        self._out = None
        self._out_lock = threading.Lock()
        self._banner = ("", WHITE, 0.0)
        self._roi_cache = None
        self._frame_no = 0
        self.face_px = []  # sizes of the faces in view (shown on the dashboard to help tuning)
        self._halt = threading.Event()

    # ---------------- row fields ----------------
    gate_id = property(lambda self: self.row["gate_id"])
    gate_name = property(lambda self: self.row["gate_name"])
    direction = property(lambda self: self.row["direction"])
    mode = property(lambda self: self.row["attendance_mode"])
    run_mode = property(lambda self: self.row["run_mode"])

    # ---------------- public ----------------
    def apply(self, row):
        """Takes new settings from the dashboard; the camera is reopened only if its source changed."""
        old = self.row
        self.row = dict(row)
        if any(str(old.get(k)) != str(row.get(k)) for k in CAMERA_KEYS):
            self._restart = True
        if old.get("roi") != row.get("roi"):
            self._roi_cache = None
        if old.get("run_mode") != row.get("run_mode"):
            self.tracker = self._new_tracker()  # fresh start: no half-collected evidence
            logger.info("[{}] mode -> {}", self.cam_id, row["run_mode"].upper())

    def _new_tracker(self):
        return FaceTracker(self.tcfg.get("iou_threshold", 0.25), self.tcfg.get("max_lost_seconds", 1.0),
                           self.icfg.get("det_thresh", 0.5), self.lcfg.get("window", 15))

    def control(self, action):
        """Video files: pause | resume | restart."""
        cam = self.camera
        if cam is None or cam.kind != "file":
            return False
        if action == "restart":
            cam.rewind()
            self.tracker = self._new_tracker()
        else:
            cam.pause(action == "pause")
        return True

    def output(self):
        with self._out_lock:
            return self._out

    def raw_frame(self):
        cam = self.camera
        return cam.latest()[1] if cam is not None else None

    def status(self):
        cam = self.camera
        st = {k: self.row.get(k) for k in ("cam_id", "gate_id", "gate_name", "direction", "source", "width",
                                           "height", "fps", "roi", "attendance_mode", "run_mode")}
        st["source"] = _mask(str(st["source"]))
        st.update(connected=bool(cam and cam.connected), input_fps=round(cam.input_fps, 1) if cam else 0.0,
                  process_fps=round(self.fps, 1) if self.run_mode != "off" else 0.0,
                  infer_ms=round(self.infer_ms, 1), tracks=len(self.tracker.tracks), face_px=self.face_px[:8],
                  is_video=bool(cam and cam.kind == "file"), paused=bool(cam and cam.paused),
                  position=round(cam.position, 1) if cam else 0.0, duration=round(cam.duration, 1) if cam else 0.0)
        return st

    def stop(self):
        self._halt.set()
        self.join(timeout=3)
        self._close_camera()
        if self._writer is not None:
            self._writer.release()

    # ---------------- loop ----------------
    def _open_camera(self):
        r = self.row
        self.camera = Camera(self.cam_id, r["source"], int(r.get("width") or 1280), int(r.get("height") or 720),
                             int(r.get("fps") or 30)).start()

    def _close_camera(self):
        if self.camera is not None:
            self.camera.stop()
            self.camera = None

    def run(self):
        seq, t_prev = 0, time.perf_counter()
        while not self._halt.is_set():
            if self.run_mode == "off":
                if self.camera is not None:
                    self._close_camera()
                    with self._out_lock:
                        self._out = None
                self._halt.wait(0.3)
                continue
            if self._restart:
                self._close_camera()
                self._restart = False
            if self.camera is None:
                self._open_camera()
                seq = 0

            got = self.camera.wait(seq, timeout=1.0)
            if got is None:
                continue
            seq, frame, ts = got
            t0 = time.perf_counter()
            try:
                out = self._process(frame, ts) if self.run_mode == "live" else self._preview(frame)
            except Exception:
                logger.exception("[{}] frame processing error", self.cam_id)
                out = frame
            now = time.perf_counter()
            self.infer_ms = 0.9 * self.infer_ms + 0.1 * (now - t0) * 1000
            self.fps = 0.9 * self.fps + 0.1 / max(now - t_prev, 1e-6)
            t_prev = now
            with self._out_lock:
                self._out = out
            if self.record_dir and self.run_mode == "live":
                self._record(out)

    def _preview(self, frame):
        """Setup mode: preview with ROI and detected face sizes (to tune distance settings), no recognition and
        no attendance."""
        boxes, _, poly = self._detect(frame)
        self.face_px = sorted((int(min(b[2] - b[0], b[3] - b[1])) for b in boxes), reverse=True)
        if not self.annotate:
            return frame
        out = frame.copy()
        self._render(out, [], poly)
        s = out.shape[1] / 1280.0
        min_px = self.qcfg.get("min_face_px", 64)
        for b in boxes:
            x1, y1, x2, y2 = b[:4].astype(int)
            px = int(min(x2 - x1, y2 - y1))
            color = WHITE if px >= min_px else GREY
            cv2.rectangle(out, (x1, y1), (x2, y2), color, max(1, int(2 * s)), cv2.LINE_AA)
            self._pill(out, f"{px}px" if px >= min_px else f"{px}px small", (x1, y1 - int(6 * s)), color, 0.5 * s)
        return out

    def _roi(self, w, h):
        if self._roi_cache is None or self._roi_cache[0] != (w, h):
            pts = (np.array(self.row.get("roi") or DEFAULT_ROI, np.float32) * [w, h]).astype(np.int32)
            x0, y0 = np.clip(pts.min(0), 0, [w - 1, h - 1])
            x1, y1 = np.clip(pts.max(0), 1, [w, h])
            self._roi_cache = ((w, h), pts, (int(x0), int(y0), int(x1), int(y1)))
        return self._roi_cache[1], self._roi_cache[2]

    def _detect(self, frame):
        """Detects on the ROI crop only (a small ROI is upscaled to det_size = digital zoom for far faces)."""
        h, w = frame.shape[:2]
        poly, (x0, y0, x1, y1) = self._roi(w, h)
        boxes, kps = self.engine.detect(frame[y0:y1, x0:x1])
        if len(boxes):
            boxes[:, [0, 2]] += x0
            boxes[:, [1, 3]] += y0
            kps += (x0, y0)
            cx, cy = (boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2
            inside = np.array([cv2.pointPolygonTest(poly, (float(a), float(b)), False) >= 0 for a, b in zip(cx, cy)])
            boxes, kps = boxes[inside], kps[inside]
        return boxes, kps, poly

    def _process(self, frame, ts):
        self._frame_no += 1
        boxes, kps, poly = self._detect(frame)
        tracks = self.tracker.update(boxes, kps, ts)
        self.face_px = sorted((int(tr.size) for tr in tracks), reverse=True)

        self._analyze([tr for tr in tracks if tr.state == PENDING], frame)
        for tr in self.tracker.removed:  # left the view while still verifying: last-chance decision
            if tr.state == PENDING and tr.samples and not tr.spoof and tr.decide(self.registry, self.rcfg, self.lcfg, True) == VERIFIED:
                self._on_verified(tr)
        self._log_spoofs(ts)

        if not self.annotate:
            return frame
        out = frame.copy()
        self._render(out, tracks, poly)
        return out

    def _analyze(self, pending, frame):
        """Runs one batched ArcFace + liveness pass for all pending tracks. Soft quality: only faces below the
        minimum size are skipped; blur, side view, low confidence, light and a face at the frame edge just give
        the frame less weight, so a person walking past fast is still recognised from the few frames there are."""
        q, h, w = self.qcfg, frame.shape[0], frame.shape[1]
        live_on = self.lcfg.get("enabled", True)
        batch, aligned, live_idx, weights = [], [], [], []
        for tr in pending:
            if tr.unknown_reported and self._frame_no % 3:
                continue  # unregistered person lingering: sample every 3rd frame only
            x1, y1, x2, y2 = tr.box
            if tr.size < q.get("min_face_px", 64):
                continue
            face = align_face(frame, tr.kps)
            fq = face_quality(face, tr.kps)
            if fq["brightness"] < q.get("min_brightness", 35):
                face = boost_if_dark(face, threshold=255)
                fq = face_quality(face, tr.kps)
            weight = (min(1.0, tr.det_score / max(q.get("min_det_score", 0.6), 0.05))
                      * min(1.0, max(0.15, fq["sharpness"] / max(q.get("min_sharpness", 25), 0.1)))
                      * max(0.15, 1.0 - 2.0 * max(0.0, abs(fq["yaw"]) - q.get("max_yaw", 0.45)))
                      * max(0.4, 1.0 - 0.5 * abs(fq["yaw"])))
            if not q.get("min_brightness", 35) <= fq["brightness"] <= q.get("max_brightness", 235):
                weight *= 0.5
            if x1 < 2 or y1 < 2 or x2 > w - 2 or y2 > h - 2:
                weight *= 0.5  # cut off at the frame edge
            batch.append(tr)
            aligned.append(face)
            weights.append(weight)
            if live_on and tr.size >= self.lcfg.get("min_face_px", 80):
                live_idx.append(len(batch) - 1)
        if not batch:
            return

        embs = self.engine.recognizer.embed(aligned)
        live = [None] * len(batch)
        if live_idx:
            probs = self.engine.liveness.predict(frame, np.array([batch[i].box for i in live_idx]))
            for i, p in zip(live_idx, probs):
                live[i] = p
        ids, best, _ = self.registry.match(embs)
        thr = self.rcfg.get("frame_threshold", 0.35)

        for i, tr in enumerate(batch):
            tr.observe(embs[i], weights[i], ids[i] if best[i] >= thr else None, live[i])
            verdict = tr.decide(self.registry, self.rcfg, self.lcfg)
            if verdict == VERIFIED:
                self._on_verified(tr)
            elif verdict == SPOOF:
                tr.spoof_snap = self._snapshot(tr, frame)
                self._set_banner("SECURITY ALERT: FAKE FACE BLOCKED", RED)
            elif verdict == "UNKNOWN":
                self._alert(tr, "UNKNOWN", "Face not in registry", self._snapshot(tr, frame))
                self._set_banner("UNREGISTERED PERSON", ORANGE)


    def _on_verified(self, tr):
        # Two people in view can never both be the same employee: the earlier, still visible match wins.
        for other in self.tracker.tracks.values():
            if other is not tr and other.state == VERIFIED and other.emp_id == tr.emp_id:
                tr.state, tr.emp_id, tr.samples, tr.message = PENDING, None, 0, "Re-verifying..."
                tr.emb_sum[:] = 0
                tr.votes.clear()
                return
        status, msg = self.attendance.record(tr.emp_id, tr.name, tr.department, self.gate_id, self.gate_name,
                                             self.cam_id, self.direction, self.mode, tr.match_score, tr.liveness)
        tr.message = msg
        if status == "MARKED":
            self._set_banner(msg.upper(), GREEN)
        elif status == "ALREADY":
            self._set_banner(msg, CYAN)

    def _log_spoofs(self, ts):
        """Attendance is blocked the moment a face is flagged; the security log entry is written once the
        flag has held for spoof_alert_seconds (or the person left still flagged), so a real person who
        recovers after a few bad frames does not create a false alarm."""
        delay = self.lcfg.get("spoof_alert_seconds", 1.5)
        for tr in list(self.tracker.tracks.values()) + self.tracker.removed:
            if tr.spoof and not tr.spoof_logged and (ts - tr.spoof_since >= delay or tr in self.tracker.removed):
                tr.spoof_logged = True
                self._alert(tr, "SPOOF", "Presentation attack blocked", tr.spoof_snap)

    @staticmethod
    def _snapshot(tr, frame):
        x1, y1, x2, y2 = tr.box.astype(int)
        mw, mh = (x2 - x1) // 2, (y2 - y1) // 2
        return frame[max(0, y1 - mh):y2 + mh, max(0, x1 - mw):x2 + mw].copy()

    def _alert(self, tr, kind, detail, snap):
        self.attendance.alert(kind, self.gate_id, self.cam_id, tr.match_score, tr.liveness, detail, snap)

    def _set_banner(self, text, color):
        self._banner = (text, color, time.time())

    # ---------------- rendering ----------------
    def _render(self, img, tracks, poly):
        h, w = img.shape[:2]
        s = w / 1280.0
        cv2.polylines(img, [poly], True, (200, 160, 40), max(1, int(2 * s)), cv2.LINE_AA)

        for tr in tracks:
            if tr.state == VERIFIED:
                color, label = GREEN, f"{tr.name}  {tr.match_score:.0%}"
                sub = tr.message
            elif tr.spoof:
                color, label, sub = RED, "FAKE FACE - BLOCKED", f"liveness {tr.liveness:.0%}"
            elif tr.unknown_reported:
                color, label, sub = ORANGE, "Unregistered", ""
            elif tr.size < self.qcfg.get("min_face_px", 64):
                color, label, sub = GREY, f"Too small {int(tr.size)}px", ""
            else:
                color, label, sub = WHITE, tr.message or "Verifying...", ""
            x1, y1, x2, y2 = tr.box.astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, max(2, int(3 * s)), cv2.LINE_AA)
            self._pill(img, label, (x1, y1 - int(8 * s)), color, 0.6 * s)
            if sub:
                self._pill(img, sub, (x1, y2 + int(28 * s)), color, 0.5 * s)

        # top bar
        bar = int(48 * s)
        cv2.rectangle(img, (0, 0), (w, bar), (24, 24, 24), -1)
        st = self.attendance.stats()
        cv2.putText(img, f"{self.gate_name.upper()}  |  {self.direction}  |  {self.cam_id}", (int(16 * s), int(32 * s)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62 * s, (255, 255, 255), max(1, int(2 * s)), cv2.LINE_AA)
        if self.run_mode == "live":
            right = (f"LIVE  {self.mode.upper()}   PRESENT {st.get('present', 0)}   "
                     f"{self.fps:4.1f} FPS   {datetime.now():%H:%M:%S}")
            rcolor = (120, 230, 120)
        else:
            right = f"SETUP MODE - ATTENDANCE PAUSED   {datetime.now():%H:%M:%S}"
            rcolor = (0, 200, 255)
        (tw, _), _ = cv2.getTextSize(right, cv2.FONT_HERSHEY_SIMPLEX, 0.58 * s, max(1, int(2 * s)))
        cv2.putText(img, right, (w - tw - int(16 * s), int(32 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.58 * s,
                    rcolor, max(1, int(2 * s)), cv2.LINE_AA)

        text, color, t = self._banner
        if text and time.time() - t < 3.5:
            bh = int(54 * s)
            cv2.rectangle(img, (0, h - bh), (w, h), (24, 24, 24), -1)
            cv2.rectangle(img, (0, h - bh), (w, h), color, max(2, int(3 * s)))
            (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8 * s, max(1, int(2 * s)))
            cv2.putText(img, text, ((w - tw) // 2, h - int(18 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * s,
                        color, max(1, int(2 * s)), cv2.LINE_AA)

    @staticmethod
    def _pill(img, text, org, color, scale):
        thick = max(1, int(scale * 3))
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        x, y = max(0, org[0]), max(th + 8, org[1])
        cv2.rectangle(img, (x, y - th - 8), (x + tw + 12, y + base - 2), color, -1)
        cv2.putText(img, text, (x + 6, y - 4), cv2.FONT_HERSHEY_SIMPLEX, scale, (20, 20, 20), thick, cv2.LINE_AA)

    def _record(self, frame):
        if self._writer is None:
            os.makedirs(self.record_dir, exist_ok=True)
            path = f"{self.record_dir}/{self.cam_id}_{datetime.now():%Y-%m-%d_%H%M%S}.mp4"
            self._writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), self.record_fps,
                                           (frame.shape[1], frame.shape[0]))
            logger.info("[{}] recording to {}", self.cam_id, path)
        self._writer.write(frame)


def _mask(source):
    if "@" in source and "://" in source:
        scheme, rest = source.split("://", 1)
        return f"{scheme}://***@{rest.split('@', 1)[1]}"
    return source


ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,40}$")


def _slug(text):
    return re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_") or "gate"


class CameraManager:
    """Owns all camera workers. Settings live in the cameras table (seeded once from config.yaml)."""

    def __init__(self, storage, engine, registry, attendance, cfg):
        self.storage, self.engine, self.registry, self.attendance, self.cfg = storage, engine, registry, attendance, cfg
        self._lock = threading.Lock()
        self.workers = {}
        if not storage.cameras():
            self._seed_from_config()
        # startup.cameras: "resume" = every camera comes back in its last mode (production, auto-restart)
        #                  "off"    = nothing opens until a camera is switched on in the dashboard (demo / POC)
        start_off = str(cfg.get("startup", {}).get("cameras", "resume")).lower() == "off"
        for row in storage.cameras():
            if start_off and row["run_mode"] != "off":
                row["run_mode"] = "off"
                storage.save_camera(row)
            self._spawn(row)
        if start_off:
            logger.info("startup.cameras = off: all cameras closed - choose and switch on a source in the dashboard")

    def _seed_from_config(self):
        default_mode = self.cfg.get("attendance", {}).get("mode", "unique")
        for gi, gate in enumerate(self.cfg.get("gates", [])):
            for ci, cam in enumerate(gate.get("cameras", [])):
                self.storage.save_camera({
                    "cam_id": cam["id"], "gate_id": _slug(gate.get("name", gate["id"])),
                    "gate_name": gate.get("name", gate["id"]),
                    "direction": str(cam.get("direction", "ENTRY")).upper(), "source": str(cam.get("source", 0)),
                    "width": cam.get("width", 1280), "height": cam.get("height", 720), "fps": cam.get("fps", 30),
                    "roi": cam.get("roi") or DEFAULT_ROI, "attendance_mode": cam.get("attendance_mode", default_mode),
                    "run_mode": "setup" if cam.get("enabled", True) else "off", "sort_order": gi * 100 + ci})
        logger.info("Cameras seeded from config: {}", [c["cam_id"] for c in self.storage.cameras()])

    def _spawn(self, row):
        w = GateWorker(row, self.engine, self.registry, self.attendance, self.cfg)
        self.workers[row["cam_id"]] = w
        w.start()

    def get(self, cam_id):
        return self.workers.get(cam_id)

    def status(self):
        ws = sorted(self.workers.values(), key=lambda w: (w.gate_name, w.row.get("sort_order") or 0, w.cam_id))
        return [w.status() for w in ws]

    def save(self, data, new):
        """Validates and creates / updates a camera. Returns (ok, message)."""
        cam_id = str(data.get("cam_id", "")).strip()
        if not ID_RE.match(cam_id):
            return False, "Camera ID may only contain letters, digits, - and _"
        with self._lock:
            existing = self.workers.get(cam_id)
            if new and existing:
                return False, f"Camera '{cam_id}' already exists"
            if not new and not existing:
                return False, f"Camera '{cam_id}' not found"
            row = dict(existing.row) if existing else {"width": 1280, "height": 720, "fps": 30, "roi": DEFAULT_ROI,
                                                       "run_mode": "setup", "attendance_mode": "unique",
                                                       "direction": "ENTRY", "sort_order": len(self.workers)}
            for k in ("gate_name", "direction", "source", "width", "height", "fps", "roi", "attendance_mode", "run_mode"):
                if k in data and data[k] not in (None, ""):
                    row[k] = data[k]
            row["cam_id"] = cam_id
            row["gate_name"] = str(row.get("gate_name", "")).strip()
            row["direction"] = str(row.get("direction", "ENTRY")).upper()
            row["source"] = str(row.get("source", "")).strip()
            if not row["gate_name"] or not row["source"]:
                return False, "Gate name and camera source are required"
            if row["direction"] not in ("ENTRY", "EXIT"):
                return False, "Direction must be ENTRY or EXIT"
            if row.get("attendance_mode") not in ("unique", "multiple"):
                return False, "Attendance mode must be unique or multiple"
            if row["run_mode"] not in ("setup", "live", "off"):
                return False, "Run mode must be setup, live or off"
            if row["run_mode"] == "live" and not len(self.registry):
                return False, "Register at least one employee before going live"
            try:
                roi = [[round(float(x), 4), round(float(y), 4)] for x, y in row.get("roi")]
                for k in ("width", "height", "fps"):
                    row[k] = int(row[k])
            except (TypeError, ValueError):
                return False, "Invalid ROI or resolution values"
            if len(roi) < 3 or not all(0 <= v <= 1 for p in roi for v in p):
                return False, "ROI needs at least 3 points with values between 0 and 1"
            row["roi"] = roi
            row["gate_id"] = _slug(row["gate_name"])
            self.storage.save_camera(row)
            if existing:
                existing.apply(row)
            else:
                self._spawn(row)
        logger.info("Camera {} {}: {} {} {} [{}]", cam_id, "added" if new else "updated", row["gate_name"],
                    row["direction"], row["attendance_mode"].upper(), row["run_mode"].upper())
        return True, f"Camera {cam_id} saved"

    def delete(self, cam_id):
        with self._lock:
            w = self.workers.pop(cam_id, None)
        if w is None:
            return False
        w.stop()
        self.storage.delete_camera(cam_id)
        logger.info("Camera {} removed", cam_id)
        return True

    def using_source(self, source):
        return [cid for cid, w in self.workers.items() if os.path.normcase(os.path.abspath(str(w.row["source"])))
                == os.path.normcase(os.path.abspath(source))]

    def set_run_mode(self, run_mode, cam_ids=None):
        """Switches the given cameras (default: every camera that is not off). Returns the changed IDs."""
        changed = []
        for cam_id, w in list(self.workers.items()):
            if (cam_ids is None and w.run_mode == "off") or (cam_ids is not None and cam_id not in cam_ids):
                continue
            if self.save({"cam_id": cam_id, "run_mode": run_mode}, new=False)[0]:
                changed.append(cam_id)
        return changed

    def stop_all(self):
        for w in list(self.workers.values()):
            w.stop()
