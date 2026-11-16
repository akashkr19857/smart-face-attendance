"""Employee enrollment: button capture from a camera and photo upload, every image quality-checked; folder import."""
import base64
import os
import re
import shutil
import threading
import time
import uuid

import cv2
import numpy as np
from loguru import logger

from core.engine import align_face, face_quality

ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,32}$")
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif")


def read_image(path):
    """cv2.imread that also works for non-ASCII Windows paths."""
    try:
        return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None
# suggested pose mix for a 12-photo enrollment (side A / side B are mirror-agnostic)
POSE_TARGET = {"front": 4, "side_a": 2, "side_b": 2, "up": 2, "down": 2}
# extra "far camera" templates per photo: the face as a CCTV camera sees it from a distance (few pixels, soft)
FAR_SIZES = (56, 40, 28)
POSE_HINT = {"front": "Look straight at the camera", "side_a": "Turn your head slightly to one side",
             "side_b": "Turn slightly to the other side", "up": "Lift your chin slightly",
             "down": "Lower your chin slightly"}


def _thumb(img, size=96):
    h, w = img.shape[:2]
    s = min(h, w)
    sq = img[(h - s) // 2:(h + s) // 2, (w - s) // 2:(w + s) // 2]
    ok, buf = cv2.imencode(".jpg", cv2.resize(sq, (size, size)), [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


class EnrollSession:
    def __init__(self):
        self.id = uuid.uuid4().hex
        self.created = time.time()
        self.samples = []  # dicts: id, emb, quality, live, crop, pose, source, thumb
        self.pitch_ref = []
        self.next_id = 1


def _far_views(face):
    """Simulates the same aligned face seen small and slightly blurred by a distant CCTV camera."""
    out = []
    for px in FAR_SIZES:
        small = cv2.resize(cv2.GaussianBlur(face, (3, 3), 0.6), (px, px), interpolation=cv2.INTER_AREA)
        out.append(cv2.resize(small, (112, 112), interpolation=cv2.INTER_LINEAR))
    return out


def _pad(img, ratio):
    """Adds a neutral border so a tightly cropped face photo gets the context the detector needs."""
    h, w = img.shape[:2]
    py, px = int(h * ratio), int(w * ratio)
    color = [int(c) for c in cv2.mean(img)[:3]]
    return cv2.copyMakeBorder(img, py, py, px, px, cv2.BORDER_CONSTANT, value=color)


class EnrollmentService:
    def __init__(self, engine, storage, registry, cfg, liveness_cfg, faces_dir, quality_cfg=None):
        self.engine, self.storage, self.registry = engine, storage, registry
        # cfg / liveness_cfg / quality_cfg are the shared config dicts, so dashboard tuning applies at once
        self.cfg, self.lcfg, self.qcfg = cfg, liveness_cfg, quality_cfg or {}
        self.recommended = int(cfg.get("samples", 12))
        self.min_samples = int(cfg.get("min_samples", 12))  # an employee is saved with 12 - 100 good photos
        self.max_samples = max(self.min_samples, int(cfg.get("max_samples", 100)))
        self.dup_thr = float(cfg.get("duplicate_threshold", 0.5))
        self.faces_dir = faces_dir
        self._sessions = {}
        self._lock = threading.Lock()

    # ---------------- session ----------------
    def start(self):
        with self._lock:
            for sid in [k for k, s in self._sessions.items() if time.time() - s.created > 1800]:
                del self._sessions[sid]
            s = EnrollSession()
            self._sessions[s.id] = s
        return s.id

    def _session(self, sid):
        s = self._sessions.get(sid)
        if s is None:
            raise KeyError("Registration session expired - press New registration")
        return s

    min_px = property(lambda self: int(self.cfg.get("min_face_px", 110)))
    check_live = property(lambda self: bool(self.cfg.get("liveness_check", True)) and self.lcfg.get("enabled", True))

    def cancel_all(self):
        with self._lock:
            self._sessions.clear()

    def cancel(self, sid):
        with self._lock:
            self._sessions.pop(sid, None)

    def summary(self, sid):
        s = self._session(sid)
        return {"count": len(s.samples), "min": self.min_samples, "recommended": self.recommended,
                "max": self.max_samples, "suggestion": self._suggest(s),
                "samples": [{"id": x["id"], "thumb": x["thumb"], "pose": x["pose"], "source": x["source"]}
                            for x in s.samples]}

    # ---------------- quality ----------------
    def _pick_face(self, img, point, size=None):
        """Largest face, or the face nearest to point (x, y normalised) when the user clicked on one."""
        boxes, kps = self.engine.detect(img, self.engine.det_thresh, size)
        if len(boxes) == 0:
            return None
        if point is None:
            i = int(((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])).argmax())
            return boxes[i], kps[i], len(boxes)
        h, w = img.shape[:2]
        px, py = point[0] * w, point[1] * h
        inside = (boxes[:, 0] <= px) & (px <= boxes[:, 2]) & (boxes[:, 1] <= py) & (py <= boxes[:, 3])
        dist = np.hypot((boxes[:, 0] + boxes[:, 2]) / 2 - px, (boxes[:, 1] + boxes[:, 3]) / 2 - py)
        dist[inside] = -1
        i = int(dist.argmin())
        return boxes[i], kps[i], 1  # the chosen face; other people in the frame are fine

    def _evaluate(self, img, from_camera, s=None, point=None, upscaled=False):
        """Returns (ok, reason, info). info holds box (normalised), face, kps, quality, live, pose."""
        h, w = img.shape[:2]
        # camera frames use the tuned detector size (far faces); close-up photos and zoomed crops the standard 640
        found = self._pick_face(img, point, None if from_camera and not upscaled else 640)
        if found is None:
            return False, "No face detected", {}
        box, kps, n = found
        info = {"box": [round(float(box[0]) / w, 4), round(float(box[1]) / h, 4),
                        round(float(box[2]) / w, 4), round(float(box[3]) / h, 4)], "raw_box": box, "kps": kps}
        if n > 1:
            return False, ("More than one face - click on the employee's face in the image to capture it"
                           if from_camera else "More than one face - only the employee should be in the photo"), info
        if box[4] < 0.6:
            return False, "Face not clear enough", info
        if min(box[2] - box[0], box[3] - box[1]) < self.min_px:
            return False, "Small face - click on it to capture (auto zoom)" if from_camera else f"Face too small (min {self.min_px} px)", info
        x1, y1, x2, y2 = box[:4]
        if x1 < 2 or y1 < 2 or x2 > w - 2 or y2 > h - 2:
            return False, "Face is cut off at the edge of the image", info
        face = align_face(img, kps)
        q = face_quality(face, kps)
        info.update(face=face, quality=q)
        if q["brightness"] < 50:
            return False, "Too dark - add light in front of the face", info
        if q["brightness"] > 225:
            return False, "Too bright - avoid direct light on the face", info
        sharp_min = float(self.qcfg.get("min_sharpness", 25)) + (10 if from_camera else 5)
        if upscaled:  # an enlarged small photo is soft by nature; the far gate camera sees faces like that too
            sharp_min *= 0.4
        if q["sharpness"] < sharp_min:
            return False, "Blurry - hold still" if from_camera else "Blurry photo", info
        if abs(q["yaw"]) > max(0.6, float(self.qcfg.get("max_yaw", 0.45))):
            return False, "Face turned too far - turn back a little", info
        info["live"] = 1.0
        # liveness needs enough pixels; a small face from a far CCTV view is checked live at the gate instead
        # (an enlarged small face is never liveness-checked here: enlarging distorts the model's input)
        if (from_camera and self.check_live and not upscaled
                and min(box[2] - box[0], box[3] - box[1]) >= self.lcfg.get("min_face_px", 80)):
            info["live"] = float(self.engine.liveness.predict(img, box[None, :])[0])
            if info["live"] < 0.5:
                return False, "Real face required (no photo or screen)", info
        info["pose"] = self._pose(q["yaw"], q["pitch"], s)
        return True, "Good - ready to capture", info

    @staticmethod
    def _pose(yaw, pitch, s):
        ref = float(np.median(s.pitch_ref)) if s and s.pitch_ref else 0.55
        if abs(yaw) > 0.18:
            return "side_a" if yaw > 0 else "side_b"
        if pitch < ref - 0.07:
            return "up"
        if pitch > ref + 0.07:
            return "down"
        return "front"

    def _suggest(self, s):
        counts = {k: 0 for k in POSE_TARGET}
        for x in s.samples:
            counts[x["pose"]] = counts.get(x["pose"], 0) + 1
        for pose, target in POSE_TARGET.items():
            if counts[pose] < target:
                return POSE_HINT[pose]
        return "Enough variety - you can save now"

    def _evaluate_photo(self, img, s, pad=None):
        """Uploaded photos: tries the photo as it is, then fixes common problems automatically -
        a tightly cropped face (adds a border), a small / low-resolution photo (upscales),
        a sideways photo (rotates). pad: None = automatic padding, or a border ratio chosen by the user
        (0.5 = 50 % of the photo size on every side) that is tried first. Returns (ok, reason, info, image_used)."""
        h, w = img.shape[:2]
        small = max(h, w) < 360
        if small:  # thumbnail / passport-size crop: enlarge so the detector can find the face
            f = 480 / max(h, w)
            img = cv2.resize(img, (int(w * f), int(h * f)), interpolation=cv2.INTER_CUBIC)
        if max(h, w) > 2400:  # huge phone photos: faster, same result
            f = 2400 / max(h, w)
            img = cv2.resize(img, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)
        if pad:
            img = _pad(img, pad)
        ok, reason, info = self._evaluate(img, False, s, upscaled=small)
        if ok:
            return ok, reason, info, img
        first = (ok, reason, info, img)
        candidates = [img, _pad(img, 0.35), _pad(img, 0.8)]
        for rot in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180):
            candidates.append(_pad(cv2.rotate(img, rot), 0.35))
        for k, cand in enumerate(candidates):
            ok, reason, info = self._evaluate(cand, False, s, upscaled=small) if k else first[:3]
            box = info.get("raw_box")
            size = min(box[2] - box[0], box[3] - box[1]) if box is not None else 0
            f = min(5.0, self.min_px * 1.25 / size) if size >= 24 else 0
            if not ok and reason.startswith(("Face too small", "Blurry")) and f > 1.0:
                # low-resolution photo: upscale so the face reaches the minimum size (softness is expected)
                cand = cv2.resize(cand, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
                ok, reason, info = self._evaluate(cand, False, s, upscaled=True)
            if ok:
                return ok, reason, info, cand
            if first[1] == "No face detected" and info:
                first = (ok, reason, info, cand)  # report the most useful reason
        return first

    # ---------------- camera ----------------
    def check(self, sid, frame):
        """Live quality indicator for the preview (no capture)."""
        s = self._session(sid)
        ok, reason, info = self._evaluate(frame, True, s)
        return {"ok": ok, "message": reason, "box": info.get("box"), "suggestion": self._suggest(s),
                "count": len(s.samples)}

    def _zoom_at(self, img, point, pad=None):
        """Digital zoom on the clicked face (far CCTV, small face): crops around it, enlarges it and adds a border.
        Returns (crop, box normalised to the full frame or None)."""
        h, w = img.shape[:2]
        px, py = point[0] * w, point[1] * h
        found, box = self._pick_face(img, point), None
        if found is not None:
            b = found[0]
            size = max(b[2] - b[0], b[3] - b[1])
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            if abs(px - cx) <= size * 1.5 and abs(py - cy) <= size * 1.5:  # the click is on / next to this face
                box = [round(float(b[0]) / w, 4), round(float(b[1]) / h, 4),
                       round(float(b[2]) / w, 4), round(float(b[3]) / h, 4)]
            else:
                found = None
        if found is None:  # face too small for the full-frame detector: zoom into the area around the click
            cx, cy, size = px, py, max(24.0, 0.07 * h)
        half = size * 1.6
        x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
        x1, y1 = int(min(w, cx + half)), int(min(h, cy + half))
        crop = img[y0:y1, x0:x1]
        if crop.size == 0:
            return None, None
        f = max(1.0, min(6.0, self.min_px * 1.4 / max(size, 1.0)))
        crop = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
        return _pad(crop, 0.3 if pad is None else pad), box

    def capture(self, sid, frame, point=None, pad=None):
        s = self._session(sid)
        ok, reason, info = self._evaluate(frame, True, s, point)
        used, box = frame, info.get("box")
        if not ok and point is not None and not reason.startswith(("Too dark", "Too bright")):
            # clicked face is small / far / at the edge: zoom in on it and check the enlarged face
            crop, zbox = self._zoom_at(frame, point, pad)
            if crop is not None:
                ok2, reason2, info2 = self._evaluate(crop, True, s, (0.5, 0.5), upscaled=True)
                if ok2 or info2 or not info:  # the zoomed check found the face: its reason is the useful one
                    ok, reason, info, used, box = ok2, reason2, info2, crop, zbox or box
                if not info2:
                    reason = "No face where you clicked - pause the video and click exactly on the face"
        if ok:
            ok, reason, sample = self._add(s, used, info, "camera")
        res = {"ok": ok, "message": reason if not ok else ("Photo captured (zoomed in on a small face)"
                                                             if used is not frame else "Photo captured"), "box": box}
        res.update(self.summary(sid))
        return res

    # ---------------- upload ----------------
    def upload(self, sid, files, pad=None):
        """files: list of (filename, image or None). Returns per-file results."""
        s = self._session(sid)
        results = []
        for name, img in files:
            if img is None:
                results.append({"file": name, "ok": False, "message": "Not an image"})
                continue
            ok, reason, info, used = self._evaluate_photo(img, s, pad)
            fixed = ok and used is not img
            if ok:
                ok, reason, sample = self._add(s, used, info, "upload")
            results.append({"file": name, "ok": ok, "message": ("Accepted (auto-fixed crop / size)" if fixed
                                                                 else "Accepted") if ok else reason,
                            "thumb": _thumb(img) if not ok else None})
        out = self.summary(sid)
        out["results"] = results
        return out

    def _add(self, s, img, info, source):
        if len(s.samples) >= self.max_samples:
            return False, f"Maximum {self.max_samples} photos - remove one first", None
        emb = self.engine.recognizer.embed([info["face"]])[0]
        if s.samples:
            embs = np.stack([x["emb"] for x in s.samples])
            centroid = embs.mean(0)
            centroid /= np.linalg.norm(centroid)
            if float(centroid @ emb) < 0.35:
                return False, "Different person than the other photos", None
            if float((embs @ emb).max()) > 0.99:  # practically identical (e.g. double click, burst photos)
                return False, ("Same as an earlier photo - change the pose slightly" if source == "camera"
                               else "Almost the same as another photo - skipped, use a different angle"), None
        x1, y1, x2, y2 = info["raw_box"][:4].astype(int)
        m = int(0.35 * (x2 - x1))
        crop = img[max(0, y1 - m):y2 + m, max(0, x1 - m):x2 + m].copy()
        sample = {"id": s.next_id, "emb": emb, "quality": float(info["quality"]["sharpness"]), "live": info["live"],
                  "crop": crop, "face": info["face"], "pose": info["pose"], "source": source, "thumb": _thumb(crop)}
        s.next_id += 1
        s.samples.append(sample)
        if sample["pose"] == "front":
            s.pitch_ref.append(info["quality"]["pitch"])
        return True, "Accepted", sample

    def remove(self, sid, sample_id):
        s = self._session(sid)
        s.samples = [x for x in s.samples if x["id"] != int(sample_id)]
        return self.summary(sid)

    # ---------------- save ----------------
    def commit(self, sid, emp_id, name, department, force=False):
        s = self._session(sid)
        n = len(s.samples)
        if n < self.min_samples:
            return False, f"At least {self.min_samples} good photos are needed ({n} so far)"
        cam = [x["live"] for x in s.samples if x["source"] == "camera"]
        if self.check_live and cam and np.mean(cam) < float(self.lcfg.get("real_threshold", 0.5)):
            return False, "Liveness check failed - please capture again with the real person"
        ok, msg = self._save(emp_id, name, department, s.samples, force)
        if ok:
            self.cancel(sid)
        return ok, msg

    def from_images(self, emp_id, name, department, images, force=False, pad=None):
        """Bulk registration: same checks and auto-fix (padding, upscale, rotation) as the upload screen.
        Needs min_samples good photos, uses up to max_samples. Returns (ok, message)."""
        if not ID_RE.match(str(emp_id).strip()):
            return False, "Employee ID may only contain letters, digits, - and _ (max 32)"
        s = EnrollSession()
        total = 0
        for img in images:
            if img is None:
                continue
            total += 1
            if len(s.samples) >= self.max_samples:
                continue  # enough: the rest are counted only
            ok, _, info, used = self._evaluate_photo(img, s, pad)
            if ok:
                self._add(s, used, info, "upload")
        n = len(s.samples)
        if n < self.min_samples:
            return False, (f"Only {n} of {total} photo(s) usable - at least {self.min_samples} good photos are needed"
                           " (clear face, different angles)")
        ok, msg = self._save(emp_id, name, department, s.samples, force)
        if ok:
            msg += f" ({n} of {total} photos usable)"
        return ok, msg

    def import_dir(self, root, with_names=False, pad=None):
        """Imports one folder per employee. Folder name = employee ID (EMP001, 1045, ...);
        with_names: folders are named <EMP_ID>_<Name>. Photos may also sit in sub-folders."""
        done, total = 0, 0
        for folder in sorted(os.listdir(root)):
            path = os.path.join(root, folder)
            if not os.path.isdir(path):
                continue
            emp_id, name = (folder.split("_", 1) if with_names and "_" in folder else (folder, ""))
            files = sorted(os.path.join(d, f) for d, _, fs in os.walk(path) for f in fs if f.lower().endswith(IMG_EXT))
            total += 1
            ok, msg = self.from_images(emp_id, name.replace("_", " "), "", [read_image(f) for f in files], True, pad)
            logger.info("Import {}: {} {}", folder, "OK " if ok else "FAILED -", msg)
            done += ok
        logger.info("Bulk import finished: {} of {} employee folder(s) registered", done, total)
        return done

    def _save(self, emp_id, name, department, samples, force):
        emp_id, name, department = emp_id.strip(), (name or "").strip(), (department or "").strip()
        if not ID_RE.match(emp_id):
            return False, "Employee ID may only contain letters, digits, - and _ (max 32)"
        name = name or emp_id  # only the employee ID is required

        embs = np.stack([x["emb"] for x in samples])
        centroid = embs.mean(0)
        centroid /= np.linalg.norm(centroid)
        if len(samples) > 1 and float((embs @ centroid).min()) < 0.30:
            return False, "Photos look like different people - please check them"
        if not force and len(self.registry):
            other, score, _ = self.registry.match_one(centroid, exclude=emp_id)
            if other is not None and score >= self.dup_thr:
                info = self.registry.info(other) or {}
                return False, f"DUPLICATE: this face is already registered as {info.get('name', other)} ({other})"

        folder = os.path.join(self.faces_dir, emp_id)
        shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder, exist_ok=True)
        rows = []
        # far-camera templates of every photo: a small face at the gate matches a small version of the photo much
        # better than the sharp close-up (one batched pass, stored next to the photo's own embedding)
        far = [v for x in samples for v in _far_views(x["face"])]
        far_embs = self.engine.recognizer.embed(far).reshape(len(samples), len(FAR_SIZES), -1)
        for i, x in enumerate(samples, 1):
            p = os.path.join(folder, f"{i:02d}.jpg").replace("\\", "/")
            cv2.imwrite(p, x["crop"])
            rows.append((x["emb"], x["quality"], p))
            rows.extend((e, x["quality"], p) for e in far_embs[i - 1])
        self.storage.save_employee(emp_id, name, department, rows)
        self.registry.reload()
        logger.info("Enrolled {} ({}) with {} photos ({} templates)", name, emp_id, len(samples), len(rows))
        return True, f"{name} ({emp_id}) registered with {len(samples)} photos"

    def delete_all(self):
        """Removes every employee and every stored face photo."""
        self.cancel_all()
        n = self.storage.clear_employees()
        shutil.rmtree(self.faces_dir, ignore_errors=True)
        os.makedirs(self.faces_dir, exist_ok=True)
        self.registry.reload()
        logger.info("All employees deleted ({})", n)
        return n

    def delete(self, emp_id):
        ok = self.storage.delete_employee(emp_id)
        shutil.rmtree(os.path.join(self.faces_dir, emp_id), ignore_errors=True)
        self.registry.reload()
        return ok
