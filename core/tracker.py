"""Face tracking (two-stage IoU association, ByteTrack style) and the per-track multi-frame verdict."""
from collections import deque

import numpy as np
from scipy.optimize import linear_sum_assignment

PENDING, VERIFIED, SPOOF = "PENDING", "VERIFIED", "SPOOF"  # SPOOF is a verdict; the track keeps a spoof flag


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


class Track:
    def __init__(self, tid, box, kps, t, live_window, vote_window=15):
        self.id = tid
        self.box = box[:4].copy()
        self.det_score = float(box[4])
        self.kps = kps
        self.vel = np.zeros(4, np.float32)
        self.t_last = t
        self.matched = True

        self.state = PENDING
        self.emb_sum = np.zeros(512, np.float32)
        self.samples = 0
        self.votes = deque(maxlen=vote_window)
        self.live = deque(maxlen=live_window)
        self.unknown_reported = False
        self.spoof = False
        self.spoof_since = 0.0
        self.spoof_snap = None
        self.spoof_logged = False

        self.emp_id = None
        self.name = ""
        self.department = ""
        self.match_score = 0.0
        self.liveness = 0.0
        self.message = ""

    @property
    def size(self):
        return float(min(self.box[2] - self.box[0], self.box[3] - self.box[1]))

    def predicted(self, t):
        return self.box + self.vel * min(t - self.t_last, 0.5)

    def update(self, box, kps, t):
        dt = max(t - self.t_last, 1e-3)
        self.vel = 0.6 * self.vel + 0.4 * (box[:4] - self.box) / dt
        self.box = box[:4].copy()
        self.det_score = float(box[4])
        self.kps = kps
        self.t_last = t
        self.matched = True

    def observe(self, emb, weight, frame_emp, live_prob):
        self.emb_sum += emb * weight
        self.samples += 1
        self.votes.append((frame_emp, weight))  # a weak frame (blur, side view, tiny) votes with less weight
        if live_prob is not None:
            self.live.append(float(live_prob))

    def decide(self, registry, rcfg, lcfg, final=False):
        """Returns "VERIFIED", "SPOOF", "UNKNOWN" (first time only) or None while still collecting.
        final: the person just left the view - decide with the frames there are (a slightly higher match is
        required when fewer than min_frames were seen), so a fast walker is not lost while "Verifying"."""
        if self.state != PENDING:
            return None

        # Liveness = the model's score only: mean of the last few readings (fast for people walking past).
        # A face too small for the liveness model has no readings and is not blocked (far CCTV views).
        live_on = lcfg.get("enabled", True)
        live_ok, verdict = True, None
        if live_on and self.live:
            n = max(1, int(lcfg.get("min_frames", 2)))
            recent = np.array(self.live, np.float32)[-n:]
            self.liveness = float(recent.mean())
            enough = len(recent) >= n or final
            if enough and self.liveness < lcfg.get("spoof_threshold", 0.30):
                if not self.spoof:
                    self.spoof, verdict, self.spoof_since = True, SPOOF, self.t_last
            elif self.spoof and self.liveness >= lcfg.get("real_threshold", 0.5):
                self.spoof = False  # model now sees a real face (e.g. bad light at first)
            if self.spoof:
                self.message = "FAKE FACE - BLOCKED"
                return verdict
            live_ok = self.liveness >= lcfg.get("real_threshold", 0.5)

        few = self.samples < rcfg.get("min_frames", 2)
        if few and not final:
            self.message = f"Verifying {self.samples}/{rcfg.get('min_frames', 5)}"
            return None

        fused = self.emb_sum / max(np.linalg.norm(self.emb_sum), 1e-9)
        emp_id, best, second = registry.match_one(fused)
        self.match_score = best
        total = sum(w for _, w in self.votes)
        agree = sum(w for e, w in self.votes if e == emp_id) / max(total, 1e-9)
        need = rcfg.get("match_threshold", 0.45) + (0.04 if few else 0.0)
        identified = (emp_id is not None and best >= need
                      and best - second >= rcfg.get("margin", 0.08) and agree >= rcfg.get("vote_ratio", 0.6))

        if identified:
            if live_ok:
                info = registry.info(emp_id) or {}
                self.state = VERIFIED
                self.emp_id = emp_id
                self.name = info.get("name", emp_id)
                self.department = info.get("department", "")
                return VERIFIED
            self.message = "Checking liveness..."
            return None

        if final:
            return None
        if not self.unknown_reported and self.samples >= rcfg.get("unknown_after_frames", 30):
            self.unknown_reported = True
            self.message = "Unregistered person"
            return "UNKNOWN"
        if not self.unknown_reported:
            self.message = "Identifying..."
        return None


class FaceTracker:
    def __init__(self, iou_threshold=0.25, max_lost_seconds=1.0, high_thresh=0.5, live_window=10):
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost_seconds
        self.high_thresh = high_thresh
        self.live_window = live_window
        self.tracks = {}
        self.removed = []  # tracks dropped in the last update
        self._next_id = 1

    def _associate(self, tracks, boxes, det_idx, t):
        if not tracks or len(det_idx) == 0:
            return tracks, det_idx
        pred = np.array([tr.predicted(t) for tr in tracks])
        iou = iou_matrix(pred, boxes[det_idx, :4])
        rows, cols = linear_sum_assignment(-iou)
        used_t, used_d = set(), set()
        for r, c in zip(rows, cols):
            if iou[r, c] >= self.iou_threshold:
                tracks[r].update(boxes[det_idx[c]], self._kps[det_idx[c]], t)
                used_t.add(r)
                used_d.add(c)
        rest_t = [tr for i, tr in enumerate(tracks) if i not in used_t]
        rest_d = np.array([d for i, d in enumerate(det_idx) if i not in used_d], dtype=int)
        return rest_t, rest_d

    def update(self, boxes, kps, t):
        """boxes (N, 5), kps (N, 5, 2). Returns tracks matched in this frame."""
        self._kps = kps
        for tr in self.tracks.values():
            tr.matched = False
        scores = boxes[:, 4] if len(boxes) else np.zeros(0)
        high = np.where(scores >= self.high_thresh)[0]
        low = np.where(scores < self.high_thresh)[0]

        pool = list(self.tracks.values())
        pool, new_dets = self._associate(pool, boxes, high, t)
        self._associate(pool, boxes, low, t)  # low-confidence boxes only extend existing tracks

        for d in new_dets:
            tr = Track(self._next_id, boxes[d], kps[d], t, self.live_window)
            self.tracks[tr.id] = tr
            self._next_id += 1

        self.removed = [tr for tr in self.tracks.values() if t - tr.t_last > self.max_lost]
        for tr in self.removed:
            del self.tracks[tr.id]
        return [tr for tr in self.tracks.values() if tr.matched]
