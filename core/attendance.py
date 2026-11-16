"""Attendance rules (unique / multiple, chosen per camera), passage de-duplication and the background DB writer.

Every confirmed passage is stored as an event. The camera's mode decides whether it counts as attendance:
  unique   -> only the first passage of the day at that camera counts
  multiple -> every passage at that camera counts
Every camera keeps its own record, so a person walking past 5 cameras gets 5 records (one per camera).
"""
import os
import queue
import shutil
import threading
import time
from datetime import date, datetime

import cv2
from loguru import logger

from core import reports

MODES = ("unique", "multiple")
RUN_MODES = ("setup", "live", "off")


class AttendanceService:
    def __init__(self, storage, cfg, reports_dir, snapshot_dir):
        self.storage = storage
        self.reports_dir = reports_dir
        self.snapshot_dir = snapshot_dir
        self.cooldown = float(cfg.get("cooldown_seconds", 30))
        self.export_every = float(cfg.get("excel_autosave_seconds", 30))

        self._lock = threading.Lock()
        self._day = None
        self._marked = {}       # (camera, emp) -> first counted time today (unique cameras)
        self._last_pass = {}    # (camera, emp) -> epoch of last stored passage
        self._stats, self._stats_at = {}, 0.0
        self._roll_day()

        self._q = queue.Queue()
        self._dirty = True
        self._writer = threading.Thread(target=self._write_loop, name="db-writer", daemon=True)
        self._writer.start()
        logger.info("Attendance service ready | cooldown {}s", self.cooldown)

    # ---------------- state ----------------
    def _roll_day(self):
        today = date.today().isoformat()
        if today == self._day:
            return
        self._day = today
        self._marked = self.storage.unique_marked(today)
        self._last_pass = {}

    def stats(self):
        """Today's counts from the DB, cached for 1 s (called by every camera's HUD on every frame)."""
        with self._lock:
            self._roll_day()
            if time.time() - self._stats_at >= 1.0:
                self._stats = dict(self.storage.counts(self._day), day=self._day)
                self._stats_at = time.time()
            return self._stats

    def clear(self):
        """Deletes all attendance events, alerts, snapshots and Excel reports (privacy wipe)."""
        with self._lock:
            self.storage.clear_attendance()
            self._marked, self._last_pass, self._stats_at = {}, {}, 0.0
            self._dirty = False
        for folder in (self.snapshot_dir, self.reports_dir):
            shutil.rmtree(folder, ignore_errors=True)
            os.makedirs(folder, exist_ok=True)
        logger.info("Attendance records, alerts, snapshots and reports deleted")

    # ---------------- events ----------------
    def record(self, emp_id, name, department, gate_id, gate_name, camera_id, direction, mode, score, liveness):
        """mode: the camera's attendance mode. Returns (status, message); status MARKED | ALREADY | COOLDOWN."""
        now = time.time()
        key = (camera_id, emp_id)  # every camera keeps its own attendance
        with self._lock:
            self._roll_day()
            if now - self._last_pass.get(key, 0.0) < self.cooldown:
                return "COOLDOWN", name
            self._last_pass[key] = now
            ts = datetime.now()
            if mode == "multiple":
                counted = True
            else:
                counted = key not in self._marked
                if counted:
                    self._marked[key] = ts.strftime("%H:%M:%S")
            first_time = self._marked.get(key)

        self._q.put(("event", {
            "ts": ts.strftime("%Y-%m-%d %H:%M:%S"), "day": ts.date().isoformat(), "emp_id": emp_id,
            "name": name, "department": department, "gate_id": gate_id, "gate_name": gate_name,
            "camera_id": camera_id, "direction": direction, "match_score": round(float(score), 4),
            "liveness": round(float(liveness), 4), "attendance_mode": mode, "counted": int(counted)}))
        logger.bind(audit=True).info("PASSAGE {} | {} ({}) | {} {} [{}] | {} | match {:.1%} | live {:.1%}",
                                     "COUNTED" if counted else "REPEAT", name, emp_id, gate_name, direction,
                                     camera_id, mode.upper(), score, liveness)
        if counted:
            return "MARKED", f"{direction}: {name}" if mode == "multiple" else f"Attendance marked: {name}"
        return "ALREADY", f"{name} - already marked at {first_time}"

    def alert(self, kind, gate_id, camera_id, score, liveness, detail, snapshot=None):
        ts = datetime.now()
        self._q.put(("alert", {
            "ts": ts.strftime("%Y-%m-%d %H:%M:%S"), "day": ts.date().isoformat(), "kind": kind,
            "gate_id": gate_id, "camera_id": camera_id, "match_score": round(float(score), 4),
            "liveness": round(float(liveness), 4), "detail": detail, "snapshot": ""}, snapshot))
        logger.bind(audit=True).warning("{} | {} {} | match {:.1%} | live {:.1%} | {}",
                                        kind, gate_id, camera_id, score, liveness, detail)

    # ---------------- writer ----------------
    def _write_loop(self):
        last_export = 0.0
        while True:
            try:
                item = self._q.get(timeout=1.0)
            except queue.Empty:
                item = None
            if item is not None and item[0] == "stop":
                break
            try:
                if item is not None:
                    self._write(item)
                if self._dirty and time.time() - last_export >= self.export_every:
                    self._export()
                    last_export = time.time()
            except Exception:
                logger.exception("Writer error")
        try:
            while not self._q.empty():
                item = self._q.get_nowait()
                if item[0] != "stop":
                    self._write(item)
            self._export()
        except Exception:
            logger.exception("Writer error during shutdown")

    def _write(self, item):
        kind, rec = item[0], item[1]
        if kind == "event":
            self.storage.add_event(rec)
        elif kind == "alert":
            img = item[2]
            if img is not None and img.size:
                folder = os.path.join(self.snapshot_dir, rec["day"])
                os.makedirs(folder, exist_ok=True)
                path = os.path.join(folder, f"{rec['ts'][11:].replace(':', '')}_{rec['kind']}_{rec['camera_id']}.jpg")
                cv2.imwrite(path, img)
                rec["snapshot"] = path.replace("\\", "/")
            self.storage.add_alert(rec)
        self._dirty = True

    def _export(self):
        if reports.export_daily(self.storage, self._day, self.reports_dir):
            self._dirty = False

    def close(self):
        self._q.put(("stop",))
        self._writer.join(timeout=10)
