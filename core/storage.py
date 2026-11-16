"""SQLite persistence (WAL mode) and the in-memory face registry used for matching."""
import json
import os
import sqlite3
import threading
from datetime import datetime

import numpy as np
from loguru import logger

SCHEMA = """
CREATE TABLE IF NOT EXISTS employees (
    emp_id      TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    department  TEXT DEFAULT '',
    active      INTEGER DEFAULT 1,
    created_at  TEXT,
    updated_at  TEXT
);
CREATE TABLE IF NOT EXISTS face_samples (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    emp_id      TEXT NOT NULL REFERENCES employees(emp_id) ON DELETE CASCADE,
    embedding   BLOB NOT NULL,
    quality     REAL DEFAULT 0,
    image_path  TEXT,
    created_at  TEXT
);
CREATE INDEX IF NOT EXISTS idx_samples_emp ON face_samples(emp_id);
CREATE TABLE IF NOT EXISTS events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           TEXT NOT NULL,
    day          TEXT NOT NULL,
    emp_id       TEXT NOT NULL,
    name         TEXT,
    department   TEXT,
    gate_id      TEXT,
    gate_name    TEXT,
    camera_id    TEXT,
    direction    TEXT,
    match_score  REAL,
    liveness     REAL,
    attendance_mode TEXT DEFAULT 'unique',
    counted      INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_events_day ON events(day, emp_id);
CREATE TABLE IF NOT EXISTS alerts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,
    day         TEXT NOT NULL,
    kind        TEXT NOT NULL,
    gate_id     TEXT,
    camera_id   TEXT,
    match_score REAL,
    liveness    REAL,
    detail      TEXT,
    snapshot    TEXT
);
CREATE INDEX IF NOT EXISTS idx_alerts_day ON alerts(day);
CREATE TABLE IF NOT EXISTS cameras (
    cam_id          TEXT PRIMARY KEY,
    gate_id         TEXT NOT NULL,
    gate_name       TEXT NOT NULL,
    direction       TEXT NOT NULL DEFAULT 'ENTRY',
    source          TEXT NOT NULL,
    width           INTEGER DEFAULT 1280,
    height          INTEGER DEFAULT 720,
    fps             INTEGER DEFAULT 30,
    roi             TEXT,
    attendance_mode TEXT NOT NULL DEFAULT 'unique',
    run_mode        TEXT NOT NULL DEFAULT 'setup',
    sort_order      INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class Storage:
    """Single shared connection guarded by a lock; WAL keeps reads and writes concurrent-safe."""

    def __init__(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(path, check_same_thread=False, timeout=30)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(SCHEMA)
            cols = {r[1] for r in self._conn.execute("PRAGMA table_info(events)")}
            for col, ddl in (("attendance_mode", "TEXT DEFAULT 'unique'"), ("counted", "INTEGER DEFAULT 0")):
                if col not in cols:  # databases created by an earlier version
                    self._conn.execute(f"ALTER TABLE events ADD COLUMN {col} {ddl}")
            self._conn.commit()
        logger.info("Database ready: {}", path)

    def execute(self, sql, params=()):
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return cur

    def query(self, sql, params=()):
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

    def close(self):
        with self._lock:
            self._conn.close()

    # ---------------- employees ----------------
    def save_employee(self, emp_id, name, department, samples):
        """Creates/updates an employee and replaces all face samples in one transaction.
        samples: list of (embedding float32[512], quality, image_path)."""
        ts = now_str()
        with self._lock:
            with self._conn:
                self._conn.execute(
                    "INSERT INTO employees (emp_id, name, department, active, created_at, updated_at) "
                    "VALUES (?, ?, ?, 1, ?, ?) ON CONFLICT(emp_id) DO UPDATE SET "
                    "name=excluded.name, department=excluded.department, active=1, updated_at=excluded.updated_at",
                    (emp_id, name, department, ts, ts))
                self._conn.execute("DELETE FROM face_samples WHERE emp_id=?", (emp_id,))
                self._conn.executemany(
                    "INSERT INTO face_samples (emp_id, embedding, quality, image_path, created_at) VALUES (?, ?, ?, ?, ?)",
                    [(emp_id, np.asarray(e, np.float32).tobytes(), float(q), p, ts) for e, q, p in samples])

    def delete_employee(self, emp_id):
        with self._lock:
            with self._conn:
                self._conn.execute("DELETE FROM face_samples WHERE emp_id=?", (emp_id,))
                return self._conn.execute("DELETE FROM employees WHERE emp_id=?", (emp_id,)).rowcount > 0

    def clear_employees(self):
        with self._lock:
            with self._conn:
                self._conn.execute("DELETE FROM face_samples")
                return self._conn.execute("DELETE FROM employees").rowcount

    def employees(self):
        return self.query(
            "SELECT e.emp_id, e.name, e.department, e.created_at, e.updated_at, COUNT(DISTINCT s.image_path) AS samples "
            "FROM employees e LEFT JOIN face_samples s ON s.emp_id = e.emp_id "
            "WHERE e.active = 1 GROUP BY e.emp_id ORDER BY e.emp_id")

    def all_samples(self):
        return self.query(
            "SELECT e.emp_id, e.name, e.department, s.embedding FROM face_samples s "
            "JOIN employees e ON e.emp_id = s.emp_id WHERE e.active = 1 ORDER BY e.emp_id, s.id")

    # ---------------- attendance ----------------
    def add_event(self, ev):
        self.execute(
            "INSERT INTO events (ts, day, emp_id, name, department, gate_id, gate_name, camera_id, direction, "
            "match_score, liveness, attendance_mode, counted) VALUES (:ts, :day, :emp_id, :name, :department, "
            ":gate_id, :gate_name, :camera_id, :direction, :match_score, :liveness, :attendance_mode, :counted)", ev)

    def add_alert(self, al):
        self.execute(
            "INSERT INTO alerts (ts, day, kind, gate_id, camera_id, match_score, liveness, detail, snapshot) "
            "VALUES (:ts, :day, :kind, :gate_id, :camera_id, :match_score, :liveness, :detail, :snapshot)", al)

    @staticmethod
    def _filters(day, gate_id=None, camera_id=None, alias=""):
        where, params = [f"{alias}day=?"], [day]
        if gate_id:
            where.append(f"{alias}gate_id=?")
            params.append(gate_id)
        if camera_id:
            where.append(f"{alias}camera_id=?")
            params.append(camera_id)
        return " AND ".join(where), params

    def events(self, day, gate_id=None, camera_id=None, counted_only=False, limit=None):
        where, params = self._filters(day, gate_id, camera_id)
        if counted_only:
            where += " AND counted=1"
        sql = f"SELECT * FROM events WHERE {where} ORDER BY id DESC" + (f" LIMIT {int(limit)}" if limit else "")
        return self.query(sql, params)

    def alerts(self, day, gate_id=None, camera_id=None, limit=None):
        where, params = self._filters(day, gate_id, camera_id)
        sql = f"SELECT * FROM alerts WHERE {where} ORDER BY id DESC" + (f" LIMIT {int(limit)}" if limit else "")
        return self.query(sql, params)

    def daily_summary(self, day, gate_id=None, camera_id=None):
        """One row per employee: first / last passage, passage count and attendance records."""
        where, params = self._filters(day, gate_id, camera_id)
        return self.query(
            f"SELECT g.*, f.gate_name || ' ' || f.direction AS first_gate, l.gate_name || ' ' || l.direction AS last_gate "
            f"FROM (SELECT emp_id, MAX(name) AS name, MAX(department) AS department, MIN(ts) AS first_seen, "
            f"MAX(ts) AS last_seen, COUNT(*) AS passages, SUM(counted) AS records, MIN(id) AS fid, MAX(id) AS lid "
            f"FROM events WHERE {where} GROUP BY emp_id) g "
            f"JOIN events f ON f.id = g.fid JOIN events l ON l.id = g.lid ORDER BY g.first_seen", params)

    def counts(self, day):
        r = self.query("SELECT COUNT(*) AS passages, COALESCE(SUM(counted), 0) AS records, "
                       "COUNT(DISTINCT CASE WHEN counted=1 THEN emp_id END) AS present FROM events WHERE day=?", (day,))[0]
        r["alerts"] = self.query("SELECT COUNT(*) AS n FROM alerts WHERE day=?", (day,))[0]["n"]
        return r

    def unique_marked(self, day):
        """(camera_id, emp_id) -> first counted time, for restoring unique-mode state after a restart."""
        rows = self.query("SELECT camera_id, emp_id, MIN(ts) AS ts FROM events "
                          "WHERE day=? AND counted=1 AND attendance_mode='unique' GROUP BY camera_id, emp_id", (day,))
        return {(r["camera_id"], r["emp_id"]): r["ts"][11:] for r in rows}

    # ---------------- cameras ----------------
    CAMERA_FIELDS = ("cam_id", "gate_id", "gate_name", "direction", "source", "width", "height", "fps", "roi",
                     "attendance_mode", "run_mode", "sort_order")

    def cameras(self):
        rows = self.query("SELECT * FROM cameras ORDER BY gate_id, sort_order, cam_id")
        for r in rows:
            r["roi"] = json.loads(r["roi"]) if r["roi"] else None
        return rows

    def save_camera(self, cam):
        row = {k: cam.get(k) for k in self.CAMERA_FIELDS}
        row["roi"] = json.dumps(row["roi"]) if row.get("roi") else None
        self.execute(
            "INSERT INTO cameras (" + ", ".join(self.CAMERA_FIELDS) + ") VALUES ("
            + ", ".join(":" + k for k in self.CAMERA_FIELDS) + ") ON CONFLICT(cam_id) DO UPDATE SET "
            + ", ".join(f"{k}=excluded.{k}" for k in self.CAMERA_FIELDS if k != "cam_id"), row)

    def delete_camera(self, cam_id):
        return self.execute("DELETE FROM cameras WHERE cam_id=?", (cam_id,)).rowcount > 0

    def clear_attendance(self):
        with self._lock:
            with self._conn:
                self._conn.execute("DELETE FROM events")
                self._conn.execute("DELETE FROM alerts")

    # ---------------- settings ----------------
    def get_setting(self, key, default=None):
        rows = self.query("SELECT value FROM settings WHERE key=?", (key,))
        return rows[0]["value"] if rows else default

    def set_setting(self, key, value):
        self.execute("INSERT INTO settings (key, value) VALUES (?, ?) "
                     "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, str(value)))


class FaceRegistry:
    """Enrolled embeddings as one matrix. Score per employee = mean(best sample, centroid).
    Reload swaps an immutable snapshot, so camera threads never see a half-updated registry."""

    def __init__(self, storage):
        self.storage = storage
        self._snap = None
        self.reload()

    def reload(self):
        rows = self.storage.all_samples()
        if not rows:
            self._snap = None
            logger.warning("Face registry is empty - enroll employees from the web dashboard")
            return
        ids, info, starts, embs = [], [], [], []
        for i, r in enumerate(rows):
            if not ids or ids[-1] != r["emp_id"]:
                ids.append(r["emp_id"])
                info.append({"emp_id": r["emp_id"], "name": r["name"], "department": r["department"]})
                starts.append(i)
            embs.append(np.frombuffer(r["embedding"], dtype=np.float32))
        E = np.stack(embs)
        starts = np.array(starts)
        C = np.add.reduceat(E, starts, axis=0)
        C /= np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-9)
        self._snap = (ids, info, E, starts, C)
        logger.info("Face registry loaded: {} employees, {} samples", len(ids), len(E))

    def __len__(self):
        return 0 if self._snap is None else len(self._snap[0])

    def info(self, emp_id):
        snap = self._snap
        if snap is None or emp_id not in snap[0]:
            return None
        return snap[1][snap[0].index(emp_id)]

    def match(self, Q, exclude=None):
        """Q: (N, 512) normalized. Returns lists (emp_ids or None, best scores, second-best scores)."""
        n = len(Q)
        snap = self._snap
        if snap is None or n == 0:
            return [None] * n, np.zeros(n), np.zeros(n)
        ids, _, E, starts, C = snap
        scores = 0.5 * (np.maximum.reduceat(Q @ E.T, starts, axis=1) + Q @ C.T)
        if exclude is not None and exclude in ids:
            scores[:, ids.index(exclude)] = -1.0
        if scores.shape[1] == 1:
            return [ids[0]] * n, scores[:, 0], np.zeros(n)
        top2 = np.argpartition(-scores, 1, axis=1)[:, :2]
        s2 = np.take_along_axis(scores, top2, axis=1)
        order = np.argsort(-s2, axis=1)
        best_idx = np.take_along_axis(top2, order, axis=1)[:, 0]
        s2 = np.take_along_axis(s2, order, axis=1)
        return [ids[i] for i in best_idx], s2[:, 0], s2[:, 1]

    def match_one(self, q, exclude=None):
        ids, best, second = self.match(q[None, :], exclude)
        return ids[0], float(best[0]), float(second[0])
