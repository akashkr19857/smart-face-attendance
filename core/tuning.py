"""Runtime tuning: detection / quality / recognition / liveness thresholds edited from the dashboard.

Values are written into the shared config dicts in place (every camera thread reads them per frame), so a
change works at once without a restart. Changes are saved in the database and survive restarts.
"""
import json

from loguru import logger

# "section.key": (kind, min, max, default when missing in config.yaml)
FIELDS = {
    "inference.det_size": ("int", 320, 1920, 640),
    "inference.det_thresh": ("float", 0.2, 0.95, 0.5),
    "inference.track_thresh": ("float", 0.1, 0.9, 0.3),
    "inference.low_light_boost": ("bool", 0, 1, True),
    "quality.min_face_px": ("int", 20, 200, 64),
    "quality.min_det_score": ("float", 0.2, 0.95, 0.6),
    "quality.min_sharpness": ("float", 0, 200, 25.0),
    "quality.max_yaw": ("float", 0.2, 1.0, 0.45),
    "recognition.match_threshold": ("float", 0.25, 0.8, 0.45),
    "recognition.margin": ("float", 0.0, 0.3, 0.08),
    "recognition.frame_threshold": ("float", 0.15, 0.7, 0.35),
    "recognition.min_frames": ("int", 1, 30, 2),
    "recognition.vote_ratio": ("float", 0.3, 1.0, 0.5),
    "liveness.enabled": ("bool", 0, 1, True),
    "liveness.min_face_px": ("int", 24, 200, 80),
    "liveness.min_frames": ("int", 1, 15, 2),
    "liveness.real_threshold": ("float", 0.3, 0.95, 0.5),
    "liveness.spoof_threshold": ("float", 0.05, 0.6, 0.3),
    "enrollment.min_face_px": ("int", 24, 200, 110),
}

# Ready-made settings. Anything not listed falls back to the config.yaml value.
PRESETS = {
    "standard": {},  # config.yaml values: face-height camera, people walk up to the gate
    "far": {         # CCTV high up / top-down / side view, people walking past at a distance
        "inference.det_size": 1280, "inference.det_thresh": 0.45, "inference.track_thresh": 0.3,
        "quality.min_face_px": 36, "quality.min_det_score": 0.5, "quality.min_sharpness": 12.0,
        "quality.max_yaw": 0.65,
        "recognition.match_threshold": 0.40, "recognition.margin": 0.08, "recognition.frame_threshold": 0.30,
        "recognition.min_frames": 2, "recognition.vote_ratio": 0.5,
        "liveness.min_face_px": 40, "liveness.min_frames": 2, "liveness.real_threshold": 0.5,
        "enrollment.min_face_px": 48,
    },
    "topdown": {     # ceiling CCTV high above a lobby / corridor: tiny faces (20-50 px), people looking down
        # 1920 = a 1080p frame is searched at full resolution (nothing shrunk away); tighten the ROI for more zoom
        "inference.det_size": 1920, "inference.det_thresh": 0.40, "inference.track_thresh": 0.25,
        "quality.min_face_px": 24, "quality.min_det_score": 0.40, "quality.min_sharpness": 6.0,
        "quality.max_yaw": 0.75,
        "recognition.match_threshold": 0.38, "recognition.margin": 0.07, "recognition.frame_threshold": 0.28,
        "recognition.min_frames": 2, "recognition.vote_ratio": 0.5,
        "liveness.min_face_px": 64, "liveness.min_frames": 2, "liveness.real_threshold": 0.5,
        "enrollment.min_face_px": 32,
    },
}


class Tuning:
    def __init__(self, cfg, storage, engine, cameras):
        self.cfg, self.storage, self.engine, self.cameras = cfg, storage, engine, cameras
        self.defaults = {k: self._get(k) for k in FIELDS}
        saved = json.loads(storage.get_setting("tuning") or "{}")
        if saved:
            self._apply(saved)
            logger.info("Tuning restored from database: {}", saved)

    def _get(self, key):
        section, name = key.split(".")
        return self.cfg.get(section, {}).get(name, FIELDS[key][3])

    def values(self):
        return {k: self._get(k) for k in FIELDS}

    def _apply(self, data):
        """Validates and writes values into the live config. Returns the cleaned changes."""
        clean = {}
        for key, value in data.items():
            if key not in FIELDS:
                raise ValueError(f"Unknown setting: {key}")
            kind, lo, hi, _ = FIELDS[key]
            try:
                if kind == "bool":
                    v = value if isinstance(value, bool) else str(value).lower() in ("1", "true", "yes", "on")
                else:
                    v = int(float(value)) if kind == "int" else round(float(value), 4)
            except (TypeError, ValueError):
                raise ValueError(f"{key}: not a number")
            if kind != "bool" and not lo <= v <= hi:
                raise ValueError(f"{key} must be between {lo} and {hi}")
            clean[key] = v
        for key, v in clean.items():
            section, name = key.split(".")
            self.cfg.setdefault(section, {})[name] = v
        self.engine.update(self.cfg["inference"])
        for w in list(self.cameras.workers.values()):
            w.tracker.high_thresh = self.cfg["inference"].get("det_thresh", 0.5)
        return clean

    def update(self, data):
        clean = self._apply(data)
        saved = json.loads(self.storage.get_setting("tuning") or "{}")
        saved.update(clean)
        saved = {k: v for k, v in saved.items() if v != self.defaults.get(k)}
        self.storage.set_setting("tuning", json.dumps(saved))
        logger.info("Tuning changed: {}", clean)
        return self.values()

    def preset(self, name):
        if name not in PRESETS:
            raise ValueError(f"Unknown preset: {name}")
        values = dict(self.defaults)
        values.update(PRESETS[name])
        return self.update(values)
