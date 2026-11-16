"""Smart Gate Attendance - entry point.

    python main.py                              run all gates + web dashboard
    python main.py --no-window                  headless (Jetson / server / cloud)
    python main.py --import-faces D:ll_faces          bulk-register: one folder per employee, named by employee ID
    python main.py --reset                      clear attendance + alerts (keeps employees)
"""
import argparse
import os
import signal
import sys
import threading

import cv2
import yaml
from loguru import logger

from core.attendance import AttendanceService
from core.engine import FaceEngine
from core.enrollment import EnrollmentService
from core.gate import CameraManager
from core.storage import FaceRegistry, Storage
from core.tuning import Tuning
from core.videos import VideoLibrary


def setup_logging(cfg):
    log_dir = cfg.get("log_dir", "logs")
    os.makedirs(log_dir, exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level=cfg.get("log_level", "INFO"),
               format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | {message}")
    logger.add(os.path.join(log_dir, "system_{time:YYYY-MM-DD}.log"), level="INFO", rotation="00:00",
               retention="30 days", enqueue=True)
    logger.add(os.path.join(log_dir, "audit_{time:YYYY-MM-DD}.log"), level="INFO", rotation="00:00",
               retention="180 days", enqueue=True, filter=lambda r: r["extra"].get("audit"),
               format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")


def main():
    ap = argparse.ArgumentParser(description="Smart Gate Attendance")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--no-window", action="store_true", help="disable OpenCV windows")
    ap.add_argument("--import-faces", metavar="DIR", help="bulk-register employees: one photo folder per employee, folder name = employee ID")
    ap.add_argument("--with-names", action="store_true", help="with --import-faces: folders are named <EMP_ID>_<Name>")
    ap.add_argument("--padding", default="auto",
                    help="with --import-faces: border around each photo, 'auto' (default) or a percent, e.g. 40")
    ap.add_argument("--reset", action="store_true", help="delete all attendance events and alerts")
    args = ap.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    app = cfg.get("app", {})
    setup_logging(app)
    data_dir = app.get("data_dir", "data")
    faces_dir = os.path.join(data_dir, "faces")
    snapshot_dir = os.path.join(data_dir, "snapshots")

    storage = Storage(os.path.join(data_dir, "attendance.db"))
    if args.reset:
        if input("Delete ALL attendance events and alerts? Type YES: ").strip() == "YES":
            storage.clear_attendance()
            logger.info("Attendance data cleared")
        return

    engine = FaceEngine(cfg["inference"])
    registry = FaceRegistry(storage)
    cfg.setdefault("enrollment", {})
    enrollment = EnrollmentService(engine, storage, registry, cfg["enrollment"], cfg["liveness"], faces_dir,
                                   cfg["quality"])
    if args.import_faces:
        if not os.path.isdir(args.import_faces):
            logger.error("Folder not found: {}", args.import_faces)
            return
        pad = None if str(args.padding).lower() == "auto" else max(0.0, min(1.5, float(args.padding) / 100))
        n = enrollment.import_dir(args.import_faces, args.with_names, pad)
        logger.info("Imported {} employee(s)", n)
        return

    attendance = AttendanceService(storage, cfg.get("attendance", {}), app.get("reports_dir", "reports"), snapshot_dir)
    cameras = CameraManager(storage, engine, registry, attendance, cfg)
    tuning = Tuning(cfg, storage, engine, cameras)
    videos = VideoLibrary(os.path.join(data_dir, "videos"), cfg.get("web", {}).get("max_video_mb", 2048))
    logger.info("{} camera(s): {}", len(cameras.workers),
                ", ".join(f"{c['cam_id']} [{c['run_mode'].upper()}]" for c in cameras.status()) or "none")
    if any(c["run_mode"] == "setup" for c in cameras.status()):
        logger.info("Cameras in SETUP mode: register employees, then press 'Start Live Attendance' on the dashboard")

    if cfg.get("web", {}).get("enabled", True):
        from web.server import start_server
        start_server({"cfg": cfg, "storage": storage, "registry": registry, "attendance": attendance,
                      "enrollment": enrollment, "cameras": cameras, "snapshot_dir": snapshot_dir,
                      "faces_dir": faces_dir, "tuning": tuning, "videos": videos})

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, lambda *_: stop.set())

    show = cfg.get("display", {}).get("window", True) and not args.no_window
    if show and not os.environ.get("DISPLAY") and sys.platform.startswith("linux"):
        logger.warning("No display found (server / SSH) - running headless")
        show = False
    try:
        shown = {}
        while show and not stop.is_set():
            for cam_id in [c for c in shown if cameras.get(c) is None or cameras.get(c).output() is None]:
                cv2.destroyWindow(f"{cam_id}  (Q to quit)")  # camera removed or switched off
                del shown[cam_id]
            for cam_id, w in list(cameras.workers.items()):
                out = w.output()
                if out is not None and shown.get(cam_id) is not out:
                    shown[cam_id] = out
                    try:
                        cv2.imshow(f"{cam_id}  (Q to quit)", out)
                    except cv2.error:
                        logger.warning("OpenCV has no GUI support (headless build) - running headless")
                        show = False
                        break
            if show and cv2.waitKey(10) & 0xFF in (ord("q"), ord("Q"), 27):
                stop.set()
        if not stop.is_set():
            logger.info("Headless mode - press Ctrl+C to stop")
            while not stop.wait(1.0):
                pass
    finally:
        logger.info("Shutting down...")
        cameras.stop_all()
        attendance.close()
        storage.close()
        if show:
            cv2.destroyAllWindows()
        logger.info("Stopped cleanly")


if __name__ == "__main__":
    main()
