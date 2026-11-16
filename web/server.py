"""Dashboard + enrollment web API (Starlette). Works on the gate PC, a Jetson, or a cloud VM."""
import asyncio
import os
import threading
import time
from datetime import date
from urllib.parse import parse_qs

import cv2
import numpy as np
import uvicorn
from loguru import logger
from starlette.applications import Starlette
from starlette.responses import FileResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from core import reports
from core.videos import safe_name

STATIC = os.path.join(os.path.dirname(__file__), "index.html")


def _decode(data):
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image")
    return img


def create_app(ctx):
    """ctx: dict with storage, registry, attendance, enrollment, cameras (CameraManager), cfg, dirs."""
    wcfg = ctx["cfg"].get("web", {})
    api_key = str(wcfg.get("api_key") or "")
    stream_fps = float(wcfg.get("stream_fps", 12))
    stream_w = int(wcfg.get("stream_max_width", 960))  # smaller preview = smooth over the internet / tunnels
    jpeg_q = [int(cv2.IMWRITE_JPEG_QUALITY), int(wcfg.get("stream_quality", 75))]
    storage, attendance, enroll, cams = ctx["storage"], ctx["attendance"], ctx["enrollment"], ctx["cameras"]
    tuning, videos = ctx["tuning"], ctx["videos"]
    snap_root = os.path.abspath(ctx["snapshot_dir"])
    enc_cache = {}  # cam_id -> (last frame, encoded MJPEG part)

    def err(msg, code=400):
        return JSONResponse({"ok": False, "message": msg}, status_code=code)

    def filters(request):
        q = request.query_params
        d = q.get("day") or date.today().isoformat()
        date.fromisoformat(d)
        return d, q.get("gate") or None, q.get("camera") or None

    async def index(request):
        return FileResponse(STATIC)

    def status(request):
        return JSONResponse({"stats": attendance.stats(), "employees": len(ctx["registry"]), "cameras": cams.status()})

    # ---------------- cameras ----------------
    def encode(worker, frame):
        """JPEG of the annotated frame, encoded once per frame for all viewers. Returns (number, jpeg) or None."""
        cached = enc_cache.get(worker.cam_id)
        if cached is not None and cached[0] is frame:
            return cached[1]
        small = frame
        if stream_w and frame.shape[1] > stream_w:
            small = cv2.resize(frame, (stream_w, int(frame.shape[0] * stream_w / frame.shape[1])),
                               interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", small, jpeg_q)
        out = ((cached[1][0] + 1) if cached and cached[1] else 1, buf.tobytes()) if ok else None
        enc_cache[worker.cam_id] = (frame, out)
        return out

    async def stream(request):
        """MJPEG stream (one long connection per viewer)."""
        worker = cams.get(request.path_params["cam_id"])
        if worker is None:
            return err("Unknown camera", 404)

        async def gen():
            period = 1.0 / stream_fps
            last = None
            while True:
                t0 = time.perf_counter()
                frame = worker.output()
                if frame is not None and frame is not last:  # paused video: nothing new to send
                    last = frame
                    out = await asyncio.to_thread(encode, worker, frame)
                    if out:
                        # waits while the viewer's connection is full, so a slow link never queues old frames
                        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + out[1] + b"\r\n"
                await asyncio.sleep(max(0.005, period - (time.perf_counter() - t0)))

        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

    async def snap(request):
        """Newest annotated frame as one JPEG (?n= last number seen: 204 when nothing is new). The Live grid polls
        this instead of MJPEG, so any number of cameras fits in the browser's 6 connections per site."""
        worker = cams.get(request.path_params["cam_id"])
        frame = worker.output() if worker else None
        if frame is None:
            return Response(status_code=204)
        out = await asyncio.to_thread(encode, worker, frame)
        if out is None or request.query_params.get("n") == str(out[0]):
            return Response(status_code=204)
        return Response(out[1], media_type="image/jpeg", headers={"X-Frame": str(out[0]), "Cache-Control": "no-store"})

    def camera_frame(request):
        """Single raw JPEG (no overlays) for drawing the ROI."""
        worker = cams.get(request.path_params["cam_id"])
        frame = worker.raw_frame() if worker else None
        if frame is None:
            return err("Camera has no frame (is it off?)", 404)
        ok, buf = cv2.imencode(".jpg", frame, jpeg_q)
        return Response(buf.tobytes(), media_type="image/jpeg")

    async def camera_create(request):
        ok, msg = await asyncio.to_thread(cams.save, await request.json(), True)
        return JSONResponse({"ok": ok, "message": msg}, status_code=200 if ok else 400)

    async def camera_update(request):
        data = await request.json()
        data["cam_id"] = request.path_params["cam_id"]
        ok, msg = await asyncio.to_thread(cams.save, data, False)
        return JSONResponse({"ok": ok, "message": msg}, status_code=200 if ok else 400)

    async def camera_delete(request):
        ok = await asyncio.to_thread(cams.delete, request.path_params["cam_id"])
        return JSONResponse({"ok": ok, "message": "Camera removed" if ok else "Not found"})

    async def camera_control(request):
        """{"action": "pause" | "resume" | "restart"} for video-file cameras."""
        worker = cams.get(request.path_params["cam_id"])
        action = (await request.json()).get("action")
        if worker is None or action not in ("pause", "resume", "restart"):
            return err("Unknown camera or action")
        if not worker.control(action):
            return err("Only video-file cameras can be paused or restarted")
        return JSONResponse({"ok": True})

    def checklist(request):
        cam_list = cams.status()
        return JSONResponse({"employees": len(ctx["registry"]), "cameras": cam_list,
                             "ready": len(ctx["registry"]) > 0 and any(c["run_mode"] != "off" for c in cam_list)})

    async def live(request):
        """{"action": "start" | "stop", "cameras": [ids] (optional, default all cameras that are not off)}"""
        body = await request.json()
        action = body.get("action")
        if action not in ("start", "stop"):
            return err("action must be start or stop")
        if action == "start" and not len(ctx["registry"]):
            return err("Register at least one employee before starting live attendance")
        changed = await asyncio.to_thread(cams.set_run_mode, "live" if action == "start" else "setup", body.get("cameras"))
        return JSONResponse({"ok": True, "changed": changed,
                             "message": f"{len(changed)} camera(s) {'LIVE' if action == 'start' else 'back to setup'}"})

    # ---------------- employees + registration ----------------
    def employees(request):
        return JSONResponse(storage.employees())

    def delete_employee(request):
        ok = enroll.delete(request.path_params["emp_id"])
        return JSONResponse({"ok": ok, "message": "Deleted" if ok else "Not found"})

    def face(request):
        emp = os.path.basename(request.path_params["emp_id"])
        path = os.path.join(ctx["faces_dir"], emp, "01.jpg")
        return FileResponse(path) if os.path.isfile(path) else err("Not found", 404)

    def session_call(fn):
        async def handler(request):
            try:
                return await fn(request, request.path_params.get("sid"))
            except KeyError as e:
                return err(str(e.args[0]), 410)
            except ValueError as e:
                return err(str(e))
        return handler

    async def _frame(request):
        """Frame from a server camera (?camera=id) or a JPEG body sent by the browser webcam."""
        cam = request.query_params.get("camera")
        if cam:
            worker = cams.get(cam)
            frame = worker.raw_frame() if worker else None
            if frame is None:
                raise ValueError("Camera has no frame yet (is it off?)")
            return frame
        return _decode(await request.body())

    def _pad(request):
        """?pad=auto (default) or a percent 0-150: border added around photos / zoomed faces."""
        v = request.query_params.get("pad", "auto")
        if v in ("", "auto"):
            return None
        try:
            return max(0.0, min(1.5, float(v) / 100))
        except ValueError:
            raise ValueError("pad must be auto or a percent")

    def enroll_start(request):
        sid = enroll.start()
        return JSONResponse(dict(enroll.summary(sid), session=sid))

    async def enroll_check(request, sid):
        return JSONResponse(await asyncio.to_thread(enroll.check, sid, await _frame(request)))

    async def enroll_capture(request, sid):
        q = request.query_params
        point = (float(q["x"]), float(q["y"])) if "x" in q and "y" in q else None  # face clicked in the preview
        return JSONResponse(await asyncio.to_thread(enroll.capture, sid, await _frame(request), point, _pad(request)))

    async def enroll_upload(request, sid):
        form = await request.form()
        files = []
        for f in form.getlist("files"):
            if hasattr(f, "read"):
                data = await f.read()
                img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) if data else None
                files.append((f.filename, img))
        return JSONResponse(await asyncio.to_thread(enroll.upload, sid, files, _pad(request)))

    async def enroll_bulk(request):
        """One employee of a bulk registration: ?emp_id=&name= plus the photos of that employee's folder."""
        q = request.query_params
        try:
            pad = _pad(request)
        except ValueError as e:
            return err(str(e))
        form = await request.form(max_files=2000)
        images = []
        for f in form.getlist("files"):
            if hasattr(f, "read"):
                data = await f.read()
                images.append(cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) if data else None)
        ok, msg = await asyncio.to_thread(enroll.from_images, q.get("emp_id", ""), q.get("name", ""), "", images, True, pad)
        return JSONResponse({"ok": ok, "message": msg})

    async def enroll_remove(request, sid):
        return JSONResponse(enroll.remove(sid, request.path_params["sample_id"]))

    async def enroll_commit(request, sid):
        body = await request.json()
        ok, msg = await asyncio.to_thread(enroll.commit, sid, body.get("emp_id", ""), body.get("name", ""),
                                          body.get("department", ""), bool(body.get("force")))
        return JSONResponse({"ok": ok, "message": msg})

    async def enroll_cancel(request, sid):
        enroll.cancel(sid)
        return JSONResponse({"ok": True})

    # ---------------- test videos ----------------
    def video_list(request):
        items = videos.list()
        for v in items:
            v["cameras"] = cams.using_source(v["path"])
        return JSONResponse({"videos": items, "max_mb": videos.max_bytes // 2**20})

    async def video_upload(request):
        """Chunked upload: ?name=&id=&offset= with the raw bytes as body; the last chunk adds &done=1."""
        q = request.query_params
        name = q.get("name", "")
        try:
            size = await videos.receive_chunk(name, q.get("id", ""), int(q.get("offset", "0")), request.stream())
            if q.get("done") != "1":
                return JSONResponse({"ok": True, "size": size})
            path, info = await asyncio.to_thread(videos.finish, name, q["id"])
        except ValueError as e:
            return err(str(e))
        cam_id = None
        if q.get("camera", "1") == "1":  # create a test camera that plays this video (setup mode)
            base = "video_" + safe_name(os.path.splitext(os.path.basename(path))[0]).lower()[:30]
            cam_id, n = base, 1
            while cams.get(cam_id):
                n += 1
                cam_id = f"{base}_{n}"
            ok, msg = await asyncio.to_thread(cams.save, {
                "cam_id": cam_id, "gate_name": "Test Gate", "direction": "ENTRY", "attendance_mode": "unique",
                "source": path, "width": info["width"], "height": info["height"], "fps": int(info["fps"]),
                "roi": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], "run_mode": "setup"}, True)
            if not ok:
                return err(msg)
        return JSONResponse({"ok": True, "name": os.path.basename(path), "info": info, "camera": cam_id,
                             "message": f"Video uploaded ({info['width']}x{info['height']}, {info['duration']} s)"})

    def _remove_video(name):
        path = videos.path(name)
        removed = [c for c in cams.using_source(path) if cams.delete(c)]
        for _ in range(20):  # the camera thread may still hold the file for a moment (Windows)
            try:
                videos.delete(name)
                break
            except PermissionError:
                time.sleep(0.25)
        else:
            raise ValueError("Video is still in use - try again in a few seconds")
        return removed

    async def video_delete(request):
        try:
            removed = await asyncio.to_thread(_remove_video, request.path_params["name"])
        except KeyError as e:
            return err(str(e.args[0]), 404)
        except ValueError as e:
            return err(str(e))
        return JSONResponse({"ok": True, "message": "Video deleted" + (f" (camera {', '.join(removed)} removed)"
                                                                       if removed else "")})

    # ---------------- tuning + privacy ----------------
    def tuning_get(request):
        return JSONResponse({"values": tuning.values(), "defaults": tuning.defaults})

    async def tuning_put(request):
        body = await request.json()
        try:
            values = tuning.preset(body["preset"]) if "preset" in body else tuning.update(body.get("values", {}))
        except ValueError as e:
            return err(str(e))
        return JSONResponse({"ok": True, "values": values, "message": "Settings applied"})

    def _wipe(parts):
        done = []
        if "videos" in parts:
            for v in videos.list():
                _remove_video(v["name"])
            done.append("test videos")
        if "employees" in parts:
            if any(c["run_mode"] == "live" for c in cams.status()):
                cams.set_run_mode("setup")
            enroll.delete_all()
            done.append("employees + face photos")
        if "attendance" in parts:
            attendance.clear()
            done.append("attendance, alerts, snapshots, Excel reports")
        return done

    async def wipe(request):
        """Privacy: deletes everything the client tested with. {"parts": ["employees", "attendance", "videos"]}"""
        body = await request.json()
        parts = set(body.get("parts") or ("employees", "attendance", "videos"))
        if body.get("confirm") != "DELETE":
            return err("Type DELETE to confirm")
        try:
            done = await asyncio.to_thread(_wipe, parts)
        except (KeyError, ValueError) as e:
            return err(str(e))
        logger.warning("Data wipe from dashboard: {}", ", ".join(done))
        return JSONResponse({"ok": True, "message": "Deleted: " + ", ".join(done)})

    # ---------------- records ----------------
    def events(request):
        d, g, c = filters(request)
        return JSONResponse(storage.events(d, g, c, request.query_params.get("counted") == "1", limit=1000))

    def alerts(request):
        d, g, c = filters(request)
        return JSONResponse(storage.alerts(d, g, c, limit=300))

    def summary(request):
        return JSONResponse(storage.daily_summary(*filters(request)))

    def report(request):
        d, g, c = filters(request)
        data = reports.workbook_bytes(storage, d, g, c)
        name = "_".join(x for x in ("attendance", d, g, c) if x)
        return Response(data, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        headers={"Content-Disposition": f'attachment; filename="{name}.xlsx"'})

    def snapshot(request):
        path = os.path.abspath(request.query_params.get("path", ""))
        if not path.startswith(snap_root + os.sep) or not os.path.isfile(path):
            return err("Not found", 404)
        return FileResponse(path)

    routes = [
        Route("/", index),
        Route("/api/status", status),
        Route("/api/stream/{cam_id}", stream),
        Route("/api/snap/{cam_id}", snap),
        Route("/api/cameras", camera_create, methods=["POST"]),
        Route("/api/cameras/{cam_id}", camera_update, methods=["PUT"]),
        Route("/api/cameras/{cam_id}", camera_delete, methods=["DELETE"]),
        Route("/api/cameras/{cam_id}/frame", camera_frame),
        Route("/api/cameras/{cam_id}/control", camera_control, methods=["POST"]),
        Route("/api/videos", video_list),
        Route("/api/videos/upload", video_upload, methods=["POST"]),
        Route("/api/videos/{name}", video_delete, methods=["DELETE"]),
        Route("/api/tuning", tuning_get),
        Route("/api/tuning", tuning_put, methods=["PUT"]),
        Route("/api/wipe", wipe, methods=["POST"]),
        Route("/api/live/checklist", checklist),
        Route("/api/live", live, methods=["POST"]),
        Route("/api/employees", employees),
        Route("/api/employees/{emp_id}", delete_employee, methods=["DELETE"]),
        Route("/api/employees/{emp_id}/face", face),
        Route("/api/enroll/start", enroll_start, methods=["POST"]),
        Route("/api/enroll/bulk", enroll_bulk, methods=["POST"]),
        Route("/api/enroll/{sid}/check", session_call(enroll_check), methods=["POST"]),
        Route("/api/enroll/{sid}/capture", session_call(enroll_capture), methods=["POST"]),
        Route("/api/enroll/{sid}/upload", session_call(enroll_upload), methods=["POST"]),
        Route("/api/enroll/{sid}/samples/{sample_id:int}", session_call(enroll_remove), methods=["DELETE"]),
        Route("/api/enroll/{sid}/commit", session_call(enroll_commit), methods=["POST"]),
        Route("/api/enroll/{sid}", session_call(enroll_cancel), methods=["DELETE"]),
        Route("/api/events", events),
        Route("/api/alerts", alerts),
        Route("/api/summary", summary),
        Route("/api/report", report),
        Route("/api/snapshot", snapshot),
    ]
    async def on_error(request, exc):
        # always JSON with a readable message (a plain "Internal Server Error" page breaks the dashboard)
        logger.opt(exception=exc).error("Request {} failed", request.url.path)
        text = str(exc).lower()
        msg = ("GPU error - restart the app (and the PC if it happens again)" if "cuda" in text
               else f"Server error: {str(exc).splitlines()[0][:200] if str(exc) else type(exc).__name__}")
        return JSONResponse({"ok": False, "message": msg}, status_code=500)

    app = Starlette(routes=routes, exception_handlers={Exception: on_error})

    if not api_key:
        return app

    async def guarded(scope, receive, send):
        """Requires X-API-Key header (or ?key= for <img>/download links) on every /api route."""
        if scope["type"] == "http" and scope["path"].startswith("/api/"):
            headers = dict(scope.get("headers") or [])
            given = headers.get(b"x-api-key", b"").decode() or parse_qs(
                scope.get("query_string", b"").decode()).get("key", [""])[0]
            if given != api_key:
                await JSONResponse({"ok": False, "message": "Unauthorized"}, status_code=401)(scope, receive, send)
                return
        await app(scope, receive, send)

    return guarded


def start_server(ctx):
    wcfg = ctx["cfg"].get("web", {})
    config = uvicorn.Config(create_app(ctx), host=wcfg.get("host", "0.0.0.0"), port=int(wcfg.get("port", 8000)),
                            log_level="warning", access_log=False)
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name="web", daemon=True)
    thread.start()
    logger.info("Dashboard: http://localhost:{}", wcfg.get("port", 8000))
    return server
