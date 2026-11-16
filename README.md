# Smart Gate Attendance

Real-time face attendance for office gates. Multi-camera, multi-gate, with anti-spoofing, a web dashboard and guided employee registration.

## How it works

```
Camera thread (newest frame only, auto-reconnect)
  -> SCRFD face detection (ROI crop only)
  -> face tracking (each person gets an ID, works with many people at once)
  -> soft quality (only faces below the minimum size are skipped; blur, side view, edge, light just lower a frame's weight)
  -> batched ArcFace embedding + MiniFASNet liveness for all faces in the frame
  -> multi-frame verdict per person:
       identity : >= 2 good frames, fused embedding >= 0.45, beats 2nd-best employee by 0.08,
                  50% of the recent frame weight agrees on the same person;
                  person leaves the view while verifying -> last-chance decision with the frames seen (+0.04 match)
       liveness : model score of the last 2 frames (mean >= 0.50); faces too small for the model are not blocked
       spoof    : mean of the last 2 frames < 0.30 -> red "FAKE FACE", attendance blocked, alert logged
  -> attendance rules (camera's unique / multiple mode) -> SQLite (background writer) -> Excel autosave
```

A single frame never marks attendance and never blocks a real person on its own.

## Project layout

| Path | Purpose |
|---|---|
| `main.py` | Entry point: starts every camera, the dashboard and the display |
| `configs/config.yaml` | All settings: gates, cameras, thresholds, attendance mode |
| `core/camera.py` | Webcam / RTSP / GStreamer / video-file reader, reconnects automatically |
| `core/engine.py` | ONNX Runtime models: detector, ArcFace, liveness (CUDA / TensorRT / CPU) |
| `core/tracker.py` | Face tracker and the multi-frame decision per person |
| `core/gate.py` | Per-camera worker (setup / live / off), rendering, CameraManager |
| `core/attendance.py` | Unique / multiple rules, cooldown, background DB writer |
| `core/storage.py` | SQLite (WAL): employees, cameras, events, alerts + in-memory face registry |
| `core/enrollment.py` | Button capture + photo upload with quality checks, folder import, duplicate-face check |
| `core/reports.py` | Excel export |
| `core/tuning.py` | Runtime thresholds and presets edited from the dashboard (saved in the DB) |
| `core/videos.py` | Test video upload (chunked), check, delete (`data/videos`) |
| `web/` | Dashboard + registration UI (`http://<host>:8000`) |
| `models/` | `det_10g.onnx`, `w600k_r50.onnx`, `liveness_minifasnet.onnx` |
| `data/` | `attendance.db`, face samples, alert snapshots (created automatically) |

## Run

```powershell
py -3.11 -m pip install -r requirements.txt
py -3.11 scripts/download_models.py   # InsightFace models: non-commercial licence, not stored in git
py -3.11 main.py                 # or double-click run_system.bat
```

Then open **http://localhost:8000**. Every camera starts in **Setup** mode, where it shows a preview but records no attendance.

**Demo / production flow**

1. **Cameras & Gates**: add or edit cameras. For each one set the gate, ENTRY / EXIT, **Unique or Multiple**, the source (webcam index or RTSP URL) and the detection area (click on the camera image).
2. **Register Employee**: enter ID / name / department, pick a camera and press **Capture photo** (or Space) for each photo, and/or **upload photos**. Every photo is quality-checked (one face, size, sharpness, light, angle, liveness, same person, not a repeat); rejected photos show the reason. Every employee needs 12 to 100 good photos. Face-only crops get padding (Auto, or a fixed percent chosen on the Register page). Saving blocks faces already registered under another ID.
3. **Start Live Attendance** (top right): a checklist shows the registered employees and the cameras, and you choose which cameras go live. **Stop Live** returns all cameras to setup. A single camera can also go live / stop / turn off from the camera bar on the Live tab or the Cameras tab, and the Live tab can show all cameras that are on or only the live ones.
4. **Live** shows counts, streams, passages and alerts. **Reports** filters by date / gate / camera and downloads Excel.

**Client test / POC flow (everything from the dashboard)**

| Tab / button | What it does |
|---|---|
| **Test Videos** | Upload a recorded video (chunked, so it works through tunnels with request limits). A test camera that plays the video is created in setup mode. Pause / Play / Restart, delete. |
| **Cameras & Gates → Edit** | Draw the ROI tightly around the walking area. The ROI crop is upscaled to the detector size, so a smaller ROI works like a digital zoom for far faces. |
| **Tuning** | Presets **Standard** (face-height camera), **Far / side view CCTV** (faces ~40-80 px) and **Top-down ceiling CCTV - tiny faces** (faces ~20-50 px, full-resolution search), plus every detection, quality, recognition and liveness threshold. Applied instantly to all cameras and saved in the database. Setup mode shows each detected face with its size in px, and the Tuning tab lists the face sizes seen right now. |
| **Register Employee** | Only the employee ID is required. **Bulk registration**: choose a main folder with one folder per employee (folder name = employee ID). Upload photos: tightly cropped faces, small photos and rotated photos are fixed automatically (border, upscale, rotation). From a video camera: pause the video and click on the employee's face to capture exactly that person. |
| **Delete all data** (top right) | Permanently deletes employees and face photos, attendance, alerts, snapshots, Excel reports, and uploaded videos with their test cameras. Tuning is kept. |

Sharing the dashboard with a client on another network (no router change): run the app, then `cloudflared tunnel --url http://localhost:8000` and send the `https://….trycloudflare.com` link. Set `web.api_key` if the link should not be open to anyone who has it.

Other commands:

```powershell
py -3.11 main.py --no-window                        # headless (Jetson / server)
py -3.11 main.py --import-faces D:\all_faces        # bulk registration: one folder per employee, folder name = employee ID (12-100 photos each)
py -3.11 main.py --import-faces D:\all_faces --padding 40   # fixed 40 % padding around face-only crops (default: auto)
py -3.11 main.py --import-faces D:\all_faces --with-names  # same, folders named <EMP_ID>_<Name>
py -3.11 main.py --reset                            # clear attendance + alerts, keep employees
py -3.11 reset_system.py                            # FULL reset before a demo: employees, photos, DB, Excel, logs, videos
```

## Attendance modes (per camera)

| Mode | Behaviour |
|---|---|
| `unique` | The first passage of the day at that **camera** counts; later passages are stored as repeats (report shows Last Seen). Every camera keeps its own record: a person passing 5 cameras gets 5 records. |
| `multiple` | Every passage on that camera is an attendance record. |

Example: Gate 1 ENTRY = Unique (morning attendance), Gate 1 EXIT = Multiple (every exit tracked). `cooldown_seconds` (default 30) merges the frames of one walk-through at one camera into one passage. A person standing in front of a camera is recorded once per visit.

The daily Excel file `reports/attendance_YYYY-MM-DD.xlsx` has the sheets Attendance Records, Employee Summary, All Passages and Security Alerts. It is rebuilt every 30 s from the DB. If the file is open in Excel, the write is retried later and no data is lost.

## Cameras and gates

Cameras are managed in the dashboard (**Cameras & Gates**) and stored in the database. The `gates` section of `configs/config.yaml` only seeds the cameras on the first start (or after `reset_system.py`). Each camera has:

| Setting | Values |
|---|---|
| Gate name | free text; cameras with the same gate name form one gate |
| Direction | ENTRY / EXIT |
| Attendance mode | unique / multiple |
| Source | `0`, `1` (USB), `rtsp://...`, GStreamer pipeline, video file |
| Run mode | setup (preview only) / live (attendance) / off (closed) |
| ROI | polygon drawn on the camera image; only faces inside are processed |

Changes apply immediately, without a restart. Each camera runs in its own thread and all cameras share one copy of the models on the GPU. Use the camera's **sub-stream** (720p) when 1080p is not needed: detection runs at 640 px anyway.

## Production / Jetson

| Target | Install | `inference.backend` |
|---|---|---|
| Windows PC / laptop GPU | `pip install -r requirements.txt` | `cuda` |
| Linux GPU server (H100, A100, L4, cloud VM) | NVIDIA driver on the host, then `pip install -r requirements.txt` (CUDA 12 + cuDNN 9 come as pip wheels). For max speed also `pip install tensorrt-cu12==10.9.*` | `tensorrt` (or `cuda`) |
| NVIDIA Jetson (JetPack 6) | JetPack OpenCV + NVIDIA's Jetson `onnxruntime-gpu` wheel, then `pip install -r requirements.txt` | `tensorrt` |
| CPU only | see the bottom of `requirements.txt` | `cpu` |

- Multi-GPU servers: `inference.device_id` picks the GPU. Run one process per GPU, each with its own config and cameras.
- On Linux servers without a display the OpenCV window turns off automatically. The dashboard and its MJPEG previews still work.
- `inference.backend: tensorrt` builds FP16 TensorRT engines on first start and caches them in `models/trt_cache`. If TensorRT cannot load, the app falls back to CUDA, never silently to CPU. The log says which provider each model runs on, and an ERROR line appears if anything runs on CPU.
- Set `display.window: false` (or use `--no-window`) and `recording.enabled: false`.
- Set `web.api_key`; the dashboard asks for it once.
- Jetson RTSP with hardware decode: use a GStreamer pipeline as `source`, e.g.
  `rtspsrc location=rtsp://... latency=100 ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1 max-buffers=1`
- Run as a service (systemd / NSSM) with auto-restart.

## Camera placement (biggest accuracy factor)

- Face height, max ~15° downward angle, faces at least 100 px wide in the image.
- WDR / HDR camera for gates with daylight behind people; IR light for night.
- Enroll people at the gate camera, or under similar light.

## License notes

- MiniFASNet liveness weights: Minivision Silent-Face-Anti-Spoofing, Apache-2.0.
- InsightFace `buffalo_l` models (`det_10g`, `w600k_r50`) are licensed for **non-commercial research**. Commercial deployment needs a license from InsightFace, or replacement models.

## Privacy and data

Face photos, embeddings, attendance records, snapshots, Excel reports and logs are **personal (biometric) data**. They
live only in `data/`, `logs/` and `reports/`, which are excluded from git. Get written consent before enrolling anyone,
keep the server on the customer's own network, set `web.api_key` in production, and use **Delete all data** or
`reset_system.py` when a pilot ends.

---

Built by **Akash Kumar**: AI engineer for computer vision, edge AI, GenAI and AI agents.
Need face-based attendance or access control? **akashkr19857@gmail.com** · [LinkedIn](https://www.linkedin.com/in/akash-kumar-883220192)
