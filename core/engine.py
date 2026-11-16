"""Face models on ONNX Runtime: SCRFD detector, ArcFace recognizer, MiniFASNet liveness.

One FaceEngine is shared by every camera thread (ORT sessions are thread-safe and release the GIL).
"""
import os
import threading

import cv2
import numpy as np
import onnxruntime as ort
from loguru import logger

# ArcFace reference landmarks for a 112x112 aligned face
ARCFACE_DST = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                        [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)

ort.set_default_logger_severity(3)  # SCRFD declares fixed output shapes; non-square inputs only trigger warnings

_GAMMA_LUT = np.array([((i / 255.0) ** 0.6) * 255 for i in range(256)], dtype=np.uint8)


def _preload_tensorrt_libs():
    """Makes TensorRT from the pip wheel (tensorrt-cu12 -> tensorrt_libs) visible to ONNX Runtime."""
    import ctypes
    import glob
    import importlib.util
    spec = importlib.util.find_spec("tensorrt_libs")
    if not spec or not spec.origin:
        return  # system TensorRT (JetPack / apt) is found through the normal library path
    lib_dir = os.path.dirname(spec.origin)
    if os.name == "nt":
        os.add_dll_directory(lib_dir)
        os.environ["PATH"] = lib_dir + os.pathsep + os.environ.get("PATH", "")
        return
    for pattern in ("libnvinfer.so.*", "libnvinfer_plugin.so.*", "libnvonnxparser.so.*"):
        for lib in sorted(glob.glob(os.path.join(lib_dir, pattern))):
            try:
                ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
                break
            except OSError:
                pass


def _preload_cuda_libs():
    """Load CUDA / cuDNN (and NVRTC, used by cuDNN fused kernels) shipped in pip wheels."""
    if hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls()
        except Exception:
            pass
    if os.name == "nt":
        import ctypes
        import importlib.util
        spec = importlib.util.find_spec("torch")
        lib_dir = os.path.join(os.path.dirname(spec.origin), "lib") if spec and spec.origin else ""
        nvrtc = os.path.join(lib_dir, "nvrtc64_120_0.dll")
        if os.path.exists(nvrtc):
            try:
                ctypes.WinDLL(nvrtc)
            except OSError:
                pass


def build_providers(backend, device_id=0, cache_dir="models/trt_cache"):
    if backend in ("tensorrt", "cuda"):
        _preload_cuda_libs()
    if backend == "tensorrt":
        _preload_tensorrt_libs()
    available = ort.get_available_providers()
    chain = []
    if backend == "tensorrt" and "TensorrtExecutionProvider" in available:
        os.makedirs(cache_dir, exist_ok=True)
        chain.append(("TensorrtExecutionProvider", {
            "device_id": device_id,
            "trt_fp16_enable": True,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": cache_dir,
            "trt_timing_cache_enable": True,
        }))
    if backend in ("tensorrt", "cuda") and "CUDAExecutionProvider" in available:
        chain.append(("CUDAExecutionProvider", {"device_id": device_id, "cudnn_conv_algo_search": "HEURISTIC"}))
    chain.append(("CPUExecutionProvider", {}))
    return chain


def _session(path, providers, trt_profile=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    resolved = []
    for name, opts in providers:
        opts = dict(opts)
        if name == "TensorrtExecutionProvider" and trt_profile:
            opts.update(trt_profile)
        resolved.append((name, opts))
    names = [n for n, _ in resolved]
    sess = None
    if "TensorrtExecutionProvider" in names:
        try:
            sess = ort.InferenceSession(path, so, providers=resolved)
        except Exception as e:
            logger.warning("TensorRT unavailable for {} ({})", os.path.basename(path), str(e).splitlines()[0])
        # ORT silently drops to CPU-only when TensorRT libs mismatch; retry on CUDA instead
        if sess is None or ("CUDAExecutionProvider" in names and "CUDAExecutionProvider" not in sess.get_providers()):
            resolved = [p for p in resolved if p[0] != "TensorrtExecutionProvider"]
            sess = None
    if sess is None:
        sess = ort.InferenceSession(path, so, providers=resolved)
    active = sess.get_providers()[0]
    if names[0] != "CPUExecutionProvider" and active == "CPUExecutionProvider":
        logger.error("{} is running on CPU - check CUDA/cuDNN installation (expect low FPS)", os.path.basename(path))
    logger.info("Loaded {} on {}", os.path.basename(path), active)
    return sess


class _Model:
    """ONNX session that survives a GPU crash (e.g. CUDA error 999 after laptop sleep or a driver reset):
    the model is reloaded on the GPU, and on the CPU if the GPU stays broken, instead of every request failing."""

    def __init__(self, path, providers, trt_profile=None):
        self.path, self.providers, self.trt_profile = path, providers, trt_profile
        self.sess = _session(path, providers, trt_profile)
        self._lock = threading.Lock()
        self._fails = 0

    def get_inputs(self):
        return self.sess.get_inputs()

    def run(self, outputs, feed):
        sess = self.sess
        try:
            return sess.run(outputs, feed)
        except Exception as e:
            if "cuda" not in str(e).lower():
                raise
            self._recover(sess, e)
            return self.sess.run(outputs, feed)

    def _recover(self, broken, error):
        with self._lock:
            if self.sess is not broken:
                return  # another camera thread already reloaded it
            self._fails += 1
            name = os.path.basename(self.path)
            logger.error("GPU error in {}: {}", name, str(error).splitlines()[0][:160])
            if self._fails <= 2:
                try:
                    self.sess = _session(self.path, self.providers, self.trt_profile)
                    logger.warning("{} reloaded on the GPU", name)
                    return
                except Exception as e:
                    logger.error("GPU reload of {} failed: {}", name, str(e).splitlines()[0][:160])
            self.sess = _session(self.path, [("CPUExecutionProvider", {})])
            logger.error("{} now runs on the CPU (slow). Restart the app (or the PC if it repeats) to use the GPU again",
                         name)


def fp16_model(path):
    """Returns a cached FP16 copy of an ONNX model (fp32 inputs/outputs kept). Falls back to the original."""
    out = path.replace(".onnx", ".fp16.onnx")
    if os.path.exists(out) and os.path.getmtime(out) >= os.path.getmtime(path):
        return out
    try:
        import onnx
        from onnxconverter_common import float16
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            onnx.save(float16.convert_float_to_float16(onnx.load(path), keep_io_types=True), out)
        logger.info("Created FP16 model {}", out)
        return out
    except Exception as e:
        logger.warning("FP16 conversion skipped for {}: {}", path, e)
        return path


def _trt_shapes(name, lo, opt, hi):
    fmt = lambda s: f"{name}:" + "x".join(map(str, s))
    return {"trt_profile_min_shapes": fmt(lo), "trt_profile_opt_shapes": fmt(opt), "trt_profile_max_shapes": fmt(hi)}


def boost_if_dark(img, threshold=70):
    """Gamma-brightens dark images (night, backlit gates). Cheap LUT, no-op for normal light."""
    small = cv2.resize(img, (64, 36), interpolation=cv2.INTER_AREA)
    if small.mean() < threshold:
        return cv2.LUT(img, _GAMMA_LUT)
    return img


def _nms(boxes, scores, thresh):
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= thresh]
    return keep


class FaceDetector:
    """SCRFD with letterboxing to an aspect-matched, 32-aligned input (less padding = faster)."""
    STRIDES = (8, 16, 32)

    MAX_SIZE = 1920  # det_size can be changed at runtime (dashboard tuning) up to this value

    def __init__(self, path, providers, det_size=640, nms_thresh=0.4):
        self.set_size(det_size)
        self.nms_thresh = nms_thresh
        self.sess = _Model(path, providers, _trt_shapes(
            "input.1", (1, 3, 96, 96), (1, 3, self.det_size, self.det_size), (1, 3, self.MAX_SIZE, self.MAX_SIZE)))
        self.input_name = self.sess.get_inputs()[0].name
        self._anchors = {}

    def set_size(self, det_size):
        self.det_size = int(min(max(int(det_size), 320), self.MAX_SIZE)) // 32 * 32

    def _anchor_centers(self, fh, fw, stride):
        key = (fh, fw, stride)
        if key not in self._anchors:
            centers = np.stack(np.mgrid[:fh, :fw][::-1], axis=-1).reshape(-1, 2).astype(np.float32) * stride
            self._anchors[key] = np.repeat(centers, 2, axis=0)  # 2 anchors per location
        return self._anchors[key]

    def detect(self, img, thresh=0.3, size=None):
        """Returns boxes (N, 5) [x1, y1, x2, y2, score] and landmarks (N, 5, 2) in image coordinates.
        size overrides det_size (e.g. 640 for close-up photos: a large size would over-zoom big faces)."""
        h, w = img.shape[:2]
        scale = (size or self.det_size) / max(h, w)
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        in_w, in_h = -(-nw // 32) * 32, -(-nh // 32) * 32
        canvas = np.zeros((in_h, in_w, 3), dtype=np.uint8)
        canvas[:nh, :nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        blob = cv2.dnn.blobFromImage(canvas, 1.0 / 128, (in_w, in_h), (127.5, 127.5, 127.5), swapRB=True)
        outs = self.sess.run(None, {self.input_name: blob})

        all_boxes, all_scores, all_kps = [], [], []
        for i, stride in enumerate(self.STRIDES):
            scores = outs[i][:, 0]
            keep = np.where(scores >= thresh)[0]
            if keep.size == 0:
                continue
            centers = self._anchor_centers(in_h // stride, in_w // stride, stride)[keep]
            d = outs[i + 3][keep] * stride
            k = outs[i + 6][keep].reshape(-1, 5, 2) * stride
            all_boxes.append(np.stack([centers[:, 0] - d[:, 0], centers[:, 1] - d[:, 1],
                                       centers[:, 0] + d[:, 2], centers[:, 1] + d[:, 3]], axis=1))
            all_kps.append(centers[:, None, :] + k)
            all_scores.append(scores[keep])

        if not all_boxes:
            return np.zeros((0, 5), np.float32), np.zeros((0, 5, 2), np.float32)
        boxes = np.concatenate(all_boxes) / scale
        kps = np.concatenate(all_kps) / scale
        scores = np.concatenate(all_scores)
        keep = _nms(boxes, scores, self.nms_thresh)
        return np.hstack([boxes[keep], scores[keep, None]]).astype(np.float32), kps[keep].astype(np.float32)


def _umeyama(src, dst):
    """Least-squares similarity transform (2x3) mapping src points onto dst."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s_c, d_c = src - mu_s, dst - mu_d
    cov = d_c.T @ s_c / len(src)
    u, sig, vt = np.linalg.svd(cov)
    d = np.ones(2)
    if np.linalg.det(cov) < 0:
        d[1] = -1
    r = u @ np.diag(d) @ vt
    scale = (sig * d).sum() / s_c.var(0).sum()
    t = mu_d - scale * r @ mu_s
    return np.hstack([scale * r, t[:, None]]).astype(np.float32)


def align_face(img, kps, size=112):
    return cv2.warpAffine(img, _umeyama(kps.astype(np.float64), ARCFACE_DST.astype(np.float64)),
                          (size, size), borderValue=0.0)


def face_pose(kps):
    """Approximate yaw (-left / +right, 0 = frontal) and pitch ratio (~0.5 frontal) from 5 landmarks."""
    le, re, nose, lm, rm = kps
    eye_mid = (le + re) / 2
    mouth_mid = (lm + rm) / 2
    eye_dist = max(np.linalg.norm(re - le), 1e-3)
    yaw = float((nose[0] - eye_mid[0]) / eye_dist)
    pitch = float((nose[1] - eye_mid[1]) / max(mouth_mid[1] - eye_mid[1], 1e-3))
    return yaw, pitch


def face_quality(aligned, kps):
    gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    center = gray[16:96, 16:96]
    yaw, pitch = face_pose(kps)
    return {
        "sharpness": float(cv2.Laplacian(center, cv2.CV_32F).var()),
        "brightness": float(center.mean()),
        "yaw": yaw,
        "pitch": pitch,
    }


class FaceRecognizer:
    """ArcFace R50 -> L2-normalized 512-D embeddings, batched."""

    def __init__(self, path, providers, max_batch=16):
        self.max_batch = max_batch
        self.sess = _Model(path, providers, _trt_shapes(
            "input.1", (1, 3, 112, 112), (4, 3, 112, 112), (max_batch, 3, 112, 112)))
        self.input_name = self.sess.get_inputs()[0].name

    def embed(self, aligned_faces, flip=True):
        """flip: also embeds the mirrored face and sums both (more stable for side views and small faces)."""
        if not aligned_faces:
            return np.zeros((0, 512), np.float32)
        n = len(aligned_faces)
        faces = list(aligned_faces) + ([cv2.flip(f, 1) for f in aligned_faces] if flip else [])
        out = []
        for i in range(0, len(faces), self.max_batch):
            blob = cv2.dnn.blobFromImages(faces[i:i + self.max_batch], 1.0 / 127.5, (112, 112),
                                          (127.5, 127.5, 127.5), swapRB=True)
            out.append(self.sess.run(None, {self.input_name: blob})[0])
        emb = np.concatenate(out).astype(np.float32)
        emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9)
        if flip:
            emb = emb[:n] + emb[n:]
        return emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9)


class LivenessModel:
    """Silent-Face MiniFASNet pair (2.7x and 4.0x context crops). Returns P(real) per face."""
    SCALES = (2.7, 4.0)
    SIZE = 80

    def __init__(self, path, providers, max_batch=16):
        self.max_batch = max_batch
        prof = {}
        for name in ("x27", "x40"):
            p = _trt_shapes(name, (1, 3, 80, 80), (4, 3, 80, 80), (max_batch, 3, 80, 80))
            for k, v in p.items():
                prof[k] = f"{prof[k]},{v}" if k in prof else v
        self.sess = _Model(path, providers, prof)

    def _crop(self, frame, box, scale):
        src_h, src_w = frame.shape[:2]
        x1, y1, x2, y2 = box[:4]
        bw, bh = x2 - x1 + 1, y2 - y1 + 1
        scale = min((src_h - 1) / bh, (src_w - 1) / bw, scale)
        nw, nh = bw * scale, bh * scale
        cx, cy = x1 + bw / 2, y1 + bh / 2
        l, t, r, b = cx - nw / 2, cy - nh / 2, cx + nw / 2, cy + nh / 2
        if l < 0:
            r, l = r - l, 0
        if t < 0:
            b, t = b - t, 0
        if r > src_w - 1:
            l, r = l - (r - src_w + 1), src_w - 1
        if b > src_h - 1:
            t, b = t - (b - src_h + 1), src_h - 1
        crop = frame[int(t):int(b) + 1, int(l):int(r) + 1]
        return cv2.resize(crop, (self.SIZE, self.SIZE), interpolation=cv2.INTER_LINEAR)

    def predict(self, frame, boxes):
        if len(boxes) == 0:
            return np.zeros(0, np.float32)
        out = []
        for i in range(0, len(boxes), self.max_batch):
            chunk = boxes[i:i + self.max_batch]
            # trained on raw 0-255 BGR, NCHW
            x27 = np.stack([self._crop(frame, b, 2.7) for b in chunk]).transpose(0, 3, 1, 2).astype(np.float32)
            x40 = np.stack([self._crop(frame, b, 4.0) for b in chunk]).transpose(0, 3, 1, 2).astype(np.float32)
            out.append(self.sess.run(None, {"x27": x27, "x40": x40})[0][:, 1])
        return np.concatenate(out)


class FaceEngine:
    def __init__(self, cfg):
        backend = cfg.get("backend", "cuda")
        providers = build_providers(backend, int(cfg.get("device_id", 0)))
        self.detector = FaceDetector(cfg["detector"], providers, cfg.get("det_size", 640))
        # On CUDA, FP16 ArcFace is ~1.5x faster with identical embeddings (cosine > 0.9999 vs FP32).
        # TensorRT does its own FP16, so it keeps the original model.
        rec_path = fp16_model(cfg["recognizer"]) if backend == "cuda" and cfg.get("fp16", True) else cfg["recognizer"]
        self.recognizer = FaceRecognizer(rec_path, providers)
        self.liveness = LivenessModel(cfg["liveness"], providers)
        self.update(cfg)
        self._warmup()

    def update(self, cfg):
        """Applies detection settings at runtime (dashboard tuning)."""
        self.detector.set_size(cfg.get("det_size", 640))
        self.det_thresh = cfg.get("det_thresh", 0.5)
        self.track_thresh = cfg.get("track_thresh", 0.3)
        self.low_light_boost = cfg.get("low_light_boost", True)

    def _warmup(self):
        dummy = np.full((720, 1280, 3), 114, np.uint8)
        self.detector.detect(dummy)
        self.recognizer.embed([dummy[:112, :112]] * 2)
        self.liveness.predict(dummy, np.array([[500, 200, 700, 450, 1.0]], np.float32))
        logger.info("Face engine ready")

    def detect(self, img, thresh=None, size=None):
        src = boost_if_dark(img) if self.low_light_boost else img
        return self.detector.detect(src, self.track_thresh if thresh is None else thresh, size)

    def largest_face(self, img):
        """Single largest confident face for enrollment / photo import: (box, kps, n_faces) or None."""
        boxes, kps = self.detect(img, self.det_thresh)
        if len(boxes) == 0:
            return None
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        i = int(areas.argmax())
        return boxes[i], kps[i], len(boxes)
