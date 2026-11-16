"""Download the face models that cannot be committed to git.

    python scripts/download_models.py

* InsightFace `buffalo_l` (SCRFD `det_10g.onnx` detector + ArcFace `w600k_r50.onnx` recognizer) from the official
  InsightFace GitHub release. LICENSE: non-commercial research only. Commercial deployment needs a licence from
  InsightFace or replacement models. `w600k_r50.onnx` is 174 MB, above GitHub's 100 MB file limit.
* `liveness_minifasnet.onnx` (Minivision Silent-Face-Anti-Spoofing, Apache-2.0) is small and ships in the repo.
* The FP16 copy of the recognizer (`w600k_r50.fp16.onnx`) is created automatically on first start (backend "cuda").
"""
from __future__ import annotations

import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

MODELS = Path(__file__).resolve().parents[1] / "models"
URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
WANTED = ("det_10g.onnx", "w600k_r50.onnx")


def main() -> None:
    MODELS.mkdir(exist_ok=True)
    missing = [n for n in WANTED if not (MODELS / n).exists()]
    if not missing:
        print("models already present:", ", ".join(WANTED))
        return
    print(f"downloading {URL} (~280 MB) ...")
    req = urllib.request.Request(URL, headers={"User-Agent": "smart-face-attendance"})
    with urllib.request.urlopen(req, timeout=600) as r:
        data = r.read()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for name in zf.namelist():
            base = Path(name).name
            if base in missing:
                (MODELS / base).write_bytes(zf.read(name))
                print(f"  {base}  sha256 {hashlib.sha256(zf.read(name)).hexdigest()[:16]}")
    still = [n for n in WANTED if not (MODELS / n).exists()]
    if still:
        sys.exit(f"not found in the archive: {still}")
    if not (MODELS / "liveness_minifasnet.onnx").exists():
        print("WARNING: models/liveness_minifasnet.onnx is missing (it ships with the repository).")
    print("done. Licence reminder: buffalo_l is for non-commercial research use only.")


if __name__ == "__main__":
    main()
