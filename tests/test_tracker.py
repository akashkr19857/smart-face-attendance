"""Unit tests for the face tracker and the multi-frame identity / liveness verdict (no models, no camera needed)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.tracker import PENDING, SPOOF, VERIFIED, FaceTracker, Track, iou_matrix  # noqa: E402

RCFG = {"min_frames": 2, "match_threshold": 0.45, "margin": 0.08, "vote_ratio": 0.6, "unknown_after_frames": 5}
LCFG = {"enabled": True, "min_frames": 2, "spoof_threshold": 0.30, "real_threshold": 0.50}


class FakeRegistry:
    """Two enrolled employees with orthogonal unit embeddings."""

    def __init__(self) -> None:
        self.emb = {"E1": np.eye(512, dtype=np.float32)[0], "E2": np.eye(512, dtype=np.float32)[1]}

    def match_one(self, v: np.ndarray):  # noqa: ANN201
        scores = sorted(((float(v @ e), k) for k, e in self.emb.items()), reverse=True)
        return scores[0][1], scores[0][0], scores[1][0]

    def info(self, emp_id: str) -> dict:
        return {"name": {"E1": "Asha", "E2": "Ravi"}[emp_id], "department": "Ops"}


def face(emp: str, noise: float = 0.0, rng: np.random.Generator | None = None) -> np.ndarray:
    v = FakeRegistry().emb[emp].copy()
    if noise:
        v = v + (rng or np.random.default_rng(0)).normal(0, noise, 512).astype(np.float32)
    return v / np.linalg.norm(v)


def track() -> Track:
    return Track(1, np.array([0, 0, 100, 100, 0.9], np.float32), None, 0.0, live_window=10)


def test_iou_matrix():
    a = np.array([[0, 0, 10, 10]], np.float32)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], np.float32)
    assert np.allclose(iou_matrix(a, b), [[1.0, 1 / 3, 0.0]], atol=1e-4)
    assert iou_matrix(a, np.zeros((0, 4))).shape == (1, 0)


def test_single_frame_never_verifies():
    tr = track()
    tr.observe(face("E1"), 1.0, "E1", 0.95)
    assert tr.decide(FakeRegistry(), RCFG, LCFG) is None and tr.state == PENDING


def test_two_good_frames_verify_the_right_person():
    tr, reg = track(), FakeRegistry()
    for _ in range(2):
        tr.observe(face("E1", 0.01), 1.0, "E1", 0.9)
    assert tr.decide(reg, RCFG, LCFG) == VERIFIED and tr.emp_id == "E1" and tr.name == "Asha"
    assert tr.decide(reg, RCFG, LCFG) is None  # a verified track never decides again


def test_low_liveness_blocks_as_spoof():
    tr = track()
    for _ in range(2):
        tr.observe(face("E1"), 1.0, "E1", 0.1)  # photo / screen attack
    assert tr.decide(FakeRegistry(), RCFG, LCFG) == SPOOF and tr.spoof and tr.emp_id is None
    assert "FAKE" in tr.message


def test_spoof_flag_clears_when_real_face_is_seen():
    tr = track()
    for p in (0.1, 0.1):
        tr.observe(face("E1"), 1.0, "E1", p)
    tr.decide(FakeRegistry(), RCFG, LCFG)
    for p in (0.9, 0.9):
        tr.observe(face("E1"), 1.0, "E1", p)
    assert tr.decide(FakeRegistry(), RCFG, LCFG) == VERIFIED and not tr.spoof


def test_ambiguous_votes_do_not_verify():
    tr = track()
    rng = np.random.default_rng(1)
    for emp in ("E1", "E2", "E1", "E2"):  # frames disagree on who it is
        tr.observe(face(emp, 0.01, rng), 1.0, emp, 0.9)
    assert tr.decide(FakeRegistry(), RCFG, LCFG) is None and tr.state == PENDING


def test_unregistered_person_reported_once():
    tr = track()
    stranger = np.eye(512, dtype=np.float32)[7]
    results = []
    for _ in range(7):
        tr.observe(stranger, 1.0, None, 0.9)
        results.append(tr.decide(FakeRegistry(), RCFG, LCFG))
    assert results.count("UNKNOWN") == 1


def test_fast_walker_final_decision_needs_higher_match():
    tr = track()
    tr.observe(face("E1", 0.02), 1.0, "E1", 0.9)  # only one frame seen before leaving
    assert tr.decide(FakeRegistry(), RCFG, LCFG, final=True) == VERIFIED


def test_liveness_can_be_disabled():
    tr = track()
    for _ in range(2):
        tr.observe(face("E2"), 1.0, "E2", 0.05)
    assert tr.decide(FakeRegistry(), RCFG, dict(LCFG, enabled=False)) == VERIFIED


def test_tracker_keeps_ids_and_drops_lost_tracks():
    ft = FaceTracker(iou_threshold=0.25, max_lost_seconds=1.0)
    kps = np.zeros((2, 5, 2), np.float32)
    ids = []
    for i in range(5):
        boxes = np.array([[10 + i * 3, 10, 60 + i * 3, 60, 0.9], [200, 50 + i * 3, 250, 100 + i * 3, 0.8]], np.float32)
        ids.append(sorted(tr.id for tr in ft.update(boxes, kps, i * 0.1)))
    assert ids[0] == ids[-1] == [1, 2]
    ft.update(np.zeros((0, 5), np.float32), np.zeros((0, 5, 2), np.float32), 2.0)  # nobody seen for > 1 s
    assert not ft.tracks and len(ft.removed) == 2


def test_low_confidence_box_only_extends_existing_track():
    ft = FaceTracker(high_thresh=0.5)
    kps = np.zeros((1, 5, 2), np.float32)
    assert not ft.update(np.array([[0, 0, 50, 50, 0.3]], np.float32), kps, 0.0)  # low score cannot start a track
    ft.update(np.array([[0, 0, 50, 50, 0.9]], np.float32), kps, 0.1)
    assert [t.id for t in ft.update(np.array([[2, 0, 52, 50, 0.3]], np.float32), kps, 0.2)] == [1]


@pytest.mark.parametrize("n", [0, 3])
def test_no_detections(n):
    ft = FaceTracker()
    for i in range(n):
        ft.update(np.zeros((0, 5), np.float32), np.zeros((0, 5, 2), np.float32), i * 0.1)
    assert ft.tracks == {}
