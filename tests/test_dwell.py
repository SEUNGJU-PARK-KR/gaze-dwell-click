import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

pytest.importorskip("cv2")
from main import DwellClicker, OneEuroFilter  # noqa: E402

FPS = 30


def run(d, points, t0=0.0, smoother=None):
    """프레임 열을 넣고 (클릭 시각 목록) 반환."""
    clicks = []
    for i, p in enumerate(points):
        t = t0 + i / FPS
        if smoother is not None and p is not None:
            p = smoother(p, t)
        if d.update(p, t)[0]:
            clicks.append(t)
    return clicks


def jitter(center, n, sigma, outlier_every=0, seed=0):
    rnd = random.Random(seed)
    out = []
    for i in range(n):
        if outlier_every and i % outlier_every == outlier_every - 1:
            out.append((center[0] + 400, center[1] - 300))   # 순간 튐
        else:
            out.append((rnd.gauss(center[0], sigma), rnd.gauss(center[1], sigma)))
    return out


def test_click_after_dwell():
    d = DwellClicker(dwell_seconds=1.0, radius=50)
    assert d.update((100, 100), 0.0) == (False, 0.0)
    clicked, p = d.update((110, 105), 0.5)
    assert not clicked and 0.45 < p < 0.55
    assert d.update((105, 100), 1.0)[0] is True


def test_jittery_gaze_still_clicks():
    # 반경에 가까운 떨림 + 0.5초마다 큰 튐이 있어도 클릭돼야 함
    d = DwellClicker(dwell_seconds=1.0, radius=80, grace=0.35)
    pts = jitter((500, 400), FPS * 2, sigma=35, outlier_every=15)
    clicks = run(d, pts, smoother=OneEuroFilter())
    assert clicks and clicks[0] < 1.5


def test_click_point_is_center_not_last_sample():
    d = DwellClicker(dwell_seconds=1.0, radius=80)
    run(d, jitter((500, 400), FPS * 2, sigma=25, seed=3))
    cx, cy = d.click_point
    assert math.hypot(cx - 500, cy - 400) < 20


def test_blink_does_not_reset():
    d = DwellClicker(dwell_seconds=1.0, radius=50, grace=0.35)
    pts = [(100, 100)] * 12 + [None] * 6 + [(100, 100)] * 20   # 0.2초 깜빡임
    clicks = run(d, pts)
    assert clicks and clicks[0] < 1.1


def test_long_look_away_resets():
    d = DwellClicker(dwell_seconds=1.0, radius=50, grace=0.35)
    pts = [(100, 100)] * 15 + [(600, 600)] * 15 + [(100, 100)] * 15
    # 0.5초 다른 곳을 본 뒤에는 처음부터 다시 세므로, 1.0초 안에 클릭 없음
    assert run(d, pts) == []


def test_lost_face_long_resets():
    d = DwellClicker(dwell_seconds=1.0, radius=50, grace=0.35)
    pts = [(100, 100)] * 15 + [None] * 15 + [(100, 100)] * 15
    assert run(d, pts) == []


def test_no_repeat_click_until_rearmed():
    d = DwellClicker(dwell_seconds=1.0, radius=50, rearm_factor=1.5)
    clicks = run(d, [(100, 100)] * (FPS * 3))
    assert len(clicks) == 1
    clicks = run(d, [(300, 100)] * (FPS * 2), t0=3.0)
    assert len(clicks) == 1


def test_one_euro_reduces_jitter_but_follows_jumps():
    f = OneEuroFilter()
    raw = jitter((500, 400), FPS * 2, sigma=30, seed=1)
    sm = [f(p, i / FPS) for i, p in enumerate(raw)]
    spread = lambda ps: sum(math.hypot(x - 500, y - 400) for x, y in ps[20:]) / len(ps[20:])
    assert spread(sm) < spread(raw) * 0.5
    # 큰 이동은 0.3초 안에 따라감
    t0 = len(raw) / FPS
    for i in range(10):
        p = f((1200, 800), t0 + i / FPS)
    assert math.hypot(p[0] - 1200, p[1] - 800) < 100
