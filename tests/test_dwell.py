import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

pytest.importorskip("cv2")
from main import DwellClicker  # noqa: E402


def test_click_after_dwell():
    d = DwellClicker(dwell_seconds=1.0, radius=50)
    assert d.update((100, 100), 0.0) == (False, 0.0)
    clicked, p = d.update((110, 105), 0.5)
    assert not clicked and 0.45 < p < 0.55
    assert d.update((105, 100), 1.0)[0] is True


def test_leaving_radius_resets():
    d = DwellClicker(dwell_seconds=1.0, radius=50)
    d.update((100, 100), 0.0)
    d.update((300, 300), 0.6)          # 반경 밖 → 새 기준점
    assert d.update((300, 300), 1.2)[0] is False
    assert d.update((300, 300), 1.6)[0] is True


def test_no_repeat_click_until_rearmed():
    d = DwellClicker(dwell_seconds=1.0, radius=50, rearm_factor=1.5)
    d.update((100, 100), 0.0)
    assert d.update((100, 100), 1.0)[0] is True
    # 같은 자리에 계속 머물러도 재클릭 없음
    assert d.update((100, 100), 3.0)[0] is False
    # 반경*1.5 밖으로 나갔다 오면 재장전
    d.update((200, 100), 3.1)
    d.update((200, 100), 3.2)
    assert d.update((200, 100), 4.2)[0] is True


def test_lost_face_resets():
    d = DwellClicker(dwell_seconds=1.0, radius=50)
    d.update((100, 100), 0.0)
    d.update(None, 0.5)
    assert d.update((100, 100), 1.0)[0] is False
