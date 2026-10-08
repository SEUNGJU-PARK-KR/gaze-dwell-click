#!/usr/bin/env python3
"""시선 마우스 — 웹캠 아이트래킹 + 응시(기본 1초) 자동 클릭

오픈소스 EyeGestures(NativeSensors)로 시선→화면 좌표를 변환하고,
pynput으로 실제 OS 커서를 움직이며, 한 지점을 일정 시간 응시하면 클릭한다.

키:
    q        종료
    c        재캘리브레이션
    s        캘리브레이션 건너뛰기(정확도 낮음, 빠른 테스트용)
    스페이스   커서 제어 일시정지/재개
"""
import argparse
import math
import os
import sys
import time
import traceback

import cv2
import numpy as np
from pynput.mouse import Button, Controller
from screeninfo import get_monitors

from eyeGestures import EyeGestures_v3
from eyeGestures.utils import VideoCapture

# 3x3 캘리브레이션 점 (화면 가장자리에서 5% 안쪽)
CALIBRATION_MAP = np.array(
    [
        [x, y]
        for y in (0.05, 0.5, 0.95)
        for x in (0.05, 0.5, 0.95)
    ]
)


def screen_size():
    m = next((m for m in get_monitors() if m.is_primary), get_monitors()[0])
    return m.width, m.height


def crash_log_path():
    base = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.getcwd()
    return os.path.join(base, "gaze-crash.log")


def pause_before_exit():
    """더블클릭 실행 시 콘솔이 바로 닫히지 않게 대기."""
    try:
        input("엔터를 누르면 종료... ")
    except Exception:
        pass


class DwellClicker:
    """한 지점에 시선이 dwell_seconds 동안 머물면 클릭한다.

    클릭 후에는 시선이 반경 * rearm_factor 밖으로 나갔다가
    다시 들어와야 재클릭이 가능하다 (연타 방지).
    """

    def __init__(self, dwell_seconds=3.0, radius=80, rearm_factor=1.5):
        self.dwell_seconds = dwell_seconds
        self.radius = radius
        self.rearm_factor = rearm_factor
        self.anchor = None
        self.started = None
        self.needs_rearm = False

    def update(self, point, now):
        """(clicked, progress) 반환. progress 는 0.0~1.0."""
        if point is None:
            self.anchor = None
            self.started = None
            return False, 0.0

        dist = math.inf if self.anchor is None else math.hypot(
            point[0] - self.anchor[0], point[1] - self.anchor[1]
        )

        if self.needs_rearm:
            if dist > self.radius * self.rearm_factor:
                self.needs_rearm = False
                self.anchor, self.started = point, now
            return False, 0.0

        if dist > self.radius:
            self.anchor, self.started = point, now
            return False, 0.0

        progress = min(1.0, (now - self.started) / self.dwell_seconds)
        if progress >= 1.0:
            self.needs_rearm = True
            self.started = None
            return True, 1.0
        return False, progress


class CalibrationScreen:
    """전체화면 캘리브레이션 창. cevent.point 위치에 점을 그린다."""

    NAME = "calibration"

    def __init__(self, width, height, total_points, auto_advance=2.5):
        self.width = width
        self.height = height
        self.total = total_points
        self.auto_advance = auto_advance
        self.done = 0
        self._prev_target = None
        self.target_started = time.time()
        cv2.namedWindow(self.NAME, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(self.NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def draw(self, cevent):
        now = time.time()
        canvas = np.zeros((self.height, self.width, 3), np.uint8)
        tx, ty = int(cevent.point[0]), int(cevent.point[1])

        # 목표 전환 감지 → 진행 카운트
        target = (tx, ty)
        if self._prev_target is not None and target != self._prev_target:
            self.done += 1
            self.target_started = now
        self._prev_target = target

        # 바깥 링(수용 반경) + 중심점 + 자동진행 카운트다운
        r = max(15, int(cevent.acceptance_radius))
        cv2.circle(canvas, target, r, (0, 90, 255), 2)
        cv2.circle(canvas, target, 8, (0, 0, 255), -1)
        cv2.circle(canvas, target, 12, (255, 255, 255), 2)
        frac = min(1.0, (now - self.target_started) / self.auto_advance)
        cv2.ellipse(canvas, target, (30, 30), -90, 0, int(360 * frac),
                    (255, 200, 0), 3)

        msg = f"Look at the dot   [{min(self.done, self.total)}/{self.total}]  auto"
        cv2.putText(canvas, msg, (self.width // 2 - 220, self.height - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
        cv2.putText(canvas, "q: quit", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (120, 120, 120), 2)
        cv2.imshow(self.NAME, canvas)

    @property
    def should_advance(self):
        """auto_advance 초 경과 — 정확히 맞추지 않아도 다음 점으로."""
        return time.time() - self.target_started >= self.auto_advance

    @property
    def finished(self):
        return self.done >= self.total

    def close(self):
        try:
            cv2.destroyWindow(self.NAME)
        except cv2.error:
            pass


class Preview:
    """추적 모드 미리보기 창: 웹캠 영상 + 시선 점 + 응시 진행 링."""

    NAME = "gaze-dwell-click"

    def __init__(self, screen_w, screen_h, preview_w=480):
        self.sw, self.sh = screen_w, screen_h
        self.pw = preview_w
        self.click_flash_until = 0.0
        cv2.namedWindow(self.NAME, cv2.WINDOW_NORMAL)

    def flash(self):
        self.click_flash_until = time.time() + 0.4

    def draw(self, frame, point, anchor, progress, paused, msg):
        h, w = frame.shape[:2]
        scale = self.pw / w
        preview = cv2.resize(cv2.flip(frame, 1), (self.pw, int(h * scale)))

        def to_preview(pt):
            return int(self.pw - pt[0] * (self.pw / self.sw)), int(pt[1] * (preview.shape[0] / self.sh))

        if anchor is not None and progress > 0:
            ax, ay = to_preview(anchor)
            cv2.circle(preview, (ax, ay), 26, (0, 200, 255), 1)
            cv2.ellipse(preview, (ax, ay), (22, 22), -90,
                        0, int(360 * progress), (0, 200, 255), 3)

        if point is not None:
            px, py = to_preview(point)
            color = (0, 0, 255) if time.time() < self.click_flash_until else (0, 255, 0)
            cv2.circle(preview, (px, py), 6, color, -1)

        status = "PAUSED" if paused else (msg or "tracking")
        cv2.putText(preview, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255) if paused else (0, 180, 0), 2)
        cv2.putText(preview, "q:quit  c:recalibrate  space:pause", (10, preview.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
        cv2.imshow(self.NAME, preview)

    def close(self):
        try:
            cv2.destroyWindow(self.NAME)
        except cv2.error:
            pass


def main():
    ap = argparse.ArgumentParser(description="웹캠 시선 마우스 — 응시(기본 1초) 자동 클릭")
    ap.add_argument("--dwell", type=float, default=1.0, help="응시 클릭 시간(초), 기본 1.0")
    ap.add_argument("--calib-time", type=float, default=2.5,
                    help="캘리브레이션 점당 자동 진행 시간(초), 기본 2.5")
    ap.add_argument("--radius", type=int, default=80, help="응시 판정 반경(px), 기본 80")
    ap.add_argument("--camera", type=int, default=0, help="카메라 인덱스")
    ap.add_argument("--video", type=str, default=None, help="웹캠 대신 영상 파일로 테스트")
    ap.add_argument("--no-move", action="store_true", help="실제 커서 이동 끄기")
    ap.add_argument("--no-click", action="store_true", help="실제 클릭 끄기(테스트용)")
    args = ap.parse_args()

    sw, sh = screen_size()
    try:
        mouse = Controller()
    except Exception as e:
        print(f"마우스 제어 불가({e}) — 추적만 동작")
        mouse = None
    gestures = EyeGestures_v3()
    gestures.uploadCalibrationMap(CALIBRATION_MAP, context="main")

    # 영상 파일은 eyeGestures VideoCapture가 EOF에서 블록되므로 cv2를 직접 사용
    if args.video:
        cap = cv2.VideoCapture(args.video)
    else:
        cap = VideoCapture(args.camera)
        if not cap.cap.isOpened():
            raise RuntimeError(f"카메라 {args.camera}을(를) 열 수 없습니다. "
                               "카메라 연결/권한 확인 후 --camera 번호를 바꿔보세요.")
    test_mode = args.video is not None
    move_cursor = not args.no_move and not test_mode and mouse is not None
    do_click = not args.no_click and not test_mode and mouse is not None

    calib = CalibrationScreen(sw, sh, total_points=len(CALIBRATION_MAP),
                              auto_advance=args.calib_time)
    preview = Preview(sw, sh)
    dwell = DwellClicker(args.dwell, args.radius)
    paused = False
    click_count = 0

    calibrating = not test_mode  # 영상 테스트 모드는 캘리브레이션 생략
    if test_mode:
        print("영상 테스트 모드 — 캘리브레이션 생략, 커서 이동/클릭 비활성")
    print(f"화면 {sw}x{sh} | 응시 {args.dwell}s" +
          (" | 시작하면 캘리브레이션 9점부터 진행" if calibrating else ""))

    step_errors = 0
    cevent = None
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if test_mode:
                    cap.release()
                    cap = cv2.VideoCapture(args.video)  # 테스트 영상 반복
                    continue
                break

            # 얼굴을 못 찾은 프레임 등에서 eyeGestures가 예외를 던질 수 있음
            # (face.py의 Face.process가 multi_face_landmarks=None에 접근) —
            # 한 프레임 오류가 앱을 죽이지 않게 스킵한다.
            try:
                event, cevent = gestures.step(frame, calibrating, sw, sh, context="main")
            except Exception as e:
                step_errors += 1
                if step_errors <= 5:
                    print(f"프레임 처리 오류(건너뜀, 계속 동작): {e}")
                event = None
            key = cv2.waitKey(1) & 0xFF

            if calibrating:
                if cevent is not None:
                    calib.draw(cevent)
                if calib.should_advance:
                    # 점을 정확히 맞추지 않아도 시간이 지나면 다음 목표로 진행
                    gestures.clb["main"].movePoint()
                if calib.finished:
                    calibrating = False
                    calib.close()
                    print("캘리브레이션 완료 — 추적 시작")
            else:
                point = None
                if event is not None and not event.blink:
                    point = (float(event.point[0]), float(event.point[1]))
                    # 화면 밖으로 나가는 추정치는 무시 (보정 전 (0,0) 등)
                    if not (1 <= point[0] <= sw - 1 and 1 <= point[1] <= sh - 1):
                        point = None

                clicked = False
                progress = 0.0
                if not paused:
                    clicked, progress = dwell.update(point, time.time())
                    if point is not None and move_cursor:
                        x = min(max(point[0], 0), sw - 1)
                        y = min(max(point[1], 0), sh - 1)
                        mouse.position = (x, y)
                    if clicked:
                        click_count += 1
                        preview.flash()
                        if do_click:
                            mouse.click(Button.left)
                        print(f"CLICK #{click_count} at {point}")

                msg = None if not test_mode else "TEST MODE (video)"
                preview.draw(frame, point, dwell.anchor, progress, paused, msg)

            if key == ord("q") or key == 27:
                break
            if key == ord("s") and calibrating:
                calibrating = False
                calib.close()
                print("캘리브레이션 건너뜀 — 추적 시작")
            if key == ord("c") and not calibrating:
                calibrating = True
                calib = CalibrationScreen(sw, sh, total_points=len(CALIBRATION_MAP),
                                          auto_advance=args.calib_time)
                dwell = DwellClicker(args.dwell, args.radius)
            if key == ord(" "):
                paused = not paused
    finally:
        if test_mode:
            cap.release()
        else:
            cap.close()
        calib.close()
        preview.close()
        cv2.destroyAllWindows()
        print(f"종료. 총 클릭 {click_count}회")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        tb = traceback.format_exc()
        path = crash_log_path()
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(tb)
            print(f"오류로 종료됨 — 로그: {path}")
        except Exception:
            print(tb)
        pause_before_exit()
        sys.exit(1)
    else:
        pause_before_exit()
