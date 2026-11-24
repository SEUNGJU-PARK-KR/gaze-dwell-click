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
import shutil
import sys
import time
import traceback

# MSMF 하드웨어 변환이 켜져 있으면 일부 웹캠이 안 열리거나 수십 초 걸린다 — cv2 import 전에 꺼야 함
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

import cv2
import numpy as np
from pynput.mouse import Button, Controller
from screeninfo import get_monitors

from eyeGestures import EyeGestures_v3

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


def _fix_mediapipe_nonascii_path():
    """윈도우 계정명이 한글 등 비ASCII일 때 mediapipe 크래시 우회.

    exe는 %TEMP%(C:\\Users\\정문\\...)에 풀리고, mediapipe C++이 한글 경로의
    모델 파일을 못 열어 'The path does not exist'로 죽는다.
    mediapipe가 쓰는 resource root를 ASCII 경로로 바꿔 우회한다."""
    if sys.platform != "win32":
        return
    try:
        import mediapipe.python.solution_base as sb
    except Exception:
        return
    root = os.sep.join(os.path.abspath(sb.__file__).split(os.sep)[:-3])
    try:
        root.encode("ascii")
        return  # 경로가 이미 ASCII면 문제없음
    except UnicodeEncodeError:
        pass
    new_root = None
    try:
        # NTFS 8.3 단축 경로가 있으면 복사 없이 그걸로 우회
        import ctypes
        buf = ctypes.create_unicode_buffer(1024)
        if ctypes.windll.kernel32.GetShortPathNameW(root, buf, 1024):
            buf.value.encode("ascii")
            new_root = buf.value
    except Exception:
        new_root = None
    if new_root is None:
        # 단축 경로도 없으면 ProgramData(순수 ASCII)에 models 복사
        new_root = os.path.join(os.environ.get("ProgramData", r"C:\ProgramData"),
                                "gaze-dwell-click")
        src = os.path.join(root, "mediapipe", "modules")
        dst = os.path.join(new_root, "mediapipe", "modules")
        try:
            if not os.path.isdir(dst):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copytree(src, dst)
                print(f"한글 계정 경로 우회 — mediapipe 리소스 복사: {dst}")
        except Exception as e:
            print(f"mediapipe 경로 우회 실패(무시): {e}")
            return
    # solution_base가 자기 __file__ 기준으로 root를 다시 계산하므로 경로를 패치
    sb.__file__ = os.path.join(new_root, "mediapipe", "python", "solution_base.py")


BACKEND_NAMES = {cv2.CAP_DSHOW: "DSHOW", cv2.CAP_MSMF: "MSMF", cv2.CAP_ANY: "ANY"}


def _camera_backends():
    if sys.platform == "win32":
        # DSHOW가 보통 빠르지만 안 되는 카메라는 MSMF로 열림
        return [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    return [cv2.CAP_ANY]


def _try_open(idx, backend, warmup=1.5):
    """카메라를 열고 실제 프레임이 나오는지 확인.

    DSHOW 웹캠은 켜진 직후 몇 프레임은 read()가 False를 돌려준다.
    첫 read 한 번만 보고 버리면 멀쩡한 카메라를 '실패'로 판정하므로
    warmup 초 동안 재시도한다."""
    cap = cv2.VideoCapture(idx, backend)
    if cap is None or not cap.isOpened():
        if cap is not None:
            cap.release()
        return None
    deadline = time.time() + warmup
    while time.time() < deadline:
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0:
            return cap
        time.sleep(0.05)
    cap.release()
    return None


def open_camera(preferred=0, max_index=5):
    """(인덱스 × 백엔드)로 카메라 탐색. 성공 시 (cap, index) 반환."""
    indices = [preferred] + [i for i in range(max_index) if i != preferred]
    for idx in indices:
        for backend in _camera_backends():
            cap = _try_open(idx, backend)
            if cap is not None:
                print(f"카메라 {idx} 연결 (백엔드 {BACKEND_NAMES.get(backend, backend)})")
                return cap, idx
    return None, None


def list_cameras(max_index=5):
    """--list-cameras: 열리는 카메라 번호/백엔드 목록 출력."""
    found = False
    for idx in range(max_index):
        for backend in _camera_backends():
            cap = _try_open(idx, backend, warmup=1.0)
            if cap is not None:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"  --camera {idx}  ({BACKEND_NAMES.get(backend, backend)}, {w}x{h})")
                cap.release()
                found = True
                break
    if not found:
        print("열리는 카메라가 없습니다.")
        print(diagnose_camera())


def _windows_camera_privacy():
    """윈도우 '카메라 액세스' 개인정보 설정. 'Deny'면 꺼진 것. 알 수 없으면 None."""
    if sys.platform != "win32":
        return {}
    import winreg
    base = r"Software\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\webcam"
    out = {}
    for hive, label in ((winreg.HKEY_LOCAL_MACHINE, "system"), (winreg.HKEY_CURRENT_USER, "user")):
        try:
            with winreg.OpenKey(hive, base) as k:
                out[label] = winreg.QueryValueEx(k, "Value")[0]
        except OSError:
            pass
    try:  # 데스크톱 앱(=이 exe) 허용 여부
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, base + r"\NonPackaged") as k:
            out["desktop"] = winreg.QueryValueEx(k, "Value")[0]
    except OSError:
        pass
    return out


def _windows_camera_devices():
    """장치 관리자에 보이는 카메라 이름 목록. 조회 실패 시 None."""
    if sys.platform != "win32":
        return None
    import subprocess
    cmd = ("Get-PnpDevice -PresentOnly -Class Camera,Image -ErrorAction SilentlyContinue | "
           "ForEach-Object { $_.Status + '|' + $_.FriendlyName }")
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", cmd],
                           capture_output=True, timeout=15,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        text = r.stdout.decode("utf-8", "replace") or r.stdout.decode("cp949", "replace")
        return [line.strip() for line in text.splitlines() if line.strip()]
    except Exception:
        return None


def diagnose_camera():
    """카메라가 안 열릴 때 원인을 좁혀서 안내문을 만든다."""
    lines = ["카메라를 열 수 없습니다."]
    if sys.platform == "win32":
        priv = _windows_camera_privacy()
        if "Deny" in (priv.get("system"), priv.get("user")):
            lines.append("▶ 원인: 윈도우 '카메라 액세스'가 꺼져 있습니다.")
            lines.append("  설정 → 개인 정보 보호 및 보안 → 카메라 → '카메라 액세스' 켜기")
            if priv.get("system") == "Deny":
                lines.append("  (관리자 정책으로 꺼져 있으면 학교 전산 담당자에게 요청해야 합니다)")
            return "\n".join(lines)
        if priv.get("desktop") == "Deny":
            lines.append("▶ 원인: '데스크톱 앱이 카메라에 액세스하도록 허용'이 꺼져 있습니다.")
            lines.append("  설정 → 개인 정보 보호 및 보안 → 카메라 → 맨 아래 항목 켜기")
            return "\n".join(lines)
        devices = _windows_camera_devices()
        if devices == []:
            lines.append("▶ 원인: 이 PC에서 카메라 장치가 검색되지 않습니다.")
            lines.append("  USB 웹캠을 꽂거나, 노트북이면 카메라 물리 스위치/Fn 키를 확인하세요.")
            return "\n".join(lines)
        if devices:
            lines.append("장치 관리자에 보이는 카메라:")
            for d in devices:
                status, _, name = d.partition("|")
                mark = "" if status == "OK" else f"  ← 상태 {status} (드라이버 문제 가능)"
                lines.append(f"  - {name}{mark}")
        lines.append("▶ 장치와 권한은 정상으로 보입니다. 확인할 것:" if devices else "▶ 확인할 것:")
        lines.append("  1. Zoom/Teams/카메라 앱/브라우저 등 다른 프로그램이 카메라를 쓰고 있지 않은지 (모두 닫고 재실행)")
        lines.append("  2. 카메라가 여러 대면 --list-cameras 로 번호 확인 후 --camera N")
        lines.append("  3. 백신/보안 프로그램의 '웹캠 보호'가 차단하고 있지 않은지")
        return "\n".join(lines)
    lines.append("카메라 연결/권한을 확인하고 --camera 번호를 바꿔보세요. (--list-cameras 로 목록 확인)")
    return "\n".join(lines)


def crash_log_path():
    base = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.getcwd()
    return os.path.join(base, "gaze-crash.log")


def pause_before_exit():
    """더블클릭 실행 시 콘솔이 바로 닫히지 않게 대기."""
    try:
        input("엔터를 누르면 종료... ")
    except Exception:
        pass


class OneEuroFilter:
    """시선 좌표 떨림 제거용 1€ 필터 (Casiez et al., 2012).

    천천히 움직이거나 한 곳을 볼 때는 강하게 평활화해 떨림을 없애고,
    시선을 크게 옮길 때는 평활화를 줄여 지연 없이 따라간다.
    """

    def __init__(self, min_cutoff=0.6, beta=0.004, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self):
        self.x = None
        self.dx = (0.0, 0.0)
        self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, point, now):
        if point is None:
            return self.x
        if self.x is None or self.t is None or now <= self.t:
            self.x, self.t = (float(point[0]), float(point[1])), now
            return self.x
        dt = now - self.t
        self.t = now
        a_d = self._alpha(self.d_cutoff, dt)
        raw_dx = ((point[0] - self.x[0]) / dt, (point[1] - self.x[1]) / dt)
        self.dx = (a_d * raw_dx[0] + (1 - a_d) * self.dx[0],
                   a_d * raw_dx[1] + (1 - a_d) * self.dx[1])
        speed = math.hypot(*self.dx)
        a = self._alpha(self.min_cutoff + self.beta * speed, dt)
        self.x = (a * point[0] + (1 - a) * self.x[0],
                  a * point[1] + (1 - a) * self.x[1])
        return self.x


class DwellClicker:
    """한 지점을 dwell_seconds 동안 응시하면 클릭한다.

    눈동자는 한 곳을 봐도 계속 미세하게 흔들리므로(고정 미동 + 추정 오차):
    - 기준점(anchor)은 첫 점이 아니라 응시 중 모인 점들의 평균(중심)으로 갱신한다.
    - 반경 밖으로 튄 점이나 깜빡임/얼굴 놓침(None)은 grace 초까지는 무시한다.
      grace 초 넘게 계속 벗어나야 응시가 끊긴 것으로 본다.
    - 클릭 위치는 흔들리는 현재 점이 아니라 기준점(중심)이다.
    - 클릭 후에는 반경 * rearm_factor 밖으로 나갔다 와야 재클릭(연타 방지).
    """

    def __init__(self, dwell_seconds=1.0, radius=80, rearm_factor=1.5, grace=0.35):
        self.dwell_seconds = dwell_seconds
        self.radius = radius
        self.rearm_factor = rearm_factor
        self.grace = grace
        self.anchor = None
        self.started = None
        self.n = 0
        self.away_since = None
        self.needs_rearm = False
        self.click_point = None

    def _restart(self, point, now):
        self.anchor = None if point is None else (float(point[0]), float(point[1]))
        self.started = None if point is None else now
        self.n = 0 if point is None else 1
        self.away_since = None

    def _progress(self, now):
        if self.started is None:
            return 0.0
        return min(1.0, (now - self.started) / self.dwell_seconds)

    def update(self, point, now):
        """(clicked, progress) 반환. progress 는 0.0~1.0. 클릭 위치는 self.click_point."""
        if self.needs_rearm:
            if point is not None and self.anchor is not None and math.hypot(
                    point[0] - self.anchor[0], point[1] - self.anchor[1]
            ) > self.radius * self.rearm_factor:
                self.needs_rearm = False
                self._restart(point, now)
            return False, 0.0

        if self.anchor is None:
            self._restart(point, now)
            return False, 0.0

        inside = point is not None and math.hypot(
            point[0] - self.anchor[0], point[1] - self.anchor[1]) <= self.radius

        if inside:
            self.away_since = None
            # 기준점을 응시 점들의 누적 평균으로 이동 (최근 점에 약간 더 가중)
            self.n += 1
            w = max(1.0 / self.n, 0.05)
            self.anchor = (self.anchor[0] + w * (point[0] - self.anchor[0]),
                           self.anchor[1] + w * (point[1] - self.anchor[1]))
        else:
            if self.away_since is None:
                self.away_since = now
            if now - self.away_since > self.grace:
                # 진짜로 다른 곳을 보는 중 → 새 응시 시작
                self._restart(point, now)
                return False, 0.0

        progress = self._progress(now)
        if progress >= 1.0:
            self.needs_rearm = True
            self.click_point = self.anchor
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
    ap.add_argument("--radius", type=int, default=None,
                    help="응시 판정 반경(px). 기본은 화면 짧은 변의 9%% (1080p≈97px)")
    ap.add_argument("--grace", type=float, default=0.35,
                    help="응시 중 시선이 튀거나 깜빡여도 봐주는 시간(초), 기본 0.35")
    ap.add_argument("--smooth", type=float, default=0.6,
                    help="떨림 보정 세기: 작을수록 더 부드럽고 느림(0.2~2.0), 기본 0.6")
    ap.add_argument("--no-snap", action="store_true",
                    help="응시 중 커서를 응시 중심에 고정하지 않기")
    ap.add_argument("--camera", type=int, default=0, help="카메라 인덱스")
    ap.add_argument("--video", type=str, default=None, help="웹캠 대신 영상 파일로 테스트")
    ap.add_argument("--no-move", action="store_true", help="실제 커서 이동 끄기")
    ap.add_argument("--no-click", action="store_true", help="실제 클릭 끄기(테스트용)")
    ap.add_argument("--list-cameras", action="store_true", help="열리는 카메라 번호 목록만 출력")
    args = ap.parse_args()

    if args.list_cameras:
        list_cameras()
        return

    sw, sh = screen_size()
    try:
        mouse = Controller()
    except Exception as e:
        print(f"마우스 제어 불가({e}) — 추적만 동작")
        mouse = None
    _fix_mediapipe_nonascii_path()
    gestures = EyeGestures_v3()
    gestures.uploadCalibrationMap(CALIBRATION_MAP, context="main")

    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise RuntimeError(f"영상 파일을 열 수 없습니다: {args.video}")
    else:
        cap, cam_idx = open_camera(args.camera)
        if cap is None:
            print(diagnose_camera())
            raise RuntimeError("카메라를 열 수 없습니다 (위 안내 참고)")
    test_mode = args.video is not None
    move_cursor = not args.no_move and not test_mode and mouse is not None
    do_click = not args.no_click and not test_mode and mouse is not None

    calib = CalibrationScreen(sw, sh, total_points=len(CALIBRATION_MAP),
                              auto_advance=args.calib_time)
    preview = Preview(sw, sh)
    radius = args.radius or max(60, int(0.09 * min(sw, sh)))
    dwell = DwellClicker(args.dwell, radius, grace=args.grace)
    smoother = OneEuroFilter(min_cutoff=args.smooth)
    paused = False
    click_count = 0

    calibrating = not test_mode  # 영상 테스트 모드는 캘리브레이션 생략
    if test_mode:
        print("영상 테스트 모드 — 캘리브레이션 생략, 커서 이동/클릭 비활성")
    print(f"화면 {sw}x{sh} | 응시 {args.dwell}s | 반경 {radius}px | 보정 {args.smooth}" +
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

                now = time.time()
                # 떨림 보정: 원시 추정치를 1€ 필터로 평활화
                point = smoother(point, now) if point is not None else None

                clicked = False
                progress = 0.0
                if not paused:
                    clicked, progress = dwell.update(point, now)
                    # 응시가 진행 중이면 커서를 응시 중심에 고정해 흔들림을 숨긴다
                    target = point
                    if (not args.no_snap and dwell.anchor is not None
                            and progress > 0.15 and not dwell.needs_rearm):
                        target = dwell.anchor
                    if clicked:
                        target = dwell.click_point
                    if target is not None and move_cursor:
                        x = min(max(target[0], 0), sw - 1)
                        y = min(max(target[1], 0), sh - 1)
                        mouse.position = (x, y)
                    if clicked:
                        click_count += 1
                        preview.flash()
                        if do_click:
                            mouse.click(Button.left)
                        cp = dwell.click_point
                        print(f"CLICK #{click_count} at ({cp[0]:.0f}, {cp[1]:.0f})")

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
                dwell = DwellClicker(args.dwell, radius, grace=args.grace)
                smoother.reset()
            if key == ord(" "):
                paused = not paused
    finally:
        cap.release()
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
