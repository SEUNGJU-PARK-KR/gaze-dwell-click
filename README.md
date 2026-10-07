# Gaze Dwell Click — 시선 마우스

웹캠으로 쳐다보는 곳에 마우스 커서가 따라가고, 한 지점을 **3초간 응시하면 자동으로 좌클릭**되는 프로그램.

오픈소스 [EyeGestures](https://github.com/NativeSensors/EyeGestures)(NativeSensors, Python)를 코어로 사용한다. 별도 하드웨어 없이 노트북 웹캠만으로 동작한다.

## 요구 사항

- Python 3.10 ~ 3.12
- 웹캠
- OS: Windows / macOS / Linux(X11)

> Linux는 Wayland에서 pynput 커서 제어가 안 된다. X11 세션에서 실행할 것.
> macOS는 처음 실행 시 시스템 설정 → 개인정보 보호 및 보안 → **손쉬운 사용** + **카메라** 권한을 터미널(또는 실행 앱)에 줘야 한다.

## 설치

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 실행

```bash
python main.py
```

1. 전체화면 캘리브레이션이 시작된다. 빨간 점 9개가 순서대로 나오는데, **점을 정확히 맞출 필요 없이** 그냥 쳐다보기만 하면 된다 — 점마다 2.5초씩 자동으로 다음으로 넘어간다(파란 원호가 카운트다운). 약 20초 걸린다.
2. 완료되면 미리보기 창이 뜨고 시선대로 커서가 움직인다.
3. 한 지점을 3초간 쳐다보면 자동으로 좌클릭. 클릭 후에는 시선을 다른 곳으로 옮겼다가 돌아와야 다음 클릭이 된다(연타 방지).

## 키

| 키 | 동작 |
|---|---|
| q / ESC | 종료 |
| c | 재캘리브레이션 |
| s | 캘리브레이션 건너뛰기(정확도 낮음, 빠른 테스트용) |
| 스페이스 | 커서 제어 일시정지/재개 |

## 옵션

```bash
python main.py --dwell 2.0          # 응시 시간 2초로
python main.py --calib-time 4.0     # 캘리브레이션 점당 자동 진행 시간 늘리기
python main.py --radius 120         # 응시 반경 120px (흔들림 크면 키우기)
python main.py --camera 1           # 카메라 번호 지정
python main.py --no-click           # 클릭 없이 시선 추적만
python main.py --video face.mp4     # 웹캠 대신 영상 파일로 테스트
```

## 팁

- 정확도가 이상하면 `c`로 재캘리브레이션. 자세/거리가 바뀌면 다시 잡는 게 좋다.
- 조명이 밝고 얼굴이 정면으로 잘 비칠수록 정확하다.
- 응시 반경(`--radius`)을 크게 하면 떨림이 있어도 클릭이 잘 되지만, 작은 버튼을 조준하기 어려워진다.
