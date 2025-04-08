# 🖐️ Sign Language Keypoint Extractor & Visualizer

이 프로젝트는 **수어(Sign Language) 영상**에서  
MediaPipe Holistic을 활용하여 손, 얼굴, 포즈의 **키포인트를 추출하고**  
👉 색상별로 시각화하여  
👉 이미지 및 영상으로 저장하고  
👉 원본과 비교 가능한 영상까지 자동 생성하는 Python 파이프라인입니다.

---

## ✅ 주요 기능 요약

- ✅ **영상에서 프레임별 키포인트 추출**
- ✅ **손(빨강), 얼굴(파랑) 시각화 및 프레임 이미지 저장**
- ✅ **시각화 영상 자동 생성**  
  ⤷ `원본이름_result.mp4`
- ✅ **원본 + 시각화 비교 영상 생성**  
  ⤷ `원본이름_comparison.mp4`
- ✅ **키포인트 좌표 JSON 저장**  
  ⤷ `원본이름_keypoints.json`

---

## 🎥 사용 예시

`like.mkv` 라는 수어 영상이 있을 때, 실행하면 다음 파일들이 생성됩니다:

| 파일명 | 설명 |
|--------|------|
| `like_result.mp4` | 시각화된 프레임으로 만든 영상 |
| `like_comparison.mp4` | 원본 영상과 시각화 영상 나란히 비교 |
| `like_keypoints.json` | 프레임별 키포인트 좌표 저장 |
| `output_frames/frame_XXXXX.png` | 시각화된 프레임 이미지들 |

---

## 🛠️ 사용 방법

### 1. 필요한 라이브러리 설치

```bash
pip install opencv-python mediapipe
```

### 2. 결과 비교 미리보기
![image](https://github.com/user-attachments/assets/571b7868-43ee-49bb-90f9-696eb9569f77)


### 3. 메인 코드 실행

```bash
python main.py
```

---

## 💻 코드 구조 설명

`main.py` 하나로 모든 기능이 실행됩니다:

| 단계 | 설명 |
|------|------|
| 🎞️ 프레임 추출 및 키포인트 인식 | MediaPipe Holistic 사용 |
| 🎨 손: 빨강, 얼굴: 파랑 | `DrawingSpec`으로 색상 지정 |
| 🖼️ 프레임 저장 | `output_frames/`에 PNG로 저장 |
| 🎬 시각화 영상 생성 | `like_result.mp4` |
| 🆚 비교 영상 생성 | `like_comparison.mp4` |
| 🧾 키포인트 JSON 저장 | `like_keypoints.json` |

---

## 🧪 향후 확장 가능성

- 실시간 웹캠 수어 인식으로 확장
- 키포인트 데이터 기반 딥러닝 수어 번역 모델 학습
- 웹 어노테이션 툴 연동 (Flask, React 등)

---

## 📂 예시 프로젝트 구조

```
project/
├── like.mkv                    # 🎥 원본 영상
├── output_frames/             # 🖼️ 시각화된 프레임 이미지들
│   ├── frame_00000.png
│   └── ...
├── like_result.mp4            # 📹 시각화 영상
├── like_comparison.mp4        # 🆚 원본+시각화 비교 영상
├── like_keypoints.json        # 📄 키포인트 좌표
├── main.py                    # 🧠 전체 실행 코드
└── README.md                  # 📖 설명 파일
```
