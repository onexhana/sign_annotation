# sign_annotation

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
