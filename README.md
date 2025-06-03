# handDoc - AI 
MediaPipe를 활용해 수어 영상으로부터 키포인트 데이터를 추출하고, PyTorch 기반 1D-CNN 모델을 학습하여 `.pth` 형태의 추론 모델을 생성합니다.

### 🛠️ 기술 스택

- Python: 모델 학습 전체 로직
- MediaPipe: 영상에서 3D 키포인트 추출 (Face, Hands, Pose)
- NumPy / Pandas: 데이터 처리 및 저장
- PyTorch: 모델 구축, 학습, 추론
