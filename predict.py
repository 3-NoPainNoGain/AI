import torch
import numpy as np
import os
from model import SignLanguage1DCNN

# 현재 파일 기준 경로 지정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 정확한 파일 경로 생성
CLASSES_PATH = os.path.join(BASE_DIR, "datasets", "processed", "classes.npy")
MODEL_PATH = os.path.join(BASE_DIR, "models", "model_val_60.61_20250518.pth")

# 클래스 정보 로딩
classes = np.load(CLASSES_PATH, allow_pickle=True)

# 모델 로드 함수
def load_model(model_path=MODEL_PATH):
    model = SignLanguage1DCNN(num_classes=len(classes))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# 예측 함수
def predict_from_keypoints(keypoints_30x154, model=None):
    if model is None:
        model = load_model()

    if isinstance(keypoints_30x154, list): 
        keypoints_30x154 = np.array(keypoints_30x154)

    input_tensor = torch.tensor(keypoints_30x154, dtype=torch.float32).unsqueeze(0) 

    with torch.no_grad():
        output = model(input_tensor)
        pred_idx = output.argmax(dim=1).item()
        return classes[pred_idx]
