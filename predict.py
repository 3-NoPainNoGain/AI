# predict.py
import torch
import numpy as np
from model import SignLanguage1DCNN

# 클래스 정보 로딩
classes = np.load("datasets/processed/classes.npy", allow_pickle=True)

# 모델 로드 함수
def load_model(model_path="models/model_val_60.61_20250518.pth"):
    model = SignLanguage1DCNN(num_classes=len(classes))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# 예측 함수
def predict_from_keypoints(keypoints_30x154, model=None):
    if model is None:
        model = load_model()

    if isinstance(keypoints_30x154, list):  # JSON 배열 대응
        keypoints_30x154 = np.array(keypoints_30x154)

    input_tensor = torch.tensor(keypoints_30x154, dtype=torch.float32).unsqueeze(0)  # (1, 30, 154)

    with torch.no_grad():
        output = model(input_tensor)
        pred_idx = output.argmax(dim=1).item()
        pred_class = classes[pred_idx]
        return pred_class
