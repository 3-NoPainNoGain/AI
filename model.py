# model.py
import torch
import torch.nn as nn

class SignLanguageBiLSTM(nn.Module):
    def __init__(self, input_size=258, hidden_size=256, num_layers=3, num_classes=11, dropout=0.5):
        super(SignLanguageBiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True,
                            dropout=dropout)
        self.bn = nn.BatchNorm1d(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)         # x: (B, 30, 258)
        out = out[:, -1, :]           # 마지막 프레임의 양방향 출력 → (B, 512)
        out = self.bn(out)
        out = self.dropout(out)
        return self.fc(out)           # (B, num_classes)   # predict.py
import os
import numpy as np
import torch
from .model import SignLanguageBiLSTM # 모델 클래스 임포트

# 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 경로 설정
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "datasets", "processed")
MODEL_PATH = os.path.join(BASE_DIR, "models", "model_bilstm_val_100_20250728.pth")

# 클래스 로딩
classes = np.load(os.path.join(DATA_DIR, "classes.npy"), allow_pickle=True)

# 모델 로딩 함수
def load_model():
    model = SignLanguageBiLSTM(num_classes=len(classes)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model, classes

# 예측 함수
def predict_from_keypoints(keypoints_30x258, model, classes):
    if isinstance(keypoints_30x258, list):
        keypoints_30x258 = np.array(keypoints_30x258)

    if keypoints_30x258.shape != (30, 258):
        raise ValueError(f"❗ 입력 shape 오류: 기대값은 (30, 258), 현재는 {keypoints_30x258.shape}")

    input_tensor = torch.tensor(keypoints_30x258, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        pred_idx = output.argmax(dim=1).item()
        return classes[pred_idx]