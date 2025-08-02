# train.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from model import SignLanguageBiLSTM

# 경로 설정
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "datasets", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# 데이터 불러오기
X = np.load(os.path.join(DATA_DIR, "X.npy"))  # (N, 30, 258)
Y = np.load(os.path.join(DATA_DIR, "Y.npy"))  # (N,)
classes = np.load(os.path.join(DATA_DIR, "classes.npy"), allow_pickle=True)

# 하이퍼파라미터
BATCH_SIZE = 32
EPOCHS = 50
LR = 0.0005
PATIENCE = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset 정의
class SignDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.long)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# 데이터 분할
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)
train_loader = torch.utils.data.DataLoader(SignDataset(X_train, Y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(SignDataset(X_val, Y_val), batch_size=BATCH_SIZE)

# 모델 초기화
model = SignLanguageBiLSTM(num_classes=len(classes)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

# 학습 루프
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        output = model(xb)
        loss = criterion(output, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            output = model(xb)
            val_loss += criterion(output, yb).item()
            correct += (output.argmax(1) == yb).sum().item()

    val_loss /= len(val_loader)
    acc = correct / len(X_val)

    print(f"[{epoch+1}/{EPOCHS}] Train Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f}, Acc: {acc:.4f}")

    # Early stopping + 모델 저장
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, "model_bilstm_val_100_20250728.pth"))
        print("✅ 모델 저장됨")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("⏹️ Early stopping")
            break
