# ai_model_258/model.py
import torch.nn as nn

class SignLanguage1DCNN(nn.Module):
    def __init__(self, num_classes):
        super(SignLanguage1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(258, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 7, 128)  # 30 → 15 → 7
        self.fc2 = nn.Linear(128, num_classes)
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, 258, 30)
        x = self.pool(self.activation(self.bn1(self.conv1(x))))
        x = self.pool(self.activation(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.activation(self.fc1(x)))
        return self.fc2(x)
