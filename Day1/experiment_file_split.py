"""
보너스 실험: 파일(결함 심각도) 단위 train/test split

세션1~3 노트북에서는 윈도우 단위로 "랜덤하게" train/test를 나눠서 1D-CNN이 100% Accuracy를 기록했다.
하지만 윈도우들은 50%씩 겹치기 때문에(window_size=2048, step=1024), 랜덤 분리 시
test 윈도우와 거의 동일한 train 윈도우가 존재할 수 있다 (데이터 누수).

이 스크립트는 그 대신, 결함 심각도(007/014 vs 021) 기준으로 "파일 단위"로 train/test를 나눈다.
즉, 모델이 학습 중에 한 번도 보지 못한 "더 큰 결함(021)" 데이터에 대해 얼마나 잘 일반화하는지 확인한다.

- Train: IR007, IR014, OR007, OR014, B007, B014, Normal(앞 80% 윈도우)
- Test : IR021, OR021, B021,           Normal(뒤 20% 윈도우)
"""

import re

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"사용할 디바이스: {device}")


def sliding_window(signal, window_size=2048, step=1024):
    segments = []
    for start in range(0, len(signal) - window_size, step):
        segments.append(signal[start:start + window_size])
    return np.array(segments)


def compute_fft(segment, fs=48000):
    n = len(segment)
    return np.abs(np.fft.rfft(segment))[:n // 2]


def get_de_signal(filepath):
    m = loadmat(filepath)
    file_num = re.search(r'_(\d+)\.mat$', filepath).group(1)
    de_key = f'X{file_num}_DE_time'
    return m[de_key].squeeze()


class_names = ['Normal', 'Inner Race', 'Outer Race', 'Ball']

# 클래스 0: Normal, 1: Inner Race, 2: Outer Race, 3: Ball
# (severity, label)
train_files = {
    'dataset/raw/IR007_1_110.mat': 1,
    'dataset/raw/IR014_1_175.mat': 1,
    'dataset/raw/OR007_6_1_136.mat': 2,
    'dataset/raw/OR014_6_1_202.mat': 2,
    'dataset/raw/B007_1_123.mat': 3,
    'dataset/raw/B014_1_190.mat': 3,
}
test_files = {
    'dataset/raw/IR021_1_214.mat': 1,
    'dataset/raw/OR021_6_1_239.mat': 2,
    'dataset/raw/B021_1_227.mat': 3,
}
normal_file = 'dataset/raw/Time_Normal_1_098.mat'


def windows_to_fft(filepath, window_size=2048, step=1024, fs=48000):
    signal = get_de_signal(filepath)
    segments = sliding_window(signal, window_size=window_size, step=step)
    return np.array([compute_fft(seg, fs=fs) for seg in segments])


# Train/Test에 사용할 (X, y) 모으기
X_train_list, y_train_list = [], []
X_test_list, y_test_list = [], []

for filepath, label in train_files.items():
    fft_feats = windows_to_fft(filepath)
    X_train_list.append(fft_feats)
    y_train_list += [label] * len(fft_feats)
    print(f"[TRAIN] {filepath:30s} | {class_names[label]:10s} | 윈도우 {len(fft_feats)}개")

for filepath, label in test_files.items():
    fft_feats = windows_to_fft(filepath)
    X_test_list.append(fft_feats)
    y_test_list += [label] * len(fft_feats)
    print(f"[TEST ] {filepath:30s} | {class_names[label]:10s} | 윈도우 {len(fft_feats)}개")

# Normal 파일은 1개뿐이므로, 윈도우를 시간 순서로 앞 80% / 뒤 20%로 분리
normal_feats = windows_to_fft(normal_file)
n_train = int(len(normal_feats) * 0.8)
X_train_list.append(normal_feats[:n_train])
y_train_list += [0] * n_train
X_test_list.append(normal_feats[n_train:])
y_test_list += [0] * (len(normal_feats) - n_train)
print(f"[TRAIN] {normal_file:30s} | {class_names[0]:10s} | 윈도우 {n_train}개 (앞 80%)")
print(f"[TEST ] {normal_file:30s} | {class_names[0]:10s} | 윈도우 {len(normal_feats) - n_train}개 (뒤 20%)")

X_train = np.vstack(X_train_list)
y_train = np.array(y_train_list)
X_test = np.vstack(X_test_list)
y_test = np.array(y_test_list)

print(f"\nX_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"y_train 클래스별 샘플 수: {dict(zip(*np.unique(y_train, return_counts=True)))}")
print(f"y_test  클래스별 샘플 수: {dict(zip(*np.unique(y_test, return_counts=True)))}")

# 정규화: train 데이터로만 fit
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 텐서 변환
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).unsqueeze(1)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

BATCH_SIZE = 32
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=BATCH_SIZE, shuffle=True)


class CNN1D(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=9, padding=4)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=9, padding=4)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(32 * 256, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.flatten(start_dim=1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


model = CNN1D(num_classes=4).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

NUM_EPOCHS = 10
for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | loss = {epoch_loss/len(train_loader):.4f}")

model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor.to(device)).argmax(dim=1).cpu().numpy()

acc = accuracy_score(y_test, y_pred)
print(f"\n=== 파일(결함 심각도) 단위 split 결과 ===")
print(f"테스트 Accuracy: {acc * 100:.2f}%")
print("\nConfusion Matrix (행=실제, 열=예측):")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
