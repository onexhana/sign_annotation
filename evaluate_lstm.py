import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.utils.rnn as rnn_utils
from train_lstm import SignLanguageDataset, SignLSTM, collate_fn  # 기존 정의 재사용

# ✅ 설정
data_dir = "processed_sequences"
model_path = "sign_lstm.pth"
input_size = 225
hidden_size = 128
batch_size = 2

# ✅ 데이터셋 및 로더
dataset = SignLanguageDataset(data_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# ✅ 모델 정의 및 불러오기
num_classes = len(dataset.label2idx)
model = SignLSTM(input_size, hidden_size, num_classes)
model.load_state_dict(torch.load(model_path))
model.eval()  # 평가 모드

# ✅ 평가
correct = 0
total = 0

with torch.no_grad():
    for x_batch, y_batch in dataloader:
        outputs = model(x_batch)  # [B, C]
        predicted = torch.argmax(outputs, dim=1)  # 예측된 클래스
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)

accuracy = correct / total * 100
print(f"✅ 정확도 (Accuracy): {accuracy:.2f}%")
