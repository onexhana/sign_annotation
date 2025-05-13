import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.rnn as rnn_utils
from collections import Counter
import re

# ✅ 1. Dataset 정의
class SignLanguageDataset(Dataset):
    def __init__(self, data_dir, label_map_path=None, min_samples=1):
        self.samples = []
        self.labels = []

        # 라벨 매핑 로드
        if label_map_path:
            with open(label_map_path, 'r', encoding='utf-8') as f:
                self.label_mapping = json.load(f)
        else:
            self.label_mapping = {}

        # 파일 탐색 및 라벨 정제
        for file in os.listdir(data_dir):
            if not file.endswith('.npy'):
                continue

            # ✅ ① chunk 제거
            filename_no_ext = file.replace('.npy', '')
            label_with_chunk = re.sub(r'_chunk\d+$', '', filename_no_ext)

            # ✅ ② 대표값 추출 (쉼표로 묶인 첫 단어)
            representative_label = label_with_chunk.split(',')[0]

            # ✅ ③ 매핑 적용
            mapped_label = self.label_mapping.get(label_with_chunk, representative_label)

            filepath = os.path.join(data_dir, file)
            self.samples.append(filepath)
            self.labels.append(mapped_label)

        # ✅ 라벨별 개수 출력
        counts = Counter(self.labels)
        print("\U0001f50d 라벨별 샘플 수:")
        for label, count in counts.items():
            print(f"{label}: {count}")

        # ✅ 샘플 수 기준 필터링
        filtered = [(x, y) for x, y in zip(self.samples, self.labels) if counts[y] >= min_samples]

        if not filtered:
            raise ValueError(f"❌ 유효한 학습 샘플이 없습니다. min_samples={min_samples} 기준을 낮추세요.")

        self.samples = [x for x, _ in filtered]
        self.labels = [y for _, y in filtered]

        self.label2idx = {label: idx for idx, label in enumerate(sorted(set(self.labels)))}
        self.idx2label = {idx: label for label, idx in self.label2idx.items()}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = np.load(self.samples[idx])
        label = self.label2idx[self.labels[idx]]
        return torch.tensor(data, dtype=torch.float32), torch.tensor(label)

# ✅ 2. Collate 함수
def collate_fn(batch):
    sequences, labels = zip(*batch)
    padded = rnn_utils.pad_sequence(sequences, batch_first=True)
    return padded, torch.tensor(labels)

# ✅ 3. LSTM 모델 정의
class SignLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SignLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn.squeeze(0))

# ✅ 4. 학습 실행
if __name__ == '__main__':
    data_dir = 'processed_sequences'
    label_map_path = 'label_mapping.json'

    dataset = SignLanguageDataset(data_dir, label_map_path=label_map_path, min_samples=1)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    input_size = 225
    hidden_size = 128
    num_classes = len(dataset.label2idx)

    model = SignLSTM(input_size, hidden_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print(f"\n\U0001f9e0 학습 시작: 총 {len(dataset)}개 샘플, {num_classes}개 클래스\n")

    for epoch in range(150):
        total_loss = 0
        for x_batch, y_batch in dataloader:
            output = model(x_batch)
            loss = criterion(output, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[Epoch {epoch+1:03d}] Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), 'sign_lstm.pth')
    print("\n✅ 모델 저장 완료: sign_lstm.pth")
    print("📌 라벨 인덱스 매핑:", dataset.label2idx)
