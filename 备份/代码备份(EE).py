import torch
import torch.nn as nn
import torch.optim as optim
import cupy as cp
import numpy as np
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
from seqeval.metrics import classification_report, f1_score
from torchcrf import CRF
import os
import json

# 配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 1e-4


# 定义双向RNN+CRF模型
class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_id, hidden_dim=256):
        super(BiLSTM_CRF, self).__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_id = tag_to_id
        self.tagset_size = len(tag_to_id)

        self.bilstm = nn.LSTM(
            bert_model.config.hidden_size,
            hidden_dim // 2,
            num_layers=1,
            bidirectional=True,
        ).to(device)
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size).to(device)
        self.crf = CRF(self.tagset_size).to(device)

    def forward(self, x):
        x = x.to(device)
        with torch.no_grad():
            x = bert_model(x)[0]
        lstm_out, _ = self.bilstm(x)
        return self.hidden2tag(lstm_out)

    def loss(self, x, tags):
        feats = self.forward(x)
        return -self.crf(feats, tags)

    def predict(self, x):
        feats = self.forward(x)
        return self.crf(feats)

    def train_model(self, train_dataloader, optimizer, num_epochs):
        self.train()  # 将模型设置为训练模式

        for _ in range(num_epochs):
            for batch in train_dataloader:
                x, y = batch

                # 梯度清零
                optimizer.zero_grad()

                # 计算损失
                loss = self.loss(x, y)

                # 反向传播
                loss.backward()

                # 更新参数
                optimizer.step()

    def evaluate_model(self, eval_dataloader):
        self.eval()  # 将模型设置为评估模式
        total_acc = 0.0
        num_samples = 0

        with torch.no_grad():
            for batch in eval_dataloader:
                x, y = batch
                predictions = self.predict(x)
                correct = (predictions == y).sum().item()
                total_acc += correct
                num_samples += y.shape[0]

        return total_acc / num_samples

    def save_information(self, file_path, output_path):
        self.eval()
        with open(file_path, "r") as f:
            text = f.read()

        tokenized_text = tokenizer.tokenize(text)
        input_ids = tokenizer.encode(
            tokenized_text, add_special_tokens=True, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            predictions = self.predict(input_ids)

        pred_tags = [id_to_tag[id] for id in predictions[0]]

        information = [
            (token, tag) for token, tag in zip(tokenized_text, pred_tags) if tag != "O"
        ]
        with open(output_path, "w") as f:
            for item in information:
                f.write(f"{item[0]}\t{item[1]}\n")


# 数据集类
class EventDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        text = row["text"]
        tags = row["tags"]

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_token_type_ids=True,
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        label_ids = [tag_to_id[tag] for tag in tags]
        label_ids = [0] * (self.max_len - len(label_ids)) + label_ids

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "labels": torch.tensor(label_ids, dtype=torch.long),
        }


# 读取训练和验证数据
with open("train_data.json", "r") as f:
    train_data = json.load(f)

with open("val_data.json", "r") as f:
    val_data = json.load(f)

"""
数据集示例
[
  {
    "text": "The event will take place in New York.",
    "tags": ["O", "O", "O", "O", "O", "O", "B-ENTITY", "I-ENTITY"]
  },
  {
    "text": "John Doe is the organizer of the event.",
    "tags": ["B-ENTITY", "I-ENTITY", "O", "O", "O", "O", "O", "B-EVENT"]
  }
]
"""


# 定义实体和标签映射
tag_to_id = {
    "O": 0,
    "B-EVENT": 1,
    "I-EVENT": 2,
    "B-ENTITY": 3,
    "I-ENTITY": 4,
    "B-ATTRIBUTE": 5,
    "I-ATTRIBUTE": 6,
}
id_to_tag = {id: tag for tag, id in tag_to_id.items()}

# 创建数据加载器
train_dataset = EventDataset(train_data, tokenizer, MAX_LEN)
val_dataset = EventDataset(val_data, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 初始化模型和优化器
model = BiLSTM_CRF(len(tokenizer), tag_to_id).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 训练和评估模型
model.train(model, train_loader, val_loader, optimizer, EPOCHS)


# 测试模型
file_path = "input_article.txt"
article_name = file_path.replace(".txt", "")
output_path = f"information of {article_name}.txt"
model.save_information(model, file_path, output_path)
