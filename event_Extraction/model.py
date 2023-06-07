import torch
import torch.nn as nn
import torch.optim as optim
import cupy as cp
import numpy as np
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
from seqeval.metrics import classification_report, f1_score
from torchcrf import CRF
from init import *


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
