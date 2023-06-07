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
import EventDataset
import model


# 配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 1e-4


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


# 读取训练和验证数据
with open("train_data.json", "r") as f:
    train_data = json.load(f)

with open("val_data.json", "r") as f:
    val_data = json.load(f)


# 创建数据加载器
train_dataset = EventDataset(train_data, tokenizer, MAX_LEN)
val_dataset = EventDataset(val_data, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 初始化模型和优化器
model = model.BiLSTM_CRF(len(tokenizer), tag_to_id).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
