import torch
from torch.utils.data import Dataset


# 数据集类
class EventDataset(Dataset):
    def __init__(self, data, tokenizer, max_len, tag_to_id):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.tag_to_id = tag_to_id

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

        label_ids = [self.tag_to_id[tag] for tag in tags]
        label_ids = [0] * (self.max_len - len(label_ids)) + label_ids

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "labels": torch.tensor(label_ids, dtype=torch.long),
        }
