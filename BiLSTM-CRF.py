"""
Author: LTH
Date: 2023-04-13 22:06:34
LastEditTime: 2023-04-19 10:13:31
FilePath: \files\python_work\dachuang\state2\BiLSTM-CRF.py
Description: 
Copyright (c) 2023 by LTH, All Rights Reserved. 

"""
import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch.optim as optim


class BiLSTM_CRF(nn.Module):
    def __init__(self, bert_path: str, tag_to_ix: dict, hidden_dim: int):
        super(BiLSTM_CRF, self).__init__()
        self.hidden_dim = hidden_dim
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.bert = BertModel.from_pretrained(bert_path)
        self.lstm = nn.LSTM(
            self.bert.config.hidden_size,
            hidden_dim // 2,
            num_layers=1,
            bidirectional=True,
        )
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000
        self.hidden = self.init_hidden()
        self.optimizer = optim.Adam(
            self.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08
        )

    def init_hidden(self) -> tuple:
       
        return (
            torch.randn(2, 1, self.hidden_dim // 2),
            torch.randn(2, 1, self.hidden_dim // 2),
        )

    def _forward_alg(self, feats: torch.FloatTensor) -> torch.FloatTensor:
        forward_var = self.init_alpha(0.0)
        for feat in feats:
            alphas_t = []
            for next_tag in range(self.tagset_size):
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                trans_score = self.transitions[next_tag].view(1, -1)
                next_tag_var = forward_var + trans_score + emit_score
                alphas_t.append(self._log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        return self._log_sum_exp(terminal_var)

    def _score_sentence(
        self, feats: torch.FloatTensor, tags: torch.LongTensor
    ) -> torch.FloatTensor:
        score = torch.zeros(1)
        tags = torch.cat(
            [torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags]
        )
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats: torch.FloatTensor) -> tuple:
        backpointers = []
        forward_var = self.init_alpha(0)
        for feat in feats:
            bptrs_t = []
            viterbivars_t = []
            for next_tag in range(self.tagset_size):
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = self._argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = self._argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]
        best_path.reverse()
        return path_score, best_path

    def init_alpha(self, arg0: int) -> torch.FloatTensor:
        init_alphas = torch.full((1, self.tagset_size), -10000.0)
        init_alphas[0][self.tag_to_ix[START_TAG]] = arg0
        return init_alphas

    def _log_sum_exp(self, vec: torch.FloatTensor) -> torch.FloatTensor:
        max_score = vec[0, self._argmax(vec)]
        max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
        return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

    def _argmax(self, vec: torch.FloatTensor) -> int:
        _, idx = torch.max(vec, 1)
        return idx.item()

    def neg_log_likelihood(
        self, sentence: str, tags: torch.LongTensor
    ) -> torch.FloatTensor:
        lstm_feats = self.embed_and_tag(sentence)
        forward_score = self._forward_alg(lstm_feats)
        gold_score = self._score_sentence(lstm_feats, tags)
        return forward_score - gold_score

    def forward(self, sentence: str) -> tuple:
        lstm_feats = self.embed_and_tag(sentence)
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq

    def embed_and_tag(self, sentence: str) -> torch.FloatTensor:
        self.hidden = self.init_hidden()
        indexed_tokens = self.bert_tokenizer.encode(sentence)
        tokens_tensor = torch.tensor([indexed_tokens])
        with torch.no_grad():
            encoded_layers, _ = self.bert(tokens_tensor)
        lstm_out, self.hidden = self.lstm(encoded_layers.permute(1, 0, 2))
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        return self.hidden2tag(lstm_out)

    def train(self, sentences: list, true_tags: list) -> None:
        for sentence, tags in zip(sentences, true_tags):
            loss = self.model.neg_log_likelihood(sentence, tags)
            loss.backward()
            self.optimizer.step()

    def evaluate(self, test_sentences: list, test_tags: list) -> tuple:
        correct = 0  
        tp, fp, fn = 0, 0, 0  
        for sentence, true_tags in zip(test_sentences, test_tags):
            _, predicted_tags = self.forward(sentence)
            for predicted_tag, true_tag in zip(predicted_tags, true_tags):
                if predicted_tag == true_tag:
                    correct += 1
                    if predicted_tag != "O":
                        tp += 1
                elif predicted_tag != "O":
                    if true_tag != "O":
                        fn += 1
                    fp += 1
        accuracy = correct / sum(len(tags) for tags in test_tags)
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1_score = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0
        )

        return accuracy, precision, recall, f1_score


bert_model = "bert-base-uncased"
tag_to_ix = {
    "O": 0,
    "B-PER": 1,
    "I-PER": 2,
    "B-ORG": 3,
    "I-ORG": 4,
    "B-LOC": 5,
    "I-LOC": 6,
    "<START>": 7,
    "<STOP>": 8,
}


START_TAG = "<START>"
STOP_TAG = "<STOP>"
hidden_dim = 1000
model = BiLSTM_CRF(bert_model, tag_to_ix, hidden_dim).cuda()
