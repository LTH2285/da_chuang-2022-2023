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
        """
        函数说明:初始化

        Args:
        - bert_path : str : bert模型路径
        - tag_to_ix : dict : 标签到ID的映射词典
        - hidden_dim : int : 隐藏层维度
        """
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

        # 输入序列的特征为bert的隐状态，双向LSTM后，长度不再是输入序列的长度，而是隐藏状态的维度
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # bi-LSTM的输出，映射到命名实体标签的概率
        # 定义了模型中的转移矩阵（transition matrix），其大小为总标签数×总标签数。
        # 转移矩阵的元素为模型预测从一个标签转移到另一个标签的概率。
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # 设置初始状态的转移概率权重，使得序列第一个标记必须是<start>，序列最后一个标记必须是<stop>
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

        self.optimizer = optim.Adam(
            self.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08
        )

    def init_hidden(self) -> tuple:
        """
        函数说明:初始化隐状态

        Args:
        - None

        Returns:
        - tuple(torch.FloatTensor,torch.FloatTensor):初始隐状态
        """
        return (
            torch.randn(2, 1, self.hidden_dim // 2),
            torch.randn(2, 1, self.hidden_dim // 2),
        )

    def _forward_alg(self, feats: torch.FloatTensor) -> torch.FloatTensor:
        """
        函数说明:前向算法

        Args:
        - feats : torch.FloatTensor : 特征矩阵

        Returns:
        - torch.FloatTensor : 前向传播最终输出
        """

        forward_var = self.init_alpha(0.0)
        for feat in feats:
            # alphas_t表示前一个单词每个可能标记的前向变量
            alphas_t = []
            for next_tag in range(self.tagset_size):
                # emit_score表示当前词每个标记的发射分数，
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # trans_score表示从每个可能的前一个标记到每个可能的当前标记的转移分数
                trans_score = self.transitions[next_tag].view(1, -1)
                # next_tag_var表示当前单词每个可能标记的发射和转移分数
                next_tag_var = forward_var + trans_score + emit_score
                alphas_t.append(self._log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        # terminal_var指的是终止步骤的前向变量向量。它用于计算整个序列的最终概率。
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        return self._log_sum_exp(terminal_var)

    def _score_sentence(
        self, feats: torch.FloatTensor, tags: torch.LongTensor
    ) -> torch.FloatTensor:
        """
        函数说明:计算真实标签下，预测标签得分

        Args:
        - feats : torch.FloatTensor : 特征矩阵
        - tags : torch.LongTensor : 真实标签

        Returns:
        - torch.FloatTensor : 得分
        """
        # 当前得分初始化为0
        score = torch.zeros(1)
        # 将真实标签tags的起始标记加入到tags中
        tags = torch.cat(
            [torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags]
        )
        # 然后遍历特征矩阵中的每个特征值，计算出score的值
        for i, feat in enumerate(feats):
            # 得分的计算方式为将score加上当前特征值的预测标签与
            # 真实标签之间的转移得分和该特征下真实标签对应的得分
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        # 将score加上最后一个特征值与STOP标记之间的转移得分
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats: torch.FloatTensor) -> tuple:
        """
        函数说明:Viterbi解码算法

        Args:
        - feats : torch.FloatTensor : 特征矩阵,包含了从前一个状态到当前状态的所有可能的转移特征。

        Returns:
        - tuple(torch.FloatTensor, list[int]) : 返回Viterbi算法得分和最优标签序列
        """
        # backpointers: 保存回溯路径的列表，
        # 用于记录从前一个标签转移到当前标签时选择的最佳前驱标签。
        backpointers = []
        # forward_var:前向变量，表示当前节点是前向算法递推的结果,
        # 用来表示前一个节点的最优标签和下一个节点的分数之和。
        forward_var = self.init_alpha(0)
        for feat in feats:
            # bptrs_t: 保存当前节点标签选择的最佳前驱的标签
            bptrs_t = []
            # viterbivars_t: 保存从前驱标签转移到当前标签的计算结果变量。
            viterbivars_t = []
            for next_tag in range(self.tagset_size):
                # next_tag: 存储每一个标签的下标。
                # next_tag_var: 存储从前一个标签到当前标签的计算结果变量。
                next_tag_var = forward_var + self.transitions[next_tag]
                # best_tag_id: 存储每一个标签的最优下标。
                best_tag_id = self._argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)
        # terminal_var: 最终变量，加上从最后一个标签到结束标签的转移特征后得到的变量。
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        # 得到最后状态的best_tag_id，为后面迭代做初始化
        best_tag_id = self._argmax(terminal_var)
        # path_score: 最终得分。
        path_score = terminal_var[0][best_tag_id]
        # best_path: 最优路径。
        best_path = [best_tag_id]
        # 找到最优路径。使用回溯方法从最后一个标签开始，
        # 根据保存的回溯路径列表backpointers找到当前标签的最佳前驱标签，
        # 然后将该标签添加到最优路径列表best_path中。
        # 最终返回最优路径得分path_score和最优路径列表best_path。
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        start = best_path.pop()
        # 判断路径的正确性，否则报错
        assert start == self.tag_to_ix[START_TAG]
        best_path.reverse()
        return path_score, best_path

    def init_alpha(self, arg0: int) -> torch.FloatTensor:
        """
        函数说明:初始化开始标记的概率

        Args:
        - arg0 : int : 初始值

        Returns:
        - torch.FloatTensor : 初始化的alpha
        """
        init_alphas = torch.full((1, self.tagset_size), -10000.0)
        init_alphas[0][self.tag_to_ix[START_TAG]] = arg0
        return init_alphas

    def _log_sum_exp(self, vec: torch.FloatTensor) -> torch.FloatTensor:
        """
        函数说明:对数求和计算，防止计算过程中出现数值下溢

        Args:
        - vec : torch.FloatTensor : 向量

        Returns:
        - torch.FloatTensor : 对数求和值
        """
        max_score = vec[0, self._argmax(vec)]
        max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
        return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

    def _argmax(self, vec: torch.FloatTensor) -> int:
        """
        函数说明:返回向量中最大值的下标

        Args:
        - vec : torch.FloatTensor : 向量

        Returns:
        - int : 最大值下标
        """
        # _是最大值，由于不需要所以写_
        _, idx = torch.max(vec, 1)
        return idx.item()

    def neg_log_likelihood(
        self, sentence: str, tags: torch.LongTensor
    ) -> torch.FloatTensor:
        """
        函数说明:计算负对数似然值，用作训练模型的损失

        Args:
        - sentence : str : 待分词长度
        - tags : torch.LongTensor : 真实标签

        Returns:
        - torch.FloatTensor : 负对数似然值
        """
        lstm_feats = self.embed_and_tag(sentence)
        forward_score = self._forward_alg(lstm_feats)
        gold_score = self._score_sentence(lstm_feats, tags)
        return forward_score - gold_score

    def forward(self, sentence: str) -> tuple:
        """
        函数说明:前向传播预测，输出路径

        Args:
        - sentence : str : 待分词字符串

        Returns:
        - tuple : 返回路径得分和对应标签序列
        """
        lstm_feats = self.embed_and_tag(sentence)
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq

    def embed_and_tag(self, sentence: str) -> torch.FloatTensor:
        """
        函数说明:输入嵌入编码和标记

        Args:
        - sentence : str : 待分词字符串

        Returns:
        - torch.FloatTensor : 返回已经嵌入并标记的特征矩阵
        """
        self.hidden = self.init_hidden()
        indexed_tokens = self.bert_tokenizer.encode(sentence)
        tokens_tensor = torch.tensor([indexed_tokens])
        # 使用no_grad函数禁止梯度反向传播，以防在推理过程中计算出梯度。
        with torch.no_grad():
            encoded_layers, _ = self.bert(tokens_tensor)
        # PyTorch中LSTM模型接受的输入形状，与encoded_layers的形状不同
        lstm_out, self.hidden = self.lstm(encoded_layers.permute(1, 0, 2))
        # 模型的输出被转换为句子长度与LSTM隐层维度相同的形状。
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        return self.hidden2tag(lstm_out)

    def train(self, sentences: list, true_tags: list) -> None:
        """
        对于每一个句子和其真实标签，计算其negative log likelihood，
        得到该句子和标签组合的损失，然后在该损失的基础上计算梯度，
        再使用优化器根据梯度的方向更新模型参数。

        Args:
        - sentences (list): 用于训练模型的句子列表（单词列表）。每个句子都是一个词的列表。
        - true_tags (list): 与每个句子相对应的真实标签列表（字符串标签列表）。

        Returns:
        - None
        """
        for sentence, tags in zip(sentences, true_tags):
            # 评估该句子及其对应的真实标签的负对数似然
            loss = self.model.neg_log_likelihood(sentence, tags)
            # 计算与模型参数有关的损失梯度
            loss.backward()
            # 根据计算出的梯度更新模型参数
            self.optimizer.step()

    def evaluate(self, test_sentences: list, test_tags: list) -> tuple:
        """
        在给定的测试句子及其相应的标签上评估模型的准确性和其他指标。

        Args:
        - test_sentences (list): 用于测试模型的句子列表（单词列表）。每个测试句子都是一个词的列表。
        - test_tags (list): 与每个测试句子相对应的真实标签列表（字符串标签列表）。

        Returns:
        - 一个评价指标的元组：
            - Accuracy (float): 正确预测的标签占所有标签的比例。
            - Precision (float): 真阳性占所有预测阳性的比例。
            - Recall (float): 真正的阳性结果占所有实际阳性结果的比例。
            - F1 Score (float): 精度和召回率的调和平均数。
        """

        correct = 0  # 正确标记的标记数
        tp, fp, fn = 0, 0, 0  # 真阳性、假阳性、假阴性的数量

        # 循环浏览所有测试句子和它们的真实标签
        for sentence, true_tags in zip(test_sentences, test_tags):
            # 预测句子中每个符号的标签，predict要改
            _, predicted_tags = self.forward(sentence)
            # 对每个标记的预测标签和真实标签进行比较，并相应地更新评估指标
            for predicted_tag, true_tag in zip(predicted_tags, true_tags):
                if predicted_tag == true_tag:
                    correct += 1
                    if predicted_tag != "O":
                        tp += 1
                elif predicted_tag != "O":
                    if true_tag != "O":
                        fn += 1
                    fp += 1

        # 计算评价指标
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
# 'O'与字符无关，主要用于标识无命名实体；'B-PER'和'I-PER'表示人名的开始和继续；
# 'B-ORG'和'I-ORG'则表示组织机构名称的开始和继续；'B-LOC'和'I-LOC'则表示地名的开始和继续。
# START_TAG和STOP_TAG也应该写入
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
model = BiLSTM_CRF(bert_model, tag_to_ix, hidden_dim)

# for param in model.parameters():
#     print(type(param), param.size())

# 优化示例，已经封装成成员函数了
# optimizer = optim.SGD(model.parameters(), lr=0.01)
# optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
# sentence = "John lives in New York City"
# true_tags = [1, 0, 0, 0, 0, 5, 5, 5]
# loss = model.neg_log_likelihood(sentence, true_tags)
# loss.backward() # compute gradients
# optimizer.step() # update the parameters
