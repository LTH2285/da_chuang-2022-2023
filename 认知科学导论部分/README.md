# 最多当个对比模型
## 认知科学导论课程实验

### 数据集
采用互联网上公开的中文多领域虚假新闻检测数据集Weibo21作为本实验的数据来源，该数据集包含来自于政治、军事、社会生活等9个领域的真假新闻数据共9123条。
### 代码
#### 第三方库安装
参考 `requirements.txt`文件安装。包括但不限于会用到的库，如果冲突需要降低至3.6版本

可以运行以下语句一键安装所需第三方库。

 `pip install -r requirements.txt` 

#### 预训练模型
本程序的预训练模型存放在以下路径中，使用时无需再次训练可以直接调用`pretrained_model/chinese_roberta_wwm_base_ext_pytorch`.

#### 如何运行
运行时在终端执行以下语句：
```python
python main.py --model_name mdfend --batchsize 32 --lr 0.0007
```
