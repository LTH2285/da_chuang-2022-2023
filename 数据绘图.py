import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 数据
classes = [
    "Science",
    "Military",
    "Education",
    "Accidents",
    "Politics",
    "Health",
    "Finance",
    "Entertainment",
    "Society",
    "All",
]
bert_only = [74.7, 77.8, 88.8, 83.1, 86.9, 90.5, 79.0, 85.9, 87.2, 83.8]
bilstm_only = [48.7, 71.6, 70.6, 76.2, 84.7, 83.7, 81.0, 83.0, 60.6, 73.4]
bert_bilstm = [81.9, 77.9, 81.3, 78.8, 81.8, 89.0, 84.6, 86.3, 82.4, 82.7]
bert_bilstm_CRF = [83.0, 93.8, 89.1, 96.0, 88.6, 84.0, 89.5, 90.6, 89.8, 91.3]

models = ["BERT Only", "BiLSTM Only", "BERT + BiLSTM", "BERT + BiLSTM + CRF"]
data = [bert_only, bilstm_only, bert_bilstm, bert_bilstm_CRF]

# 计算整体表现和模型稳定性
avg_performance = [np.mean(model_data) for model_data in data]
std_dev = [np.std(model_data) for model_data in data]


def prepare_data(classes, models, data):
    df = pd.DataFrame(columns=["Category", "Model", "F1 Score"])
    for i, model_name in enumerate(models):
        for j, category in enumerate(classes):
            df = df.append(
                {"Category": category, "Model": model_name, "F1 Score": data[i][j]},
                ignore_index=True,
            )
    return df


df = prepare_data(classes, models, data)


# 创建柱状图
def create_bar_chart(df, x_label, y_label, title, ylabel, ylim=None):
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x=x_label, y=y_label, hue="Model", data=df)
    ax.set_xlabel("", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=16)
    if ylim:
        ax.set_ylim(ylim)
    plt.xticks(rotation=45)
    plt.show()


# 类别表现
create_bar_chart(
    df, "Category", "F1 Score", "F1 Score by Category and Model", "F1 Score"
)

# 整体表现
create_bar_chart(
    df.groupby("Model").mean().reset_index(),
    "Model",
    "F1 Score",
    "Average F1 Score by Model",
    "Average F1 Score",
    ylim=(0, 100),
)

# 模型稳定性
create_bar_chart(
    df.groupby("Model").std().reset_index(),
    "Model",
    "F1 Score",
    "Standard Deviation of F1 Score by Model",
    "Standard Deviation",
    ylim=(0, 20),
)

# 热力图
plt.figure(figsize=(12, 6))
sns.heatmap(
    np.array(data),
    # annot=True,
    annot=False,
    fmt=".1f",
    cmap="coolwarm",
    xticklabels=classes,
    yticklabels=models,
    vmin=50,
    vmax=100,
)
plt.xlabel("Categories", fontsize=14)
plt.ylabel("Models", fontsize=14)
plt.title("F1 Score Heatmap", fontsize=16)
plt.show()
