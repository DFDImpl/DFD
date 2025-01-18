from os import path

import numpy as np
import torch
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch import nn

X_tensor,y_tensor=#import ur data here
model= #load ur pretrained model here 



np.random.seed(45)



# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 将数据移动到 GPU
X_tensor = X_tensor.to(device)

# 提取特征
with torch.no_grad():  # 不需要计算梯度
    features = model(X_tensor)

tsne = TSNE(n_components=2, random_state=38,perplexity=50)
X = tsne.fit_transform(features.cpu().numpy())  # 转换为 NumPy 数组

# tsne = TSNE(n_components=2, random_state=38,perplexity=50)
# X = tsne.fit_transform(X)  # 转换为 NumPy 数组

# 可视化结果
plt.figure(figsize=(14, 12))
scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='tab10', alpha=0.6)
handles, labels = scatter.legend_elements()  # 自动获取图例
#plt.legend(handles, [0,1,2,3,4,5,6],ncol=7,fontsize=20, loc='upper center')  # 手动指定每个类别的名称



plt.xticks([])
plt.yticks([])


plt.savefig('tsne1.png', dpi=600,bbox_inches='tight')
#plt.show()
