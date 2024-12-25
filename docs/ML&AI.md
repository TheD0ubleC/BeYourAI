# 机器学习与AI的实践

*实践将会是你最好的练习*

---

## **1. 什么是机器学习与 AI？**

### **1.1 人工智能（AI）**
人工智能是让计算机模仿人类的学习、推理和决策能力的技术。

典型应用：
- 图像识别（如人脸检测）。
- 语言处理（如聊天机器人）。
- 自动驾驶。

### **1.2 机器学习（ML）**
机器学习是 AI 的一个子领域，通过算法从数据中学习并做出预测。

典型任务：
- 分类：如垃圾邮件识别。
- 回归：如房价预测。
- 聚类：如客户分组。

机器学习的核心思想是利用数据驱动的方法解决问题，从而提高预测的准确性和模型的效率。

---

## **2. 第一个机器学习项目：鸢尾花分类**

### **2.1 数据加载与探索**
**需要安装的依赖：**
```bash
pip install pandas scikit-learn
```

使用 Scikit-learn 提供的鸢尾花数据集：

```python
from sklearn.datasets import load_iris
import pandas as pd

# 加载数据
iris = load_iris()
# 转换为 DataFrame
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target

# 查看数据
print(iris_df.head())
print(iris_df.info())
print(iris_df.describe())
```

**作用**：
- 加载和了解数据集的基本结构。
- 使用 `head` 查看前几行数据，`info` 显示数据类型和缺失情况，`describe` 生成统计摘要。

**运行输出**：
- 数据表的前几行。
- 数据类型和列信息。
- 各列的统计信息，如均值、标准差。

---

### **2.2 数据可视化**
**需要安装的依赖：**
```bash
pip install matplotlib seaborn
```

分析特征之间的关系，观察数据的分布：

```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import pandas as pd

# 加载数据并转换为 DataFrame
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target

# 绘制散点图矩阵
sns.pairplot(iris_df, hue='target', diag_kind='hist')
plt.show()

# 绘制特征之间的关系图
sns.heatmap(pd.DataFrame(data=iris.data, columns=iris.feature_names).corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation')
plt.show()
```

**作用**：
- 通过散点图矩阵了解特征与类别之间的关系。
- 使用热力图查看特征之间的相关性。

**运行输出**：
- 一组彩色散点图，显示不同类别的数据分布。
- 一个热力图，标注各特征之间的相关系数。

通过这些可视化，我们可以更直观地观察数据特性，例如哪些特征对分类有更高的区分度。

---

### **2.3 构建分类模型**
**需要安装的依赖：**
```bash
pip install scikit-learn
```

使用 K 最近邻（KNN）分类算法：

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 加载数据
iris = load_iris()
X = iris.data
y = iris.target

# 数据分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 模型训练
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# 模型预测
y_pred = knn.predict(X_test)

# 输出评估结果
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
```

**作用**：
- 数据分割：将数据分为训练集和测试集。
- 模型训练：使用 KNN 算法拟合训练数据。
- 模型预测：对测试集进行预测并评估性能。

**运行输出**：
- 混淆矩阵：显示正确分类和错误分类的数量。
- 分类报告：包括精确率、召回率和 F1 分数。

通过这些评估指标，我们能够了解模型在不同类别上的表现，从而找到优化的方向。

---

### **2.4 模型优化**
**需要安装的依赖：**
```bash
pip install scikit-learn
```

通过交叉验证选择最佳参数：

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# 加载数据
iris = load_iris()
X = iris.data
y = iris.target

# 数据分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义参数网格
param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance']
}

# 网格搜索
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

# 输出最佳参数
print(f"Best Parameters: {grid_search.best_params_}")

# 使用最佳模型预测
y_best_pred = grid_search.predict(X_test)
print("Optimized Model Classification Report:")
print(classification_report(y_test, y_best_pred))
```

**作用**：
- 使用网格搜索找到 KNN 的最佳超参数组合。
- 提升模型性能，确保分类效果最优。

**运行输出**：
- 最佳参数设置，如 `n_neighbors` 和 `weights`。
- 优化后的分类报告。

---

## **3. 实际项目实践案例**

### **3.1 房价预测**
**需要安装的依赖：**
```bash
pip install scikit-learn
```

利用波士顿房价数据集进行回归分析：

```python
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 加载数据
boston = fetch_openml(name="boston", as_frame=True)
X = boston.data.to_numpy()
y = boston.target.to_numpy()

# 数据分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 模型训练
model = LinearRegression()
model.fit(X_train, y_train)

# 模型预测
y_pred = model.predict(X_test)

# 输出评估结果
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
```

**作用**：
- 学习回归模型，预测房价等连续值。
- 评估模型的均方误差（MSE）。

**运行输出**：
- 均方误差，衡量模型预测与实际值的差异。

---

### **3.2 情感分析**
**需要安装的依赖：**
```bash
pip install scikit-learn
```

基于文本数据对评论的情感进行分类：

*英文版本*

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

# 数据集
texts = [
    "I love this product!", "This is the worst thing I bought.",
    "Absolutely fantastic!", "Not good at all.",
    "I am extremely satisfied!", "Terrible experience, would not recommend.",
    "Highly recommended, great value!", "Awful product, it broke after one use.",
    "Very pleased with this purchase.", "Horrible quality, very disappointed.",
    "Exceptional quality, exceeded expectations!", "Waste of money, do not buy.",
    "Amazing performance and quality!", "Never buying this again, waste of money.",
    "The best I've used in years!", "Completely useless, stopped working in a day."
]
labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]

# 再加点正面
texts += [
    "This product exceeded my expectations!",
    "Absolutely love it, highly recommend.",
    "Amazing quality, will definitely buy again.",
    "Very satisfied, great purchase.",
    "Fantastic product, worth every penny!"
]
labels += [1, 1, 1, 1, 1]

# 再加点负面
texts += [
    "This is a terrible product, very disappointed.",
    "Not worth the money, complete waste.",
    "I hate this item, it broke after one use.",
    "Awful experience, I will never buy this again."
]
labels += [0, 0, 0, 0]


# 文本向量化
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=100, min_df=2, max_df=0.9)
X = vectorizer.fit_transform(texts)

# 数据分割
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42, stratify=labels
)

# 逻辑回归模型（调整类别权重）
model = LogisticRegression(max_iter=200, class_weight='balanced')
model.fit(X_train, y_train)

# 模型预测
y_pred = model.predict(X_test)

# 输出评估结果
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# 交叉验证评分
cross_val_scores = cross_val_score(model, X, labels, cv=5)
print(f"Cross-Validation Accuracy: {cross_val_scores.mean() * 100:.2f}%")

```

**作用**：
- 使用朴素贝叶斯对文本分类，解决情感分析问题。
- 转换文本为数值特征以供机器学习使用。

**运行输出**：
- 模型准确率，显示分类效果。

---

*中文版本*
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import jieba

# 中文数据集
texts = [
    "我很喜欢这个产品！", "这是我买过最糟糕的东西。",
    "简直太棒了！", "一点都不好。",
    "我非常满意！", "糟糕的体验，不推荐。",
    "强烈推荐，性价比很高！", "垃圾产品，用了一次就坏了。",
    "对这次购买非常满意。", "质量非常差，非常失望。",
    "卓越的品质，超出预期！", "浪费钱，千万不要买。",
    "性能和质量都很棒！", "再也不会买这个了，浪费钱。",
    "这是我多年用过的最好的！", "完全没用，一天就坏了。",
    "这个产品完全超出了我的预期，非常满意！", "真心喜欢，强烈推荐给大家。",
    "品质优秀，下次还会购买。", "购买体验很好，很值得推荐。",
    "非常棒的产品，每一分钱都很值！", "非常失望，这真的是个垃圾产品。",
    "完全不值这个价钱，浪费我的时间。", "我很讨厌这个东西，用了一次就坏了。",
    "糟糕透了，再也不会买这种东西了。"
]
labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]

# 增加更多的正面评论
texts += [
    "这个产品真的很不错，值得购买！", "非常满意的一次购物体验。",
    "质量很好，使用起来很方便。", "物超所值，下次还会再来。",
    "非常喜欢这个产品，推荐给大家。"
]
labels += [1, 1, 1, 1, 1]

# 增加更多的负面评论
texts += [
    "非常糟糕的产品，完全不推荐。", "用了几次就坏了，质量太差。",
    "非常失望，完全不值这个价钱。", "体验很差，不会再买了。",
    "这个产品真的是垃圾，浪费钱。"
]
labels += [0, 0, 0, 0, 0]

# 中文分词
texts = [" ".join(jieba.lcut(text)) for text in texts]

# 文本向量化
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=100, min_df=2, max_df=0.9)
X = vectorizer.fit_transform(texts)

# 数据分割
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42, stratify=labels
)

# 逻辑回归模型（调整类别权重）
model = LogisticRegression(max_iter=200, class_weight='balanced')
model.fit(X_train, y_train)

# 模型预测
y_pred = model.predict(X_test)

# 输出评估结果
accuracy = accuracy_score(y_test, y_pred)
print(f"准确率: {accuracy * 100:.2f}%")
print("分类报告:")
print(classification_report(y_test, y_pred, zero_division=0, target_names=["负面", "正面"]))

# 交叉验证评分
cross_val_scores = cross_val_score(model, X, labels, cv=5)
print(f"交叉验证准确率: {cross_val_scores.mean() * 100:.2f}%")

```



---

### **3.3 图像分类：手写数字识别**
**需要安装的依赖：**
```bash
pip install scikit-learn
```

利用 MNIST 数据集进行简单的图像分类：

```python
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 加载数据
digits = load_digits()
X = digits.data
y = digits.target

# 数据分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 模型训练
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 模型预测
y_pred = model.predict(X_test)

# 输出评估结果
print("Classification Report:")
print(classification_report(y_test, y_pred))
```

**作用**：
- 学习分类模型，应用于图像识别。
- 利用随机森林提高分类性能。

**运行输出**：
- 分类报告，显示每类数字的精确率、召回率等指标。

---

## **4. 参考资源**

- [Scikit-learn 官方文档](https://scikit-learn.org/stable/)
- [Kaggle 数据科学社区](https://www.kaggle.com/)
- [Google 的机器学习课程](https://developers.google.com/machine-learning)

---

*机器学习的实践是从理解数据到构建模型的全过程，通过不断练习与优化，你将逐步掌握核心技能。*

