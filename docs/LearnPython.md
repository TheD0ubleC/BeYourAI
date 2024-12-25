# Python 入门与语法讲解

---

## **1. 基础语法** 🐍

## **1.1 变量与数据类型**  
- 什么是变量？  
- 基本数据类型：`int`, `float`, `str`, `bool`, `list`, `dict` 等  
- **小练习**：创建变量并打印它们的值。

> ⭐️如果您使用的是网页文档，那么您的右下角会有一个"打开WebPy"按钮。那是我们帮您准备的快速练习的*网页Python执行器*！你可以粘贴示例代码进去测试&练习！不过注意它无法运行带有复杂库的代码。

示例：  
```python
name = "BeYourAI"
age = 18
is_active = True
print(f"Name: {name}, Age: {age}, Active: {is_active}")
```
---
## **1.2 数据类型详解**
#### Python 提供了丰富的数据类型，下面是几种常见类型的示例：

 - 1. 整数（```int```）
```python
age = 25
year = 2024
print(type(age))  # 输出：<class 'int'>
```


 - 2. 浮点数（```float```）
```python
price = 1145.14
pi = 3.14159
print(type(price))  # 输出：<class 'float'>
```


 - 3. 字符串（```str```）
```python
message = "Hello, BeYourAI!"
name = 'Python'
print(f"Message: {message}, Name: {name}")
```


 - 4. 布尔值（```bool```）
```python
is_active = True
is_finished = False
print(is_active, is_finished)  # 输出：True False
```


 - 5. 列表（```list```）

 列表是一种 有序、可变 的数据集合。
 ```python
 numbers = [1, 2, 3, 4, 5]
 names = ["Alice", "Bob", "Charlie"]
 print(numbers[0])  # 输出第一个元素：1
 names.append("David")  # 添加新元素
 print(names)  # 输出：['Alice', 'Bob', 'Charlie', 'David']
```


 - 6. 字典（```dict```）
```python
person = {
    "name": "BeYourAI",
    "age": 18,
    "language": "Python"
}
print(person["name"])  # 输出：BeYourAI
person["job"] = "AI Developer"  # 添加新键值对
print(person)
```
---
## **1.3 小练习：综合应用**
### 任务
创建一个包含个人信息的字典，包括 ```name```、```age```、```is_active``` 和 ```skills```（一个列表），然后输出所有信息。
> **💡 小提示**：Python 中的索引是从 0 开始的，`numbers[0]` 表示列表的第一个元素。

**参考**
```python
person = {
    "name": "BeYourAI",
    "age": 18,
    "is_active": True,
    "skills": ["Python", "Machine Learning", "Data Analysis"]
}

print(f"Name: {person['name']}")
print(f"Age: {person['age']}")
print(f"Active: {person['is_active']}")
print(f"Skills: {', '.join(person['skills'])}")
```
### 输出结果应为：
```
Name: BeYourAI
Age: 18
Active: True
Skills: Python, Machine Learning, Data Analysis
```