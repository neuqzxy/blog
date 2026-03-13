---
title: 搜广推入门
date: 2026-03-11 07:18:35
tags:
  - Math
  - ML
  - AI
categories:
  - ML
---
**CTR** (Click-Through Rate，点击率) 和 **CVR** (Conversion Rate，转化率) 预估是计算广告、推荐系统和电商平台的核心技术，用于预测用户行为概率，驱动商业化决策与用户体验优化。

| 指标  | 定义                         | 公式                          | 业务含义                       |
| ----- | ---------------------------- | ----------------------------- | ------------------------------ |
| CTR   | 用户看到内容后点击的概率     | CTR = 点击次数 / 展示次数    | 衡量内容吸引力与相关性         |
| CVR   | 用户点击后完成转化的概率     | CVR = 转化次数 / 点击次数    | 衡量购买意愿与转化路径质量     |
| CTCVR | 展示到转化的全链路概率       | CTCVR = CTR × CVR            | 直接反映商业价值               |

# 传统模型阶段 (2012 年前)
## 逻辑回归 LR（Logistic Regression）
我们有特征 $x^(i)_j$，建立

$$
\hat{y} = \sigma(W x + b)
$$

使用二分类交叉熵作为损失函数
$$
L = - \frac{1}{N} \sum_{i=1}{N} y_i log(\hat{y}) + (1 - y_i) log(1 - \hat{y_i})
$$

## FM（Factorization Machine）因子分解机
在真实场景中，很多特征之间是有相关性的（比如 “男性 + 篮球鞋”），FM 在传统逻辑回归的基础上添加了**二阶交叉特征**的因子分解项。每个特征 $i$ 对应一个隐向量 $\mathbf{v}_i \in \mathbb{R}^k$（$k \ll n$）,通常 $k$ 在(8 - 64)维即可，而特征维度 $n$ 可能是(百万 ~ 亿)级别，用内积表示特征间交互强度，从而在稀疏数据下也能估计未共现的特征组合：

$$
\hat{y} = w_0 + \sum_{i=1}^n w_i x_i + \sum_{i=1}^{n} \sum_{j=i+1}^{n} \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j
$$

其中二阶项的内积展开为：
$$
\langle \mathbf{v}_i, \mathbf{v}_j \rangle = \sum_{f=1}^{k} v_{i,f} v_{j,f}
$$

**特点**：参数量从显式二阶的 $O(n^2)$ 降为 $O(n \cdot k)$，且可线性时间计算（化简后为 $O(n \cdot k)$）。

## FFM（Field-aware Factorization Machine）域感知因子机
FM 中每个特征只有一个隐向量，对所有其它特征一视同仁。FFM 引入**域（Field）**：先按业务把特征划分到不同域（如用户域、物品域、上下文域），每个特征在**与不同域交互时使用不同的隐向量**。即特征 $i$ 有 $|\mathcal{F}|$ 个隐向量 $\mathbf{v}_{i,f}$（$f$ 为域编号），与特征 $j$ 交互时用 $\mathbf{v}_{i, f_j}$ 和 $\mathbf{v}_{j, f_i}$（各自在对方所属域下的向量）：

$$
\hat{y} = w_0 + \sum_{i=1}^n w_i x_i + \sum_{i=1}^{n} \sum_{j=i+1}^{n} \langle \mathbf{v}_{i, f_j}, \mathbf{v}_{j, f_i} \rangle x_i x_j
$$

**特点**：同一特征在不同域下有不同的表示，表达能力更强，适合多域异构特征；参数量约为 $O(n \cdot |\mathcal{F}| \cdot k)$，训练和推理成本更高。
