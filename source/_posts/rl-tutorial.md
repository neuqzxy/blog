---
title: WIP 强化学习
date: 2026-03-04 23:22:36
tags:
  - Math
  - ML
  - AI
  - LLM
categories:
  - ML
---

强化学习（Reinforcement Learning，RL）是机器学习的一个重要分支，通过让智能体在与环境的持续交互中学习最优决策策略。它擅长解决监督学习难以处理的问题——例如游戏、棋类等场景无法穷举所有状态打标签，以及 LLM 安全对齐中"偏好"这类无法用固定标签衡量的目标。

## RL vs 监督学习

两者的本质区别在于**反馈信号的性质**：

| | 监督学习 | 强化学习 |
|---|---|---|
| 反馈信号 | 精确标签（正确答案） | 稀疏奖励（好/坏，甚至延迟） |
| 数据来源 | 人工标注的静态数据集 | 与环境实时交互产生 |
| 决策性质 | 单步预测（输入→输出） | 序列决策（当前动作影响未来状态） |
| 目标 | 拟合输入输出映射 | 最大化长期累积奖励 |

监督学习的前提是**能给出"正确答案"**。当问题满足以下任一条件时，监督学习就会遇到瓶颈，此时 RL 更适合：

- **状态空间过大，无法穷举标注**：围棋有 $10^{170}$ 种可能局面，Atari 游戏的像素组合近乎无穷，根本无法为每个状态打上"最优动作"标签
- **正确答案本身不存在或难以定义**：LLM 对话中什么回答更好，只能通过人类偏好打分（奖励）来衡量，而非唯一正确答案——RLHF 正是利用这一点做安全对齐
- **决策存在长期依赖**：下棋时某步"看似吃亏"的让子可能20步后致胜，监督学习只能拟合单步，无法对延迟奖励建模
- **环境会响应动作**：机器人控制、自动驾驶中，智能体的动作会改变环境，下一个状态由当前动作决定，数据分布是非静态的

当然，RL 也有明显代价：样本效率低（需要大量交互），训练不稳定，奖励设计（reward shaping）困难。因此实践中往往先用监督学习做预训练，再用 RL 微调，例如 ChatGPT 的 SFT → RLHF 流程。

## 基本框架：马尔可夫决策过程（MDP）

强化学习的标准数学框架是**马尔可夫决策过程（Markov Decision Process，MDP）**，形式化定义为一个五元组：

$$
\mathcal{M} = (\mathcal{S},\ \mathcal{A},\ P,\ R,\ \gamma)
$$

- $\mathcal{S}$：状态空间（所有可能状态的集合）
- $\mathcal{A}$：动作空间（所有可能动作的集合）
- $P(s' \mid s, a)$：状态转移概率，执行动作 $a$ 后从状态 $s$ 转移到 $s'$ 的概率
- $R(s, a, s')$：奖励函数，通常简写为 $r_t$，表示该转移获得的即时奖励
- $\gamma \in [0, 1)$：折扣因子，控制未来奖励的权重衰减

"马尔可夫"性质意味着状态转移只依赖当前状态，与历史无关：

$$
P(s_{t+1} \mid s_t, a_t, s_{t-1}, a_{t-1}, \ldots) = P(s_{t+1} \mid s_t, a_t)
$$

### 轨迹与累积回报

智能体与环境交互产生一条**轨迹（Trajectory）**：

$$
\tau = (s_0,\ a_0,\ r_0,\ s_1,\ a_1,\ r_1,\ \ldots)
$$

从时刻 $t$ 起的**折扣累积回报（Return）** 定义为：

$$
G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k} = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots
$$

$\gamma$ 的作用是双重的：数学上保证无限序列收敛（$\gamma < 1$）；语义上体现"近期奖励比远期更确定"的偏好。

### 策略

**策略（Policy）** $\pi$ 是智能体的决策函数，分为两类：

- **随机策略**：$\pi(a \mid s) = P(a_t = a \mid s_t = s)$，输出动作的概率分布
- **确定性策略**：$a = \mu(s)$，直接输出动作

### 价值函数

**动作价值函数**（Q 函数）进一步条件化到具体动作：

$$Q^{\pi}(s, a) = \mathbb{E}_{\pi}\left[G_t \mid s_t = s,\ a_t = a\right]$$

**注意**：由于 $G_t$​ 是针对**某一条具体轨迹** $\tau$ 计算的累积回报，而轨迹会因环境转移、策略采样而具有随机性，因此价值函数中引入数学期望，本质是对从当前状态（或状态 - 动作对）出发的所有可能轨迹求平均，得到期望累积回报。

在策略 $\pi$ 下，**状态价值函数** 定义为从状态 $s$ 出发的期望累积回报：

$$
V^{\pi}(s) = \sum_{a} \pi(a \mid s)\, Q^{\pi}(s, a)
$$

### Bellman 方程

价值函数满足递推结构，即 **Bellman 期望方程**：

$$
V^{\pi}(s) = \sum_{a} \pi(a \mid s) \sum_{s'} P(s' \mid s, a)\left[R(s,a,s') + \gamma V^{\pi}(s')\right]
$$

$$
Q^{\pi}(s, a) = \sum_{s'} P(s' \mid s, a)\left[R(s,a,s') + \gamma \sum_{a'} \pi(a' \mid s') Q^{\pi}(s', a')\right]
$$

这一递推关系是大多数 RL 算法（动态规划、TD 学习等）的理论基础。

### 优化目标

RL 的终极目标是找到**最优策略** $\pi^*$，使期望累积回报最大, 对应的最优价值函数满足 **Bellman 最优方程**：

$$
V^{\*}(s) = \max_{a} \sum_{s'} P(s' \mid s, a)\left[R(s,a,s') + \gamma V^{*}(s')\right]
$$

$$
Q^{\*}(s, a) = \sum_{s'} P(s' \mid s, a)\left[R(s,a,s') + \gamma \max_{a'} Q^{*}(s', a')\right]
$$

最优策略可直接从 $Q^{\*}$ 中贪心提取：$\pi^{\*}(s) = \arg \max_{a}\, Q^{\*}(s, a)$

---

根据求解路径的不同，RL 算法分为三大流派：**Policy-Based** 直接对 $\pi$ 参数化并梯度上升；**Value-Based** 先估计 $Q^*$ 再贪心得到策略；**Model-Based** 先学习转移模型 $P$ 和 $R$，再通过规划求解。

## 三大流派

### Value-Based（基于价值）
> Value-Based 算法（以 Q-Learning/DQN/Rainbow 为代表的 Value-Based 方法属于 Off-policy），天然支持用离线数据训练；但Value-Based 家族同时包含 Sarsa 等 On-policy 算法。

**核心思路**：不直接学习策略，而是学习最优动作价值函数 $Q^{\*}(s,a)$，策略由 $Q^{\*}$ 贪心导出（注意此时相当于执行 $\pi^{\*}$ 策略，该策略下动作不再随机，而是取最优路径，Q从期望退化成确定值）：

$$
\pi^*(s) = \arg\max_{a} Q^{\*}(s, a)
$$

**求解方式**：对 Bellman 最优方程做迭代逼近。以最经典的 **Q-Learning** 为例，每次交互后对 $Q$ 表做 TD 更新：

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \underbrace{\left[r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t)\right]}_{\text{TD 误差 } \delta_t}
$$

其中 $\alpha$ 为学习率，方括号内称为 **TD 误差**——当前估计与 Bellman 目标之间的差距，驱动 $Q$ 值收敛到 $Q^*$。

> 通用定义补充：TD 误差是时序差分目标（即时奖励与下一状态 / 动作价值估计的折扣和）与当前状态 / 动作价值估计之间的差值，用于衡量价值函数估计的即时预测误差，是时序差分学习的核心更新信号。

当状态空间连续或规模过大无法用表格存储时，用神经网络 $Q_θ(s,a)$ 拟合，即 **DQN**，损失函数为均方TD误差：

$$
\mathcal{L}(θ) = \mathbb{E}\left[\left(r + \gamma \max_{a'} Q_{θ^-}(s', a') - Q_θ(s, a)\right)^2\right]
$$

其中 $θ^-$ 为周期性同步的目标网络参数，用于稳定训练。

**特点**：
- 只适用于**离散动作空间**（$\text{argmax}$ 在连续动作上不可行，因为**连续动作空间里没法穷举**）；
- 样本效率较高（可用**经验回放**）。

##### MC 误差（蒙特卡洛误差）
MC 必须等到一整局结束，拿到**真实完整回报** $G_t$​：

$$
G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1} = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \cdots
$$

**MC 误差**：

$$
\delta_t^{MC} = G_t - V(s_t)
$$

##### TD 误差（时序差分误差）
TD 不等结束，每走一步就更新一次：

$$
\delta_t^{TD} = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)
$$

---

### Policy-Based（基于策略）

**核心思路**：将策略参数化为 $\pi_{θ}(a \mid s)$，直接对期望回报做梯度上升：

$$
J(θ) = \mathbb{E}_{\tau \sim \pi_θ}\left[G(\tau)\right] = \int p_θ(\tau) G(\tau) d\tau
$$

$$
p_θ(\tau) = \pi_{θ}(a \mid s)
$$

**Policy Gradient 定理** 给出梯度的解析形式：

$$
\nabla_θ J(θ) = \mathbb{E}_{\pi_θ}\left[\nabla_θ \log \pi_θ(a_t \mid s_t) \cdot G_t\right]
$$

直觉上：若轨迹回报 $G_t$ 高，就增大该动作的概率（$\log \pi_θ$ 梯度方向）；反之则压低。这是 **REINFORCE** 算法的核心。

实践中用**基线（Baseline）** 减少方差，常取状态价值函数 $b(s) = V(s)$：

$$
\nabla_θ J(θ) = \mathbb{E}_{\pi_θ}\left[\nabla_θ \log \pi_θ(a_t \mid s_t) \cdot \left(G_t - b(s_t)\right)\right]
$$

**特点**：天然支持**连续动作空间**；策略可直接表达随机性；但方差大、样本效率低。代表算法：REINFORCE、PPO、TRPO。

---

### Actor-Critic（演员-评论家）

Value-Based 和 Policy-Based 的结合体，也是目前最主流的框架。

- **Actor**：参数化策略 $\pi_θ$，负责选动作（Policy-Based）
- **Critic**：参数化价值函数 $V_\phi$ 或 $Q_\phi$，负责评估动作好坏（Value-Based）

引入**优势函数（Advantage Function）** 替代原始回报，进一步降低方差：

$$
A^{\pi}(s, a) = Q^{\pi}(s, a) - V^{\pi}(s)
$$

$A > 0$ 表示该动作比平均水平好，$A < 0$ 则相反。Actor 的梯度更新变为：

$$
\nabla_θ J(θ) = \mathbb{E}_{\pi_θ}\left[\nabla_θ \log \pi_θ(a_t \mid s_t) \cdot A^{\pi}(s_t, a_t)\right]
$$

实践中 $A$ 用 **TD 误差**近似：$A \approx \delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$，**Critic** 同步用 **TD 误差**更新 $V_\phi$。

**PPO（Proximal Policy Optimization）** 是 **Actor-Critic** 的主流变体，通过 Clip 机制限制每次策略更新的幅度，兼顾稳定性与效率：

$$\mathcal{L}^{\text{CLIP}}(θ) = \mathbb{E}\left[\min\left(r_t(θ)\, A_t,\ \text{clip}(r_t(θ), 1-\epsilon, 1+\epsilon)\, A_t\right)\right]$$

其中 $r_t(θ) = \dfrac{\pi_θ(a_t \mid s_t)}{\pi_{θ_{\text{old}}}(a_t \mid s_t)}$ 为新旧策略的概率比。

---

### Model-Based（基于模型）

**核心思路**：上述两类方法都是 **Model-Free**——直接从交互数据中学策略/价值，不对环境本身建模。Model-Based 则额外用数据拟合一个**环境模型** $\hat{P}$、$\hat{R}$：

$$\hat{P}(s' \mid s, a) \approx P(s' \mid s, a), \qquad \hat{R}(s, a) \approx R(s, a)$$

有了模型，就可以在"想象"的轨迹上做**规划（Planning）**，无需每步都与真实环境交互，大幅提升样本效率。

#### 模型学习

用真实交互收集到的数据 $\{(s_t, a_t, r_t, s_{t+1})\}$，以监督学习的方式训练模型：

$$\mathcal{L}_P = \mathbb{E}\left[\|\hat{P}(s, a) - s'\|^2\right], \qquad \mathcal{L}_R = \mathbb{E}\left[(\hat{R}(s, a) - r)^2\right]$$

模型可以是表格（小状态空间）、线性模型，也可以是神经网络（高维状态空间）。

#### 规划方式

拥有模型后，有两类主流规划路径：

**① 虚拟展开（Dyna 框架）**：用学到的模型生成虚拟样本，与真实样本一起送入标准的 Model-Free 更新器（如 Q-Learning）。核心思想是：

> 真实交互 → 更新模型 $\hat{P}, \hat{R}$；从模型中采样虚拟转移 $(\hat{s}, \hat{a}, \hat{r}, \hat{s}')$ → 更新 $Q$

每次真实交互后执行 $k$ 步虚拟更新，等效将样本利用率提升 $k$ 倍。

**② 树搜索规划（AlphaZero / MuZero）**：以当前状态为根节点，在模型构建的搜索树上运行 **MCTS（蒙特卡洛树搜索）**，对每个候选动作进行多步前向推演，选择期望价值最高的动作。

- **AlphaZero**：模型即真实规则（棋类游戏），价值网络和策略网络辅助 MCTS 剪枝
- **MuZero**：模型也用神经网络学习，无需已知规则，在 Atari 和棋类上均达到超人水平

#### 核心挑战：模型误差累积

规划时每一步都在 $\hat{P}$ 上展开，误差会随步数指数级累积（**compounding error**）：

$$\text{第}\ k\ \text{步误差} \sim O(\epsilon^k)$$

其中 $\epsilon$ 是单步模型误差。因此规划步数不能太长，否则虚拟轨迹与真实轨迹严重偏离。这也是 Model-Based 方法在复杂、高维环境（如像素输入的游戏）中落地困难的根本原因。

**特点**：样本效率高；但模型训练本身需要数据，且误差累积限制规划深度。代表算法：Dyna-Q、MBPO、AlphaZero、MuZero。

---

| | Value-Based | Policy-Based | Model-Based |
|---|---|---|---|
| 学习对象 | $Q^*(s,a)$ | $\pi_θ(a\|s)$ | $\hat{P}, \hat{R}$ |
| 动作空间 | 离散为主 | 离散 + 连续 | 均可 |
| 样本效率 | 中（可回放） | 低 | 高 |
| 代表算法 | DQN, Rainbow | PPO, TRPO | Dyna, MuZero |
