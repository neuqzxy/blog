---
title: 强化学习算法
date: 2026-03-08 13:18:22
tags:
  - Math
  - ML
  - AI
  - LLM
categories:
  - ML
---
## PPO
> OpenAI的GPT-3.5、GPT-4，二者在RLHF阶段均主要采用PPO算法完成偏好对齐，是PPO在LLM领域应用的标杆，也是目前闭源大模型中对齐效果最稳定的案例。

### GAE

PPO基于Actor-Critic基础框架进行优化而来，同样包含`Actor`、`Critic`模型，传统的Actor-Critic会通过TD误差来近似奖励函数：

$$
A \approx \delta_t =  r_{t+1} + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)
$$

PPO使用**GAE（广义优势估计）**，融合**未来多步 TD 误差**的加权和：

$$
A_t^{GAE_{\lambda}} = \sum_{l=0}^{\inf} (\gamma λ)^l \delta_{t+l}
$$

- $λ=0$：退化为单步 TD 误差（低方差、高偏差）
- $λ=1$：退化为蒙特卡洛估计（低偏差、高方差）

### 裁剪 & 重要性采样（ $L^{CLIP}$ ）

传统的策略梯度 (如 REINFORCE) 用策略 $\pi_θ$ 采样 $\rightarrow$ 计算梯度 $\rightarrow$ 更新 $θ$。而PPO基于训练速度的考量，希望使用离线数据并进行多次学习，针对这种诉求，传统的策略梯度方法就会非常不稳定。如果一次更新让策略 $\pi_θ$ 发生了剧烈变化，可能会导致模型陷入崩溃，无法恢复。
PPO 的核心逻辑是：
- 添加重要性采样，修正新旧两个分布之间的偏差
- 限制策略更新的幅度，给重要性采样打补丁，避免方差爆炸，当新策略 $\pi_θ$ 偏离旧策略 $\pi_{θ_{old}}$ 太远时，直接忽略

我们定义新旧策略的概率比值为 $r_t(θ)$：
$$
r_t(θ) = \frac{\pi_θ(a_t|s_t)}{\pi_{θ_{old}}(a_t|s_t)}
$$

- 如果 $r_t > 1$，说明在当前状态下，新策略选择该动作的概率比旧策略大。
- 如果 $r_t < 1$，则相反。PPO 的目标函数（Clipped Surrogate Objective）如下：

$$
L^{CLIP}(θ) = \hat{\mathbb{E}}_t \left[ \min(r_t(θ) \hat{A}_t, \text{clip}(r_t(θ), 1-\epsilon, 1+\epsilon) \hat{A}_t) \right]
$$

| 优势 $\hat{A}_t$ | 动作表现 | 策略变化 $r_t$ | PPO 的态度 |
|------------------|----------|----------------|------------|
| >0 (好) | 优于平均 | 很大 (>1+ϵ) | 压制 (Clip) |
| >0 (好) | 优于平均 | 很小 (<1) | 不限制 |
| <0 (差) | 低于平均 | 很大 (>1) | 猛拉 (Unclipped) |
| <0 (差) | 低于平均 | 很小 (<1−ϵ) | 压制 (Clip) |

### 目标函数（The Total Loss）

PPO的总损失函数是将三个目标合成一个 **Loss 最小化**：

$$
L_t(θ, \phi) = \hat{\mathbb{E}}_t \left[ -L_t^{CLIP}(θ) + c_1 L_t^{VF}(\phi) - c_2 H(s_t) \right]
$$

- **策略损失 (Policy Loss)**：即 $-L_t^{CLIP}(θ)$。通过剪裁机制，确保策略更新平滑。
- **价值损失 (Value Function Loss)**：即 $L_t^{VF}(\phi)$。通常是均方误差（MSE），用来训练 Critic 网络，让 $V_\phi(s)$ 越来越接近实际观测到的奖励回报。

$$
L^{VF} = (V_\phi(s_t) - R_t)^2
$$

- **熵奖励 (Entropy Bonus)**：即 $H$。这是为了鼓励探索。如果策略输出的概率分布太集中（比如只选某一个动作），熵就小。加上这一项能防止模型过早陷入局部最优解。

## DPO (Direct Preference Optimization)
> 它把强化学习问题变成了一个二元交叉熵 (Binary Cross Entropy) 问题，它让模型在看到问题 $x$ 时，最大化“好回答” $y_w$ 出现的概率，同时最小化“烂回答” $y_l$ 出现的概率。

在 RLHF 中，我们的目标是最大化奖励并保留 KL 惩罚：

$$
\max_{\pi} \mathbb{E}_{s \sim D, a \sim \pi} [r(s, a)] - \beta \mathbb{D}_{KL} [\pi(a|s) || \pi_{ref}(a|s)]
$$

### 公式展开
我们展开这个式子，并推导

$$
\begin{aligned}
J(\pi) &= \max_{\pi} \sum_a \pi(a|s) r(s, a) - \beta \sum_a \pi(a|s) \log \frac{\pi(a|s)}{\pi_{ref}(a|s)}\\\\
&= \max_{\pi} \beta \sum_a \pi(a|s) \left[ \frac{1}{\beta} r(s, a) - \log \frac{\pi(a|s)}{\pi_{ref}(a|s)} \right]\\\\
&= \max_{\pi} \beta \sum_a \pi(a|s) \left[ \log e^{\left( \frac{1}{\beta} r(s, a) \right)} - \log \frac{\pi(a|s)}{\pi_{ref}(a|s)} \right]\\\\
&= \max_{\pi} \beta \sum_a \pi(a|s) \log \left( \frac{\pi_{ref}(a|s) e^{\left( \frac{1}{\beta} r(s, a) \right)}}{\pi(a|s)} \right)\\\\
&= \min_{\pi} \beta \sum_a \pi(a|s) \log \left( \frac{\pi(a|s)}{\pi_{ref}(a|s) e^{\left( \frac{1}{\beta} r(s, a) \right)}} \right)
\end{aligned}
$$

我们发现 $\log \left( \frac{\pi(a|s)}{\pi_{ref}(a|s) e^{\left( \frac{1}{\beta} r(s, a) \right)}} \right)$ 的形式非常类似一个KL散度，如果我们能将分母改造成一个概率分布函数 $\pi^{*}$ 那么其最优解就是 $\pi = \pi^{*}$ 了

### 引入配分函数 (Normalization Constant)

为了让分母部分符合一个概率分布的形式，我们定义一个归一化常数（即配分函数）$Z(s)$ 使得其符合概率和为1的性质：

$$
Z(s) = \sum_a \pi_{ref}(a|s) \exp \left( \frac{1}{\beta} r(s, a) \right)
$$

我们可以定义一个新的概率分布 $\pi^*(a|s)$：

$$
\pi^*(a|s) = \frac{1}{Z(s)} \pi_{ref}(a|s) \exp \left( \frac{1}{\beta} r(s, a) \right)
$$

将最新的分母表示 $\pi_{ref}(a|s) \exp \left( \frac{1}{\beta} r(s, a) \right) = Z(s) \pi^*(a|s)$ 带入原式，我们可以得到：

$$
\begin{aligned}
J(\pi) &= \min_{\pi} \beta \sum_a \pi(a|s) \log \left( \frac{\pi(a|s)}{Z(s) \pi^*(a|s)} \right)\\\\
&= \min_{\pi} \beta \sum_a \pi(a|s) \left[ \log \frac{\pi(a|s)}{\pi^*(a|s)} - \log Z(s) \right]\\\\
&= \min_{\pi} -\beta \log Z(s) + \beta \sum_a \pi(a|s) \log \frac{\pi(a|s)}{\pi^*(a|s)}\\\\
&= \min_{\pi} -\beta \log Z(s) + \beta \mathbb{D}_{KL}(\pi(a|s) || \pi^*(a|s))\\\\
\end{aligned}
$$

其中第一项 $\beta \log Z(s)$ 与当前策略 $\pi$ 无关，所以我们只需要让 $\pi(a|s) = \pi^*(a|s)$ 即可，即最优策略 $\pi_r$ 为：

$$
\pi_r(a|s) = \frac{1}{Z(s)} \pi_{ref}(a|s) e^{\frac{1}{\beta} r(s, a)}
$$

依据该公式，可以反算出 $r(s, a)$ 以便于下面BT模型使用：

$$
r(s, a) = \beta \log \frac{\pi(a|s)}{\pi_{ref}(a|s)} + \beta \log Z(s)
$$


虽然我们推导出了最优策略 $\pi_r(a|s)$，但由于 $Z(s)$ 的积分/求和不可算，且 $r(s, a)$ 本身未知，这个公式在物理世界里是“悬在空中”的。
但是BT模型巧妙的解决了这个问题，通过引入BT模型，我们可以：

**1. 利用“相对值”消灭“绝对项” ($Z(s)$)：**

根据 BT 模型，人类的偏好只取决于两个回复的奖励差值：$r(s, a_w) - r(s, a_l)$。当我们把推导出的 $r(s, a) = \beta \log \frac{\pi(a|s)}{\pi_{ref}(a|s)} + \beta \log Z(s)$ 代入差值时，同一个 Prompt 下的 $Z(s)$ 是完全一样的。在减法中，这个不可算的“幽灵项”被神奇地抵消了。

**2. 将“奖励函数”映射为“策略概率”：**

我们不再需要去苦苦寻找 $r(s, a)$ 的显式表达式。通过代换，我们将原本需要拟合奖励模型（Reward Model）的训练，转化成了直接拟合当前模型 $\pi_\theta$ 与参考模型 $\pi_{ref}$ 之间的对数概率比（Log Ratio）。

### Bradley-Terry(BT) 模型与Loss
Bradley-Terry 模型的核心就是给每个对象分配一个正的实力参数 $\alpha_i$​，然后用比值来定义 “i 比 j 强” 的概率：

$$
P(i > j) = \frac{\alpha_i}{\alpha_i + \alpha_j}
$$

我们正好有偏好数据：$(x,y_w​,y_l​)$ 表示 “在输入 $x$ 下，$y_w$​ 比 $y_l$​ 更优”：

$$
\begin{aligned}
p^∗(y_w ​> y_l ​∣ x) &= \frac{r​(x,y_w​)​}{r​(x,y_w​) + r​(x,y_l​)}\\\\
&= \sigma(r​(x,y_w​) - r​(x,y_l​))
\end{aligned}
$$

我们发现，如果将之前算出来的 $r(s, a) = \beta \log \frac{\pi(a|s)}{\pi_{ref}(a|s)} + \beta \log Z(s)$ 带入到 $p^∗(a_w ​> a_l ​∣ s)$ 中，可以直接消去 $\log Z(s)$。而且Loss函数也是现成的，我们只需要最大化 $p^∗(a_w ​> a_l ​∣ s)$ 即可，也就是最大化对数似然（最小化负对数似然）：

$$
\mathcal{L}_{DPO}(\pi_\theta; \pi_{ref}) = -\mathbb{E}_{(x, y_w, y_l) \sim D} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]
$$


## GRPO

在标准 PPO 中，为了计算优势函数 $A(s, a)$，我们需要一个和 Actor 同样大的 Critic 网络 来估算 $V(s)$。多维护一个 Critic 模型意味着：
- 显存翻倍：两个庞然大物并存，显存捉襟见肘。
- 训练变慢：需要额外的计算资源来更新 Critic。
- 价值估计难：在推理任务（如奥数、代码）中，状态 $s$ 到奖励 $r$ 的映射非常复杂，Critic 往往很难估准。

GRPO中优势函数不依赖 $V(s)$，而是计算这组分数在当前组内的标准化分值：
$$
A_i = \frac{r_i - \text{mean}(r_1, \dots, r_G)}{\text{std}(r_1, \dots, r_G)}
$$

GRPO 损失函数就可以直接删掉critic那一项

$$
L_t(θ, \phi) = \hat{\mathbb{E}}_t \left[ -L_t^{CLIP}(θ) - c_2 H(s_t) \right]
$$

## GDPO
DPO 虽然优雅在实际应用中存在两个硬伤：
- **Bradley-Terry (BT) 模型的局限性**：DPO 假设人类偏好严格遵循 $P(a_w \succ a_l) = \sigma(r_w - r_l)$。但在现实中，好的回答（$a_w$）可能只比坏的（$a_l$）好一点点，也可能好非常多。DPO 强制用一个固定的 Sigmoid 去拟合，会导致模型在 $a_w$ 和 $a_l$ 差距很小时也过度推高前者的概率。
- **忽略了“边界” (The Margin Problem)**：DPO 只看谁更好，不看好多少。这导致模型可能会为了微小的奖励提升，不惜大幅偏离参考模型（Reference Model），产生严重的 KL 散度漂移。

GDPO 核心是对 DPO 的损失函数进行了泛化。最常见的 GDPO 变体（如带有 Margin 的版本）如下：

$$
\mathcal{L}_{GDPO}(\theta) = -\mathbb{E}_{(s, a_w, a_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(a_w|s)}{\pi_{ref}(a_w|s)} - \beta \log \frac{\pi_\theta(a_l|s)}{\pi_{ref}(a_l|s)} - \delta \right) \right]
$$

这里引入的关键变量 $\delta$ (Margin)：
- $\delta = 0$：这就是标准的 DPO。
- $\delta > 0$：这代表一个“安全阈值”。模型不仅要让 $a_w$ 的隐含奖励高于 $a_l$，而且必须高出至少 $\delta$ 这么多，Loss 才会显著下降。
