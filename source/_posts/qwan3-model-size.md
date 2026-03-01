---
title: Qwen3 架构深度拆解：从模型参数推导到推理显存评估全攻略
date: 2026-02-27 22:22:36
tags:
  - Math
  - ML
  - AI
  - LLM
categories:
  - ML
---
## 实际案例（Qwen3）

### Qwen3-0.6B
基本信息
```python
Type: Causal Language Models
Training Stage: Pretraining & Post-training
Number of Parameters: 0.6B
Number of Paramaters (Non-Embedding): 0.44B
Number of Layers: 28
Number of Attention Heads (GQA): 16 for Q and 8 for KV
Context Length: 32,768
```
架构信息
```python
{
  "architectures": [
    "Qwen3ForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 1024,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "max_position_embeddings": 40960,
  "max_window_layers": 28,
  "model_type": "qwen3",
  "num_attention_heads": 16,
  "num_hidden_layers": 28,
  "num_key_value_heads": 8,
  "rms_norm_eps": 1e-06,
  "rope_scaling": null,
  "rope_theta": 1000000,
  "sliding_window": null,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.51.0",
  "use_cache": true,
  "use_sliding_window": false,
  "vocab_size": 151936
}
```

### Qwen3-1.7B

基本信息
```python
Type: Causal Language Models
Training Stage: Pretraining & Post-training
Number of Parameters: 1.7B
Number of Paramaters (Non-Embedding): 1.4B
Number of Layers: 28
Number of Attention Heads (GQA): 16 for Q and 8 for KV
Context Length: 32,768
```
架构信息
```python
{
  "architectures": [
    "Qwen3ForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 2048,
  "initializer_range": 0.02,
  "intermediate_size": 6144,
  "max_position_embeddings": 40960,
  "max_window_layers": 28,
  "model_type": "qwen3",
  "num_attention_heads": 16,
  "num_hidden_layers": 28,
  "num_key_value_heads": 8,
  "rms_norm_eps": 1e-06,
  "rope_scaling": null,
  "rope_theta": 1000000,
  "sliding_window": null,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.51.0",
  "use_cache": true,
  "use_sliding_window": false,
  "vocab_size": 151936
}
```

上述两个模型，架构都一样，但参数差异巨大，借着这个Case展开讨论一下哪些因素会影响模型的参数大，以及如何评估训练/推理的所需显存

## 模型参数量

模型的参数量由很多因素决定：

1. **模型架构** 这是很显然的，不同架构模型本身都不一样（下面只讨论相似架构下的情况）
2. **层数** 网络的层级越深，需要的参数越多，一般成线性增长关系o(N)
3. **hidden dim** 隐藏层维度是最容易控制参数量的，他在保持架构层级的前提下高效控制参数量，而且是平方增长关系o(D^2)

在具体推导之前，可以给一个结论，decoder-only的模型每层（注意力+FFN）的参数大小：

$
P_{layer} \approx 4d_{model}^2 + 2 \times (d_{model} \times d_{ff})
$

如果是 GQA（Grouped Query Attention），Attention 部分则变为：

$
P_{attn} = d_{model} \times d_{head} \times (n_q + n_{kv} + n_{kv}) + (n_q \times d_{head} \times d_{model})
$

如果是SwiGLU，FFN部分则变为：

$
P_{ffn} = 3 \times (d_{model} \times d_{ff})
$

回到Qwen的这个例子🌰，**Qwen3-0.6B**整体参数是0.6B，非embedding为0.44B，embedding为0.16B。我们走一遍流程

* $d_{model} = 1024$ 

* $n_{\text{heads_q}} = 16$

* $n_{\text{heads_kv}} = 8$

* $d_{head} = 128$

**第一步：线性投影**
Qwen的自注意力层是GQA（Grouped Query Attention）的架构，Key 和 Value 的头数与 Query 不同。
1. **生成 Q**: $Q = X \cdot W_Q$ $(B, L, 1024) \times (1024, 2048) \rightarrow \mathbf{(B, L, 2048)}$
2. **生成 K**: $K = X \cdot W_K$ $(B, L, 1024) \times (1024, 1024) \rightarrow \mathbf{(B, L, 1024)}$
3. **生成 V**: $V = X \cdot W_V$ $(B, L, 1024) \times (1024, 1024) \rightarrow \mathbf{(B, L, 1024)}$

**第二步：分头与转置（View & Permute）**
为了让每个“头”独立计算，我们需要对矩阵进行重塑。

- **Q**: $(B, L, 2048) \xrightarrow{reshape} (B, L, 16, 128) \xrightarrow{permute} \mathbf{(B, 16, L, 128)}$
- **K**: $(B, L, 1024) \xrightarrow{reshape} (B, L, 8, 128) \xrightarrow{permute} \mathbf{(B, 8, L, 128)}$
- **V**: $(B, L, 1024) \xrightarrow{reshape} (B, L, 8, 128) \xrightarrow{permute} \mathbf{(B, 8, L, 128)}$

**第三步：广播与点积**

8 个 $KV$ 组被广播（Broadcast）以匹配 16 个 $Q$ 头。计算 $\text{Softmax}(\frac{QK^T}{\sqrt{d_k}})V$ 后，我们得到每个头独立计算的结果：
$$
\text{Attention Output}: (B, 16, L, 128) \xrightarrow{permute} (B, L, 16, 128)
$$

**第四步：合并与输出投影（Concat & Output Projection）**

**还原形状**:

$$
O_{reshaped} = (B, L, 16, 128) \xrightarrow{reshape} \mathbf{(B, L, 2048)}
$$

**输出投影 ($W_O$)**:
从多头的拼接维度投影回 $d_{model}$
$$
Y = O_{reshaped} \cdot W_{O}
$$

$$
(B, L, 2048) \times (2048, 1024) \rightarrow \mathbf{(B, L, 1024)}
$$

### embedding层（0.16B）
embedding参数主要有如下影响：

- 词嵌入矩阵(V, dim)：$151936 \times 1024 = 155582464$
- 位置编码（Qwen是旋转位置编码RoPE）是无参数的 = 0

约为0.16B

注意：配置中`tie_word_embeddings: true`，代表`Input Embedding`和`Output LM Head`共享参数，所以只算一份。

### 非embedding层（隐藏层，0.44B）
非embedding参数主要有如下影响：

- 注意力层：
    - $W_Q$ 的形状是 $(hidden_{size}, \text{num_attention_heads} * head_{dim}) = (1024, 2048)$
    - $W_K$、$W_V$ 的形状是 $(hidden_{size}, \text{num_key_value_heads} × head_{dim}) = (1024, 1024)$。
    - 输出投影参数形状是 $(\text{num_attention_heads} * head_{dim}, hidden_{size}) = (2048, 1024)$

注意力层每一层的总参数量是：$1024 \times 2048 \times 3 = 6291456$

- 前馈网络（SwiGLU）:
    - **$W_{gate}$ (w1)**: 形状为 $(1024, 3072)$ —— 负责生成“门”信号。
    - **$W_{up}$ (w3)**: 形状为 $(1024, 3072)$ —— 负责提取特征信息。
    - **$W_{down}$ (w2)**: 形状为 $(3072, 1024)$ —— 负责将维度投影回原大小。

前馈网络每一层的总参数量是：$1024 \times 3072 \times 3 = 9437184$

- 归一化层（RMSNorm）:
RMSNorm 的参数量极小，每层 2 个，称之为Pre-Norm（前置归一化），分别是注意力层之前、FFN层之前，每个大小为 $\text{hidden_size}$，可得 $2 \times 1024 = 2048$

总共28层：$(6291456 + 9437184 + 2048) \times 28 = 440459264$
约为0.44B

## 所需显存大小

### 推理显存

推理所需显存主要包含以下几块：

| 组成部分 | 说明 | 是否随序列长度变化 |
|---------|------|-------------------|
| **模型权重** | 模型加载到显存中的参数，与 batch、seq 无关 | 否 |
| **KV Cache** | 自回归解码时每层缓存的 K、V，用于避免重复计算 | 是（∝ seq_len × batch） |
| **激活值** | 前向传播时的中间激活，prefill 阶段较大，decode 每步很小 | 是（prefill 大，decode 小） |
| **框架/上下文** | CUDA 上下文、kernel 等，通常几百 MB 量级 | 基本固定 |

对 Qwen 这类 decoder-only 模型，推理时的大头是：**模型权重 + KV Cache**。激活在 decode 阶段只占一层，相对可忽略；prefill 时若 batch 或 seq 很大，激活会暂时变大。

#### 模型权重显存

$$
M_{\text{weights}} = P \times b
$$

其中 $P$ 为模型参数量（个），$b$ 为每个参数的字节数（如 bfloat16 / float16 为 2，int8 为 1）。

**Qwen3-0.6B**（bf16）：$0.6 \times 10^9 \times 2 \approx 1.2\text{GB}$  
**Qwen3-1.7B**（bf16）：$1.7 \times 10^9 \times 2 \approx 3.4\text{GB}$

#### KV Cache 显存

自回归解码时，每层都要保存当前序列的 Key 和 Value，以便下一 token 的注意力计算复用。单层、单头、单 token 的 K 或 V 形状为 $(1, d_{head})$，整序列为 $(L, d_{head})$。

总 KV Cache 显存（字节）可写为：

$$
M_{\text{kv}} = 2 \times n_{layers} \times n_{\text{heads_kv}} \times d_{head} \times L \times B \times b
$$

- $n_{layers}$：层数（num_hidden_layers）
- $n_{\text{heads_kv}}$：KV 头数（num_key_value_heads）
- $d_{head}$：头维度（head_dim）
- $L$：当前序列长度（已生成 token 数 + 输入长度），与形状 $(B, L, d_{model})$ 中的 $L$ 一致
- $B$：batch size
- $b$：每元素字节数（如 bf16 取 2）
- 前面的 $2$ 表示 K 和 V 各一份

**Qwen3-0.6B 示例**：$n_{layers}=28$，$n_{\text{heads_kv}}=8$，$d_{head}=128$，$b=2$，$B=1$，$L=2048$：

$$
M_{\text{kv}} = 2 \times 28 \times 8 \times 128 \times 2048 \times 1 \times 2 \approx 224\text{MB}
$$

若 $L=32768$（满上下文），则 KV Cache 约 $3.6\text{GB}$。

#### 推理显存粗估

$$
M_{\text{infer}} \approx M_{\text{weights}} + M_{\text{kv}} + M_{\text{overhead}}
$$

$M_{\text{overhead}}$ 为框架与临时张量等，通常预留 0.5–1 GB 即可。因此：

- **Qwen3-0.6B**：$L=2048$、$B=1$ 时约 $1.2 + 0.22 + 0.5 \approx 2\text{GB}$。
- **Qwen3-1.7B**：同样条件下，权重约 3.4 GB，KV 与 0.6B 同配置下相同（层数、KV 头、$d_{head}$ 一致），总显存约比 0.6B 多 2.2 GB 左右。

