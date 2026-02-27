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
