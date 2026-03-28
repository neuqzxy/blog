---
title: word2vec实战
date: 2026-02-25 18:41:22
tags:
  - Math
  - ML
  - AI
  - LLM
categories:
  - ML
---
## 数据处理

假设我们已经有了大量的语料，我们可以先预处理将其按句子拆分，每行一个句子。然后做两件事：

1. 清洗语料，将一些写错/非法的词转成未知词元类似'<unk>’
2. 通过空格进行词级的tokenization。

现在我们有了tokenization后的数据列表，为了加速训练，我们通常还需要**采样**，比如“the”“a”和“in”等高频词。例如，考虑上下文窗口中的词“chip”，它与低频单词“intel”的共现比与高频单词“a”的共现在训练中更有用。所以，降低高频词出现的频率可以在保证训练结果的前提下加速训练速度

### 下采样（降采样）

我们希望我们的采样能实现如下目标：

1. 要有一个阈值，词频高于该阈值时采允许采样，这样能保留低频词的训练效果。
2. 采样要平滑，不能太过分。

对于目标1，我们提供了超参数t来控制，$f(w_i) < t$的，采样概率一律为0。

对于目标2，我们通过开方，让丢弃概率更 “温和”。举个🌰，假设$t = 10^-4$，$f(w_i) = 10^-2$，$1 - 0.01 = 0.99$，而$1 - \sqrt{0.01}= 0.9$。这样能避免过度移除

$$
P(w_i) = max(1 - \sqrt{\frac{t}{f(w_i)}}, 0)
$$

$f(w_i)$是$w_i$的词数与数据集中的总词数的比率

$$
f(w_i) = \frac{count(w_i)}{\sum_j count(w_j)}
$$

## 代码

```py
from typing import List, Tuple, Dict
import collections
import math
import random
import time
import os
import torch
from torch import nn
import torch.utils.data as Data

# ========== 固定随机种子（可复现） ==========
seed = 2025
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

assert 'ptb.train.txt' in os.listdir('ptb'), "Error: ptb.train.txt not found in ptb directory"

# 按行读取单词
def read_words(filename: str) -> List[List[str]]:
    with open(filename, 'r') as f:
        lines = f.readlines()
        raw_dataset = [l.split() for l in lines]
        return raw_dataset

# 建立词索引
def build_vocab(raw_dataset: List[List[str]]) -> Tuple[Dict[str, int], List[str], Dict[str, int], int, List[List[int]], int]:
    counter = collections.Counter([vocab for l in raw_dataset for vocab in l])
    filtered_vocab = {word: count for word, count in counter.items() if count >= 5}

    idx_to_vocab = [vocab for vocab in filtered_vocab.keys()]
    vocab_to_idx = {vocab: idx for idx, vocab in enumerate(idx_to_vocab)}
    vocab_size = len(idx_to_vocab)

    dataset = [[vocab_to_idx[vocab] for vocab in l if vocab in vocab_to_idx] for l in raw_dataset]
    token_count = sum([len(l) for l in dataset])

    return filtered_vocab, idx_to_vocab, vocab_to_idx, vocab_size, dataset, token_count

# 二次采样
def discard_high_freq_words(vocab_idx: int, filtered_vocab: Dict[str, int], idx_to_vocab: List[str], token_count: int) -> bool:
    vocab = idx_to_vocab[vocab_idx]
    vocab_count = filtered_vocab[vocab]
    threshold = 1e-4
    vocab_prob = vocab_count / token_count
    p = max(1 - math.sqrt(threshold / vocab_prob), 0)
    return random.uniform(0, 1) < p

# 提取中心词和背景词
def get_centers_and_contexts(dataset: List[List[int]], max_window_size: int) -> Tuple[List[int], List[List[int]]]:
    centers: List[int] = []
    contexts: List[List[int]] = []
    for line in dataset:
        if len(line) < 2:
            continue
        centers += line
        for center_i in range(len(line)):
            window_size = random.randint(1, max_window_size)
            indices = list(range(
                max(0, center_i - window_size),
                min(len(line), center_i + 1 + window_size)
            ))
            indices.remove(center_i)
            contexts.append([line[idx] for idx in indices])
    return centers, contexts

# 负采样（修复：排除中心词）
def get_negatives(all_centers, all_contexts, sampling_weights: List[float], K: int) -> List[List[int]]:
    all_negatives: List[List[int]] = []
    neg_candidates: List[int] = []
    i = 0
    population = list(range(len(sampling_weights)))

    for center, contexts in zip(all_centers, all_contexts):
        negatives: List[int] = []
        while len(negatives) < len(contexts) * K:
            if i == len(neg_candidates):
                i, neg_candidates = 0, random.choices(population, sampling_weights, k=int(1e5))

            neg, i = neg_candidates[i], i + 1
            # ========== 修复：负样本不能是背景词 也不能是中心词 ==========
            if neg not in contexts and neg != center:
                negatives.append(neg)

        all_negatives.append(negatives)
    return all_negatives

# DataSet
class PTBDataset(Data.Dataset):
    def __init__(self, centers, contexts, negatives):
        assert len(centers) == len(contexts) == len(negatives)
        self.centers = centers
        self.contexts = contexts
        self.negatives = negatives

    def __getitem__(self, index) -> Tuple[int, List[int], List[int]]:
        return self.centers[index], self.contexts[index], self.negatives[index]

    def __len__(self):
        return len(self.centers)

# 按batch处理数据
def batchify(data: List[Tuple[int, List[int], List[int]]]):
    max_len = max(len(context) + len(negative) for _, context, negative in data)

    centers: List[int] = []
    contexts_negatives: List[int] = []
    masks: List[int] = []
    labels: List[int] = []

    for center, context, negative in data:
        centers += [center]
        cur_len = len(context) + len(negative)
        contexts_negatives += [context + negative + [0] * (max_len - cur_len)]
        masks += [[1] * cur_len + [0] * (max_len - cur_len)]
        labels += [[1] * len(context) + [0] * (max_len - len(context))]

    return (torch.tensor(centers).view(-1, 1), torch.tensor(contexts_negatives), torch.tensor(masks), torch.tensor(labels))

embed_size = 128

# 跳元模型
def skip_gram(center: torch.Tensor, contexts_and_negatives: torch.Tensor, embed_u: nn.Embedding, embed_v: nn.Embedding):
    u = embed_u(center)
    v = embed_v(contexts_and_negatives)
    pred = torch.bmm(u, v.permute(0, 2, 1))
    return pred

# 二元交叉熵
class SigmoidBinaryCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor):
        inputs, targets, mask = inputs.float(), targets.float(), mask.float()
        res = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none", weight=mask)
        return res.mean(dim=1)

CHECKPOINT_PATH = 'checkpoints/word2vec.pt'

def save_checkpoint(net: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, loss: float):
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, CHECKPOINT_PATH)
    print(f'checkpoint saved -> {CHECKPOINT_PATH} (epoch {epoch})')

def load_checkpoint(net: nn.Module, optimizer: torch.optim.Optimizer) -> int:
    if not os.path.exists(CHECKPOINT_PATH):
        return 0
    checkpoint = torch.load(CHECKPOINT_PATH, weights_only=True)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    print(f'checkpoint loaded <- {CHECKPOINT_PATH} (epoch {start_epoch}, loss {checkpoint["loss"]:.2f})')
    return start_epoch

def train(net: nn.Module, lr: float, num_epochs: int, data_iter: Data.DataLoader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("train on", device)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = SigmoidBinaryCrossEntropyLoss()

    start_epoch = load_checkpoint(net, optimizer)

    for epoch in range(start_epoch, num_epochs):
        t_start, l_sum, n = time.time(), 0.0, 0
        for batch in data_iter:
            center, context_negative, mask, label = [d.to(device) for d in batch]

            pred = skip_gram(center, context_negative, net[0], net[1])
            l = (loss(pred.view(label.shape), label, mask) * mask.shape[1] / mask.float().sum(dim=1)).mean()
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            l_sum += l.cpu().item()
            n += 1

        avg_loss = l_sum / n
        print('epoch %d, loss %.2f, time %.2fs' % (epoch + 1, avg_loss, time.time() - t_start))
        save_checkpoint(net, optimizer, epoch + 1, avg_loss)

def get_similar_tokens(query_token: str, k: int, embed: nn.Embedding, token_to_idx: Dict[str, int], idx_to_token: List[str]):
    W = embed.weight.data
    x = W[token_to_idx[query_token]]
    cos = torch.matmul(W, x) / (torch.sum(W * W, dim=1) * torch.sum(x * x) + 1e-9).sqrt()
    _, topk = torch.topk(cos, k=k+1)
    topk = topk.cpu().numpy()
    for i in topk[1:]:
        print('cosine sim=%.3f: %s' % (cos[i], idx_to_token[i]))

if __name__ == '__main__':
    raw_dataset = read_words('ptb/ptb.train.txt')
    filtered_vocab, idx_to_vocab, vocab_to_idx, vocab_size, dataset, token_count = build_vocab(raw_dataset)

    subsampled_dataset = [
        [vocab_idx for vocab_idx in l if not discard_high_freq_words(vocab_idx, filtered_vocab, idx_to_vocab, token_count)]
        for l in dataset
    ]

    def compare_counts(token):
        return '# %s: before=%d, after=%d' % (
            token,
            sum([st.count(vocab_to_idx[token]) for st in dataset]),
            sum([st.count(vocab_to_idx[token]) for st in subsampled_dataset])
        )

    print(f"高频词被高强度采样，低频词保留：{compare_counts('the')}, {compare_counts('join')}")

    all_centers, all_contexts = get_centers_and_contexts(subsampled_dataset, 5)

    sampling_weights = [filtered_vocab[word]**0.75 for word in idx_to_vocab]
    # ========== 修复：传入中心词 ==========
    all_negatives = get_negatives(all_centers, all_contexts, sampling_weights, 5)

    batch_size = 512
    # ========== 修复：Windows 兼容 ==========
    num_workers = 0 if os.name == 'nt' else 4

    ptb_dataset = PTBDataset(all_centers, all_contexts, all_negatives)
    data_iter = Data.DataLoader(ptb_dataset, batch_size, shuffle=True, num_workers=num_workers, collate_fn=batchify)

    net = nn.Sequential(
        nn.Embedding(vocab_size, embed_size),   # center embedding (used finally)
        nn.Embedding(vocab_size, embed_size)   # context embedding
    )

    # ========== 修复：学习率 ==========
    train(net, 0.002, 15, data_iter)

    get_similar_tokens('stock', 10, net[0], vocab_to_idx, idx_to_vocab)
```
