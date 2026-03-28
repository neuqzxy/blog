---
title: VLLM 入门
date: 2026-03-28 19:36:27
tags:
  - Math
  - ML
  - AI
categories:
  - ML
---
## vLLM vs 普通 Transformers 推理对比

### 1. KV Cache 内存管理 (PagedAttention)

> PagedAttention核心价值在于通过减少显存碎片，提升显存的利用率

**普通 Transformers 推理**在生成阶段，每个 token 都需要将之前所有 token 的 Key/Value 向量缓存起来（即 KV Cache），以避免重复计算。传统实现会为每条请求**预先分配连续的最大长度内存块**，例如一条最大支持 2048 token 的请求，无论实际生成了多少 token，都会一次性占用 2048 个 token 对应的 KV Cache 空间。这带来了严重的内存碎片和浪费：

- **内部碎片**：预分配过多，实际只用了一小部分
- **外部碎片**：不同请求的内存块大小不一，难以复用

**vLLM** 引入了 `PagedAttention`，借鉴操作系统虚拟内存分页的思想，将 KV Cache 切分为固定大小的**物理块（Block）**，每个块存储固定数量 token 的 KV 向量。逻辑上连续的 KV Cache 可以映射到不连续的物理块，按需分配，几乎消除了内存碎片。

### 2. 批处理策略

| 特性 | 普通 Transformers | vLLM |
|------|-----------------|------|
| 批处理方式 | 静态批处理（Static Batching） | 连续批处理（Continuous Batching） |
| 请求调度 | 一批请求必须全部完成才能接收新请求 | 每生成完一个 token 就可以插入新请求 |
| GPU 利用率 | 长请求阻塞短请求，GPU 经常空闲 | 持续高负载，GPU 利用率显著提升 |

**静态批处理**的问题在于：一个 batch 中如果有一条极长的请求，其他已完成的请求只能等待，GPU 算力被白白浪费。

**连续批处理**（也叫 iteration-level scheduling）以每次迭代（每生成一个 token）为粒度调度请求，已完成的序列立即释放槽位给新请求，服务吞吐量大幅提升。

### 3. 吞吐量与延迟

根据 vLLM 论文的评测数据，与 HuggingFace Transformers 相比：

- **吞吐量提升**：在相同硬件下，vLLM 的请求吞吐量可达 HuggingFace Transformers 的 **24 倍**
- **内存利用率**：PagedAttention 将 KV Cache 的内存浪费从约 60-80% 降低到 **4% 以下**

### 4. 适用场景对比

| 场景 | 推荐方案 | 原因 |
|------|---------|------|
| 研究/调试单条推理 | HuggingFace Transformers | 简单易用，调试方便 |
| 小规模批量离线推理 | HuggingFace Transformers | 依赖少，部署简单 |
| 在线服务/高并发推理 | vLLM | 高吞吐、低延迟、内存高效 |
| 多请求并发服务 | vLLM | 连续批处理显著提升 GPU 利用率 |

## 代码演示

```py
import os
import sys
import subprocess
import time
from types import SimpleNamespace

from vllm import LLM, SamplingParams
from openai import OpenAI
from util import models_root

# 要使用的模型名称（远程仓库标识，仅用于缺省时 huggingface-cli 下载）
model_name = "Qwen_Qwen3-0.6B"
local_root_dir = models_root()
# 推理用权重目录（全量微调示例：os.path.join(local_root_dir, "qwen3-0.6b-medical-full-final")）
local_model_dir = os.path.join(local_root_dir, model_name)

example_messages = [
    {"role": "system", "content": "你是一个专业的中文助手"},
    {"role": "user", "content": "你好啊"},
    {
        "role": "assistant",
        "content": "您好！有什么可以帮助您的吗？需要我为您提供支持或提供帮助吗？",
    },
    {"role": "user", "content": "吸烟对身体有什么影响"},
]

# 使用 vLLM 直接进行推理
def reasoning() -> None:
    # vLLM 会通过 multiprocessing spawn 起子进程，会再次 import 本模块；
    # LLM() 必须只在主进程、在 __main__ 里创建，否则触发 bootstrapping 报错。
    if not os.path.isdir(local_model_dir) or not os.listdir(local_model_dir):
        os.system(
            f"huggingface-cli download --resume-download {model_name} --local-dir {local_model_dir}"
        )

    llm = LLM(model=local_model_dir)
    sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=32768)

    response = llm.chat(example_messages, sampling_params=sampling_params)
    print(response)

# 使用 OpenAI 接口进行推理
class OpenAIReasoning:
    def __init__(self, model_path: str):
        # 用 SimpleNamespace 才能用 self.cfg.seed 这种属性写法；dict 要用 self.cfg["seed"]
        self.cfg = SimpleNamespace(
            seed=2025,
            model_path=model_path,
            served_model_name=os.path.basename(model_path.rstrip(os.sep)),
            batch_size=3,
            gpu_memory_utilization=0.9,
            port=8123,
            dtype="auto",
            kv_cache_dtype="fp8_e4m3",
            context_tokens=32768,
            stream_interval=200,
            session_timeout=600,
            server_timeout=600,
        )
        self.cmd = [
            sys.executable,
            '-m',
            'vllm.entrypoints.openai.api_server',
            '--seed',                   str(self.cfg.seed),
            '--model',                  self.cfg.model_path,
            '--served-model-name',      self.cfg.served_model_name,
            '--tensor-parallel-size',   '1',
            '--max-num-seqs',           str(self.cfg.batch_size),
            '--gpu-memory-utilization', str(self.cfg.gpu_memory_utilization),
            '--host',                   '0.0.0.0',
            '--port',                   str(self.cfg.port),
            '--dtype',                  self.cfg.dtype,
            '--kv-cache-dtype',         self.cfg.kv_cache_dtype,
            '--max-model-len',          str(self.cfg.context_tokens),
            '--stream-interval',        str(self.cfg.stream_interval),
            '--async-scheduling',       # 启用异步调度，提升吞吐
            '--disable-log-stats',      # 关闭统计日志，减少输出噪音
            '--enable-prefix-caching',  # 开启前缀缓存，加速相同前缀的重复请求
        ]

        self.base_url = f'http://0.0.0.0:{self.cfg.port}/v1'
        self.api_key = 'sk-local'

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.cfg.session_timeout,
        )

        self.log_file = open('vllm_server.log', 'w')

        self.server_process = subprocess.Popen(
            self.cmd,
            stdout=self.log_file,
            stderr=subprocess.STDOUT,  # 将 stderr 合并到同一日志文件
            start_new_session=True     # 新建会话，避免父进程信号传播到子进程
        )
    
    def wait_for_server(self):
        """轮询等待 vLLM 服务就绪，每秒检查一次，超时或进程意外退出则抛出异常。

        通过调用 `models.list()` 接口探测服务是否可响应；
        若子进程已退出，则读取日志并在异常中附上完整输出，便于排查启动失败原因。
        """
        print('Waiting for vLLM server...')
        start_time = time.time()

        for _ in range(self.cfg.server_timeout):
            return_code = self.server_process.poll()


            if return_code is not None:
                # 进程已退出（非预期），读取日志后抛出详细错误
                self.log_file.flush()

                with open(self.log_file.name, 'r') as log_file:
                    logs = log_file.read()

                raise RuntimeError(f'Server died with code {return_code}. Full logs:\n{logs}\n')
            
            try:
                self.client.models.list()
                elapsed = time.time() - start_time
                print(f'Server is ready (took {elapsed:.2f} seconds).\n')

                return


            except Exception:
                time.sleep(1)  # 服务尚未就绪，等待 1 秒后重试
    
    def reasoning(self, messages: list[dict[str, str]]) -> str:
        """推理：发送请求并返回模型响应。"""
        try:
            response = self.client.chat.completions.create(
                model=self.cfg.served_model_name,
                messages=messages,
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f'Failed to reason: {e}') from e
    
    def shutdown(self, wait_seconds: float = 15.0) -> None:
        """结束本类启动的 vLLM 子进程并关闭日志文件。"""
        proc = getattr(self, "server_process", None)
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=wait_seconds)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5.0)
        log = getattr(self, "log_file", None)
        if log is not None and not log.closed:
            log.close()

def reasoning_openai() -> None:
    try:
        openai_reasoning = OpenAIReasoning(model_path=local_model_dir)
        openai_reasoning.wait_for_server()
        response = openai_reasoning.reasoning(messages=example_messages)
        print(response)
    finally:
        openai_reasoning.shutdown()

if __name__ == "__main__":
    # reasoning()
    reasoning_openai()
```
