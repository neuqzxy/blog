---
title: 数学语料
date: 2026-03-20 21:18:26
tags:
  - Math
  - ML
  - AI
  - LLM
categories:
  - ML
---
## 开源数学语料（用于后训练）

| 语料 | 描述 |
|---|---|
| [nvidia/Nemotron-Math-v2](https://huggingface.co/datasets/nvidia/Nemotron-Math-v2) | 大规模数学推理数据集：约 34.7 万高质量题目与约 700 万推理轨迹；结合多模式监督与工具配置生成路径，并通过 LLM-as-a-judge 与通过率筛选保证质量；可商用。 |
| [nvidia/Nemotron-Math-HumanReasoning](https://huggingface.co/datasets/nvidia/Nemotron-Math-HumanReasoning) | 紧凑的人写解题数据，模拟 DeepSeek-R1 等“扩展推理风格”；提供多版本解答并用于对比人写与模型生成推理；非商用。 |
| [nvidia/Nemotron-Math-Proofs-v1](https://huggingface.co/datasets/nvidia/Nemotron-Math-Proofs-v1) | 证明类数学推理数据集：约 58 万自然语言证明题、约 55 万 Lean 4 定理形式化、约 90 万最终落到 Lean 4 证明的推理轨迹；经 Lean 编译器验证；可商用。 |
| [AI-MO/NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) | 约 86 万道数学题的 CoT 格式解答数据（题目→链式推理→最终答案）；来源覆盖中文中学题与美国/国际奥数题，包含 OCR、切分为题-解对、翻译与 CoT 重排等流程。 |
| [agentica-org/DeepScaleR-Preview-Dataset](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset) | 约 4 万条唯一“题目-答案”对；主要来自 AIME（1984-2023）、AMC（2023 前）、Omni-MATH 与 Still 等；JSON 包含 LaTeX 题面、LaTeX 格式官方解答（含 boxed 最终答案）与提取的 answer。 |
| [DeepMath-103K](https://go.hyper.ai/dquTu) | 腾讯与上交联合发布（2025）的数学推理数据集：聚焦难度 5-9，覆盖代数/微积分/数论/几何/概率/离散数学等；通过语义匹配进行去污染，尽量减少测试集泄露并提升评估公平性。 |
| [Project Euler](https://projecteuler.net/) | 数学 + 编程结合的挑战题平台；多数题需要的不仅是数学直觉，还要借助计算机与编程获得高效解法，适合训练“可计算解题思路”。 |
| [OpenEvals/IMO-AnswerBench](https://huggingface.co/datasets/OpenEvals/IMO-AnswerBench) | IMO 相关短答基准：包含 400 道来自 IMO 及其他来源的高难度短答案题；任务是给出简短且可验证的答案；属于 DeepMind 的 IMO-Bench 套件。 |
| [OpenDataArena/ODA-Math-460k](https://huggingface.co/datasets/OpenDataArena/ODA-Math-460k) | 基于 OpenDataArena 领先语料精选并深度清洗的数据集；含去重、竞赛基准去污染、LLM 过滤与验证器支持的响应蒸馏；训练集约 46 万题，格式为题目→逐步推理轨迹→最终答案。 |
| [BytedTsinghua-SIA/DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k) | 面向 AIME 与 AMC10/12 等竞赛体系的数学题数据集（约 1.7 万条）；适合训练竞赛风格数学推理与答案对齐。 |
| [MathArena](https://huggingface.co/MathArena) | 用于评测 LLM 在最新数学竞赛/奥数题表现的平台；matharena.ai 提供数据同步，HuggingFace 仓库包含原始竞赛题与对应模型答案。 |
