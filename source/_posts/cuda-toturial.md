---
title: WIP CUDA 编程入门：线程层次与索引
date: 2026-02-29 18:37:23
tags:
  - CUDA
  - GPU
  - 并行计算
  - ML
categories:
  - ML
---
# CUDA 编程入门：线程层次与索引

## 1. 线程层次结构

在 CUDA 的编程模型中，线程按层次组织为 **Grid → Block → Thread** 三级结构：

- **Grid（网格）**：顶层容器，可包含大量 Block；
- **Block（线程块）**：每个 Block 最多包含 **1024** 个 Thread；
- **Thread（线程）**：最小执行单元，每个 Thread 独立执行 Kernel 函数。

Kernel 通过 `func_name<<<grid_dim, block_dim>>>(...args)` 语法在 GPU 上启动。各 Thread 之间**并行且执行顺序不确定**，可通过内置变量 `blockIdx`、`threadIdx` 等唯一确定当前线程的身份。

## 2. 内置索引变量与存储顺序

CUDA 提供以下内置变量用于线程定位：

| 变量 | 含义 |
|------|------|
| `blockIdx` | 当前 Block 在 Grid 中的索引 |
| `threadIdx` | 当前 Thread 在 Block 中的索引 |
| `gridDim` | Grid 在各维度上的大小 |
| `blockDim` | Block 在各维度上的大小 |

上述多维索引变量（`dim3` 类型）在底层采用 **列主序（Column-major）** 存储，与 C/C++ 数组的行主序不同：

- **行主序（C/C++）**：`arr[x][y][z]` 的存储顺序为 x → y → z，即先变化 x，再 y，最后 z；
- **列主序（CUDA）**：`dim3(x, y, z)` 的底层存储顺序为 z → y → x，按 z/y/x 顺序遍历可获得更好的访存效率（与硬件设计一致）。

![](/images/cuda_idx_01.png)
![](/images/cuda_idx_02.png)

## 3. 示例代码

以下示例展示如何启动 Kernel 并打印各线程的索引信息：

```cpp
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void print_idx_kernel() {
    printf("block idx: (%3u, %3u, %3u), thread idx: (%3u, %3u, %3u)\n",
        blockIdx.z, blockIdx.y, blockIdx.x,
        threadIdx.z, threadIdx.y, threadIdx.x);
}

__global__ void print_dim_kernel() {
    printf("block dim: (%3u, %3u, %3u), thread dim: (%3u, %3u, %3u)\n",
        gridDim.z, gridDim.y, gridDim.x,
        blockDim.z, blockDim.y, blockDim.x);
}

__global__ void print_coord_kernel() {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int idx_in_block = threadIdx.z * (blockDim.x * blockDim.y) +
        threadIdx.y * blockDim.x +
        threadIdx.x;

    printf("block idx: (%3u, %3u, %3u), thread idx in block: %3u, coord: (%3u, %3u)\n",
        blockIdx.z, blockIdx.y, blockIdx.x,
        idx_in_block, x, y);
}

int main()
{
    dim3 grid{ 2, 2 };
    dim3 block{ 2, 3 };

    print_coord_kernel<<<grid, block>>>();

    cudaDeviceSynchronize();    

    return 0;
}

/*
block idx: (  0,   0,   0), thread idx in block:   0, coord: (  0,   0)
block idx: (  0,   0,   0), thread idx in block:   1, coord: (  1,   0)
block idx: (  0,   0,   0), thread idx in block:   2, coord: (  0,   1)
block idx: (  0,   0,   0), thread idx in block:   3, coord: (  1,   1)
block idx: (  0,   0,   0), thread idx in block:   4, coord: (  0,   2)
block idx: (  0,   0,   0), thread idx in block:   5, coord: (  1,   2)
block idx: (  0,   0,   1), thread idx in block:   0, coord: (  2,   0)
block idx: (  0,   0,   1), thread idx in block:   1, coord: (  3,   0)
block idx: (  0,   0,   1), thread idx in block:   2, coord: (  2,   1)
block idx: (  0,   0,   1), thread idx in block:   3, coord: (  3,   1)
block idx: (  0,   0,   1), thread idx in block:   4, coord: (  2,   2)
block idx: (  0,   0,   1), thread idx in block:   5, coord: (  3,   2)
block idx: (  0,   1,   0), thread idx in block:   0, coord: (  0,   3)
block idx: (  0,   1,   0), thread idx in block:   1, coord: (  1,   3)
block idx: (  0,   1,   0), thread idx in block:   2, coord: (  0,   4)
block idx: (  0,   1,   0), thread idx in block:   3, coord: (  1,   4)
block idx: (  0,   1,   0), thread idx in block:   4, coord: (  0,   5)
block idx: (  0,   1,   0), thread idx in block:   5, coord: (  1,   5)
block idx: (  0,   1,   1), thread idx in block:   0, coord: (  2,   3)
block idx: (  0,   1,   1), thread idx in block:   1, coord: (  3,   3)
block idx: (  0,   1,   1), thread idx in block:   2, coord: (  2,   4)
block idx: (  0,   1,   1), thread idx in block:   3, coord: (  3,   4)
block idx: (  0,   1,   1), thread idx in block:   4, coord: (  2,   5)
block idx: (  0,   1,   1), thread idx in block:   5, coord: (  3,   5)
*/
```
