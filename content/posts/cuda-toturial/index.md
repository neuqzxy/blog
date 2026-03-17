---
title: CUDA 编程入门
date: 2026-03-01 18:37:23
tags:
  - CUDA
  - GPU
  - 并行计算
  - ML
categories:
  - ML
---
# 线程层次与索引

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

![](/img/cuda_idx_01.png)
![](/img/cuda_idx_02.png)

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

# 矩阵计算

对于 $\mathbb{R}^{m \times n}$ 和 $\mathbb{R}^{n \times k}$ 的矩阵乘法，大约需要 $m \cdot k \cdot n$ 个乘加运算，在CPU中，我们通常通过两层大循环( $ m \times n $ )，内部再加一个n次乘加的小循环，计算出 $\mathbb{R}^{m \times k}$ 的output

```c++
// width维方阵乘法（CPU）
void matMulOnHost(const float* matM, const float* matN, float* matP, int width) {
	for (int i = 0; i < width; ++i) {
		for (int j = 0; j < width; ++j) {
			float sum = 0;
			for (int k = 0; k < width; ++k) {
				sum += matM[i * width + k] * matN[k * width + j];
			}
			matP[i * width + j] = sum;
		}
	}
}
```

使用CUDA并发编程，就可以创建 $m \cdot k$ 个threads来并行处理，每个threads中完成一个n次乘加得到output矩阵中的一个cell

```c++
// width维方阵乘法（GPU）
__global__ void matMulKernel(const float* matM, const float* matN, float* matP, int width) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	
	float value = 0;
	for (int k = 0; k < width; ++k) {
		value += matM[row * width + k] * matN[k * width + col];
	}
	matP[row * width + col] = value;
}

void matMulOnDevice(const float* matM, const float* matN, float* matP, int width, unsigned int blockSize) {
	int size = width * width * sizeof(float);

	float *m_device;
	float *n_device;
	float *p_device;

	// 在GPU上分配内存
	cudaMalloc(&m_device, size);
	cudaMalloc(&n_device, size);
	cudaMalloc(&p_device, size);

	// 将数据从CPU复制到GPU
	cudaMemcpy(m_device, matM, size, cudaMemcpyKind::cudaMemcpyHostToDevice);
	cudaMemcpy(n_device, matN, size, cudaMemcpyKind::cudaMemcpyHostToDevice);

	dim3 dimBlock(blockSize, blockSize);
	dim3 dimGrid(width / dimBlock.x, width / dimBlock.y);

	// 在GPU上执行矩阵乘法
	matMulKernel <<<dimGrid, dimBlock>>>(m_device, n_device, p_device, width);

	// 将结果从GPU复制回CPU
	cudaMemcpy(matP, p_device, size, cudaMemcpyKind::cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	// 释放GPU内存
	cudaFree(p_device);
	cudaFree(n_device);
	cudaFree(m_device);
}
```

## Block与Warp调度

由于 GPU 硬件架构的限制，每个 block 的最大线程数不能超过 1024（绝大多数现代 CUDA 设备）。GPU 会将一个 block 内的线程按照Warp（线程束） 为基本单元拆分和调度，每个 Warp 固定包含 32 个连续的线程 —— 这是 CUDA 硬件层面的核心设计，目的是在调度效率和执行性能之间达到最优平衡：

1. **调度层面**：GPU 的 SM（流多处理器）不以单个线程为调度单位，而是以 Warp 为最小调度单元。相比调度单个线程，调度 32 线程的 Warp 能大幅降低调度开销（比如减少指令分发、上下文切换的成本）；
2. **执行层面**：一个 Warp 内的 32 个线程会同步执行相同的指令（SIMT 架构） —— 并非简单的 “并行执行”，而是 “单指令多线程”：同一个 Warp 内的所有线程在同一时钟周期执行同一条指令，只是处理不同的数据；
3. **访存优化层面**：当某个 Warp 中的线程需要访存（比如从全局内存读取数据）时，访存操作会有延迟（几十到几百个时钟周期）。此时 GPU 会将这个等待访存的 Warp挂起，并调度同一个 SM 上的其他就绪 Warp 执行，直到原 Warp 的访存完成。这种 “延迟隐藏” 机制能让 SM 的计算核心始终处于忙碌状态，最大化 GPU 的利用率。

一个Block不允许运行在多个SM中，但是一个SM允许运行多个Block，但会有一个**最大 Block 驻留数量（Max Blocks per SM）**，否则Block和Warp的界限就不明确了

**按 Compute Capability 区分的 GPU 规格：**

| 规格项 | 7.5 | 8.0 | 8.6 | 8.7 | 8.9 | 9.0 | 10.0 | 10.3 | 11.0 | 12.x |
|--------|-----|-----|-----|-----|-----|-----|------|------|------|------|
| Ratio of FP32 to FP64 Throughput [2] | 32:1 | 2:1 | 64:1 | 2:1 | 64:1 | — | — | — | — | — |
| Maximum number of resident blocks per SM | 16 | 32 | 16 | 24 | 32 | 24 | — | — | — | — |
| Maximum number of resident Warps per SM | 32 | 64 | 48 | 64 | 48 | — | — | — | — | — |
| Maximum number of resident threads per SM | 1024 | 2048 | 1536 | 2048 | 1536 | — | — | — | — | — |
| Green contexts: minimum SM partition size (useFlags 0) | 2 | 4 | 8 | — | — | — | — | — | — | — |
| Green contexts: SM co-scheduled alignment per partition (useFlags 0) | 2 | 8 | — | — | — | — | — | — | — | — |

**所有 Compute Capability 通用规格：**

| 规格项 | 值 |
|--------|-----|
| Maximum number of resident grids per device (Concurrent Kernel Execution) | 128 |
| Maximum dimensionality of a grid | 3 |
| Maximum x-dimension of a grid | 2³¹-1 |
| Maximum y- or z-dimension of a grid | 65535 |
| Maximum dimensionality of a thread block | 3 |
| Maximum x- or y-dimensionality of a thread block | 1024 |
| Maximum z-dimension of a thread block | 64 |
| Maximum number of threads per block | 1024 |
| Warp size | 32 |

```c++
void printCudaInfo() {
	int deviceId;
	cudaGetDevice(&deviceId); // 获取当前使用的GPU设备ID

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, deviceId); // 获取设备属性

	std::cout << "=== GPU设备属性 ===" << std::endl;
    std::cout << "GPU名称: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "每个Warp的线程数 (WarpSize): " << prop.WarpSize << std::endl; // 关键字段
    std::cout << "每个SM的最大线程数: " << prop.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "SM数量: " << prop.multiProcessorCount << std::endl;
    std::cout << "每个block最大线程数: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "各维度最大线程数限制：" << std::endl;
    std::cout << "  x维度: " << prop.maxThreadsDim[0] << std::endl;
    std::cout << "  y维度: " << prop.maxThreadsDim[1] << std::endl;
    std::cout << "  z维度: " << prop.maxThreadsDim[2] << std::endl;
}

// === GPU设备属性 ===
// GPU名称: NVIDIA GeForce RTX 5090 D v2
// Compute Capability: 12.0
// 每个Warp的线程数 (WarpSize): 32
// 每个SM的最大线程数: 1536
// SM数量: 170
// 每个block最大线程数: 1024
// 各维度最大线程数限制：
//   x维度: 1024
//   y维度: 1024
//   z维度: 64
```

## blockDim取值

SM 能够同时 **驻留（Resident）多个 Warp，这些 Warp 共享 SM 的寄存器和 Shared Memory 资源。虽然硬件为每个 Warp 预留了 32 个物理线程的执行通道，但由于 SIMT（单指令多线程） 的执行特性，硬件利用率取决于 Warp 内活跃线程（Active Threads）** 的数量。
当某个正在执行的 Warp 因为长延迟操作（如 Global Memory 访存）被挂起时，SM 的调度器会从池子里挑选另一个就绪（Ready）的 Warp 填补指令发射空隙。因此，通过合理的线程块（Block）配置来保证足够的 Warp 驻留量和 Warp 满载率，是实现延迟隐藏（Latency Hiding）、提升吞吐量的关键。”

> 我们考虑两种计算 $\mathbb{R}^{1024 \times 1024}$ 方阵乘法方案的例子🌰：
- **方案A**: 如果blockDim为1，那就是会有 $1024 \times 1024$ 个block，每个block有1个Warp，这个Warp中有1个threads，其他31个threads在空跑

对于方案A，每个SM最多分配 $1536 / 1 = 1536$ 个block，但由于**最大 Block 驻留数量（Max Blocks per SM）**的限制，每个 SM 最多只能同时驻留 24 个 Block（12.x的Compute Capability）。理论最多分配最多分配 $1536 / 32 = 48$ 个Warp，实际上则是 $\text{min}(48, 24) = 24$ 个Warp，同一时刻并行执行24个有效线程
同时方案A还有一个问题，每个 Warp 只有一个活动线程。这意味着当 GPU 发出访存指令时，它为了取 1 个 float（4 bytes），可能也要触发一个完整的 Cache Line 加载。带宽浪费极其严重。

- **方案B**：如果blockDim为32，那就是会有 $1024$ 个block，每个block有32个Warp，每个Warp中有32个threads

对于方案B，每个SM最多分配 $1536 / 1024 = 1$ 个block，最多分配 $1024 / 32 = 32$ 个Warp（每个Warp里有32个有效线程），同一时刻并行执行1024个有效线程

**更好的方案？**
方案B看似跑满了block，但是实际上并没有跑满SM的总线程数量，**Occupancy（占用率）**为 $1024/1536 \approx 66.7\%$。
**方案C**：如果设置blockDim为16，理论上会有那就是会有 $4096$ 个block，每个block有8个Warp，每个Warp中有32个threads
对于方案C，每个SM最多分配 $1536 / 256 = 6$ 个block，每个block分配 $256 / 32 = 8$ 个Warp，SM最多分配 $6 \times 8 = 48$ 个Warp（每个Warp里有32个有效线程）

| 特性 | 方案 A (blockDim=1) | 方案 B (blockDim=32x32) | 方案 C (blockDim=16x16) |
|------|---------------------|-------------------------|-------------------------|
| 线程分布/Block | 1 线程 | 1024 线程 | 256 线程 |
| Warp 数/Block | 1 Warp | 32 Warps | 8 Warps |
| SM 驻留 Block 数 | 24 (受限于 Max Blocks) | 1 (受限于 Max Threads) | 6 (完美契合) |
| SM 驻留 Warp 数 | 24 | 32 | 48 |
| 理论 Occupancy | 1.56% | 66.7% | 100% |
| 延迟隐藏能力 | 极差 | 一般 | 优秀 |

```text
# blockDim = 32
CPU uses: 4224.96 ms
GPU warmup uses: 74.894 ms
GPU uses: 1.9467 ms

# blockDim = 1
CPU uses: 4787.8 ms
GPU warmup uses: 82.4443 ms
GPU uses: 12.6243 ms
```

# 共享内存

## 方案

对于矩阵计算为例，假设两个 $\mathbb{R}^{4096 \times 4096}$ 维度的方阵相乘，生成的 $4096 \times 4096$ 维度矩阵中，每一个cell都需要 $4096 \times 2$ 次访存。总共需要 $ 4096 \times 4096 \times 4096 \times 2 $ 次访存。我们通过cudaMalloc分配内存，并在核函数中通过指针访问的时候，其实是访问的DRAM（global memory）属于片外内存（off-inch memory）。GPU中所有缓存的耗时可参考下图：

![](/img/cuda_memory_list.jpg)

所以需要再计算前，将一批数据缓存到shared memory中，Shared Memory 的大小并不是固定的，它取决于你的 GPU 架构，在矩阵乘法 $C = A \times B$ 中，我们通常切出 $\text{TILE_WIDTH} \times \text{TILE_WIDTH}$ 的小块。选择这个尺寸时，需要权衡以下三个核心因素：

- **A. 必须是 Warp 的倍数**
由于一个 Warp 是 32 个线程，你的线程块总数（Threads per Block）最好是 32 的倍数。$16 \times 16$： 256 个线程（8 个 Warp）。常用，平衡性好。$32 \times 32$： 1024 个线程（32 个 Warp）。这是单 Block 线程数的上限，性能通常最强，但会挤占资源。

- **B. 内存占用的计算**
假设你选 $32 \times 32$ 的 Tile，每个元素是 float（4 字节）：矩阵 A 的 Tile：$32 \times 32 \times 4 = 4$ KB矩阵 B 的 Tile：$32 \times 32 \times 4 = 4$ KB总计： 一个 Block 需要 8 KB 的 Shared Memory。对于 RTX 4090（128 KB/SM）来说，这远远不够塞满。看起来我们可以开更大的 Tile？不一定。

- **C. 占用率（Occupancy）的博弈**
GPU 的强大在于“并行掩盖延迟”。如果你把 Tile 设得非常大（比如耗尽了 128KB），那么一个 SM 同时只能运行 1 个 Block。如果这个 Block 因为某些原因阻塞了，SM 就没牌可换了，性能反而下降。黄金法则： 尽量让一个 SM 能同时跑 2~4 个 Blocks。这样当一个 Block 在搬运数据时，另一个 Block 可以在计算，实现流水线化。

```text
[核心计算单元]
  设备名称:                   NVIDIA GeForce RTX 5090 D v2
  计算能力:                   12.0
  SM 数量:                    170
  核心主频:                   2467 MHz
  Warp 大小:                  32 threads

[存储层级]
  总显存容量:                 23.88 GB
  L2 缓存大小:                96.00 MB
  显存位宽:                   384 bit
  理论峰值带宽:               1344.10 GB/s

[SM 资源限制]
  每个 SM 最大共享内存:       100.00 KB
  每个 Block 最大共享内存:    48.00 KB
  每个 SM 最大寄存器数:       65536
  每个 Block 最大寄存器数:    65536
  每个 SM 最大活跃线程:       1536
  每个 SM 最大活跃 Block:     24
```

## 实现
实现很简单，首先我们将需要计算的两个矩阵 $M$ 和 $N$ 拆分成 $\text{TILE\_WIDTH} \times \text{TILE\_WIDTH}$ 的小矩阵 $M_{tile}$ 和 $N_{tile}$，并且保障TILE_WIDTH是小于BLOCK_DIM的，这样每个线程刚好负责从 Global Memory 搬运 1 个 元素到 Shared Memory。然后将 $M_{tile}$ 和 $N_{tile}$ 的数据平均拆给每个线程去读写到shared memory中，等都读完之后就使用shared memory中的数据开始计算。

**涉及到两次阻塞：**

1. **第一次同步**： 在所有线程完成从 Global Memory 到 Shared Memory 的搬运后。

	- 原因： 必须保证 Tile 里的所有数据都到位了，计算线程才能开始读，否则会读到旧数据。

2. **第二次同步**： 在所有线程完成当前 Tile 的乘加计算后。

	- 原因： 必须保证所有线程都用完了当前的 Shared Memory 数据，才能开始搬运下一个 Tile 的数据覆盖它，否则会把还没参与计算的数据给覆盖掉（Write-after-Read 冲突）。

当然阻塞分别独立的发生在BLOCK中的，并不会产生跨BLOCK影响

```c++
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include "cuda_err.h"

#define BLOCKSIZE 16

extern void __syncthreads();

__global__ void matMulStaticKernel(const float* matM, const float* matN, float* matP, int width) {
	// 声明共享内存
	__shared__ float M_deviceShared[BLOCKSIZE][BLOCKSIZE];
	__shared__ float N_deviceShared[BLOCKSIZE][BLOCKSIZE];

	// 计算当前线程在输出矩阵 P 中的行 (y) 和列 (x)
	int x = blockIdx.x * BLOCKSIZE + threadIdx.x;
	int y = blockIdx.y * BLOCKSIZE + threadIdx.y;

	// 线程在块内的局部索引
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	float sum = 0.0f;
	
	for (int m = 0; m < width / BLOCKSIZE; m++) {
		M_deviceShared[ty][tx] = matM[y * width + tx + m * BLOCKSIZE];
		N_deviceShared[ty][tx] = matN[(ty + BLOCKSIZE * m) * width + x];

		__syncthreads();

		// 具体计算
		for (int k = 0; k < BLOCKSIZE; k++) {
			sum += M_deviceShared[ty][k] * N_deviceShared[k][tx];
		}

		__syncthreads();
	}

	matP[y * width + x] = sum;
}

__global__ void matMulDynamicKernel(const float* matM, const float* matN, float* matP, int width) {
	// 动态共享变量必须是一维，而且只能声明一个
	extern __shared__ float deviceShared[];
	int stride = BLOCKSIZE * BLOCKSIZE;

	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	float sum = 0.0f;

	for (int m = 0; m < width / BLOCKSIZE; m++) {
		// 将矩阵 M 和 N 的子块加载到共享内存中
		deviceShared[ty * BLOCKSIZE + tx] = matM[y * width + tx + m * BLOCKSIZE];
		deviceShared[stride + ty * BLOCKSIZE + tx] = matN[(BLOCKSIZE * m + ty) * width + x];

		__syncthreads();

		// 具体计算
		for (int k = 0; k < BLOCKSIZE; k++) {
			sum += deviceShared[ty * BLOCKSIZE + k] * deviceShared[stride + tx + BLOCKSIZE * k];
		}

		__syncthreads();
	}

	matP[y * width + x] = sum;
}

void matMulOnShareDevice(const float* matM, const float* matN, float* matP, int width, unsigned int blockSize, bool staticMem) {
	int size = width * width * sizeof(float);

	float* m_device;
	float* n_device;
	float* p_device;

	// 在GPU上分配内存
	CUDA_CHECK(cudaMalloc(&m_device, size));
	CUDA_CHECK(cudaMalloc(&n_device, size));
	CUDA_CHECK(cudaMalloc(&p_device, size));

	// 将数据从CPU复制到GPU
	CUDA_CHECK(cudaMemcpy(m_device, matM, size, cudaMemcpyKind::cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(n_device, matN, size, cudaMemcpyKind::cudaMemcpyHostToDevice));

	dim3 dimBlock(blockSize, blockSize);
	dim3 dimGrid(width / dimBlock.x, width / dimBlock.y);

	// 在GPU上执行矩阵乘法
	if (staticMem) {
		matMulStaticKernel <<<dimGrid, dimBlock>>> (m_device, n_device, p_device, width);
	} else {
		size_t sharedMemSize = blockSize * blockSize * sizeof(float) * 2; // M 和 N 各占一个块
		matMulDynamicKernel <<<dimGrid, dimBlock, sharedMemSize>>> (m_device, n_device, p_device, width);
	}

	CUDA_CHECK_KERNEL();

	// 将结果从GPU复制回CPU
	CUDA_CHECK(cudaMemcpy(matP, p_device, size, cudaMemcpyKind::cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();

	// 释放GPU内存
	CUDA_CHECK(cudaFree(p_device));
	CUDA_CHECK(cudaFree(n_device));
	CUDA_CHECK(cudaFree(m_device));
}
```
