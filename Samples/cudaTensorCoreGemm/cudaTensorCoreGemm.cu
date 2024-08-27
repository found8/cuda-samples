/* Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// CUDA sample demonstrating a GEMM computation using the Warp Matrix Multiply
// and Accumulate API introduced in CUDA 9.

// In this program, the compute_gemm kernel computes the result of a matrix
// multiplication and addition: D = alpha * A * B + beta * C. The dimensions of
// both C and D matrices are M_GLOBAL x N_GLOBAL. The A matrix is M_GLOBAL x
// K_GLOBAL (row-major), the B matrix is K_GLOBAL x N_GLOBAL (column-major). In
// that kernel, each CTA computes one 128 x 128 tile of the resulting matrix per
// iteration. When the tile is computed, the CTA stores it to the global memory
// and begins a new iteration, selecting a new 128 x 128 tile to compute.
// Each CTA consists of eight warps. For the 128 x 128 tile, each warp computes
// eight 16 x 16 subtiles, organized in a 2 x 4 two-dimensional array. Warps
// compute the 16 x 16 subtiles using nvcuda::wmma::mma_sync operations by
// moving through the K_GLOBAL dimension of the A and B matrices and
// accumulating the intermediate result in the local thread state.

// There are a number of simple optimizations used in the algorithm:
// - The CTA copies the 128 x 128 tile of the C matrix from the global memory to
//   shared memory. After that is done, each warp loads the C matrix fragments
//   from shared memory, thus avoiding a random global memory access.
// - On each internal iteration, the CTA copies a portion of the A and B
// matrices from
//   global memory to shared memory. After that, all warps in the CTA reuse the
//   A and B data from shared memory, thus reducing the number of data copies
//   from global memory.
// - The portions of the A and B matrices are stored in shared memory with an
// additional
//   padding (skew) to reduce the number of shared memory access bank conflicts.
//   (See a detailed explanation near the SKEW_HALF macro definition.)
// - When the CTA finishes computing the tiles of the resulting matrix, each
// warp stores
//   its subtiles to shared memory. The CTA then copies the shared memory
//   contents to global memory, again avoiding redundant random global memory
//   accesses.
// - Note that the CTA tile size is chosen to maximize the GPU register
// utilization,
//   but carefully enough to avoid local memory use.

#include <assert.h>
#include <cuda.h>
#include <mma.h>
#include <stdio.h>

// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

// Externally configurable parameters.

#ifndef CPU_DEBUG
// Set this to 1 to verify the correctness of the GPU-computed matrix.
#define CPU_DEBUG 0
#endif

#ifndef SHARED_MEMORY_LIMIT_64K
// Set this to 0 to use more than 64 Kb of shared memory to cache data, to
// improve the performance of the computations on GPU.
// Note that you need a GPU that can have more than 64 Kb of shared memory
// per multiprocessor.
#define SHARED_MEMORY_LIMIT_64K 1
#endif

// GPU configuration.

#define WARP_SIZE 32

// MMA matrix tile dimensions.

#define M 16
#define N 16
#define K 16

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// GEMM configuration.

#define M_TILES 256
#define N_TILES 256
#define K_TILES 256

#define M_GLOBAL (M * M_TILES)
#define N_GLOBAL (N * N_TILES)
#define K_GLOBAL (K * K_TILES)

#define C_LAYOUT wmma::mem_row_major

// Implementation constants.

#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)

#if SHARED_MEMORY_LIMIT_64K
// 由于 64K 共享内存的限制，可以装下两个 128 * 128 的 A 和 B 矩阵
// 因为  16 * 16 * 8 * 8 * 2 = 32 Kb // 这个地方感觉有问题啊，half是多大？不是 sizeof(half) = 2 ?
// 但是 还有 为了避免 bank conflicts 的共享内存开销，如果没有这个开销，性能将受到严重影响
// 所以我们选择将 块的大小 减半。 这会使得 K 维度上的循环开销增加一倍，这只是轻微的影响性能。

// With only 64 Kb shared memory available, we can fit two 8-tile chunks of
// the A and B matrix data, that are 16 * 16 * 8 * 8 * 2 = 32 Kb each
// (i.e. two 8x8 arrays of tiles of 16x16 half-typed elements per CTA).
// But we cannot account the 8 Kb total skew overhead, without which the
// performance would be severely impacted. So we choose to reduce the chunk size
// in half, i.e. the amount of A and B matrix data we cache in shared memory.
// Accordingly, this doubles the number of outer iterations across the global K
// dimension, which only slightly impacts the performance.
#define CHUNK_K 4
#else
#define CHUNK_K 8
#endif

#define CHUNK_LINE_BYTES (CHUNK_K * K * sizeof(half)) // 每个 chunk 数据占 4 * 16 * 2 = 8 * 16
#define WARP_COPY_BYTES (WARP_SIZE * sizeof(int4))    // 32 * 16
#define CHUNK_COPY_LINES_PER_WARP (WARP_COPY_BYTES / CHUNK_LINE_BYTES) // 32 * 16 / (8 * 16) = 4
#define CHUNK_COPY_LINE_LANES (WARP_SIZE / CHUNK_COPY_LINES_PER_WARP) // 32 / 4 = 8

#define BLOCK_ROW_WARPS 2
#define BLOCK_COL_WARPS 4

#define WARP_ROW_TILES 4
#define WARP_COL_TILES 2

#define BLOCK_ROW_TILES (WARP_ROW_TILES * BLOCK_ROW_WARPS) // 8
#define BLOCK_COL_TILES (WARP_COL_TILES * BLOCK_COL_WARPS) // 8

#define GLOBAL_MEM_STRIDE N_GLOBAL

#define SHMEM_STRIDE (N * BLOCK_ROW_TILES) // shmem_stride 共享内存步长 N * 8
#define SHMEM_OFFSET (N * WARP_ROW_TILES)  // shmem_offset 共享内存偏移 N * 4

// 下面这个宏，是用来使 矩阵 A 的 行 和 矩阵 B 的列 产生偏移，来最小化共享内存的 bank conflicts。
// 在执行 nvcuda::wmma::mma_sync 操作之前，必须执行 nvcuda::wmma::load_matrix_sync 操作来加载数据
// 虽然该函数未指定内存访问模式，但每个warp中的每个lane都可以读取一个或多个矩阵 来自不同矩阵行或列的元素
// 对于共享内存，如果同一warp下的不同lane，访问同一bank下的不同数据，会产生bank conflicts，
// 即 如果同一warp下的不同lane，访问同一bank下的 不同矩阵的不同行/列，就会有bank conflicts
// 通过是矩阵的行/列在共享内存中偏移一些字节，可以尽可能的减小这种冲突。
// 在本次计算中，最小偏移量是 16 个 双字节（half） 的元素，因为我们必须保证 行/列 是256-bit 对齐。
// 256-bit对齐是 nvcuda::wmma::load_matrix_sync 所需要的

// The macro below is used to shift rows of the A matrix and columns of the B
// matrix in shared memory to minimize possible bank conflicts. Before
// performing the nvcuda::wmma::mma_sync operation, the warp must load the
// matrix data using the nvcuda::wmma::load_matrix_sync operation. Although the
// memory access pattern is not specified for that function, each lane in the
// warp can read one or multiple matrix elements from different matrix rows or
// columns. For shared memory, such access can result in bank conflicts if
// different rows / columns of the matrix map to the same bank. By shifting each
// row and column by a few bytes, we make sure that they map to different banks,
// thus reducing the number of possible bank conflicts. The number of 8 two-byte
// "half" elements is chosen as the minimum possible shift because we must keep
// each row and column 128-bit aligned, as required by
// nvcuda::wmma::load_matrix_sync.
#define SKEW_HALF 16 // 偏移量 防止bank conflicts

#define checkKernelErrors(expr)                             \
  do {                                                      \
    expr;                                                   \
                                                            \
    cudaError_t __err = cudaGetLastError();                 \
    if (__err != cudaSuccess) {                             \
      printf("Line %d: '%s' failed: %s\n", __LINE__, #expr, \
             cudaGetErrorString(__err));                    \
      abort();                                              \
    }                                                       \
  } while (0)

using namespace nvcuda;

__host__ void init_host_matrices(float *a, float *b, float *c) {
  for (int i = 0; i < M_GLOBAL; i++) {
    for (int j = 0; j < K_GLOBAL; j++) {
      a[i * K_GLOBAL + j] = static_cast<float>(rand() % 3);
    }
  }

  for (int i = 0; i < N_GLOBAL; i++) {
    for (int j = 0; j < K_GLOBAL; j++) {
      b[i * K_GLOBAL + j] = static_cast<float>(rand() % 3);
    }
  }

  for (int t = 0; t < M_GLOBAL * N_GLOBAL; t++) {
    c[t] = static_cast<float>(rand() % 3);
  }
}

__global__ void init_device_matrices(const float *A_h, const float *B_h,
                                     const float *C_h, half *A, half *B,
                                     float *C, float *D) {
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < M_GLOBAL * K_GLOBAL;
       i += gridDim.x * blockDim.x)
    A[i] = __float2half(A_h[i]);

  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < N_GLOBAL * K_GLOBAL;
       i += gridDim.x * blockDim.x)
    B[i] = __float2half(B_h[i]);

  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < M_GLOBAL * N_GLOBAL;
       i += gridDim.x * blockDim.x)
    C[i] = C_h[i];

  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < M_GLOBAL * N_GLOBAL;
       i += gridDim.x * blockDim.x)
    D[i] = 0;
}

__global__ void compute_gemm(const half *A, const half *B, const float *C,
                             float *D, float alpha, float beta) {
  // 4 * 16 + 16
  // 每行储存 4 * 16 个 half的数据， 8 * 16 个字节
  // 还有 16的偏移量
  // 256bits(位)对齐，256 = 8 * 32 即32字节对齐: 32 * 5 = 160;
  // sizeof(half) * 5 * 16 = 32 * 5 = 160;
  extern __shared__ half shmem[][CHUNK_K * K + SKEW_HALF];

  // Warp and lane identification.
  const unsigned int warpId = threadIdx.x / WARP_SIZE; // 0 ~ 8
  const unsigned int laneId = threadIdx.x % WARP_SIZE; // 0 ~ 32

  // B 矩阵从 8 * M = 8 * 16 这个地方开始储存
  // Offset in shared memory from which the B matrix is stored.
  const size_t shmem_idx_b_off = BLOCK_COL_TILES * M; // 8 * 16

  // 下面开始的是，每个 warp 将 共享内存中的 128 * 128 的数据，加载到 寄存器中。
  // 每个 warp 负责的数据 8 * 16 * 16 ，后面会提到
  // warp 0 -> (0/2)*8*16*16*2 + (0%2)*8*16 = 0
  // warp 1 -> (1/2)*8*16*16*2 + (1%2)*8*16 = 4 * 16
  // warp 2 -> (2/2)*8*16*16*2 + (2%2)*8*16 = 8 * 16 * 16 * 2 + 0
  // warp 3 -> (3/2)*8*16*16*2 + (3%2)*8*16 = 8 * 16 * 16 * 2 + 4 * 16
  // warp 3 -> (4/2)*8*16*16*2 + (4%2)*8*16 = 8 * 16 * 16 * 4 + 0
  // ...
  // warp 7 -> (7/2)*8*16*16*2 + (7%2)*8*16 = 8 * 16 * 16 * 6 + 4 * 16
  // 以上的可以看出来
  // This pointer is used to access the C and D matrix tiles this warp computes.
  float *shmem_warp_tile_ptr = (float *)&shmem[0][0] +
                               (warpId / 2) * SHMEM_STRIDE * K * 2 +
                               (warpId % 2) * SHMEM_OFFSET;

  // warpId 范围 : 0 ~ 8
  // 每个 warpId 负责 8 * 16 * 16
  // 这样 8 个 warp 可以负责 8 * 8 * 16 * 16
  // 注意： 这个地方被转换为了float，而不是 half
  // This pointer is used to stream the C and D matrices block-wide tile to and
  // from shared memory.
  float *shmem_warp_stream_ptr =
      (float *)&shmem[0][0] + warpId * SHMEM_STRIDE * K;

  // Adjust the beta scaler, as it'll be multiplied by alpha at the end of
  // each tile computation. Technically this is not generally correct (may
  // result in a loss of precision). Zero still needs to be specially handled
  // though.
  beta /= alpha;

  // Each CTA slides along the 128 x 128 tiles from the top left corner of the
  // matrix to the right and down, and selects the next tile to compute. Once
  // there's no such tile, all warps in this CTA exit.
  for (unsigned int block_pos = blockIdx.x;; block_pos += gridDim.x) {
    // 每个小块的大小是 16 * 16
    // 有 256 * 256 个块
    // 现在开始分块处理数据
    // M 、N 、K都是分成了 256 个块, 现在开始加载(i,j)块，(i,j)都在0 - 256之间
    // i = ((block_pos * 8)/256) * 8  // 这个是 8 的倍数，每次增加8，直到 大于 256 处理完成
    // j = (block_pos * 8) %256   // 这个也是 8 的倍数，每次增加8，在 0 ~ 256 之间循环
    // 下面开始枚举block_pos
    //    0  1  2  3  4  5          255  256  257  258
    // i: 0  0  0  0  0  0     ...   0    8    8    8
    // j: 0  8  16 24 32 40    ...  248   0    8    16
    const unsigned int block_tile_i =
        ((block_pos * BLOCK_ROW_TILES) / N_TILES) * (BLOCK_COL_TILES);
    const unsigned int block_tile_j = (block_pos * BLOCK_COL_TILES) % N_TILES;

    // Stop when there are no more D matrix tiles to compute in this CTA.
    if (block_tile_i >= M_TILES) { // 如果多于 256 个块，就处理完了
      break;
    }

    // This warp's pointer to the C matrix data to copy memory from to shared
    // memory.
    // 每个 warp 将 C 矩阵拷贝到共享内存
    // C 矩阵的大小为 4096 * 4096，每个小块 16 * 16 大小，，一共 256 * 256 个小块。
    // (warpId, i, j) 开始加载 C矩阵的 小块数据
    // warpId:0 ~ 8, 不同的warp ，处理不同的行，每行256 个块
    // (warpId, i = i + 8, j): i 增加 8 的情况下，j 增长一轮，配合warp，有 8 * 256 个小块 被处理。
    // (warpId, i, j = j + 8)：i 不变，j 增加 8 的情况下，指针增加一共8个小块的步长：8 * 16，读取 8 个 16 的数据
    // j : 0 ~ 248 ，那么，j 一轮循环过后，256 个块就被加载。

    // 所以，本轮循环内(i, j)固定，8个warp 加载 8个不同行 的，每行 8 个小块。
    /*
          一次加载 矩阵C 128 * 128 大块 的示意图， 下面说的最小单位都是 16 * 16 的小块
          0               --> j                    -> j+8                     256
        0 ----------------------------------------------------------------------
          |                   |                       |                         |
          |                   |                       |                         |
          |                   |                       |                         |
          |                   |        8 x 16         |                         |
     -> i |-------------------|-----------------------|                         |
          |         warpId 0->|--|--|--|--|--|--|--|--|  这里的每个小格子，      |
          |         warpId 1->|--|--|--|--|--|--|--|--|  代表一个小块，16*16     |
          |               .   |--|--|--|--|--|--|--|--|  每轮循环，加载          |
          | warpId        .   |--|--|--|--|--|--|--|--|  8 * 8 个小格子         |
          | 0 ~ 8         .   |--|--|--|--|--|--|--|--|  即 128 * 128 的大块    |
          |       8 x 16  .   |--|--|--|--|--|--|--|--|                         |
          |               .   |--|--|--|--|--|--|--|--|                         |
   -> i+8 |-------------------|-----------------------|                         |
          |                   |                       |                         |
          |                   |                       |                         |
          |                   |                       |                         |
          |                   |                       |                         |
          |                   |                       |                         |
          |                   |                       |                         |
          |                   |                       |                         |
          |                   |                       |                         |
          |                   |                       |                         |
          |                   |                       |                         |
      256 ----------------------------------------------------------------------

      C矩阵的 8 * 8 * 16 * 16，即 8 行 8 列 共64个小块，加载进 shared memory 后的示意图
      |<------------------------------------------ warp 0  8*16*16 ------------------------------->|....  共 8 个 warp
      |<-- 原矩阵第0行 8*16 -->|<-- 原矩阵第1行 8*16 -->||<-- 原矩阵第2行 8*16 -->|...
      ---------------------------------------------------------------------------------------------------------------------
      |                      这里是shared memory 加载进来的 C矩阵的数据                            ..........
      |                      这里是shared memory 加载进来的 C矩阵的数据                            ..........
      ---------------------------------------------------------------------------------------------------------------------
    */

    // 现在该warp要选择哪 8 个 16 * 16的块：(i + warpId) * 16 * 4096 + j * 16
    const size_t gmem_idx =
        (block_tile_i + warpId) * M * GLOBAL_MEM_STRIDE + block_tile_j * N;
    const float *src_gmem_warp_stream_ptr = &C[gmem_idx];

    // Stream multiple C tiles to shared memory.
    // 读入一个块的 C 矩阵数据，C矩阵的数据是 float 的
    // 一个warp 负责 8 * 16 * 16 个float数据的读入
    // 其中 16 行是循环来读入的
#pragma unroll                    // 循环 16 次
    for (int i = 0; i < K; i++) { // 每次8个warp， 一个warp一次一行，每行负责读取 8 * 16 个 float，8 * 16 * 4字节
      typedef int4 copy_t;        // sizeof(int4) = 16 , 所以下面读入 ： 16  * 32  = 8 * 16 * 4 字节

      *((copy_t *)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId) =
          *((copy_t *)(src_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) +
            laneId);
    }

    __syncthreads();

    // 每个warp 都有自己的 c[2][4] 矩阵
    // 所以每个warp 需要加载 8 * 16 * 16
    // 这大小为 2 * 4 = 8的二维数组，是为了加速 A 和 B 矩阵乘法
    // These fragments will accumulate the result of A and B matrix fragment
    // multiplications along the K_GLOBAL dimension.  // 这里需要的数据量 ： 8 * 16 * 16
    wmma::fragment<wmma::accumulator, M, N, K, float> c[WARP_COL_TILES]
                                                       [WARP_ROW_TILES]; // c[2][4]

    // 将C矩阵的块，加载到 数据段中
    // wmma::load_matrix_sync(c, 数据指针，行步长，行主序)
    // 对照上面的 shared memory 数据排列，每一行的数据是 8 * 16，所以行步长是 8 * 16
    // 下面的两层循环，i x j 共 8 次 :  可以加载 8 * 16 * 16 的数据
    // 配合着 shmem_warp_tile_ptr 这个指针，随着 warp Id 也在变化，就可以加载 8 * 8 * 16 * 16 个数据
    /*
      下面是 load_matrix_sync 通过 warp 0， warp 1，配合一个 j 0 ~4 循环，加载 8 * 16 * 16 的示意图

      | j :  0   1   2   3 | j :  0   1   2   3 |
      |<--warp0 4 * 16     |<--warp1 4 * 16     |<-----后面的数据是 load_matrix_sync 设置行步长，自动加载，直到 8 * 16 * 16
      |<------------ 原矩阵第0行 8*16 ---------->|<------------ 原矩阵第1行 8*16 ---------->|...
      ---------------------------------------------------------------------------------------------------------------------
      |                      这里是shared memory 加载进来的 C矩阵的数据                            ..........
      |                      这里是shared memory 加载进来的 C矩阵的数据                            ..........
      ---------------------------------------------------------------------------------------------------------------------
    */
    // Load the C matrix tiles into fragments from shared memory.
#pragma unroll
    for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
      for (int j = 0; j < WARP_ROW_TILES; j++) {
        const float *tile_ptr =
            shmem_warp_tile_ptr + i * SHMEM_STRIDE * K + j * N;

        // 这个地方，按照道理来讲，会有bank conflicts 吧，共享内存偏移，只是给 A B的加载用的，现在是加载C
        wmma::load_matrix_sync(c[i][j], tile_ptr, SHMEM_STRIDE, C_LAYOUT);
      }
    }

    __syncthreads();

    // Scale the C matrix.
#pragma unroll
    for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
      for (int j = 0; j < WARP_ROW_TILES; j++) {
#pragma unroll
        for (int t = 0; t < c[i][j].num_elements; t++) {
          c[i][j].x[t] *= beta;
        }
      }
    }

    // 将 矩阵 A 和 矩阵 B拷贝到 共享内存
    // warps 0-3 负责 A， warps 4-7 负责 B
    // i: 0 - 256 来遍历 A 的第一个维度， j: 0 - 256 来遍历 B 的第一个维度， 步长为 8
    // 所以在 一轮 (i, j) 中，需要 处理 A、B 的 8 行/列
    // 由于 A、B 各 4 个 warp ，所以 每个 warp 要负责拷贝 2 行/列

    // 以矩阵 A 为例
    //                i 每次加 8 直到 >= 256              warpId 0/1/2/3
    //                      i     * 16 * 4096     + 16 * 4096 * warpId * 2
    // warp_ptr = &A[block_tile_i * M * K_GLOBAL] + M * K_GLOBAL * (warpId % 4) * 2
    // 故 warp 0 和 warp 1 相差 2 * 16 * 4096， 即 2 * 16 * 16 * 256,
    // 每个warp 负责 拷贝  2 * 16 = 32 行
    // 4 个 warp 负责 8 * 16 行
    // Select what warp copies what matrix to shared memory.
    // Warps 0-3 copy the A matrix, warps 4-7 copy the B matrix.
    const half *warp_ptr = (warpId < 4) ? (&A[block_tile_i * M * K_GLOBAL] +
                                           M * K_GLOBAL * (warpId % 4) * 2)
                                        : (&B[block_tile_j * N * K_GLOBAL] +
                                           N * K_GLOBAL * (warpId % 4) * 2);

    // Go through the global K dimension by a fixed step at a time.
    // 以步长 4 来 遍历 256(m,n,k中的k) 这个维度
#pragma unroll
    for (int tile_k = 0; tile_k < K_TILES; tile_k += CHUNK_K) {
      // Copy slices of the A and B matrices to shared memory.
      // The first half of the warps in the CTA copy the A matrix, the rest copy
      // the B matrix.
      // 下面开始计算，拷贝 矩阵 A 到 共享内存的 第一维
      //  M * (warpId % (WARPS_PER_BLOCK / 2)) * 2 = 16 * (warpId % 4) * 2
      //  shmem_idx -> warp 0: 0 * 16,   warp 1: 2 * 16,    warp 2: 4 * 16,   warp 3: 6 * 16
      // 论行，显然，每个 warp 负责加载  2 * 16 的数据，共占  8 * 16
      // 所以 A、B的每行，就拷贝到 shmem 每行里面放着
      // 矩阵B 在共享内存中的起始地址，就是在 A 的基础上 偏移 shmem_idx_b_off = 8 * 16
      size_t shmem_idx =
          warpId < (WARPS_PER_BLOCK / 2)
              ? (M * (warpId % (WARPS_PER_BLOCK / 2)) * 2)
              : (N * (warpId % (WARPS_PER_BLOCK / 2)) * 2 + shmem_idx_b_off);

      // First half of the warp copies the first row / column of the matrix,
      // the second half of the warp copies the next.
      //                                      0  ~  4            0  ~  8
      // lane_ptr = warp_ptr + tile_k * 16 + (laneId/8) * 4096 + (int4*)(laneId%8)
      //          = warp_ptr + (laneId/8) * 4096 + tile_k * 16 + (int4*)(laneId%8)
      //                       -----------------  ------------          ----------
      //                              ①                 ②                   ③
      // ① (laneId/8) * 4096 ： 4 * 4096, 4行, 不同的lane 间隔是一行
      // ② tile_k * 16 : tile_k 0 ~ 256 , 以 4 递增，则每一轮 256 * 16 = 4096，即 一小行， 不同的tile_k, 间隔是 4 * 16 = 64
      // ③ laneId%8 :  0 ~ 8 * sizeof(int4)/sizeof(half) = 8 * 8 = 64。
      // 当 tile_k 确定的时候， 一个lane 是可以加载 4 行中，每行的 4 * 16 个 half
      int4 *lane_ptr = (int4 *)(warp_ptr + tile_k * K +
                                (laneId / CHUNK_COPY_LINE_LANES) * K_GLOBAL) +
                       (laneId % CHUNK_COPY_LINE_LANES);

      // shmem_idx ：shared memory
      // lane_ptr : global memory

      // Shift the second half of the warp to the next row / column in the
      // shared memory.
      // 每 8 个 lane 加载 8 * sizeof(int4) / sizeof(half) = 8 * 16 / 2 = 4 * 16 ，就是一行的数据
      // 故，一个warp 填充 4 行，每行 4 * 16 个 half 的数据
      shmem_idx += laneId / CHUNK_COPY_LINE_LANES;
      // 下面再循环 8 次，处理 8 * 4 = 32 = 2 * 16 行 的 数据

#pragma unroll
      for (int i = 0; i < ((WARP_SIZE / 2) / CHUNK_COPY_LINES_PER_WARP) * 2;
           i++) { // 这里有 8 个循环
        // 每个循环，shmem 填充 4 行，每行 4 * 16 个 half
        // Copy 16 bytes at once in each lane.
        *((int4 *)&shmem[shmem_idx][0] + (laneId % CHUNK_COPY_LINE_LANES)) =
            *lane_ptr;

        // Advance the global memory pointer and the shared memory index.
        lane_ptr = // 偏移到下一个 4 行，循环 8 次， 那么一个 warp 读入 8 * 4 = 32 行，即 一个warp 处理 2 * 16行的数据
            (int4 *)((half *)lane_ptr + K_GLOBAL * CHUNK_COPY_LINES_PER_WARP); // lane_ptr + 4 * 4096 half
        shmem_idx += CHUNK_COPY_LINES_PER_WARP; // shmem_idx += 4;
        // 一轮下来，shared memory 一行填充了 4 * 16 个half，填充了 4 行，这是同一 warp 内。
        // 8轮下来，填充了 8 * 4 行，一共 8 * 4 * 4 * 16 个half ，= 8 * 16 * 16 half ，这是一个warp 处理的一个 矩阵。
      }

      // 此处同步
      // shared memory 中，存放了 A、B各 8 * 16 行，每行 4 * 16 half的数据
      //
      __syncthreads();

      // Compute a grid of C matrix tiles in each warp.
      // 开始将 A、B 从 shared memory 加载到 wmma
      // 现在 需要处理的数据  A : 8 * 16 * 4 * 16, B : 8 * 16 * 4 * 16 ， 即 A 有 32 个块，B 有 32 个块
      // 一次 wmma 调用 可以处理 一个块 16 * 16
#pragma unroll
      for (int k_step = 0; k_step < CHUNK_K; k_step++) {
        wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major>
            a[WARP_COL_TILES]; // a[2] // 每个 warp 都有一个 ，所以有 2 * 4 (A的warp) = 8 个
        wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major>
            b[WARP_ROW_TILES]; // b[4]

#pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i++) { // 2            // 遍历 A 的 8 这个维度
          size_t shmem_idx_a = (warpId / 2) * M * 2 + (i * M);     // shmem_idx_a = (warpId / 2) * 16 * 2 + (i * 16)
          const half *tile_ptr = &shmem[shmem_idx_a][k_step * K];  // 取得 A 的 (8 * 16) * (4 * 16) 中的某一个 (16 * 16)

          wmma::load_matrix_sync(a[i], tile_ptr, K * CHUNK_K + SKEW_HALF);

#pragma unroll
          for (int j = 0; j < WARP_ROW_TILES; j++) {
            if (i == 0) {
              // Load the B matrix fragment once, because it is going to be
              // reused against the other A matrix fragments.
              size_t shmem_idx_b = shmem_idx_b_off +  // 遍历 B 的 8 这个维度
                                   (WARP_ROW_TILES * N) * (warpId % 2) +  // （4 * 16） * （0 , 1） + (j * 16)
                                   (j * N);
              const half *tile_ptr = &shmem[shmem_idx_b][k_step * K];

              wmma::load_matrix_sync(b[j], tile_ptr, K * CHUNK_K + SKEW_HALF); // 加载 A 的 某个 16 * 16
            }

            wmma::mma_sync(c[i][j], a[i], b[j], c[i][j]); // 大功告成
          }
        }
      }

      __syncthreads();
    }

      // Store the D fragments to shared memory.
#pragma unroll
    for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
      for (int j = 0; j < WARP_ROW_TILES; j++) {
#pragma unroll
        // Uniform, point-wise transformations of ALL fragment elements by ALL
        // threads in the warp are well-defined even though element indices
        // within fragment storage are not defined.
        for (int t = 0; t < c[i][j].num_elements; t++) c[i][j].x[t] *= alpha;

        float *tile_ptr = shmem_warp_tile_ptr + i * SHMEM_STRIDE * K + j * N; // shmem_warp_tile_ptr + (0 ~ 2) * 8 * 16 * 16 + (0 ~ 4) * 16

        wmma::store_matrix_sync(tile_ptr, c[i][j], SHMEM_STRIDE, C_LAYOUT); // 步长 8 * 16
      }
    }

    __syncthreads();

    // Now that shared memory contains all the D tiles, stream them to global
    // memory.
    float *dst_gmem_warp_stream_ptr = &D[gmem_idx];

#pragma unroll
    for (int i = 0; i < K; i++) {
      *((int4 *)(dst_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) + laneId) =
          *((int4 *)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId);
    }

    __syncthreads();
  }
}

// Performs an MxNxK GEMM (C=alpha*A*B + beta*C) assuming:
//  1) Matrices are packed in memory.
//  2) M, N and K are multiples of 16.
//  3) Neither A nor B are transposed.
// Note: This is a less performant version of the compute_gemm kernel. It is
// designed for
//       demonstration purposes only to show the CUDA WMMA API use without
//       relying on availability of the shared memory.
__global__ void simple_wmma_gemm(half *a, half *b, float *c, float *d, int m_ld,
                                 int n_ld, int k_ld, float alpha, float beta) {
  // Leading dimensions. Packed with no transpositions.
  int lda = m_ld;
  int ldb = k_ld;
  int ldc = n_ld;

  // Tile using a 2D grid
  int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
  int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

  // Declare the fragments
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>
      a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major>
      b_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

  wmma::fill_fragment(acc_frag, 0.0f);

  // Loop over k
  for (int i = 0; i < k_ld; i += WMMA_K) {
    int aCol = i;
    int aRow = warpM * WMMA_M;

    int bCol = i;
    int bRow = warpN * WMMA_N;

    // Bounds checking
    if (aRow < m_ld && aCol < k_ld && bRow < k_ld && bCol < n_ld) {
      // Load the inputs
      wmma::load_matrix_sync(a_frag, a + aCol + aRow * lda, lda);
      wmma::load_matrix_sync(b_frag, b + bCol + bRow * ldb, ldb);

      // Perform the matrix multiplication
      wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
  }

  // Load in the current value of c, scale it by beta, and add this our result
  // scaled by alpha
  int cCol = warpN * WMMA_N;
  int cRow = warpM * WMMA_M;

  if (cRow < m_ld && cCol < n_ld) {
    wmma::load_matrix_sync(c_frag, c + cCol + cRow * ldc, ldc,
                           wmma::mem_row_major);

    for (int i = 0; i < c_frag.num_elements; i++) {
      c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
    }

    // Store the output
    wmma::store_matrix_sync(d + cCol + cRow * ldc, c_frag, ldc,
                            wmma::mem_row_major);
  }
}

__host__ void matMultiplyOnHost(float *A, float *B, float *C, float alpha,
                                float beta, int numARows, int numAColumns,
                                int numBRows, int numBColumns, int numCRows,
                                int numCColumns) {
  for (int i = 0; i < numCRows; i++) {
    for (int j = 0; j < numCColumns; j++) {
      float temp = 0.0;

      for (int k = 0; k < numAColumns; k++) {
        temp += A[i * numAColumns + k] * B[j * numBRows + k];
      }

      C[i * numCColumns + j] = temp * alpha + beta * C[i * numCColumns + j];
    }
  }
}

int main(int argc, char **argv) {
  printf("Initializing...\n");

  int dev = findCudaDevice(argc, (const char **)argv);

  cudaDeviceProp deviceProp;
  checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));

  // Tensor cores require a GPU of Volta (SM7X) architecture or higher.
  if (deviceProp.major < 7) {
    printf(
        "cudaTensorCoreGemm requires requires SM 7.0 or higher to use Tensor "
        "Cores.  Exiting...\n");
    exit(EXIT_WAIVED);
  }

  printf("M: %d (%d x %d)\n", M_GLOBAL, M, M_TILES);
  printf("N: %d (%d x %d)\n", N_GLOBAL, N, N_TILES);
  printf("K: %d (%d x %d)\n", K_GLOBAL, K, K_TILES);

  float *A_h = NULL;
  float *B_h = NULL;
  float *C_h = NULL;
#if CPU_DEBUG
  float *result_hD = NULL;
  float *result_host = NULL;
#endif

  checkCudaErrors(cudaMallocManaged(reinterpret_cast<void **>(&A_h),
                                    sizeof(float) * M_GLOBAL * K_GLOBAL));
  checkCudaErrors(cudaMallocManaged(reinterpret_cast<void **>(&B_h),
                                    sizeof(float) * K_GLOBAL * N_GLOBAL));
  checkCudaErrors(cudaMallocManaged(reinterpret_cast<void **>(&C_h),
                                    sizeof(float) * M_GLOBAL * N_GLOBAL));
#if CPU_DEBUG
  checkCudaErrors(cudaMallocManaged((void **)&result_hD,
                                    sizeof(float) * M_GLOBAL * N_GLOBAL));
  checkCudaErrors(cudaMallocManaged((void **)&result_host,
                                    sizeof(float) * M_GLOBAL * N_GLOBAL));
#endif

  half *A = NULL;
  half *B = NULL;
  float *C = NULL;
  float *D = NULL;

  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&A),
                             sizeof(half) * M_GLOBAL * K_GLOBAL));
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&B),
                             sizeof(half) * N_GLOBAL * K_GLOBAL));
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&C),
                             sizeof(float) * M_GLOBAL * N_GLOBAL));
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&D),
                             sizeof(float) * M_GLOBAL * N_GLOBAL));

  assert(((unsigned long long)A) % 128 == 0);
  assert(((unsigned long long)B) % 128 == 0);
  assert(((unsigned long long)C) % 128 == 0);
  assert(((unsigned long long)D) % 128 == 0);

  init_host_matrices(A_h, B_h, C_h);

  printf("Preparing data for GPU...\n");

  checkKernelErrors(
      (init_device_matrices<<<deviceProp.multiProcessorCount,
                              THREADS_PER_BLOCK>>>(A_h, B_h, C_h, A, B, C, D)));

  checkCudaErrors(cudaDeviceSynchronize());

  enum {
    // Compute the right amount of shared memory to request.
    // We need shared memory to hold per-CTA C and D matrix tiles, and to cache
    // per-CTA chunks
    // of the A and B matrices. Therefore, the right amount to request is the
    // maximum of those
    // two numbers.
    SHMEM_SZ = MAX(
        sizeof(half) * (BLOCK_COL_TILES * M) * (CHUNK_K * K + SKEW_HALF) * 2,
        M * (BLOCK_ROW_WARPS * WARP_ROW_TILES) * N *
            (BLOCK_COL_WARPS * WARP_COL_TILES) * sizeof(float))
  };

  printf("Required shared memory size: %lu Kb\n", SHMEM_SZ / 1024UL);

  const float alpha = 1.1f;
  const float beta = 1.2f;

  cudaEvent_t start, stop;

  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));
  checkCudaErrors(cudaEventRecord(start));

  // If enough shared memory available on the GPU use high performant kernel
  if (deviceProp.sharedMemPerMultiprocessor >= SHMEM_SZ) {
    printf("Computing... using high performance kernel compute_gemm \n");

    checkCudaErrors(cudaFuncSetAttribute(
        compute_gemm, cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ));
    checkKernelErrors(
        (compute_gemm<<<deviceProp.multiProcessorCount, THREADS_PER_BLOCK,
                        SHMEM_SZ>>>(A, B, C, D, alpha, beta)));
#if CPU_DEBUG
    checkCudaErrors(cudaMemcpy(result_hD, D,
                               sizeof(float) * M_GLOBAL * N_GLOBAL,
                               cudaMemcpyDeviceToHost));
#endif
  } else {
    dim3 gridDim;
    dim3 blockDim;

    // blockDim.x must be a multple of warpSize
    // 128x4 means we have 16 warps and a block computes a 64x64 output tile
    blockDim.x = 128;
    blockDim.y = 4;

    gridDim.x = (M_GLOBAL + (WMMA_M * blockDim.x / 32 - 1)) /
                (WMMA_M * blockDim.x / 32);
    gridDim.y = (N_GLOBAL + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);

    printf("Computing... using simple_wmma_gemm kernel\n");
    simple_wmma_gemm<<<gridDim, blockDim>>>(A, B, C, D, M_GLOBAL, N_GLOBAL,
                                            K_GLOBAL, alpha, beta);
#if CPU_DEBUG
    checkCudaErrors(cudaMemcpy(result_hD, D,
                               sizeof(float) * M_GLOBAL * N_GLOBAL,
                               cudaMemcpyDeviceToHost));
#endif
  }

  checkCudaErrors(cudaEventRecord(stop));
  checkCudaErrors(cudaEventSynchronize(stop));

#if CPU_DEBUG
  printf("Verifying correctness of the computations...\n");

  memcpy(result_host, C_h, sizeof(float) * M_GLOBAL * N_GLOBAL);

  matMultiplyOnHost(A_h, B_h, result_host, alpha, beta, M_GLOBAL, K_GLOBAL,
                    K_GLOBAL, N_GLOBAL, M_GLOBAL, N_GLOBAL);

  for (int i = 0; i < N_GLOBAL * M_GLOBAL; i++) {
    if (fabs(result_hD[i] - result_host[i]) > 0.1f)
      printf("mismatch i=%d result_hD=%f result_host=%f\n", i, result_hD[i],
             result_host[i]);
  }
#endif

  float milliseconds = 0;

  checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));

  printf("Time: %f ms\n", milliseconds);
  printf("TFLOPS: %.2f\n", static_cast<double>((static_cast<double>(M_GLOBAL) *
                                                N_GLOBAL * K_GLOBAL * 2) /
                                               (milliseconds / 1000.)) /
                               1e12);

  checkCudaErrors(cudaFree(reinterpret_cast<void *>(A_h)));
  checkCudaErrors(cudaFree(reinterpret_cast<void *>(B_h)));
  checkCudaErrors(cudaFree(reinterpret_cast<void *>(C_h)));
  checkCudaErrors(cudaFree(reinterpret_cast<void *>(A)));
  checkCudaErrors(cudaFree(reinterpret_cast<void *>(B)));
  checkCudaErrors(cudaFree(reinterpret_cast<void *>(C)));
  checkCudaErrors(cudaFree(reinterpret_cast<void *>(D)));

  return 0;
}
