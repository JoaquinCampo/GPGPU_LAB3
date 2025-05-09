/**
 * @file ej2_part2.cu
 * @brief Exercise 2 Part II: Shared-memory tiled transpose kernel with padding.
 *
 * Implements a tiled transpose using shared memory with an extra column (padding)
 * to avoid bank conflicts. Measures execution time and verifies correctness.
 *
 * Usage:
 *   ./ej2_part2 [rows cols [blockX blockY]]
 * Defaults: rows=1024, cols=1024, blockX=32, blockY=32.
 */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <nvtx3/nvToolsExt.h>

#define MAX_ROWS 4096
#define MAX_COLS 4096

#define CUDA_CHK(ans) do { gpuAssert((ans), __FILE__, __LINE__); } while(0)
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


/**
 * @brief Shared-memory tiled transpose kernel with padding.
 *
 * Loads a tile of size blockX x blockY into shared memory with stride (blockX+1)
 * to prevent bank conflicts, then writes the tile transposed back to global memory.
 *
 * @param in    Input matrix in row-major order (rows x cols).
 * @param out   Output matrix in row-major order (cols x rows).
 * @param rows  Number of rows in input matrix.
 * @param cols  Number of columns in input matrix.
 */
__global__ void transposeSharedPad(const int* in, int* out, int rows, int cols) {
    extern __shared__ int tile[]; // shared memory sized (blockX+1)*blockY
    int blockX = blockDim.x;
    int blockY = blockDim.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * blockY + ty;
    int col = blockIdx.x * blockX + tx;

    int stride = blockX + 1; // padded stride
    // Phase 1: Load to shared memory with padding
    if (row < rows && col < cols) {
        tile[ty * stride + tx] = in[row * cols + col];
    }
    __syncthreads();

    // Phase 2: Write transposed from shared memory
    int trow = blockIdx.x * blockX + ty;
    int tcol = blockIdx.y * blockY + tx;
    if (trow < cols && tcol < rows) {
        out[trow * rows + tcol] = tile[tx * stride + ty];
    }
}


/**
 * @brief Host entry point for shared-memory transpose test.
 *
 * - Parses optional command-line arguments for rows, cols, blockX and blockY.
 * - Allocates host and device buffers and initializes input matrix with linear values.
 * - Launches transposeSharedPad kernel with dynamic shared memory.
 * - Measures execution time over multiple iterations using CUDA events.
 * - Computes average and standard deviation of the runs.
 * - Verifies correctness by comparing output matrix to expected transpose.
 *
 * Usage:
 *   ./ej2_part2 [rows cols [blockX blockY]]
 * Defaults: rows=1024, cols=1024, blockX=32, blockY=32.
 */
int main(int argc, char* argv[]) {
    int rows = 1024, cols = 1024;
    int blockX = 32, blockY = 32;
    if (argc >= 3) {
        rows = std::atoi(argv[1]);
        cols = std::atoi(argv[2]);
    }
    if (argc >= 5) {
        blockX = std::atoi(argv[3]);
        blockY = std::atoi(argv[4]);
    }
    if (rows > MAX_ROWS || cols > MAX_COLS) {
        std::cerr << "Error: Matrix size exceeds MAX_ROWS or MAX_COLS." << std::endl;
        return 1;
    }
    std::cout << "Ej2 Part2: Matrix " << rows << "x" << cols
              << ", Block " << blockX << "x" << blockY << std::endl;

    size_t size = static_cast<size_t>(rows) * cols;
    size_t bytes = size * sizeof(int);

    int h_in[MAX_ROWS][MAX_COLS];
    int h_out[MAX_ROWS][MAX_COLS];
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            h_in[i][j] = i * cols + j;
        }
    }

    // Allocate device buffers
    int *d_in = nullptr, *d_out = nullptr;
    nvtxRangePushA("Malloc in");
    CUDA_CHK(cudaMalloc(&d_in, bytes));
    nvtxRangePop();

    nvtxRangePushA("Malloc out");
    CUDA_CHK(cudaMalloc(&d_out, bytes));
    nvtxRangePop();

    nvtxRangePushA("H2D memcpy");
    CUDA_CHK(cudaMemcpy(d_in, &h_in[0][0], bytes, cudaMemcpyHostToDevice));
    nvtxRangePop();

    dim3 blockDim(blockX, blockY);
    // Calculate how many blocks are needed in each dimension
    int remainder_x = cols % blockDim.x;
    int remainder_y = rows % blockDim.y;
    int numBlocksX = cols / blockDim.x + (remainder_x > 0 ? 1 : 0);
    int numBlocksY = rows / blockDim.y + (remainder_y > 0 ? 1 : 0);
    dim3 gridDim(numBlocksX, numBlocksY);

    // 5) Launch kernel once
    size_t sharedBytes = (blockX + 1) * blockY * sizeof(int);
    nvtxRangePushA("Kernel launch");
    transposeSharedPad<<<gridDim, blockDim, sharedBytes>>>(d_in, d_out, rows, cols);
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaDeviceSynchronize());
    nvtxRangePop();

    // 6) Copy back & verify correctness
    nvtxRangePushA("D2H memcpy");
    CUDA_CHK(cudaMemcpy(&h_out[0][0], d_out, bytes, cudaMemcpyDeviceToHost));
    nvtxRangePop();
    bool ok = true;
    for (int r = 0; r < rows && ok; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (h_out[c][r] != h_in[r][c]) {
                std::cerr << "FAILED at (" << r << "," << c << "): "
                          << h_out[c][r] << " != " << h_in[r][c] << "\n";
                ok = false;
                break;
            }
        }
    }
    std::cout << (ok ? "Transpose OK\n" : "Transpose FAILED\n");

    // 7) Cleanup
    CUDA_CHK(cudaFree(d_in));
    CUDA_CHK(cudaFree(d_out));
    return ok ? 0 : 2;
} 