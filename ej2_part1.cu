/**
 * @file ej2_part1.cu
 * @brief Exercise 2 Part I: Shared-memory tiled transpose kernel without padding.
 *
 * Implements a tiled transpose using shared memory to improve global memory coalescing.
 * No padding is used, so potential bank conflicts may occur.
 * Measures execution time over multiple runs and verifies correctness.
 *
 * Usage:
 *   ./ej2_part1 [rows cols [blockX blockY]]
 * Defaults: rows=1024, cols=1024, blockX=32, blockY=32.
 */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

#define MAX_ROWS 4096
#define MAX_COLS 4096

#define CUDA_CHK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


/**
 * @brief Shared-memory tiled transpose kernel (no padding).
 *
 * Loads a tile of size blockX x blockY into shared memory and writes it transposed.
 * Shared memory layout:
 *   tile[ty * blockX + tx] stores element (row,col).
 *   After sync, write tile[tx * blockX + ty] back to out.
 *
 * @param in    Input matrix in row-major order (rows x cols).
 * @param out   Output matrix in row-major order (cols x rows).
 * @param rows  Number of rows in input.
 * @param cols  Number of columns in input.
 */
__global__ void transposeSharedNoPad(const int* in, int* out, int rows, int cols) {
    extern __shared__ int tile[];
    int blockX = blockDim.x;
    int blockY = blockDim.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * blockY + ty;
    int col = blockIdx.x * blockX + tx;

    // Phase 1: Load to shared memory
    if (row < rows && col < cols) {
        tile[ty * blockX + tx] = in[row * cols + col];
    }
    __syncthreads();

    // Phase 2: Write transposed from shared memory
    int trow = blockIdx.x * blockX + ty;
    int tcol = blockIdx.y * blockY + tx;
    if (trow < cols && tcol < rows) {
        out[trow * rows + tcol] = tile[tx * blockX + ty];
    }
}

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
    std::cout << "Ej2 Part1: Matrix " << rows << "x" << cols
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

    int *d_in = nullptr, *d_out = nullptr;
    CUDA_CHK(cudaMalloc(&d_in, bytes));
    CUDA_CHK(cudaMalloc(&d_out, bytes));
    CUDA_CHK(cudaMemcpy(d_in, &h_in[0][0], bytes, cudaMemcpyHostToDevice));

    dim3 blockDim(blockX, blockY);
    dim3 gridDim((cols + blockX - 1) / blockX,
                 (rows + blockY - 1) / blockY);

    cudaEvent_t start, stop;
    CUDA_CHK(cudaEventCreate(&start));
    CUDA_CHK(cudaEventCreate(&stop));

    const int iterations = 10;
    std::vector<float> times(iterations);

    // Warm-up
    size_t sbytes = blockX * blockY * sizeof(int);
    transposeSharedNoPad<<<gridDim, blockDim, sbytes>>>(d_in, d_out, rows, cols);
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaDeviceSynchronize());

    // Timed runs
    for (int i = 0; i < iterations; ++i) {
        CUDA_CHK(cudaEventRecord(start));
        transposeSharedNoPad<<<gridDim, blockDim, sbytes>>>(d_in, d_out, rows, cols);
        CUDA_CHK(cudaGetLastError());
        CUDA_CHK(cudaEventRecord(stop));
        CUDA_CHK(cudaEventSynchronize(stop));
        CUDA_CHK(cudaEventElapsedTime(&times[i], start, stop));
    }

    float sum = 0.0f;
    for (float t : times) sum += t;
    float avg = sum / iterations;
    float sq = 0.0f;
    for (float t : times) sq += (t - avg) * (t - avg);
    float stddev = std::sqrt(sq / iterations);

    std::cout << "Average time: " << avg << " ms ± " << stddev << " ms" << std::endl;

    CUDA_CHK(cudaMemcpy(&h_out[0][0], d_out, bytes, cudaMemcpyDeviceToHost));
    bool ok = true;
    for (int r = 0; r < rows && ok; ++r) {
        for (int c = 0; c < cols && ok; ++c) {
            if (h_out[c][r] != h_in[r][c]) ok = false;
        }
    }
    std::cout << "TransposeSharedNoPad " << (ok ? "PASSED" : "FAILED") << std::endl;

    CUDA_CHK(cudaFree(d_in));
    CUDA_CHK(cudaFree(d_out));
    CUDA_CHK(cudaEventDestroy(start));
    CUDA_CHK(cudaEventDestroy(stop));
    return ok ? 0 : 1;
} 