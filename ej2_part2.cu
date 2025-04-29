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
    std::cout << "Ej2 Part2: Matrix " << rows << "x" << cols
              << ", Block " << blockX << "x" << blockY << std::endl;

    size_t size = static_cast<size_t>(rows) * cols;
    size_t bytes = size * sizeof(int);

    // Allocate host buffers
    std::vector<int> h_in(size), h_out(size);
    for (size_t i = 0; i < size; ++i) h_in[i] = static_cast<int>(i);

    // Allocate device buffers
    int *d_in = nullptr, *d_out = nullptr;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice);

    dim3 blockDim(blockX, blockY);
    dim3 gridDim((cols + blockX - 1) / blockX,
                 (rows + blockY - 1) / blockY);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int iterations = 10;
    std::vector<float> times(iterations);

    // Warm-up
    size_t sharedBytes = (blockX + 1) * blockY * sizeof(int);
    transposeSharedPad<<<gridDim, blockDim, sharedBytes>>>(d_in, d_out, rows, cols);
    cudaDeviceSynchronize();

    // Timed runs
    for (int i = 0; i < iterations; ++i) {
        cudaEventRecord(start);
        transposeSharedPad<<<gridDim, blockDim, sharedBytes>>>(d_in, d_out, rows, cols);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&times[i], start, stop);
    }

    // Compute statistics
    float sum = 0.f;
    for (float t : times) sum += t;
    float avg = sum / iterations;
    float sq = 0.f;
    for (float t : times) sq += (t - avg) * (t - avg);
    float stddev = std::sqrt(sq / iterations);

    std::cout << "Average time: " << avg << " ms ± " << stddev << " ms" << std::endl;

    // Verify correctness
    cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost);
    bool ok = true;
    for (int r = 0; r < rows && ok; ++r) {
        for (int c = 0; c < cols && ok; ++c) {
            if (h_out[c * rows + r] != h_in[r * cols + c]) ok = false;
        }
    }
    std::cout << "TransposeSharedPad " << (ok ? "PASSED" : "FAILED") << std::endl;

    // Cleanup
    cudaFree(d_in);
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ok ? 0 : 1;
} 