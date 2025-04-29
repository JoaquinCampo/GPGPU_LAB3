/**
 * @file ej1_part2.cu
 * @brief GPU matrix transpose using shared memory tile for coalesced access.
 *
 * Implements a tiled transpose kernel that loads a block of size blockX x blockY
 * into shared memory, then writes it transposed back to global memory.
 * Host code measures execution time over multiple runs and verifies correctness.
 *
 * Usage:
 *   ./programa_part2 <rows> <cols> <blockX> <blockY>
 * Default: 1024x1024 matrix, 32x32 block.
 */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

/**
 * @brief Shared-memory tiled transpose kernel (no padding).
 *
 * Each thread block loads a tile of size blockX x blockY from global memory
 * into shared memory, then writes the tile transposed to the output matrix.
 * This optimizes coalesced reads and writes by accessing shared memory for the transpose.
 *
 * Shared memory layout:
 *   tile[ty * blockX + tx] holds element at (row, col) for this thread.
 * Transposed write uses tile[tx * blockX + ty].
 *
 * @param in    Input matrix in global memory (rows x cols, row-major).
 * @param out   Output matrix in global memory (cols x rows, row-major).
 * @param rows  Number of rows in the input matrix.
 * @param cols  Number of columns in the input matrix.
 */
__global__ void transposeShared(const int* in, int* out, int rows, int cols) {
    extern __shared__ int tile[]; // dynamic shared memory array of size blockX*blockY
    int blockX = blockDim.x;
    int blockY = blockDim.y;
    int tx = threadIdx.x;  // thread's x coordinate in block
    int ty = threadIdx.y;  // thread's y coordinate in block
    int row = blockIdx.y * blockY + ty;
    int col = blockIdx.x * blockX + tx;

    // Phase 1: Load global memory into shared tile
    if (row < rows && col < cols) {
        tile[ty * blockX + tx] = in[row * cols + col];
    }
    __syncthreads();

    // Phase 2: Write tile transposed back to global memory
    int trow = blockIdx.x * blockX + ty;
    int tcol = blockIdx.y * blockY + tx;
    if (trow < cols && tcol < rows) {
        out[trow * rows + tcol] = tile[tx * blockX + ty];
    }
}

/**
 * @brief Host entry point for shared-memory transpose test.
 *
 * - Parses optional command-line arguments for rows, cols, blockX and blockY.
 * - Allocates host and device buffers and initializes input matrix with linear values.
 * - Launches transposeShared kernel with dynamic shared memory.
 * - Measures execution time over multiple iterations using CUDA events.
 * - Computes average and standard deviation of the runs.
 * - Verifies correctness by comparing output matrix to expected transpose.
 *
 * Usage:
 *   ./programa_part2 [rows cols [blockX blockY]]
 * Defaults: rows=1024, cols=1024, blockX=32, blockY=32.
 *
 * @param argc  Argument count.
 * @param argv  Argument vector: argv[1]=rows, argv[2]=cols, argv[3]=blockX, argv[4]=blockY.
 * @return      Zero on success, non-zero on failure.
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
    std::cout << "Matrix: " << rows << "x" << cols
              << ", Block: " << blockX << "x" << blockY << std::endl;

    size_t size = static_cast<size_t>(rows) * cols;
    size_t bytes = size * sizeof(int);

    // Allocate and initialize host buffers
    std::vector<int> h_in(size), h_out(size);
    for (size_t i = 0; i < size; ++i) {
        h_in[i] = static_cast<int>(i);
    }

    // Allocate device buffers
    int *d_in = nullptr, *d_out = nullptr;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice);

    // Compute grid dimensions
    dim3 blockDim(blockX, blockY);
    dim3 gridDim((cols + blockX - 1) / blockX,
                 (rows + blockY - 1) / blockY);

    // Prepare CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int iterations = 10;
    std::vector<float> times(iterations);

    // Warm-up run
    size_t sharedBytes = blockX * blockY * sizeof(int);
    transposeShared<<<gridDim, blockDim, sharedBytes>>>(d_in, d_out, rows, cols);
    cudaDeviceSynchronize();

    // Timed iterations
    for (int i = 0; i < iterations; ++i) {
        cudaEventRecord(start);
        transposeShared<<<gridDim, blockDim, sharedBytes>>>(d_in, d_out, rows, cols);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&times[i], start, stop);
    }

    // Compute statistics
    float sum = 0.0f;
    for (float t : times) sum += t;
    float avg = sum / iterations;
    float sq_sum = 0.0f;
    for (float t : times) sq_sum += (t - avg) * (t - avg);
    float stddev = std::sqrt(sq_sum / iterations);
    std::cout << "Average: " << avg << " ms ± " << stddev << " ms" << std::endl;

    // Verify correctness of transpose
    cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost);
    bool ok = true;
    for (int r = 0; r < rows && ok; ++r) {
        for (int c = 0; c < cols && ok; ++c) {
            if (h_out[c * rows + r] != h_in[r * cols + c]) {
                ok = false;
            }
        }
    }
    std::cout << "TransposeShared " << (ok ? "PASSED" : "FAILED") << std::endl;

    // Cleanup
    cudaFree(d_in);
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ok ? 0 : 1;
} 