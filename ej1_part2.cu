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
    if (rows > MAX_ROWS || cols > MAX_COLS) {
        std::cerr << "Error: Matrix size exceeds MAX_ROWS or MAX_COLS." << std::endl;
        return 1;
    }
    std::cout << "Matrix: " << rows << "x" << cols
              << ", Block: " << blockX << "x" << blockY << std::endl;

    size_t size = static_cast<size_t>(rows) * cols;
    size_t bytes = size * sizeof(int);

    // Allocate and initialize host buffers
    int h_in[MAX_ROWS][MAX_COLS];
    int h_out[MAX_ROWS][MAX_COLS];
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            h_in[i][j] = i * cols + j;
        }
    }

    // Allocate device buffers
    int *d_in = nullptr, *d_out = nullptr;
    CUDA_CHK(cudaMalloc(&d_in, bytes));
    CUDA_CHK(cudaMalloc(&d_out, bytes));
    CUDA_CHK(cudaMemcpy(d_in, &h_in[0][0], bytes, cudaMemcpyHostToDevice));

    // Compute grid dimensions
    dim3 blockDim(blockX, blockY);
    dim3 gridDim((cols + blockX - 1) / blockX,
                 (rows + blockY - 1) / blockY);

    // Prepare CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHK(cudaEventCreate(&start));
    CUDA_CHK(cudaEventCreate(&stop));

    const int iterations = 10;
    std::vector<float> times(iterations);

    // Warm-up run
    size_t sharedBytes = blockX * blockY * sizeof(int);
    transposeShared<<<gridDim, blockDim, sharedBytes>>>(d_in, d_out, rows, cols);
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaDeviceSynchronize());

    // Timed iterations
    for (int i = 0; i < iterations; ++i) {
        CUDA_CHK(cudaEventRecord(start));
        transposeShared<<<gridDim, blockDim, sharedBytes>>>(d_in, d_out, rows, cols);
        CUDA_CHK(cudaGetLastError());
        CUDA_CHK(cudaEventRecord(stop));
        CUDA_CHK(cudaEventSynchronize(stop));
        CUDA_CHK(cudaEventElapsedTime(&times[i], start, stop));
    }

    // Compute statistics
    float sum = 0.0f;
    for (float t : times) sum += t;
    float avg = sum / iterations;
    float sq_sum = 0.0f;
    for (float t : times) sq_sum += (t - avg) * (t - avg);
    float stddev = std::sqrt(sq_sum / iterations);
    std::cout << "Average: " << avg << " ms ± " << stddev << " ms" << std::endl;

    // After kernel runs, copy result back to host
    CUDA_CHK(cudaMemcpy(&h_out[0][0], d_out, bytes, cudaMemcpyDeviceToHost));
    // Verify correctness of transpose
    bool ok = true;
    for (int r = 0; r < rows && ok; ++r) {
        for (int c = 0; c < cols && ok; ++c) {
            if (h_out[c][r] != h_in[r][c]) {
                ok = false;
            }
        }
    }
    std::cout << "TransposeShared " << (ok ? "PASSED" : "FAILED") << std::endl;

    // Cleanup
    CUDA_CHK(cudaFree(d_in));
    CUDA_CHK(cudaFree(d_out));
    CUDA_CHK(cudaEventDestroy(start));
    CUDA_CHK(cudaEventDestroy(stop));

    return ok ? 0 : 1;
} 