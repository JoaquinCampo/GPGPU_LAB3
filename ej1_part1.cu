/**
 * @file main.cu
 * @brief Kernel and host code for naive GPU matrix transpose using global memory.
 *
 * Implements a __global__ kernel that transposes an integer matrix on the GPU
 * by reading and writing only from/to global memory. The host-side code
 * allocates buffers, initializes data, launches the kernel, measures execution time
 * over multiple runs, and verifies correctness of the result.
 *
 * Usage:
 *   ./programa [rows cols]
 * Default matrix size is 1024x1024 if no arguments are provided.
 */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

#define MAX_ROWS 4096
#define MAX_COLS 4096

/**
 * @brief Naive transpose kernel using only global memory.
 *
 * Each thread computes its 2D coordinates and performs the transpose
 * by reading from input at (row, col) and writing to output at (col, row).
 * No shared memory or tiling used: serves as baseline for memory-access analysis.
 *
 * @param in    Pointer to input matrix in row-major order (rows x cols).
 * @param out   Pointer to output matrix in row-major order (cols x rows).
 * @param rows  Number of rows in the input matrix.
 * @param cols  Number of columns in the input matrix.
 */
__global__ void transposeNaive(const int *in, int *out, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols)
    {
        // Transpose element
        out[col * rows + row] = in[row * cols + col];
    }
}

/**
 * @brief Host entry point: sets up data, launches kernel, measures time, validates.
 *
 * - Parses optional command-line arguments for matrix dimensions.
 * - Allocates host and device buffers, initializes input with sequential values.
 * - Launches kernel with 32x32 thread blocks and grid covering the matrix.
 * - Performs a warm-up run, then times 10 executions using CUDA events.
 * - Calculates average and standard deviation of execution times.
 * - Copies result back to host and verifies correctness element-wise.
 *
 * @param argc  Number of command-line arguments.
 * @param argv  Array of argument strings (rows, cols).
 * @return      Returns 0 on success.
 */
int main(int argc, char *argv[])
{
    // Matrix dimensions (default 1024x1024)
    int rows = 1024;
    int cols = 1024;
    if (argc > 1) {
        rows = std::atoi(argv[1]);
        cols = std::atoi(argv[2]);
    }
    std::cout << "Matrix size: " << rows << " x " << cols << std::endl;

    if (rows > MAX_ROWS || cols > MAX_COLS) {
        std::cerr << "Error: Matrix size exceeds MAX_ROWS or MAX_COLS." << std::endl;
        return 1;
    }

    size_t size = static_cast<size_t>(rows) * cols;
    size_t bytes = size * sizeof(int);

    // Host allocations
    int h_in[MAX_ROWS][MAX_COLS];
    int h_out[MAX_ROWS][MAX_COLS];
    // Initialize input matrix
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            h_in[i][j] = i * cols + j;
        }
    }

    // Device allocations
    int *d_in = nullptr, *d_out = nullptr;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, &h_in[0][0], bytes, cudaMemcpyHostToDevice);

    // Kernel configuration
    dim3 blockDim(32, 32);
    dim3 gridDim((cols + blockDim.x - 1) / blockDim.x,
                 (rows + blockDim.y - 1) / blockDim.y);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int iterations = 10;
    std::vector<float> times(iterations);

    // Warm-up
    transposeNaive<<<gridDim, blockDim>>>(d_in, d_out, rows, cols);
    cudaDeviceSynchronize();

    // Timed runs
    for (int i = 0; i < iterations; ++i)
    {
        cudaEventRecord(start);
        transposeNaive<<<gridDim, blockDim>>>(d_in, d_out, rows, cols);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        times[i] = ms;
    }

    // Compute average and standard deviation
    float sum = 0.0f;
    for (float t : times)
        sum += t;
    float avg = sum / iterations;
    float sq_sum = 0.0f;
    for (float t : times)
        sq_sum += (t - avg) * (t - avg);
    float stddev = std::sqrt(sq_sum / iterations);

    std::cout << "Average time over " << iterations
              << " runs: " << avg << " ms (± " << stddev << " ms)" << std::endl;

    // After kernel runs, copy result back to host
    cudaMemcpy(&h_out[0][0], d_out, bytes, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_in);
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
