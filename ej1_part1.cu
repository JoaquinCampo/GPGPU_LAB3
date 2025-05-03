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

#define MAX_DIM 4096

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
 * @brief Host entry point for naive GPU matrix transpose (global memory only).
 *
 * Steps performed:
 * - Parses optional command-line arguments for matrix dimensions (rows, cols).
 * - Allocates and initializes host input matrix with sequential values.
 * - Allocates device memory for input and output matrices.
 * - Copies input data to device.
 * - Configures and launches the transposeNaive kernel (32x32 thread blocks).
 * - Synchronizes and checks for kernel errors.
 * - Copies the transposed result back to host.
 * - Verifies correctness by comparing each element to the expected value.
 * - Prints whether the transpose succeeded or failed.
 * - Frees device memory before exit.
 *
 * Usage:
 *   ./programa [rows cols]
 *   (Defaults: rows=1024, cols=1024)
 *
 * @param argc  Number of command-line arguments.
 * @param argv  Array of argument strings (optionally: rows, cols).
 * @return      0 if transpose is correct, 1 for usage/dimension error, 2 for failed verification.
 */
int main(int argc, char *argv[]) {
    // 1) Parse arguments
    int rows = 1024, cols = 1024;
    if (argc == 3) {
        rows = std::atoi(argv[1]);
        cols = std::atoi(argv[2]);
    } else if (argc != 1) {
        std::cerr << "Usage: " << argv[0] << " [rows cols]\n";
        return 1;
    }
    if (rows > MAX_DIM || cols > MAX_DIM) {
        std::cerr << "Error: dims must be ≤ " << MAX_DIM << "\n";
        return 1;
    }
    std::cout << "Matrix size: " << rows << " x " << cols << "\n";

    size_t size = size_t(rows) * cols;
    size_t bytes = size * sizeof(int);

    // 2) Allocate & init host arrays
    std::vector<int> h_in(size), h_out(size);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            h_in[i * cols + j] = i * cols + j;

    // 3) Allocate device arrays
    int *d_in = nullptr, *d_out = nullptr;
    CUDA_CHK(cudaMalloc(&d_in,  bytes));
    CUDA_CHK(cudaMalloc(&d_out, bytes));
    CUDA_CHK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    // 4) Kernel launch config
    dim3 blockDim(32, 32);

    // Calculate how many blocks are needed in each dimension
    int remainder_x = cols % blockDim.x;
    int remainder_y = rows % blockDim.y;

    // If there is a remainder, we need one extra block to cover the edge
    int numBlocksX = cols / blockDim.x + (remainder_x > 0 ? 1 : 0);
    int numBlocksY = rows / blockDim.y + (remainder_y > 0 ? 1 : 0);
    dim3 gridDim(numBlocksX, numBlocksY);

    // 5) Launch once
    transposeNaive<<<gridDim, blockDim>>>(d_in, d_out, rows, cols);
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaDeviceSynchronize());

    // 6) Copy back & verify correctness
    CUDA_CHK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
    bool ok = true;
    for (int i = 0; i < rows && ok; ++i) {
      for (int j = 0; j < cols; ++j) {
        int expected = i * cols + j;
        if (h_out[j * rows + i] != expected) {
          std::cerr << "FAILED at ("<<i<<","<<j<<"): "
                    << h_out[j * rows + i]
                    << " != " << expected << "\n";
          ok = false; break;
        }
      }
    }
    std::cout << (ok ? "Transpose OK\n" : "Transpose FAILED\n");

    // 7) Cleanup
    CUDA_CHK(cudaFree(d_in));
    CUDA_CHK(cudaFree(d_out));
    return ok ? 0 : 2;
}