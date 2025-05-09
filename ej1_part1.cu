/**
 * @file main.cu
 * @brief kernel and host code for naive gpu matrix transpose using global memory.
 *
 * implements a __global__ kernel that transposes an integer matrix on the gpu
 * by reading and writing only from/to global memory. the host-side code
 * allocates buffers, initializes data, launches the kernel, measures execution time
 * over multiple runs, and verifies correctness of the result.
 *
 * usage:
 *   ./programa [rows cols]
 * default matrix size is 1024x1024 if no arguments are provided.
 */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <nvtx3/nvToolsExt.h>


#define MAX_DIM 4096

#define CUDA_CHK(ans) do { gpuAssert((ans), __FILE__, __LINE__); } while(0)
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"gpuAssert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


/**
 * @brief naive transpose kernel using only global memory.
 *
 * each thread computes its 2d coordinates and performs the transpose
 * by reading from input at (row, col) and writing to output at (col, row).
 * no shared memory or tiling used: serves as baseline for memory-access analysis.
 *
 * @param in    pointer to input matrix in row-major order (rows x cols).
 * @param out   pointer to output matrix in row-major order (cols x rows).
 * @param rows  number of rows in the input matrix.
 * @param cols  number of columns in the input matrix.
 */
__global__ void transposeNaive(const int *in, int *out, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols)
    {
        // transpose element
        out[col * rows + row] = in[row * cols + col];
    }
}


/**
 * @brief host entry point for naive gpu matrix transpose (global memory only).
 *
 * steps performed:
 * - parses optional command-line arguments for matrix dimensions (rows, cols).
 * - allocates and initializes host input matrix with sequential values.
 * - allocates device memory for input and output matrices.
 * - copies input data to device.
 * - configures and launches the transposenaive kernel (32x32 thread blocks).
 * - synchronizes and checks for kernel errors.
 * - copies the transposed result back to host.
 * - verifies correctness by comparing each element to the expected value.
 * - prints whether the transpose succeeded or failed.
 * - frees device memory before exit.
 *
 * usage:
 *   ./programa [rows cols]
 *   (defaults: rows=1024, cols=1024)
 *
 * @param argc  number of command-line arguments.
 * @param argv  array of argument strings (optionally: rows, cols).
 * @return      0 if transpose is correct, 1 for usage/dimension error, 2 for failed verification.
 */
int main(int argc, char *argv[]) {
    // 1) Parse arguments
    int rows = 1024, cols = 1024;
    int blockX = 32, blockY = 32; // Default block dimensions
    // Use 1D vectors for host arrays to avoid stack overflow
    // A REVISAR - CHEQUEAR SI ESTA IMPLEMENTACION DE MATRICES ESTA BIEN, O SI TENEMOS QUE USAR [] []. CUANDO USO [] [] ME TIRA ERROR DE EJECUCIÓN POR ALOCACIÓN DE MEMORIA.
    std::vector<int> h_in(rows * cols);
    std::vector<int> h_out(rows * cols);
    if (argc == 3) {
        rows = std::atoi(argv[1]);
        cols = std::atoi(argv[2]);
    } else if (argc == 5) {
        rows = std::atoi(argv[1]);
        cols = std::atoi(argv[2]);
        blockX = std::atoi(argv[3]);
        blockY = std::atoi(argv[4]);
    } else if (argc != 1) {
        std::cerr << "Usage: " << argv[0] << " [rows cols [blockX blockY]]\n";
        return 1;
    }

    if (rows > MAX_DIM || cols > MAX_DIM) {
        std::cerr << "Error: dims must be ≤ " << MAX_DIM << "\n";
        return 1;
    }
    if (blockX <= 0 || blockY <= 0 || blockX * blockY > 1024) { // Check block size validity
         std::cerr << "Error: Invalid block dimensions (" << blockX << "x" << blockY
                   << "). Product must be > 0 and <= 1024.\n";
         return 1;
    }

    std::cout << "Matrix size: " << rows << " x " << cols
              << ", Block size: " << blockX << " x " << blockY << "\n";

    size_t size = static_cast<size_t>(rows) * cols;
    size_t bytes = size * sizeof(int);

    // 2) Allocate & init host arrays
    nvtxRangePushA("Init in");
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            h_in[i * cols + j] = i * cols + j;
        }
    }
    nvtxRangePop();

    // 3) Allocate device arrays
    int *d_in = nullptr, *d_out = nullptr;
    nvtxRangePushA("Malloc in");
    std::cout << "[DEBUG] Allocating d_in with cudaMalloc, bytes: " << bytes << std::endl;
    cudaError_t err_in = cudaMalloc(&d_in,  bytes);
    if (err_in != cudaSuccess) {
        std::cerr << "[ERROR] cudaMalloc for d_in failed: " << cudaGetErrorString(err_in) << std::endl;
        return 1;
    }
    nvtxRangePop();
    nvtxRangePushA("Malloc out");
    std::cout << "[DEBUG] Allocating d_out with cudaMalloc, bytes: " << bytes << std::endl;
    cudaError_t err_out = cudaMalloc(&d_out, bytes);
    if (err_out != cudaSuccess) {
        std::cerr << "[ERROR] cudaMalloc for d_out failed: " << cudaGetErrorString(err_out) << std::endl;
        cudaFree(d_in);
        return 1;
    }
    nvtxRangePop();

    nvtxRangePushA("H2D memcpy");
        CUDA_CHK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));
    nvtxRangePop();

    // 4) Kernel launch config
    dim3 blockDim(blockX, blockY); // Use parsed block dimensions

    // Calculate how many blocks are needed in each dimension
    int remainder_x = cols % blockDim.x;
    int remainder_y = rows % blockDim.y;

    // If there is a remainder, we need one extra block to cover the edge
    int numBlocksX = cols / blockDim.x + (remainder_x > 0 ? 1 : 0);
    int numBlocksY = rows / blockDim.y + (remainder_y > 0 ? 1 : 0);
    dim3 gridDim(numBlocksX, numBlocksY);

    // 5) Launch once
    nvtxRangePushA("Kernel launch");
        transposeNaive<<<gridDim, blockDim>>>(d_in, d_out, rows, cols);
        CUDA_CHK(cudaGetLastError());
        CUDA_CHK(cudaDeviceSynchronize());
    nvtxRangePop();

    // 6) Copy back & verify correctness
    nvtxRangePushA("D2H memcpy");
        CUDA_CHK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
    nvtxRangePop();

    bool ok = true;
    for (int r = 0; r < rows && ok; ++r) {
      for (int c = 0; c < cols; ++c) {
        int expected = r * cols + c;
        if (h_out[c * rows + r] != expected) {
            std::cerr << "FAILED at ("<<r<<","<<c<<"): " << h_out[c * rows + r] << " != " << expected << "\n";
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