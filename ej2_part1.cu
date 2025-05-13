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
      fprintf(stderr,"gpuAssert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


__global__ void transposeSharedNoPad(const int* in, int* out, int rows, int cols) {
    extern __shared__ int tile[];
    int blockX = blockDim.x;
    int blockY = blockDim.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * blockY + ty;
    int col = blockIdx.x * blockX + tx;

    if (row < rows && col < cols) {
        tile[ty * blockX + tx] = in[row * cols + col];
    }
    __syncthreads();

    int trow = blockIdx.x * blockX + ty;
    int tcol = blockIdx.y * blockY + tx;
    if (trow < cols && tcol < rows) {
        out[trow * rows + tcol] = tile[tx * blockX + ty];
    }
}

int main(int argc, char* argv[]) {
    int rows = 1024, cols = 1024;
    int blockX = 32, blockY = 32; 
    std::vector<int> h_in(rows * cols);
    std::vector<int> h_out(rows * cols);

    rows = std::atoi(argv[1]);
    cols = std::atoi(argv[2]);
    blockX = std::atoi(argv[3]);
    blockY = std::atoi(argv[4]);

    std::cout << "Matrix size: " << rows << " x " << cols << ", Block size: " << blockX << " x " << blockY << "\n";

    size_t size = static_cast<size_t>(rows) * cols;
    size_t bytes = size * sizeof(int);

    nvtxRangePushA("Init in");
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            h_in[i * cols + j] = i * cols + j;
        }
    }
    nvtxRangePop();

    int *d_in = nullptr, *d_out = nullptr;
    
    nvtxRangePushA("Malloc in");
    cudaError_t err_in = cudaMalloc(&d_in,  bytes);
    if (err_in != cudaSuccess) {
        std::cerr << "[ERROR] cudaMalloc for d_in failed: " << cudaGetErrorString(err_in) << std::endl;
        return 1;
    }
    nvtxRangePop();

    nvtxRangePushA("Malloc out");
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


    dim3 blockDim(blockX, blockY); 

    int remainder_x = cols % blockDim.x;
    int remainder_y = rows % blockDim.y;

    int numBlocksX = cols / blockDim.x + (remainder_x > 0 ? 1 : 0);
    int numBlocksY = rows / blockDim.y + (remainder_y > 0 ? 1 : 0);
    dim3 gridDim(numBlocksX, numBlocksY);

    size_t sbytes = blockX * blockY * sizeof(int);
    nvtxRangePushA("Kernel launch");
        transposeSharedNoPad<<<gridDim, blockDim, sbytes>>>(d_in, d_out, rows, cols);
        CUDA_CHK(cudaGetLastError());
        CUDA_CHK(cudaDeviceSynchronize());
    nvtxRangePop();

    nvtxRangePushA("D2H memcpy");
        CUDA_CHK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
    nvtxRangePop();

    bool ok = true;
    for (int r = 0; r < rows && ok; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (h_out[c * rows + r] != h_in[r * cols + c]) {
                ok = false;
                break;
            }
        }
    }
    std::cout << (ok ? "Transpose OK\n" : "Transpose not OK\n");

    CUDA_CHK(cudaFree(d_in));
    CUDA_CHK(cudaFree(d_out));
    return ok ? 0 : 2;
} 
