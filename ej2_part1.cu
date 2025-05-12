/**
 * @file ej2_part1.cu
 * @brief Ejercicio 2 Parte I: Kernel de transposición usando memoria compartida sin padding.
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
 * @brief Kernel de transposición usando memoria compartida sin padding.
 * @param in    Matriz de entrada en orden fila-mayor (filas x columnas)
 * @param out   Matriz de salida en orden fila-mayor (columnas x filas)
 * @param rows  Número de filas en la entrada
 * @param cols  Número de columnas en la entrada
 */
__global__ void transposeSharedNoPad(const int* in, int* out, int rows, int cols) {
    extern __shared__ int tile[];
    int blockX = blockDim.x;
    int blockY = blockDim.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * blockY + ty;
    int col = blockIdx.x * blockX + tx;

    // Fase 1: Cargar en memoria compartida
    if (row < rows && col < cols) {
        tile[ty * blockX + tx] = in[row * cols + col];
    }
    __syncthreads();

    // Fase 2: Escribir transpuesto desde memoria compartida
    int trow = blockIdx.x * blockX + ty;
    int tcol = blockIdx.y * blockY + tx;
    if (trow < cols && tcol < rows) {
        out[trow * rows + tcol] = tile[tx * blockX + ty];
    }
}

/**
 * @brief Punto de entrada del host para transposición con memoria compartida.

 * @param argc  Número de argumentos de línea de comando
 * @param argv  Array de strings de argumentos (opcionalmente: filas, columnas, blockX, blockY)
 * @return      0 si la transposición es correcta, 1 para error de uso/dimensión, 2 para verificación fallida
 */
int main(int argc, char* argv[]) {
    // 1) Parsear argumentos
    int rows = 1024, cols = 1024;
    int blockX = 32, blockY = 32;  // Dimensiones de bloque por defecto
    
    // Parsear argumentos de línea de comando
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

    // Validar dimensiones
    if (rows > MAX_ROWS || cols > MAX_COLS) {
        std::cerr << "Error: Matrix size exceeds MAX_ROWS or MAX_COLS." << std::endl;
        return 1;
    }
    if (blockX <= 0 || blockY <= 0 || blockX * blockY > 1024) {
        std::cerr << "Error: Invalid block dimensions (" << blockX << "x" << blockY << ")." << std::endl;
        return 1;   
    }

    std::cout << "Matrix size: " << rows << " x " << cols
              << ", Block size: " << blockX << " x " << blockY << "\n";

    // 2) Asignar e inicializar arrays en host
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

    // 3) Asignar arrays en dispositivo
    int *d_in = nullptr, *d_out = nullptr;
    nvtxRangePushA("Malloc in");
    CUDA_CHK(cudaMalloc(&d_in, bytes));
    nvtxRangePop();

    nvtxRangePushA("Malloc out");
    CUDA_CHK(cudaMalloc(&d_out, bytes));
    nvtxRangePop();

    nvtxRangePushA("H2D memcpy");
    CUDA_CHK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));
    nvtxRangePop();

    // 4) Configuración del lanzamiento del kernel
    dim3 blockDim(blockX, blockY);
    
    // Calcular dimensiones de la grilla
    int remainder_x = cols % blockDim.x;
    int remainder_y = rows % blockDim.y;
    int numBlocksX = cols / blockDim.x + (remainder_x > 0 ? 1 : 0);
    int numBlocksY = rows / blockDim.y + (remainder_y > 0 ? 1 : 0);
    dim3 gridDim(numBlocksX, numBlocksY);

    // 5) Lanzar kernel
    size_t sbytes = blockX * blockY * sizeof(int);
    nvtxRangePushA("Kernel launch");
    transposeSharedNoPad<<<gridDim, blockDim, sbytes>>>(d_in, d_out, rows, cols);
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaDeviceSynchronize());
    nvtxRangePop();

    // 6) Copiar de vuelta y verificar corrección
    nvtxRangePushA("D2H memcpy");
    CUDA_CHK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
    nvtxRangePop();

    // Verificar resultados
    bool ok = true;
    for (int r = 0; r < rows && ok; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (h_out[c * rows + r] != h_in[r * cols + c]) {
                std::cerr << "FAILED at (" << r << "," << c << "): " << h_out[c * rows + r] << " != " << h_in[r * cols + c] << "\n";
                ok = false;
                break;
            }
        }
    }
    std::cout << (ok ? "Transpose OK\n" : "Transpose FAILED\n");

    // 7) Limpieza
    CUDA_CHK(cudaFree(d_in));
    CUDA_CHK(cudaFree(d_out));
    return ok ? 0 : 2;
} 