/**
 * @file ej1_part1.cu
 * @brief Kernel y código host para transposición de matriz usando memoria global en GPU.
 */

// Inclusión de bibliotecas necesarias
#include <cuda_runtime.h>  // Para funciones de CUDA
#include <iostream>        // Para operaciones de entrada/salida
#include <vector>         // Para manejo de vectores
#include <cmath>          // Para funciones matemáticas
#include <nvtx3/nvToolsExt.h>  // Para profiling

// Constantes del programa
#define MAX_DIM 4096  // Dimensión máxima de la matriz

// Macro para verificar errores de CUDA
#define CUDA_CHK(ans) do { gpuAssert((ans), __FILE__, __LINE__); } while(0)

// Función para manejar errores de CUDA
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/**
 * @brief Kernel de transposición naive usando solo memoria global.
 *
 * Cada thread calcula sus coordenadas 2D y realiza la transposición
 * leyendo desde la entrada en (fila, columna) y escribiendo en la salida en (columna, fila).
 * No se usa memoria compartida ni tiling: sirve como línea base para análisis de acceso a memoria.
 *
 * @param in    Puntero a matriz de entrada en orden fila-mayor (filas x columnas)
 * @param out   Puntero a matriz de salida en orden fila-mayor (columnas x filas)
 * @param rows  Número de filas en la matriz de entrada
 * @param cols  Número de columnas en la matriz de entrada
 */
__global__ void transposeNaive(const int *in, int *out, int rows, int cols)
{
    // Calcula las coordenadas 2D del thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Verifica que el thread esté dentro de los límites de la matriz
    if (row < rows && col < cols)
    {
        // Realiza la transposición del elemento
        out[col * rows + row] = in[row * cols + col];
    }
}

/**
 * @brief Punto de entrada del host para transposición de matriz en GPU.
 *
 * @param argc  Número de argumentos de línea de comando
 * @param argv  Array de strings de argumentos (opcionalmente: filas, columnas, blockX, blockY)
 * @return      0 si la transposición es correcta, 1 para error de uso/dimensión, 2 para verificación fallida
 */
int main(int argc, char *argv[]) 
{
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
    if (rows > MAX_DIM || cols > MAX_DIM) {
        std::cerr << "Error: Matrix dimensions must be ≤ " << MAX_DIM << "\n";
        return 1;
    }
    if (blockX <= 0 || blockY <= 0 || blockX * blockY > 1024) {
        std::cerr << "Error: Invalid block dimensions (" << blockX << "x" << blockY
                  << "). Product must be > 0 and <= 1024.\n";
        return 1;
    }

    std::cout << "Matrix size: " << rows << " x " << cols
              << ", Block size: " << blockX << " x " << blockY << "\n";

    // 2) Asignar e inicializar arrays en host
    size_t size = static_cast<size_t>(rows) * cols;
    size_t bytes = size * sizeof(int);
    std::vector<int> h_in(size);
    std::vector<int> h_out(size);

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
    nvtxRangePushA("Kernel launch");
    transposeNaive<<<gridDim, blockDim>>>(d_in, d_out, rows, cols);
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
            int expected = r * cols + c;
            if (h_out[c * rows + r] != expected) {
                std::cerr << "FAILED at (" << r << "," << c << "): "
                          << h_out[c * rows + r] << " != " << expected << "\n";
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