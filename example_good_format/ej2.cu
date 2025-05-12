#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"

#define CUDA_CHK(ans) { gpuAssert((ans), __FILE__, __LINE__) }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


__global__ void modifySubmatrix(int *A, int n,
                                int i1, int j1,
                                int i2, int j2,
                                int val) {
    // Calcula los índices globales de la matriz basados en la posición del thread
    int row = blockIdx.y * blockDim.y + threadIdx.y + i1;
    int col = blockIdx.x * blockDim.x + threadIdx.x + j1;

    // Verifica si el thread actual está dentro de los límites de la submatriz
    if (row < n && col < n && row >= i1 && row <= i2 && col >= j1 && col <= j2) {
        // Suma el valor especificado al elemento de la matriz
        A[row * n + col] += val;
    }
}


void leer_archivo(const char* nombre_archivo, int** A,
                           int* n, int* i1, int* j1, int* i2, int* j2, int* val) {
    FILE* f = fopen(nombre_archivo, "r");
    if (!f) {
        fprintf(stderr, "No se pudo abrir %s\n", nombre_archivo);
        exit(1);
    }

    // Lee los parámetros de la primera línea
    fscanf(f, "%d %d %d %d %d %d", n, i1, j1, i2, j2, val);

    // Reserva memoria para la matriz
    int total = (*n) * (*n);
    int* matriz = (int*)malloc(total * sizeof(int));

    // Lee los elementos de la matriz
    for (int i = 0; i < total; i++) {
        fscanf(f, "%d", &matriz[i]);
    }

    fclose(f);
    *A = matriz;
}



void imprimir_matriz(int* A, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%4d ", A[i * n + j]);
        }
        printf("\n");
    }
    printf("\n");
}


int main(int argc, char *argv[]) {
    // Verifica que se haya proporcionado el archivo de entrada
    if (argc < 2) {
        fprintf(stderr, "Uso: %s archivo_entrada\n", argv[0]);
        return 1;
    }

    // Variables para la matriz y parámetros
    int* h_A;  // Matriz en el host (CPU)
    int n, i1, j1, i2, j2, val;

    // Lee el archivo de entrada
    leer_archivo(argv[1], &h_A, &n, &i1, &j1, &i2, &j2, &val);

    // Reserva memoria en la GPU y copia la matriz
    int *d_A = NULL;  // Matriz en el device (GPU)
    size_t size = n * n * sizeof(int);  // Tamaño total de la matriz en bytes
    CUDA_CHK(cudaMalloc((void**)&d_A, size));
    CUDA_CHK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));

    // Configuración de la ejecución del kernel
    dim3 blockSize(16, 16);  // Cada bloque tiene 16x16 threads

    // Calcula las dimensiones de la submatriz
    int height = i2 - i1 + 1;
    int width = j2 - j1 + 1;

    // Calcula el número de bloques necesarios para cubrir la submatriz
    int residuo_x = width % blockSize.x;    
    int residuo_y = height % blockSize.y;

    int cant_threads_x = width / blockSize.x + (residuo_x > 0);
    int cant_threads_y = height / blockSize.y + (residuo_y > 0);

    dim3 gridSize(cant_threads_x, cant_threads_y);

    // Ejecuta el kernel
    modifySubmatrix<<<gridSize, blockSize>>>(d_A, n, i1, j1, i2, j2, val);
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaDeviceSynchronize());

    // Copia el resultado de la GPU a la CPU
    CUDA_CHK(cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost));
    CUDA_CHK(cudaFree(d_A));

    // Imprime la matriz modificada
    printf("Matriz modificada:\n");
    imprimir_matriz(h_A, n);

    // Libera la memoria
    free(h_A);
    return 0;
} 