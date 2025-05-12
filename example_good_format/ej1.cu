// Inclusión de bibliotecas necesarias
#include <stdio.h>    // Para operaciones de entrada/salida
#include <stdlib.h>   // Para funciones de memoria dinámica
#include "cuda.h"     // Para funciones de CUDA

// Macro para verificar errores de CUDA
#define CUDA_CHK(ans) { gpuAssert((ans), __FILE__, __LINE__) }
// Función para manejar errores de CUDA
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// Declaración de funciones auxiliares
void read_file(const char*, int*);  // Función para leer el archivo
int get_text_length(const char * fname);  // Función para obtener la longitud del archivo

// Constantes para el algoritmo de descifrado
#define A 15         // Constante A del algoritmo
#define B 27         // Constante B del algoritmo
#define M 256        // Módulo para operaciones
#define A_MMI_M -17  // Inverso multiplicativo modular de A módulo M

// Función auxiliar para calcular el módulo correctamente con números negativos
__device__ int modulo(int a, int b){
	int r = a % b;
	r = (r < 0) ? r + b : r;
	return r;
}

// Kernel CUDA para descifrar el mensaje
__global__ void decrypt_kernel(int *d_message, int length)
{
	// Calcula el índice global del thread
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < length) {
		int c = d_message[i];  // Obtiene el carácter cifrado
		// Aplica la fórmula de descifrado: A^-1 * (c - B) mod M
		int p = modulo(A_MMI_M * (c - B), M);
		d_message[i] = p;  // Guarda el resultado descifrado
	}
}

int main(int argc, char *argv[])
{
	// Variables para almacenar el mensaje en CPU (host) y GPU (device)
	int *h_message;  // Puntero al mensaje en CPU
	int *d_message;  // Puntero al mensaje en GPU
	unsigned int size;

	const char * fname;  // Nombre del archivo a procesar

	// Verifica que se haya proporcionado el nombre del archivo
	if (argc < 2) printf("Debe ingresar el nombre del archivo\n");
	else
		fname = argv[1];

	// Obtiene la longitud del archivo
	int length = get_text_length(fname);
	size = length * sizeof(int);

	// Reserva memoria en CPU para el mensaje
	h_message = (int *)malloc(size);

	// Lee el archivo y guarda su contenido en h_message
	read_file(fname, h_message);

	// Reserva memoria en GPU y copia el mensaje
	CUDA_CHK(cudaMalloc((void**)&d_message, size));
	CUDA_CHK(cudaMemcpy(d_message, h_message, size, cudaMemcpyHostToDevice));

	// Configuración de la ejecución del kernel
	int threadsPerBlock = 256;  // Número de threads por bloque
	int residuo = length % threadsPerBlock;
	int blocks = length / threadsPerBlock + (residuo > 0);  // Calcula número de bloques
	
	// Ejecuta el kernel de descifrado
	decrypt_kernel<<<blocks, threadsPerBlock>>>(d_message, length);
	CUDA_CHK(cudaGetLastError());
	CUDA_CHK(cudaDeviceSynchronize());  // Espera a que termine la ejecución

	// Copia el resultado de vuelta a CPU
	CUDA_CHK(cudaMemcpy(h_message, d_message, size, cudaMemcpyDeviceToHost));
	CUDA_CHK(cudaFree(d_message));  // Libera memoria de GPU

	// Imprime el mensaje descifrado
	for (int i = 0; i < length; i++) {
		printf("%c", (char)h_message[i]);
	}
	printf("\n");

	// Libera memoria de CPU
	free(h_message);

	return 0;
}

// Función para obtener la longitud del archivo
int get_text_length(const char * fname)
{
	FILE *f = NULL;
	f = fopen(fname, "r");  // Abre el archivo en modo lectura

	size_t pos = ftell(f);     // Guarda la posición actual
	fseek(f, 0, SEEK_END);     // Va al final del archivo
	size_t length = ftell(f);  // Obtiene la longitud
	fseek(f, pos, SEEK_SET);   // Regresa a la posición original

	fclose(f);

	return length;
}

// Función para leer el contenido del archivo
void read_file(const char * fname, int* input)
{
	FILE *f = NULL;
	f = fopen(fname, "r");  // Abre el archivo en modo lectura
	if (f == NULL){
		fprintf(stderr, "Error: Could not find %s file \n", fname);
		exit(1);
	}

	// Lee el archivo carácter por carácter
	int c; 
	while ((c = getc(f)) != EOF) {
		*(input++) = c;  // Guarda cada carácter en el array
	}

	fclose(f);
}
