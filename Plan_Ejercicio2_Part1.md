# Plan Detallado para Ejercicio 2 – Parte I: Memoria Compartida (sin Padding)

Este plan se centra en implementar y medir un kernel que utiliza memoria compartida para la transposición de matrices, sin aplicar padding contra conflictos de banco.

---

## 1. Objetivo

- Implementar un kernel CUDA que cargue tiles de la matriz en memoria compartida y realice la transposición en este buffer.
- Medir tiempos de ejecución (promedio y desviación estándar de 10 runs) usando bloques de 32×32.
- Detectar el grado de conflictos de banco en la memoria compartida.

## 2. Desglose de Tareas

### 2.1 Preparación del Proyecto

- Clonar o crear un directorio específico para el Ejercicio 2.
- Crear/ajustar Makefile o CMake:
  - Target `build2_1` que compile `ej1_part2.cu` (sin padding).
  - Flags: `-O3 -arch=sm_60`.
- Escribir un script `run2_1.sh` que lance el programa con `nsys profile --stats=true` y guarde resultados.

### 2.2 Implementación del Kernel

- Definir `__global__ void transposeShared(int *in, int *out, int rows, int cols)`:
  1. Calcular índices globales.
  2. Reservar shared memory dinámicamente (`extern __shared__ int tile[]`).
  3. Cargar tile desde `in` a shared memory.
  4. `__syncthreads()`.
  5. Escribir tile transpuesto a `out`.
- Comprobar límites (`if (row<rows && col<cols)` y condición simétrica para escritura).

### 2.3 Host Harness y Medición

- Adaptar `main` para:
  - Aceptar `rows`, `cols`, `blockX`, `blockY` desde argumentos.
  - Configurar `dim3 block(32,32); dim3 grid((cols+31)/32,(rows+31)/32)`.
  - Reservar, inicializar y copiar buffers host→GPU.
  - Warm-up y bucle de 10 iteraciones con `cudaEvent_t`.
  - Calcular promedio y desviación estándar.
  - Verificar trasposición copiando de vuelta y comparando.

### 2.4 Profiling de Conflictos de Banco

- Con Nsight Compute:
  1. Capturar kernel `transposeShared`.
  2. Recoger métricas de conflictos de banco: `shared_load_conflict`, `shared_store_conflict`.
  3. Anotar número promedio de conflictos por warp.

### 2.5 Recogida de Resultados

- Ejecutar para matrices 1024×1024 y 2048×2048.
- Registrar:
  - Tiempo promedio ± σ.
  - Número de conflictos de banco.
- Guardar logs y capturas de Nsight Compute.

## 3. Entregables de la Parte I

1. Código fuente `ej1_part2.cu` (sin padding).
2. Makefile/script (`run2_1.sh`).
3. Tabla de resultados:
   | Tamaño matriz | Tiempo (ms) ± σ | Conflictos banco (avg/warp) |
   |-------------:|---------------:|----------------------------:|
   | 1024×1024    |               |                             |
   | 2048×2048    |               |                             |
4. Capturas de Nsight Compute mostrando métrica de conflictos.

## 4. Preguntas clave

1. ¿Confirmamos usar bloques de 32×32 para las primeras pruebas?  
2. ¿Necesitamos soporte para matrices no cuadradas o dimensiones que no sean múltiplos de 32?  
3. ¿Qué herramienta de profiler específica (GPU Compute Capability) usamos para contar conflictos?  
4. ¿Requerimos automatizar más la extracción de métricas con scripts o basta con capturas manuales?  

---

*Fin del plan para la Parte I del Ejercicio 2.* 