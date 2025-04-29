# Plan Detallado para Ejercicio 2 – Parte II: Resolución de Conflictos con Padding

Este plan describe los pasos necesarios para modificar el kernel de transposición con memoria compartida añadiendo padding (columna dummy), medir su impacto en conflictos de banco y rendimiento.

---

## 1. Objetivo

- Adaptar el kernel `transposeSharedNoPad` para incluir padding en el tile de memoria compartida y evitar conflictos de banco.
- Medir de nuevo tiempos de ejecución (promedio y desviación estándar de 10 runs) para bloques 32×32.
- Perfilado con Nsight Compute para verificar la reducción (idealmente eliminación) de conflictos de banco.
- Comparar los resultados con la versión sin padding.

## 2. Desglose de Tareas

### 2.1 Modificación del Kernel

1. Reemplazar la declaración de memoria compartida dinámica por una estática con padding:
   ```cpp
   __shared__ int tile[blockY][blockX + 1];
   ```
2. Ajustar las indexaciones en las fases de carga y escritura:
   - Acceso a `tile[ty * (blockX+1) + tx]` y escritura desde `tile[tx * (blockX+1) + ty]`.
3. Asegurar que el tamaño reservado (`shared memory size`) en el lanzamiento del kernel sea `(blockX+1)*blockY*sizeof(int)`.
4. Compilar y verificar que el output (corrida y validación) siga pasando.

### 2.2 Host Harness y Medición

- Adaptar script o Makefile:
  - Target `build2_2` compilando `ej2_part2.cu`.
  - Target `run2_2` ejecutando y profilando con Nsight Systems.
- En `main`, lanzar kernel `transposeSharedPad` con parámetros de matriz y bloque.
- Medir:
  1. Warm-up run.
  2. 10 iteraciones cronometradas con eventos CUDA.
  3. Cálculo de promedio y desviación estándar.
- Ejecutar para matrices de prueba (1024×1024, 2048×2048) con bloque 32×32.

### 2.3 Profiling de Conflictos y Rendimiento

- Con Nsight Compute:
  1. Capturar métricas `shared_load_conflict`, `shared_store_conflict` para el kernel con padding.
  2. Confirmar que el número de conflictos se reduce respecto a la versión sin padding.
- Con Nsight Systems:
  1. Obtener tiempo de ejecución y comparar con la versión sin padding.

### 2.4 Comparativa Final

- Completar una tabla con:
  | Configuración          | Tiempo (ms) ± σ | Conflictos (avg/warp) | Mejora vs sin pad (%) |
  |------------------------|---------------:|----------------------:|----------------------:|
  | Sin padding            |             -- |                   --  |                    -- |
  | Con padding (X+1 stride)|            -- |                   --  |                    -- |

- Analizar:
  - Reducción de conflictos.
  - Impacto en el tiempo de ejecución.
  - Costo adicional de memoria compartida (stride +1).

## 3. Entregables de la Parte II

1. Código fuente `ej2_part2.cu` con kernel con padding.
2. Makefile/script (`run2_2.sh`).
3. Tabla comparativa de tiempos y conflictos.
4. Capturas de Nsight Compute y Nsight Systems para evidenciar la mejora.

## 4. Preguntas Clave

1. ¿Confirmamos usar padding de +1 columna o considerar +2 para alineamiento a 64 bytes?  
2. ¿Deseamos medir también otros tamaños de bloque (e.g., 16×16) con padding?  
3. ¿Tenemos límite de shared memory por SM que pueda restringir bloques grandes?  
4. ¿Requerimos validar en distintas arquitecturas (compute capabilities)?

---

*Fin del plan para la Parte II del Ejercicio 2.* 