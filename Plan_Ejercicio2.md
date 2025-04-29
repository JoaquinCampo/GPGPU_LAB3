# Plan Detallado para Ejercicio 2 - Memoria Compartida y Conflictos de Banco

Este documento describe la estrategia paso a paso para implementar, perfilar y optimizar un kernel CUDA que transpone matrices usando memoria compartida, y analiza/soluciona los conflictos de banco.

---

## 1. Objetivo

- **Implementar** un kernel CUDA que utilice memoria compartida como buffer temporal para mejorar la coalescencia de accesos en la transposición de matrices.
- **Medir** tiempos de ejecución (promedio y desviación estándar de 10 runs) para distintas configuraciones de bloque (32×32).
- **Analizar** patrones de acceso en memoria compartida para detectar conflictos de banco.
- **Resolver** conflictos de banco añadiendo padding (columna dummy) y medir el impacto en rendimiento.

## 2. Puntos clave y supuestos

1. **Tile en compartida**: reservamos un array `tile[blockY][blockX]` en memoria compartida para cargar sub-bloques de la matriz.
2. **Acceso coalesced**: las lecturas iniciales y las escrituras finales deben ser coalesced.
3. **Conflictos de banco**: comprender la disposición de bancos (32 bancos, 4 B/banco) y cómo evitar accesos simultáneos al mismo banco.
4. **Padding**: añadir una columna extra (stride = blockX+1) para romper alineamientos conflictivos.
5. **Herramientas**: usar Nsight Systems para medir tiempos y Nsight Compute para contar conflictos de banco.

## 3. Fase 1: Kernel con memoria compartida (sin padding)

### 3.1 Implementación del kernel

- Definir `__global__ void transposeShared(int *in, int *out, int rows, int cols)` que:
  1. Carga un tile de tamaño `blockY×blockX` desde `in` a `shared int tile[blockY][blockX]`.
  2. Sincroniza con `__syncthreads()`.
  3. Escribe el bloque transpuesto desde `tile` a `out`.
- Lanzar con bloques `32×32` y grid calculado para cubrir la matriz.
- Reservar memoria compartida dinámica con `extern __shared__` o estática.

### 3.2 Host y medición

- Reusar host-harness de la Parte I, pero reemplazar kernel.
- Ejecutar warm-up y 10 iteraciones con eventos CUDA.
- Calcular promedio y desviación estándar para matrices de prueba (1024×1024 y 2048×2048).

### 3.3 Profiling de conflictos de banco

- Con Nsight Compute:
  1. Capturar métrica `shared/local/load` y `shared/local/store` para detectar conflictos.
  2. Observar la tasa de `bank_conflict` o `shared_store_conflict` según la arquitectura.
- Documentar el número de conflictos por warp y por banco.

## 4. Fase 2: Resolución de conflictos de banco (padding)

### 4.1 Modificación del kernel

- Ajustar el tile en compartida para tener stride = `blockX + 1`:
  ```cpp
  __shared__ int tile[blockY][blockX+1];
  ```
- Actualizar los accesos a `tile` (indexación) para usar el nuevo stride.

### 4.2 Medición y profiling

- Repetir warm-up y 10 runs, recolectar tiempos promedio y desviación.
- Con Nsight Compute, volver a medir conflictos de banco.
- Comparar las métricas de conflictos y tiempos antes y después del padding.

### 4.3 Comparativa final

- Completar tabla con:
  | Configuración          | Transacciones globales | Confl. bancos | Tiempo (ms) ± σ |
  |------------------------|-----------------------:|-------------:|---------------:|
  | Global naïve           |                      - |            - |               |
  | Compartida (sin pad)   |                      - |            - |               |
  | Compartida (con pad)   |                      - |            - |               |

- Analizar cómo el padding elimina los conflictos y mejora el rendimiento.

## 5. Documentación de conclusiones

- Redactar conclusiones en base a los resultados:
  1. Impacto de la memoria compartida en la coalescencia de accesos.
  2. Efectividad del padding para eliminar conflictos de banco.
  3. Balance entre overhead de memoria compartida y ganancia en throughput.

## 6. Preguntas clave para avanzar

1. ¿Qué tamaño de tile (blockX, blockY) confirmamos usar inicialmente (32×32)?
2. ¿Deseamos probar otros tamaños de tile (p.ej. 16×16) para medir efectos de ocupación?
3. ¿Qué GPU/arquitectura hostname usaremos para ajustar métricas (bancos de 32, 4 B)?
4. ¿Disponemos de Nsight Compute y permisos para capturar métricas de conflicto?

---

*Fin del plan para Ejercicio 2.* 