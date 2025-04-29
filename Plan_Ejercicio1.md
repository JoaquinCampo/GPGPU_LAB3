# Plan Detallado para Ejercicio 1 - Memoria Global

Este documento describe paso a paso la estrategia para implementar, perfilar y optimizar un kernel CUDA que transpone matrices usando únicamente memoria global. Incluye objetivos, tareas, dependencias y preguntas clave.

---

## 1. Objetivo

- **Construir** un kernel CUDA que reciba una matriz de enteros en memoria global y devuelva su transpuesta, sin utilizar memoria compartida.
- **Medir** tiempos de ejecución (promedio y desviación estándar de 10 runs) y analizar patrones de acceso a memoria global (coalesced vs. no-coalesced).
- **Comparar** dos configuraciones de bloque para evaluar el impacto del tamaño en la coalescencia y el rendimiento.

## 2. Puntos clave y supuestos

1. **Dimensión de las matrices**: decidir tamaños representativos (e.g. 1024×1024, 2048×2048) y garantizar que sean múltiples de las dimensiones de bloque.
2. **Arquitectura de GPU**: conocer Compute Capability para interpretar correctamente la agrupación de transacciones de 32 B.
3. **Herramientas de profiling**: usar `nsys profile --stats=true` para obtener tiempos y Nsight Systems para inspeccionar transacciones por warp.
4. **Entorno de ejecución**: Slurm (`lanzar.sh`) para automatizar ejecuciones y recolección de estadísticas.

## 3. Fase 1: Configuración inicial con bloque 32×32

### 3.1 Scaffold del proyecto

- Crear o actualizar scripts de compilación (`Makefile` o `CMakeLists`) con targets:
  - `build`: compila el programa.
  - `run`: ejecuta el programa con parámetros de prueba.
- Ajustar `lanzar.sh` para Slurm, invocando `nsys profile --stats=true ./programa`.

### 3.2 Implementación del kernel "naïve"

- Definir `__global__ void transposeNaive(int *in, int *out, int n, int m)`:
  1. Calcular índices globales a partir de `blockIdx`, `threadIdx`.
  2. Leer `in[row * m + col]` y escribir en `out[col * n + row]`.
- Asegurarse de que `n` y `m` sean múltiplos de 32 para evitar condiciones de contorno.

### 3.3 Host code y medición

- En host:
  1. Reservar y copiar memoria en dispositivo.
  2. Inicializar matrix de prueba.
  3. Ejecutar kernel 10 veces:
     - Sincronizar GPU.
     - Registrar tiempo con `nsys`.
  4. Calcular promedio y desviación estándar.

### 3.4 Profiling de accesos

- Con Nsight Systems:
  1. Capturar sesión de profiler.
  2. Filtrar por kernel `transposeNaive`.
  3. Inspeccionar transacciones por warp en lectura y escritura.

## 4. Análisis de resultados iniciales

- Preparar tabla comparativa:

  | Métrica               | Valor (32×32) |
  |-----------------------|-------------:|
  | Transacciones lectura |      
  | Transacciones escritura |    
  | Tiempo promedio (ms)  |    ±       |
  | Desviación (ms)       |             |

- Identificar patrones no-coalesced:
  - Cuántas transacciones por warp.
  - Alineamiento de direcciones.

## 5. Fase 2: Optimización del tamaño de bloque

### 5.1 Selección de nuevos bloques

- Proponer configuraciones alternativas, por ejemplo:
  - Bloque 16×32 (1 warp por fila).
  - Bloque 32×16 (1 warp por columna).
  - Bloques cuadrados menores (16×16).
- Justificar teóricamente cómo cada warp accede a 32×4 B contiguos.

### 5.2 Implementación mínima

- Cambiar únicamente la llamada al kernel (`dim3 blockDim`) sin alterar la lógica interna.

### 5.3 Medición y profiling

- Repetir pasos de la fase 1 (3.3 y 3.4) para cada configuración.
- Completar tabla comparativa:

  | Bloque (x×y) | Lectura (trans.) | Escritura (trans.) | Tiempo (ms) ± σ |
  |-------------:|-----------------:|-------------------:|---------------:|
  |    32×32     |                  |                    |                |
  |    16×32     |                  |                    |                |
  |    32×16     |                  |                    |                |

## 6. Documentación de conclusiones

- Redactar conclusiones:
  1. Impacto del tamaño de bloque en la coalescencia.
  2. Mejora relativa en rendimiento.
  3. Lecciones aprendidas sobre accesos a memoria global.
- Preparar gráficos (bar charts / líneas) para tiempos y transacciones.
- Integrar en informe PDF (máx. 4 páginas).

## 7. Preguntas clave para avanzar

1. ¿Qué tamaños de matriz iniciales confirmamos para las pruebas?
2. ¿Debemos contemplar matrices no cuadradas o con dimensiones no múltiplos de 32?
3. ¿Disponemos ya de la GPU (compute capability) en la que correremos el práctico?
4. ¿Hay restricciones de tiempo o recursos para la fase de profiling?

---

*Fin del plan para Ejercicio 1.* 