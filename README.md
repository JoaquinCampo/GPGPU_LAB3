# Laboratorio de Patrones de Acceso a Memoria

Este repositorio contiene la implementación y análisis de patrones de acceso a memoria y técnicas de optimización de caché. El trabajo se divide en dos ejercicios principales que demuestran el impacto de la localidad espacial y temporal en el rendimiento de los programas.

## Estructura del Repositorio

### Documentación
- `informe_anterior.tex`: Informe LaTeX anterior que contiene análisis detallado, detalles de implementación y resultados de ambos ejercicios
- `informe.tex`: Informe LaTeX actual
- `Letra.md`: Descripción y requisitos de la tarea
- `Plan_Ejercicio1.md`: Documento de planificación para el Ejercicio 1
- `Plan_Ejercicio2.md`, `Plan_Ejercicio2_Part1.md`, `Plan_Ejercicio2_Part2.md`: Documentos de planificación para el Ejercicio 2
- `Recursos_CUDA.md`: Recursos y referencias de CUDA

### Implementación
- `ej1_part1.cu`: Implementación del Ejercicio 1 Parte 1 (Acceso Secuencial)
- `ej1_part2.cu`: Implementación del Ejercicio 1 Parte 2 (Acceso Aleatorio)
- `ej2_part1.cu`: Implementación del Ejercicio 2 Parte 1 (Multiplicación de Matrices Básica)
- `ej2_part2.cu`: Implementación del Ejercicio 2 Parte 2 (Multiplicación de Matrices Optimizada por Bloques)

### Sistema de Compilación
- `Makefile`: Configuración de compilación para archivos fuente CUDA
- `lanzar.sh`: Script de shell para ejecutar experimentos

## Ejercicio 1: Localidad Espacial

Este ejercicio demuestra el impacto de la localidad espacial en el rendimiento del programa comparando dos patrones diferentes de acceso a memoria:

- Acceso Secuencial: Lectura de elementos en orden contiguo
- Acceso Aleatorio: Lectura de elementos en orden aleatorio

La implementación trabaja con un arreglo de 100MB y mide la diferencia de rendimiento entre estos patrones de acceso, mostrando el impacto de la utilización de líneas de caché.

### Hallazgos Principales
- El acceso secuencial fue en promedio 28.44 veces más rápido que el acceso aleatorio
- Las pruebas se realizaron en diferentes niveles de caché:
  - Caché L1 (384 KB): mejora de 1.98x
  - Caché L2 (10 MB): mejora de 8.97x
  - Caché L3 (24 MB): mejora de 20.71x
  - Memoria Principal (100 MB): mejora de 28.44x

## Ejercicio 2: Localidad Temporal

Este ejercicio se centra en la optimización de la multiplicación de matrices utilizando algoritmos basados en bloques para mejorar la utilización de la caché. Compara una implementación básica de multiplicación de matrices con una versión optimizada usando bloques.

### Implementaciones
1. Multiplicación de matrices básica (`ej2_part1.cu`)
2. Multiplicación optimizada por bloques con tamaños configurables (`ej2_part2.cu`)

### Características Principales
- Selección dinámica del tamaño de bloque basada en la jerarquía de caché
- Optimización para niveles de caché L1, L2 y L3
- Pruebas de rendimiento con varios tamaños de matrices:
  - 115x115 (cabe en caché L1)
  - 594x594 (cabe en caché L2)
  - 921x921 (cabe en caché L3)
  - 1842x1842 (excede los niveles de caché)

### Resultados
- La versión optimizada por bloques mostró mejoras significativas:
  - 574.8% de mejora para matrices de 115x115
  - 567.2% de mejora para matrices de 594x594
  - 621.5% de mejora para matrices de 921x921
  - 1279.9% de mejora para matrices de 1842x1842

## Especificaciones del Sistema

Los experimentos se realizaron en un sistema con la siguiente jerarquía de caché:
- Caché L1: 192 KB
- Caché L2: 5120 KB
- Caché L3: 12288 KB
- Tamaño de línea de caché: 64 bytes

## Compilación y Ejecución

1. Asegúrese de tener el kit de herramientas CUDA instalado
2. Compile el proyecto:
   ```bash
   make
   ```
3. Ejecute los experimentos:
   ```bash
   ./lanzar.sh
   ```

## Autores

- Joaquin Campo (5.280.080-4)
- Mateo Daneri (5.660.750-1)
- Santiago Rodriguez (5.221.151-4)

## Licencia

[Información de licencia por agregar] 