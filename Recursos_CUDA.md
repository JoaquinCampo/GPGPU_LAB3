# Documentación y Recursos Esenciales para CUDA

Este documento agrupa las referencias y guías más importantes para comenzar y profundizar en el desarrollo con CUDA.

---

## 1. Guías Oficiales de Programación

- **CUDA C/C++ Programming Guide**: descripción detallada del modelo de programación, extensiones de C/C++ y detalles de hardware. [Leer guía](https://docs.nvidia.com/cuda/)  
- **Best Practices Guide**: recomendaciones de paralelización, optimización y patrones de código para obtener máximo rendimiento. [Leer guía](https://docs.nvidia.com/cuda/)  

## 2. Compatibilidad y Tuning por Arquitectura

- **Pascal, Volta, Turing, Ampere, Hopper, Ada, Blackwell Compatibility Guides**: pautas para asegurar que el código sea compatible con cada arquitectura NVIDIA. [Ver lista](https://docs.nvidia.com/cuda/)  
- **Tuning Guides**: optimizaciones específicas para cada generación de GPU (Maxwell, Pascal, Volta, etc.). [Ver lista de tuning guides](https://docs.nvidia.com/cuda/)  

## 3. Herramientas de Profiling y Debugging

- **Nsight Systems**: perfilado de todo el stack (CPU/GPU), trazas de API y análisis de memoria. [Documentación](https://docs.nvidia.com/cuda/)  
- **Nsight Compute**: perfilado detallado de kernels, métricas de ocupación y análisis de rendimiento por instrucción. [Documentación](https://docs.nvidia.com/cuda/)  
- **CUDA Profiler (CUPTI)**: API para crear herramientas de perfilado personalizadas. [Documentación](https://docs.nvidia.com/cuda/)  
- **CUDA-GDB**: depurador para aplicaciones CUDA en Linux. [Documentación](https://docs.nvidia.com/cuda/)  

## 4. API References

- **CUDA Runtime API**: funciones de gestión de memoria, ejecución de kernels y sincronización. [Referencia](https://docs.nvidia.com/cuda/)  
- **CUDA Driver API**: APIs de bajo nivel para mayor flexibilidad y control. [Referencia](https://docs.nvidia.com/cuda/)  

## 5. Instalación y Primeros Pasos

- **Quick Start Guide e Installation Guide (Linux/Windows)**: pasos para instalar el CUDA Toolkit y verificar la correcta operación. [Ver instalación](https://docs.nvidia.com/cuda/)  
- **CUDA Zone**: portal con descargas de toolkit, ejemplos, laboratorios y webinars. [Visitar CUDA Zone](https://developer.nvidia.com/cuda-zone)

## 6. Recursos Adicionales

- **CUDA Samples**: ejemplos de código que ilustran patrones comunes. Incluidos en el CUDA SDK.  
- **CUB, Thrust, cuBLAS, cuFFT, cuSPARSE, cuRAND**: bibliotecas de alto nivel para funciones matemáticas y de parallel primitives.  
- **Foros y Comunidad**: NVIDIA Developer Forums, Stack Overflow (`tag: cuda`).

---

*Referencias relacionadas extraídas de la documentación oficial de NVIDIA CUDA.* 