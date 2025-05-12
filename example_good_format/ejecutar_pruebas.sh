#!/bin/bash
#SBATCH --job-name=grupo34_test
#SBATCH --output=logs/resultados.txt
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --mem=1G

set -e
mkdir -p logs

EJ1_EJECUTABLE=ejercicio1
EJ1_FUENTE=ejercicios/ej1.cu

echo "Compilando $EJ1_FUENTE..."
nvcc -o $EJ1_EJECUTABLE $EJ1_FUENTE || { echo "❌ Falló la compilación de $EJ1_FUENTE"; exit 1; }

echo ""
echo "Ejecutando Ejercicio 1 (desencriptar mensaje)..."
./$EJ1_EJECUTABLE material/secreto.txt
echo ""

EJ2_EJECUTABLE=ejercicio2
EJ2_FUENTE=ejercicios/ej2.cu

echo "Compilando $EJ2_FUENTE..."
nvcc -o $EJ2_EJECUTABLE $EJ2_FUENTE || { echo "❌ Falló la compilación de $EJ2_FUENTE"; exit 1; }

echo ""
echo "Ejecutando pruebas del Ejercicio 2..."

total=0
ok=0

for dir in pruebas/prueba*/; do
    input_file="${dir}entrada.txt"
    expected_file="${dir}salida_esperada.txt"
    output_file="${dir}salida_obtenida.txt"

    total=$((total + 1))
    ./$EJ2_EJECUTABLE "$input_file" > "$output_file"

    if diff -q "$output_file" "$expected_file" > /dev/null; then
        echo "✅ ${dir}OK"
        ok=$((ok + 1))
        [ -f "$output_file" ] && rm "$output_file"
    else
        echo "❌ ${dir}DIFERENCIAS ENCONTRADAS"
    fi
done

echo ""
echo "📊 Resultado: $ok de $total pruebas pasaron"
