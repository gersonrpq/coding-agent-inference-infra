#!/bin/bash

PID_FILE="$HOME/.k8s_forwards.pids"

echo "================================================="
echo " Deteniendo Port-Forwards activos..."
echo "================================================="

# Verificar si el archivo de PIDs existe y no está vacío
if [ ! -s "$PID_FILE" ]; then
    echo "⚠ No se encontraron PIDs registrados. ¿Ya los habías apagado?"
    echo "================================================="
    exit 0
fi

# Leer PIDs uno a uno y matarlos de forma segura
while IFS= read -r pid; do
    if [ -n "$pid" ]; then
        # -0 verifica si el proceso sigue existiendo
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid"
            echo "✔ Proceso PID $pid detenido con éxito."
        else
            echo "⚠ El proceso PID $pid ya se había cerrado."
        fi
    fi
done < "$PID_FILE"

# Limpiar el archivo de registro
rm -f "$PID_FILE"

echo "================================================="
echo "¡Listo! Todos los port-forwards han sido cerrados."
echo "================================================="
