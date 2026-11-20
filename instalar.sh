#!/bin/bash

set -e

echo "=================================="
echo " Instalando dependencias de Control4"
echo "=================================="

# Verificar Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 no está instalado."
    exit 1
fi

# Instalar soporte para entornos virtuales
sudo apt update
sudo apt install -y python3-venv python3-full

# Crear entorno virtual
python3 -m venv venv

# Activar entorno
source venv/bin/activate

# Instalar dependencias
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "=================================="
echo " Instalación completada."
echo "=================================="

echo "Para ejecutar el proyecto:"
echo "source venv/bin/activate"
echo 'python "codigo pregunta 4.py"'
