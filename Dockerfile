# Dockerfile para el Backend de LangGraph

# 1. Imagen base: python:3.11-slim es una excelente elección, ligera y moderna.
FROM python:3.11-slim

# 2. Establece el directorio de trabajo en el contenedor
WORKDIR /app

# 3. Copia el archivo de requisitos primero para aprovechar la caché de Docker.
# Esto significa que si solo cambias tu código Python pero no las dependencias,
# Docker no necesitará reinstalar todo cada vez que construyas la imagen.
COPY requirements.txt .

# 4. Instala las dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copia el resto del código de la aplicación.
# Esto incluye tu carpeta 'src', el archivo '.env' si lo tienes localmente (aunque es mejor manejarlo como secreto), etc.
COPY . /app

# 6. Expone el puerto que usará Uvicorn. Usaremos 8000 como estándar para APIs.
EXPOSE 8000

# 7. Comando para ejecutar la aplicación en producción.
# Este es el comando que se ejecutará cuando el contenedor se inicie.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]