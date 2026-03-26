# Backend

```bash
conda create -n LangGraph-Perfilador-Copywriter python=3.11 -y
conda activate LangGraph-Perfilador-Copywriter
cd Backend
cp .env.example .env   # Configurar tus API keys
pip install -r requirements.txt
uvicorn main:app --reload

## Instalar solo las dependencias necesarias para langgraph dev
pip install "langgraph-cli[inmem]"

## Usar en este entorno
langgraph dev
langgraph dev --allow-blocking

## Dockerfile:
docker login

docker docker buildx build -t fedecanesa/scraping-canesa:latest --push .
```

# Frontend

```bash
cd Frontend
npm install
npm run dev
```
