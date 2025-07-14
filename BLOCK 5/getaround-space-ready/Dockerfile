FROM python:3.10-slim

WORKDIR /app

# Installer les dépendances
COPY api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install streamlit requests

# Copier tous les fichiers nécessaires
COPY app.py /app/
COPY api /app/api
COPY model /app/model
COPY data /app/data

# Exposer le port du dashboard
EXPOSE 7860
EXPOSE 8501

# Lancer l'API FastAPI en arrière-plan + Streamlit
CMD uvicorn api.app:app --host 0.0.0.0 --port 7860 & streamlit run app.py --server.port=8501 --server.address=0.0.0.0
