# Immagine base Python
FROM python:3.10-slim

# Imposta la directory di lavoro
WORKDIR /app

# Copia i file di requirements
COPY requirements.txt ./

# Installa le dipendenze
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && rm -rf /root/.cache/pip

# Esponi la porta 8000
EXPOSE 8000

# Comando per avviare l'applicazione
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
