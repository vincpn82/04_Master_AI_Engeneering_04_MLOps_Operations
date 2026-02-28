# 🏢 MachineInnovators - Sentiment Analysis MLOps

## 📊 Monitoraggio della Reputazione Online

![MLOps](https://img.shields.io/badge/MLOps-Sentiment%20Analysis-blue)
![Python](https://img.shields.io/badge/Python-3.10+-green)
![FastAPI](https://img.shields.io/badge/FastAPI-Framework-teal)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)

Soluzione end-to-end di MLOps per l'analisi del sentiment sui social media, finalizzata al monitoraggio della reputazione online di **MachineInnovators Inc.**

Repository GitHub: https://github.com/vincpn82/04_Master_AI_Engeneering_04_MLOps_Operations.git

---

## 🎯 Obiettivi del Progetto

- **Automazione dell'Analisi del Sentiment**: Classificazione automatica dei sentiment in positivo, neutrale o negativo  
- **Monitoraggio Continuo**: Sistema di monitoraggio continuo per valutare l'andamento del sentiment nel tempo
- **Pipeline CI/CD Automatizzata**: Testing e deployment automatico
- **Retraining del Modello**: Sistema per mantenere alta l'accuratezza predittiva del modello

---

## 🚀 Caratteristiche Principali

- ✅ **API RESTful con FastAPI**: Endpoint per inferenza in tempo reale e batch processing
- ✅ **Modello Pre-addestrato**: Utilizzo di `cardiffnlp/twitter-roberta-base-sentiment-latest`
- ✅ **Interfaccia Gradio**: Interfaccia web interattiva deployata su HuggingFace Spaces
- ✅ **Containerizzazione Docker**: Ambiente isolato e riproducibile
- ✅ **CI/CD con GitHub Actions**: Pipeline automatizzata per testing e deployment
- ✅ **Monitoraggio Continuo**: Valutazione automatica delle performance del modello

---

## 📂 Struttura del Progetto

```
sentiment-analysis-mlops/
├── .github/
│   └── workflows/
│       ├── ci-cd.yml              # Pipeline CI/CD principale
│       └── monitoring.yml         # Monitoraggio automatico
├── app/
│   ├── __init__.py
│   ├── main.py                    # Applicazione FastAPI
│   ├── model.py                   # Logica del modello
│   └── schema.py                  # Schemi Pydantic
├── tests/
│   ├── __init__.py
│   ├── test_model.py              # Test del modello
│   └── test_api.py                # Test dell'API
├── monitoring/
│   ├── monitoring.py              # Script monitoraggio
│   └── reports/                   # Report generati
├── data/
│   └── sample_data.json           # Dati di esempio
├── hf_spaces/                     # 🚀 HuggingFace Spaces deployment
│   ├── README.md                  # Card con metadata YAML per Gradio
│   ├── app.py                     # App Gradio (importa da ../app/)
│   └── requirements.txt           # Dipendenze minimal per HF Spaces
├── Dockerfile                     # Container configuration
├── .dockerignore                  # File esclusi dal container
├── requirements.txt               # Dipendenze Python complete
├── app.py                         # App Gradio locale (test)
├── .gitignore                     # File esclusi da Git
└── README.md                      # Documentazione completa
```

---

## 🛠️ Installazione e Utilizzo

### Prerequisiti

- Python 3.10+
- Docker (opzionale)
- Git

### Setup Locale

```bash
# Clone il repository
git clone https://github.com/vincpn82/AI_Engeneering_04_MLOps_Operations.git
cd sentiment-analysis-mlops

# Crea ambiente virtuale
python -m venv venv
source:
 - unix: venv/bin/activate  
 - Windows: venv\Scripts\activate

# Installa le dipendenze
pip install -r requirements.txt
```

### Avvio dell'API

```bash
# Avvio con uvicorn
uvicorn app.main:app --reload

# Oppure con Docker
docker build -t sentiment-api .
docker run -p 8000:8000 sentiment-api
```

L'API sarà disponibile su `http://localhost:8000`

- 📝 Documentazione interattiva (Swagger): `http://localhost:8000/docs`  
- 📚 Documentazione alternativa (ReDoc): `http://localhost:8000/redoc`

---

## 🔌 API Endpoints

### 1. Health Check

```bash
GET /health
```

### 2. Predizione Singola

```bash
POST /predict
Content-Type: application/json

{
  "text": "I love this product!"
}
```

**Response:**
```json
{
  "text": "I love this product!",
  "sentiment": "positive",
  "confidence": 0.99
}
```

### 3. Predizione Batch

```bash
POST /predict/batch
Content-Type: application/json

{
  "texts": [
    "Great service!",
    "Not satisfied.",
    "It's okay."
  ]
}
```

**Response:**
```json
{
  "results": [
    {"text": "Great service!", "sentiment": "positive", "confidence": 0.99},
    {"text": "Not satisfied.", "sentiment": "negative", "confidence": 0.97},
    {"text": "It's okay.", "sentiment": "neutral", "confidence": 0.85}
  ],
  "total": 3
}
```

---

## 🧪 Testing

```bash
# Esegui tutti i test
pytest tests/ -v

# Test del modello
pytest tests/test_model.py -v

# Test dell'API
pytest tests/test_api.py -v

# Test con coverage
pytest tests/ --cov=app --cov-report=html
```

---

## 📊 Monitoraggio

Il sistema di monitoraggio valuta automaticamente le performance del modello:

```bash
# Esegui il monitoraggio manualmente
python monitoring/monitoring.py
```

I report vengono salvati in `monitoring/reports/` e includono:

- ✅ Metriche di performance (accuracy, precision, recall, F1-score)
- ✅ Matrice di confusione
- ✅ Report di classificazione dettagliato
- ✅ Predizioni complete per analisi

---

## 🎨 Interfaccia Gradio su HuggingFace Spaces

Il progetto include un'interfaccia web interattiva costruita con Gradio e deployata automaticamente su HuggingFace Spaces.

### Caratteristiche dell'Interfaccia

- 🎯 **Analisi in Tempo Reale**: Inserisci un testo e ottieni immediatamente il sentiment
- 📝 **Esempi Pre-caricati**: Esempi di testi per testare rapidamente il modello
- 😊 **Risultati Visualizzati**: Sentiment mostrato con emoji e percentuale di confidenza
- 🎨 **Design Moderno**: Tema Soft di Gradio per un'esperienza utente ottimale

### Esecuzione Locale dell'App Gradio

```bash
# Avvia l'interfaccia Gradio
python app.py
```

L'interfaccia sarà disponibile su `http://localhost:7860`

### Deploy Automatico

Ad ogni push sul branch `main`, la pipeline CI/CD:
1. Esegue tutti i test
2. Valida le performance del modello
3. Effettua il push automatico su HuggingFace Spaces
4. L'app Gradio viene automaticamente deployata e resa pubblica

---

## 🐳 Docker

### Build dell'immagine

```bash
docker build -t sentiment-analysis .
```

### Run del container

```bash
docker run -p 8000:8000 sentiment-analysis
```

---

## 🔄 CI/CD Pipeline

Le GitHub Actions automatizzano:

1. **Testing**: Esecuzione automatica dei test ad ogni push
2. **Build & Push**: Creazione e pubblicazione dell'immagine Docker (solo su branch `main`)
3. **Deploy**: Deploy automatico su HuggingFace Spaces
4. **Monitoring**: Valutazione giornaliera delle performance del modello (schedulata alle 02:00 UTC)

### Secrets necessari

- `HF_TOKEN`: Token HuggingFace
- `HF_SPACE_NAME`: Nome dello Space HuggingFace

---

## 📈 Metriche e Performance

Il modello viene valutato su:

- **Accuracy**: Precisione complessiva
- **Precision**: Precisione per classe
- **Recall**: Richiamo per classe
- **F1-Score**: Media armonica di precision e recall

**Soglia di alert**: Accuracy < 0.69