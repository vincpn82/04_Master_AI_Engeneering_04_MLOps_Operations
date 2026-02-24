# 📚 Documentazione del Progetto MLOps - Sentiment Analysis

## Indice

- [STEP 1: Setup e Introduzione](#step-1-setup-e-introduzione)
- [STEP 2: Implementazione Modello Sentiment Analysis](#step-2-implementazione-modello-sentiment-analysis)
- [STEP 3: Creazione Struttura Progetto](#step-3-creazione-struttura-progetto)
- [STEP 4: Sviluppo API REST con FastAPI](#step-4-sviluppo-api-rest-con-fastapi)
- [STEP 5: Testing e Qualità del Codice](#step-5-testing-e-qualità-del-codice)
- [STEP 6: Containerizzazione con Docker](#step-6-containerizzazione-con-docker)
- [STEP 7: Pipeline CI/CD con GitHub Actions](#step-7-pipeline-cicd-con-github-actions)
- [STEP 8: Sistema di Monitoraggio](#step-8-sistema-di-monitoraggio)
- [STEP 9: Documentazione e Deploy su HuggingFace](#step-9-documentazione-e-deploy-su-huggingface)
- [Link di Approfondimento](#link-di-approfondimento)

---

## STEP 1: Setup e Introduzione

### 🎯 Obiettivi

Configurare l'ambiente di sviluppo con tutte le dipendenze necessarie per costruire un sistema MLOps completo.

### ✅ Cosa viene fatto

1. **Installazione delle dipendenze principali**:
   - `transformers` (≥4.35.0): Libreria di HuggingFace per utilizzare modelli transformer come RoBERTa, BERT, GPT
   - `torch` (≥2.0.0): Backend per modelli transformer, supporta GPU/CPU
   - `fastapi` (≥0.104.0): Framework moderno e veloce per API REST, genera automaticamente documentazione OpenAPI
   - `uvicorn` (≥0.24.0): Server ASGI ad alte prestazioni per FastAPI
   - `pydantic` (≥2.0.0): Validazione dati e schemi con typing Python
   - `pytest` (≥7.4.0): Standard de-facto per testing in Python
   - `httpx` (≥0.25.0): Client HTTP async per testare API FastAPI
   - `scikit-learn` (≥1.3.0): Metriche ML (accuracy, precision, recall, F1-score)
   - `pandas` (≥2.0.0): Manipolazione e analisi dati
   - `matplotlib` (≥3.7.0): Visualizzazione dati
   - `seaborn` (≥0.12.0): Visualizzazioni statistiche avanzate
   - `gradio` (≥4.0.0): Permette di creare interfacce web per modelli ML in poche righe di codice
   - `python-multipart` (≥0.0.6): Per gestire file uploads in FastAPI

2. **Verifica delle installazioni**: Controllo versioni e disponibilità GPU/CPU

### 📚 Concetti Chiave

- **Ambiente virtuale**: Isolamento delle dipendenze per evitare conflitti
- **GPU vs CPU**: Le GPU accelerano il training e l'inferenza dei modelli transformer (fino a 100x più veloci)
- **Gestione dipendenze**: requirements.txt per riproducibilità

### 🔗 Link di Approfondimento

- [Python Virtual Environments](https://docs.python.org/3/tutorial/venv.html)
- [HuggingFace Transformers Documentation](https://huggingface.co/docs/transformers/index)
- [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/)
- [Pytest Documentation](https://docs.pytest.org/)

---

## STEP 2: Implementazione Modello Sentiment Analysis

### 🎯 Obiettivi

Implementare un sistema di sentiment analysis utilizzando un modello transformer pre-addestrato, creando un'interfaccia pulita e testabile.

### 🤖 Il Modello RoBERTa

**RoBERTa** (Robustly Optimized BERT Approach) è un modello transformer sviluppato da Facebook AI (Meta) come miglioramento di BERT.

#### Caratteristiche principali:
- **Pre-training**: Addestrato su enormi quantità di testo (160GB di testo non compresso)
- **Architettura**: 12 layer, 768 dimensioni hidden, 12 attention heads
- **Parametri**: ~125 milioni
- **Tokenization**: Byte-Pair Encoding (BPE) con vocabolario di 50k token

#### Perché `cardiffnlp/twitter-roberta-base-sentiment-latest`?

1. **Specializzato per Social Media**: Fine-tuned specificamente su testi di Twitter (linguaggio informale, emoji, abbreviazioni)
2. **Aggiornato regolarmente**: Versione "latest" con le migliori performance
3. **Multi-classe**: Classifica in 3 categorie (Positive, Neutral, Negative)
4. **Alta accuratezza**: State-of-the-art performance su sentiment analysis
5. **Production-ready**: Modello ottimizzato e testato dalla community HuggingFace

#### Architettura del Modello

```
Input Text 
    ↓
Tokenization (BPE)
    ↓
RoBERTa Encoder (12 layers)
    ↓
Classification Head (Linear + Softmax)
    ↓
[Negative, Neutral, Positive] + Confidence Scores
```

#### Metriche del Modello (Benchmark)

| Dataset | Accuracy | F1-Score | Note |
|---------|----------|----------|------|
| TweetEval | 0.720 | 0.710 | Dataset di tweet reali |
| SST-2 | 0.945 | 0.945 | Stanford Sentiment Treebank |
| IMDB | 0.950 | 0.950 | Movie reviews |

### ✅ Cosa viene fatto

1. **Caricamento del modello**: Download da HuggingFace Hub (automatico con cache locale)
2. **Creazione pipeline**: Wrapper high-level di transformers per semplificare l'uso
3. **Classe SentimentAnalyzer**: Incapsula la logica del modello con interfaccia pulita
4. **Testing**: Verifica funzionamento con esempi reali
5. **Caso d'uso pratico**: Analisi di commenti social media simulati con visualizzazioni

### 🏗️ Design Pattern

**Singleton Pattern** per il caricamento del modello:
- Il modello viene caricato una sola volta in memoria (occupa ~500MB)
- Tutte le richieste utilizzano la stessa istanza
- Ottimizza RAM e tempo di inizializzazione

### 📊 Performance Attese

| Operazione | CPU | GPU (CUDA) |
|-----------|-----|-----------|
| Predizione singola | 50-100ms | 5-10ms |
| Batch 10 testi | 200-300ms | 20-30ms |
| Batch 100 testi | 2-3s | 100-200ms |

**Speedup con batch processing**: 5-10x più veloce rispetto a predizioni singole sequential

### 📚 Concetti Chiave

- **Transfer Learning**: Utilizzo di modelli pre-addestrati riduce tempi e costi di training
- **Fine-tuning**: Specializzazione del modello su task specifici
- **Attention Mechanism**: Permette al modello di "focalizzarsi" su parti rilevanti del testo
- **Confidence Score**: Probabilità (0-1) che la predizione sia corretta
- **Batch Processing**: Elaborazione simultanea di più testi per ottimizzare l'uso della GPU

### 🔗 Link di Approfondimento

- [RoBERTa Paper (arXiv)](https://arxiv.org/abs/1907.11692)
- [Model Card HuggingFace](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)
- [Sentiment Analysis Guide](https://huggingface.co/blog/sentiment-analysis-python)
- [Attention Is All You Need (Transformer paper)](https://arxiv.org/abs/1706.03762)
- [BERT Paper](https://arxiv.org/abs/1810.04805)
- [Transfer Learning in NLP](https://ruder.io/transfer-learning/)

---

## STEP 3: Creazione Struttura Progetto

### 🎯 Obiettivi

Organizzare il codice in una struttura professionale per un progetto MLOps production-ready, seguendo best practices di software engineering.

### 📂 Struttura del Progetto

```
sentiment-analysis-mlops/
├── .github/
│   └── workflows/              # GitHub Actions (CI/CD)
│       ├── ci-cd.yml          # Pipeline principale: test, build, deploy
│       └── monitoring.yml     # Monitoraggio automatico schedulato
├── app/
│   ├── __init__.py           # Package marker
│   ├── main.py               # Applicazione FastAPI
│   ├── model.py              # Logica del modello (SentimentAnalyzer)
│   └── schema.py             # Schemi Pydantic (validazione I/O)
├── tests/
│   ├── __init__.py
│   ├── test_model.py         # Test unitari del modello
│   └── test_api.py           # Test integration API
├── monitoring/
│   ├── monitoring.py         # Script di monitoraggio performance
│   └── reports/              # Report generati (metriche, grafici)
├── data/
│   └── sample_data.json      # Dati di esempio per testing
├── Dockerfile                # Containerizzazione
├── requirements.txt          # Dipendenze Python
├── app.py                    # App Gradio (HuggingFace Spaces)
├── .gitignore               # File da escludere da Git
└── README.md                # Documentazione completa
```

### ✅ Cosa viene fatto

1. **File di configurazione**:
   - `requirements.txt`: Lista completa delle dipendenze con versioni
   - `.gitignore`: Esclude file non necessari (modelli scaricati, cache, __pycache__, etc.)
   
2. **Moduli Python organizzati**:
   - Separazione responsabilità (model, schema, main)
   - Facilita testing e manutenzione
   - Rende il codice riutilizzabile

3. **Struttura per testing**:
   - Test unitari separati dal codice di produzione
   - Facile integrazione con pytest e CI/CD

4. **Preparazione CI/CD**:
   - Struttura compatibile con GitHub Actions
   - File YAML per workflow automatici

### 📚 Concetti Chiave

- **Separation of Concerns**: Ogni file ha uno scopo ben definito
- **DRY Principle** (Don't Repeat Yourself): Codice riutilizzabile
- **Package Structure**: Organizzazione standard Python
- **Configuration Management**: Centralizzazione delle configurazioni

### 🔗 Link di Approfondimento

- [Python Package Structure](https://packaging.python.org/tutorials/packaging-projects/)
- [Project Structure Best Practices](https://docs.python-guide.org/writing/structure/)
- [.gitignore templates](https://github.com/github/gitignore)

---

## STEP 4: Sviluppo API REST con FastAPI

### 🎯 Obiettivi

Creare un'API RESTful production-ready per esporre il modello di sentiment analysis, con documentazione automatica e validazione dei dati.

### 🚀 Perché FastAPI?

- **Performance**: Uno dei framework Python più veloci (paragonabile a NodeJS/Go)
- **Async/Await**: Supporto nativo per operazioni asincrone
- **Documentazione automatica**: OpenAPI (Swagger) e ReDoc generate automaticamente
- **Validazione automatica**: Integrazione con Pydantic per type checking e validazione
- **Type hints**: Supporto completo per Python typing
- **Production-ready**: Utilizzato da Netflix, Uber, Microsoft

### 📡 Endpoint Implementati

| Endpoint | Metodo | Descrizione | Input | Output |
|----------|--------|-------------|-------|--------|
| `/` | GET | Root con info API | - | Messaggio benvenuto |
| `/health` | GET | Health check | - | Status + info modello |
| `/predict` | POST | Predizione singola | Testo | Sentiment + confidence |
| `/predict/batch` | POST | Predizione batch | Lista testi | Lista sentiment |

### ✅ Cosa viene fatto

1. **Schemi Pydantic** (`schema.py`):
   - Validazione automatica input/output
   - Documentazione dei campi
   - Type safety

2. **Applicazione FastAPI** (`main.py`):
   - Configurazione CORS per accesso cross-origin
   - Logging strutturato
   - Error handling

3. **Integrazione modello**:
   - Singleton pattern per efficienza
   - Gestione errori robusta

### 📚 Concetti Chiave

- **REST API**: Architectural style per web services
- **CORS**: Cross-Origin Resource Sharing per sicurezza
- **Pydantic**: Data validation usando Python type annotations
- **OpenAPI**: Specifica standard per documentazione API
- **HTTP Status Codes**: 200 (OK), 400 (Bad Request), 500 (Server Error)

### 🔗 Link di Approfondimento

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [REST API Best Practices](https://restfulapi.net/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [OpenAPI Specification](https://swagger.io/specification/)

---

## STEP 5: Testing e Qualità del Codice

### 🎯 Obiettivi

Garantire affidabilità e qualità del codice attraverso testing automatico completo.

### 🧪 Tipologie di Test

1. **Test Unitari** (`test_model.py`):
   - Testano singole funzioni/metodi in isolamento
   - Verificano logica del modello
   - Rapidi da eseguire

2. **Test di Integrazione** (`test_api.py`):
   - Testano interazione tra componenti
   - Verificano endpoint API
   - Simulano richieste reali

### ✅ Cosa viene testato

- Inizializzazione del modello
- Predizione singola
- Predizione batch
- Endpoint API (200, 400, 500)
- Validazione input
- Format output

### 📊 Coverage Target

- **Obiettivo**: >80% code coverage
- **Critico**: 100% coverage per funzioni core

### 📚 Concetti Chiave

- **TDD** (Test-Driven Development): Scrivere test prima del codice
- **Mocking**: Simulare dipendenze esterne
- **Fixtures**: Setup riutilizzabile per test
- **Assertions**: Verifiche automatiche dei risultati
- **Code Coverage**: Percentuale di codice testato

### 🔗 Link di Approfondimento

- [Pytest Documentation](https://docs.pytest.org/)
- [Testing FastAPI](https://fastapi.tiangolo.com/tutorial/testing/)
- [TDD Best Practices](https://testdriven.io/blog/modern-tdd/)

---

## STEP 6: Containerizzazione con Docker

### 🎯 Obiettivi

Creare un container Docker per garantire riproducibilità e facilitare deployment.

### 🐳 Perché Docker?

- **Riproducibilità**: Stesso ambiente in dev, test, production
- **Isolamento**: Dipendenze isolate dal sistema host
- **Portabilità**: Funziona ovunque (local, cloud, on-premise)
- **Scalabilità**: Facile orchestrazione con Kubernetes
- **Versioning**: Immutabilità delle immagini

### ✅ Cosa viene fatto

1. **Dockerfile multi-stage**:
   - Stage 1: Build dependencies
   - Stage 2: Production (minimal)

2. **Ottimizzazioni**:
   - Image leggera (Python slim)
   - Layer caching
   - .dockerignore per ridurre context

3. **Best practices**:
   - Non-root user
   - Health checks
   - Logs a stdout

### 📚 Concetti Chiave

- **Container vs VM**: Container condividono kernel, VM no
- **Image Layers**: Sistema a layer per efficienza
- **Docker Hub**: Registry pubblico per immagini
- **Multi-stage builds**: Riduce dimensione finale immagine

### 🔗 Link di Approfondimento

- [Docker Get Started](https://docs.docker.com/get-started/)
- [Dockerfile Best Practices](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)
- [Docker for Python](https://docs.docker.com/language/python/)

---

## STEP 7: Pipeline CI/CD con GitHub Actions

### 🎯 Obiettivi

Automatizzare testing, building e deployment attraverso pipeline CI/CD.

### 🔄 Cos'è CI/CD?

- **CI** (Continuous Integration): Integrazione continua del codice con test automatici
- **CD** (Continuous Deployment): Deploy automatico in produzione

### ✅ Pipeline Implementate

1. **ci-cd.yml** - Pipeline principale:
   - Trigger: Push su main, Pull Request
   - Steps: Test → Build Docker → Push Registry → Deploy

2. **monitoring.yml** - Monitoraggio schedulato:
   - Trigger: Cron (es. ogni giorno alle 00:00)
   - Steps: Run monitoring script → Generate report → Upload artifacts

### 📊 Workflow

```
Push Code → GitHub Actions Triggered
    ↓
Run Tests (pytest)
    ↓
Build Docker Image
    ↓
Push to Docker Hub
    ↓
Deploy (optional)
```

### 📚 Concetti Chiave

- **GitHub Actions**: Piattaforma CI/CD integrata in GitHub
- **Workflow**: File YAML che definisce pipeline
- **Runners**: Server che eseguono i job
- **Secrets**: Gestione sicura di credenziali
- **Artifacts**: Output persistenti (report, build)

### 🔗 Link di Approfondimento

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [CI/CD Best Practices](https://about.gitlab.com/topics/ci-cd/)
- [YAML Syntax](https://yaml.org/spec/1.2.2/)

---

## STEP 8: Sistema di Monitoraggio

### 🎯 Obiettivi

Implementare monitoraggio continuo delle performance del modello per rilevare data drift e performance degradation.

### 📊 Metriche Monitorate

1. **Performance Metrics**:
   - Accuracy
   - Precision, Recall, F1-Score
   - Confusion Matrix

2. **Data Quality**:
   - Distribution drift
   - Confidence score distribution

3. **System Metrics**:
   - Response time
   - Error rate
   - Throughput (requests/sec)

### ✅ Cosa viene fatto

1. **Script di monitoraggio**:
   - Valutazione automatica su test set
   - Generazione metriche
   - Visualizzazioni (matplotlib/seaborn)

2. **Report automatici**:
   - Salvati in `monitoring/reports/`
   - Timestamped
   - Includono grafici e statistiche

3. **Alerting** (opzionale):
   - Notifiche se accuracy < threshold
   - Email/Slack per anomalie

### 📚 Concetti Chiave

- **Model Drift**: Degradation delle performance nel tempo
- **Data Drift**: Cambiamento nella distribuzione dei dati
- **Concept Drift**: Cambiamento nella relazione input-output
- **Monitoring vs Observability**: Monitoraggio reattivo vs comprensione proattiva

### 🔗 Link di Approfondimento

- [MLOps Monitoring](https://ml-ops.org/content/three-levels-of-ml-software)
- [Evidently AI](https://www.evidentlyai.com/)
- [Model Monitoring Best Practices](https://christophergs.com/machine%20learning/2020/03/14/how-to-monitor-machine-learning-models/)

---

## STEP 9: Documentazione e Deploy su HuggingFace

### 🎯 Obiettivi

Creare documentazione completa e deployare l'applicazione su HuggingFace Spaces per demo pubblica.

### 📝 README.md

Documentazione completa del progetto includendo:
- Descrizione e obiettivi
- Architettura del sistema
- Istruzioni di installazione
- Esempi di utilizzo
- API reference
- Contribuzione guidelines

### 🤗 HuggingFace Spaces

Piattaforma per hostare demo ML gratuitamente:
- **Gradio**: Framework per creare UI interattive
- **Auto-deploy**: Da repository GitHub
- **Free tier**: Sufficiente per demo
- **Custom domains**: Possibilità di usare dominio personale

### ✅ Cosa viene fatto

1. **App Gradio** (`app.py`):
   - Interfaccia web interattiva
   - Input testo → Output sentiment
   - Esempi pre-caricati

2. **Deploy HuggingFace**:
   - Collegamento repository GitHub
   - Automatic rebuild on push
   - Public URL per condivisione

3. **README completo**:
   - Badge status CI/CD
   - Quick start guide
   - Architecture diagram
   - Screenshots

### 📚 Concetti Chiave

- **Documentation as Code**: Documentazione versionata con il codice
- **API Documentation**: OpenAPI/Swagger per API REST
- **Demo Applications**: Importanza di demo interattive per adoption
- **Markdown**: Linguaggio di markup per documentazione

### 🔗 Link di Approfondimento

- [HuggingFace Spaces](https://huggingface.co/spaces)
- [Gradio Documentation](https://gradio.app/docs/)
- [Writing Great Documentation](https://documentation.divio.com/)
- [Markdown Guide](https://www.markdownguide.org/)

---

## Link di Approfondimento

### MLOps e Best Practices

- [MLOps.org](https://ml-ops.org/) - Guida completa a MLOps
- [Google Cloud MLOps](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
- [AWS MLOps](https://aws.amazon.com/sagemaker/mlops/)
- [Made With ML](https://madewithml.com/) - Corso completo MLOps

### Machine Learning e NLP

- [HuggingFace Course](https://huggingface.co/course) - Corso gratuito su Transformers
- [Fast.ai](https://www.fast.ai/) - Deep Learning for Coders
- [Papers With Code](https://paperswithcode.com/) - Paper ML con implementazioni
- [Stanford CS224N](https://web.stanford.edu/class/cs224n/) - NLP con Deep Learning

### DevOps e CI/CD

- [The DevOps Handbook](https://itrevolution.com/product/the-devops-handbook-second-edition/)
- [GitLab CI/CD Documentation](https://docs.gitlab.com/ee/ci/)
- [Jenkins Documentation](https://www.jenkins.io/doc/)

### Python e Software Engineering

- [Real Python](https://realpython.com/) - Tutorial Python avanzati
- [Python Design Patterns](https://refactoring.guru/design-patterns/python)
- [Clean Code in Python](https://github.com/zedr/clean-code-python)

### Monitoring e Observability

- [Prometheus](https://prometheus.io/) - Monitoring system
- [Grafana](https://grafana.com/) - Visualization platform
- [OpenTelemetry](https://opentelemetry.io/) - Observability framework

### Community e Risorse

- [Kaggle](https://www.kaggle.com/) - Competizioni ML e dataset
- [Reddit r/MachineLearning](https://www.reddit.com/r/MachineLearning/)
- [MLOps Community](https://mlops.community/)
- [AI Stack Exchange](https://ai.stackexchange.com/)

---

**Ultimo aggiornamento**: Febbraio 2026  
**Autore**: Progetto Finale MLOps - Profession AI Master
