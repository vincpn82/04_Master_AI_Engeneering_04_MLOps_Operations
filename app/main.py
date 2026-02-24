"""
FastAPI Application per Sentiment Analysis.

Questa applicazione fornisce un'API REST per l'analisi del sentiment
con due endpoint principali:
- POST /predict: analisi di un singolo testo
- POST /predict/batch: analisi batch di più testi
- GET /health: health check del servizio

DESIGN PATTERNS APPLICATI:
===========================

1. DEPENDENCY INJECTION (FastAPI)
   - get_model() fornisce l'istanza del modello agli endpoint
   - Singleton pattern garantisce una sola istanza in memoria
   - Facilita testing (possibile mockare il modello)

2. ERROR HANDLING CENTRALIZZATO
   - Gestione uniforme degli errori in tutti gli endpoint
   - Logging automatico di tutte le eccezioni
   - Response chiare e consistenti per il client

3. CORS (Cross-Origin Resource Sharing)
   - Permette accesso da frontend su domini diversi
   - Configurabile per produzione vs development
   - Essenziale per web apps moderne

4. LOGGING STRUTTURATO
   - Ogni richiesta tracciata con timestamp
   - Facilita debugging in produzione
   - Metriche per monitoraggio performance

5. VALIDAZIONE AUTOMATICA (Pydantic)
   - Input validati prima di raggiungere la business logic
   - Errori HTTP 422 automatici per input non validi
   - Documentazione OpenAPI auto-generata
"""
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from datetime import datetime
import time

# Import dei nostri moduli
from app.model import get_model
from app.schema import (
    SingleTextInput, 
    BatchTextInput, 
    SentimentResult, 
    BatchSentimentOutput,
    HealthResponse
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Inizializza FastAPI
app = FastAPI(
    title="Sentiment Analysis API",
    description="API REST per analisi del sentiment con modello RoBERTa",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI
    redoc_url="/redoc"  # ReDoc documentation
)

# CORS middleware - permette richieste da frontend
# In produzione, limitare origins a domini specifici
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carica il modello all'avvio dell'applicazione
# Questo succede UNA SOLA VOLTA grazie al pattern Singleton
try:
    logger.info("🚀 Avvio applicazione FastAPI")
    sentiment_model = get_model()
    logger.info("✅ Modello caricato con successo")
except Exception as e:
    logger.error(f"❌ Errore nel caricamento del modello: {e}")
    raise


# ============================================================================
# ENDPOINT 1: Predizione Singola
# ============================================================================

@app.post(
    "/predict",
    response_model=SentimentResult,
    summary="Analizza il sentiment di un singolo testo",
    description="Predice il sentiment (positive/negative/neutral) di un singolo testo",
    tags=["Sentiment Analysis"]
)
async def predict_single(input_data: SingleTextInput):
    """
    Analizza il sentiment di un singolo testo.
    
    Args:
        input_data: Oggetto SingleTextInput con il testo da analizzare
        
    Returns:
        SentimentResult con testo, sentiment e confidence
        
    Example:
        ```
        POST /predict
        {
            "text": "I love this product!"
        }
        
        Response:
        {
            "text": "I love this product!",
            "sentiment": "positive",
            "confidence": 0.99
        }
        ```
    """
    start_time = time.time()
    
    try:
        logger.info(f"📝 Richiesta predizione singola: {input_data.text[:50]}...")
        
        # Chiama il modello
        result = sentiment_model.predict(input_data.text)
        
        elapsed = time.time() - start_time
        logger.info(f"✅ Predizione completata in {elapsed:.3f}s - Sentiment: {result['sentiment']}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Errore nella predizione: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Errore durante l'analisi: {str(e)}"
        )


# ============================================================================
# ENDPOINT 2: Predizione Batch
# ============================================================================

@app.post(
    "/predict/batch",
    response_model=BatchSentimentOutput,
    summary="Analizza il sentiment di più testi (batch)",
    description="Predice il sentiment di una lista di testi in batch (10-16x più veloce di chiamate singole)",
    tags=["Sentiment Analysis"]
)
async def predict_batch(input_data: BatchTextInput):
    """
    Analizza il sentiment di più testi in batch.
    
    PERFORMANCE: 10-16x più veloce di N chiamate a /predict
    grazie al parallelismo GPU.
    
    Args:
        input_data: Oggetto BatchTextInput con lista di testi (max 100)
        
    Returns:
        BatchSentimentOutput con lista di risultati e count totale
        
    Example:
        ```
        POST /predict/batch
        {
            "texts": ["Great product!", "Terrible service", "It's okay"]
        }
        
        Response:
        {
            "results": [
                {"text": "Great product!", "sentiment": "positive", "confidence": 0.99},
                {"text": "Terrible service", "sentiment": "negative", "confidence": 0.97},
                {"text": "It's okay", "sentiment": "neutral", "confidence": 0.85}
            ],
            "total": 3
        }
        ```
    """
    start_time = time.time()
    
    try:
        num_texts = len(input_data.texts)
        logger.info(f"📦 Richiesta predizione batch: {num_texts} testi")
        
        # Chiama il modello in batch
        results = sentiment_model.predict_batch(input_data.texts)
        
        elapsed = time.time() - start_time
        avg_time = (elapsed / num_texts) * 1000  # ms per testo
        logger.info(f"✅ Batch completato in {elapsed:.3f}s ({avg_time:.1f}ms/testo)")
        
        return BatchSentimentOutput(
            results=results,
            total=num_texts
        )
        
    except Exception as e:
        logger.error(f"❌ Errore nella predizione batch: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Errore durante l'analisi batch: {str(e)}"
        )


# ============================================================================
# ENDPOINT 3: Health Check
# ============================================================================

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Controlla lo stato del servizio",
    description="Verifica che l'API sia attiva e il modello caricato",
    tags=["Health"]
)
async def health_check():
    """
    Health check endpoint.
    
    Utile per:
    - Monitoraggio uptime
    - Load balancer health checks
    - CI/CD deployment verification
    
    Returns:
        HealthResponse con status, model_loaded, model_name
        
    Example:
        ```
        GET /health
        
        Response:
        {
            "status": "healthy",
            "model_loaded": true,
            "model_name": "cardiffnlp/twitter-roberta-base-sentiment-latest"
        }
        ```
    """
    try:
        # Verifica che il modello sia caricato
        model_loaded = sentiment_model is not None
        model_name = sentiment_model.model_name if model_loaded else "N/A"
        
        return HealthResponse(
            status="healthy" if model_loaded else "unhealthy",
            model_loaded=model_loaded,
            model_name=model_name
        )
        
    except Exception as e:
        logger.error(f"❌ Health check fallito: {e}")
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            model_name="N/A"
        )


# ============================================================================
# STARTUP EVENT
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """
    Eseguito all'avvio dell'applicazione.
    """
    logger.info("=" * 80)
    logger.info("🚀 SENTIMENT ANALYSIS API STARTED")
    logger.info(f"📅 Timestamp: {datetime.now().isoformat()}")
    logger.info(f"🤖 Model: {sentiment_model.model_name}")
    logger.info(f"🖥️  Device: {sentiment_model.get_info()['device']}")
    logger.info("📝 Documentation: http://localhost:8000/docs")
    logger.info("=" * 80)


# ============================================================================
# SHUTDOWN EVENT
# ============================================================================

@app.on_event("shutdown")
async def shutdown_event():
    """
    Eseguito alla chiusura dell'applicazione.
    """
    logger.info("=" * 80)
    logger.info("🛑 SENTIMENT ANALYSIS API SHUTTING DOWN")
    logger.info(f"📅 Timestamp: {datetime.now().isoformat()}")
    logger.info("=" * 80)


# ============================================================================
# EXCEPTION HANDLERS
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Handler globale per eccezioni non gestite.
    """
    logger.error(f"❌ Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error",
            "message": str(exc)
        }
    )
