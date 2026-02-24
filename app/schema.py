"""
Schemi Pydantic per la validazione dell'input/output API.

Pydantic fornisce:
- Validazione automatica dei dati in ingresso
- Serializzazione JSON
- Documentazione automatica OpenAPI
- Type safety con mypy

DESIGN: Dual Endpoint API
=========================
Supportiamo DUE endpoint per massima flessibilità:

1. POST /predict (singolo testo)
   - Input: {"text": "testo"}
   - Output: {"text": "...", "sentiment": "...", "confidence": 0.99}
   - Caso d'uso: real-time, chat, form feedback

2. POST /predict/batch (lista testi)
   - Input: {"texts": ["testo1", "testo2"]}
   - Output: {"results": [...], "total": 2}
   - Caso d'uso: dataset analysis, bulk operations
   - Performance: 10-16x più veloce di N chiamate singole

Vantaggi:
- API chiara e intuitiva
- Validazione specifica per use case
- Metriche separate (tracking singole vs batch)
- Backward compatibility
"""
from pydantic import BaseModel, Field, field_validator
from typing import List


class SingleTextInput(BaseModel):
    """
    Schema per l'input dell'API - singolo testo.
    
    Usato dall'endpoint POST /predict
    """
    text: str = Field(
        ...,
        min_length=1,
        max_length=512,
        description="Testo da analizzare",
        examples=["I love this product!"]
    )


class BatchTextInput(BaseModel):
    """
    Schema per l'input dell'API - batch processing.
    
    Usato dall'endpoint POST /predict/batch
    
    Valida che:
    - texts sia una lista di stringhe
    - La lista contenga almeno 1 elemento
    - La lista non contenga più di 100 elementi (limita batch size)
    """
    texts: List[str] = Field(
        ...,
        description="Lista di testi da analizzare (min 1, max 100)",
        examples=[
            ["Great service!", "Terrible experience", "It's okay"]
        ]
    )
    
    @field_validator('texts')
    @classmethod
    def validate_texts(cls, v):
        """Valida la lista di testi."""
        if not v:
            raise ValueError("La lista 'texts' non può essere vuota")
        if len(v) > 100:
            raise ValueError("Massimo 100 testi per richiesta")
        if any(not text.strip() for text in v):
            raise ValueError("Tutti i testi devono contenere almeno un carattere")
        return v


class SentimentResult(BaseModel):
    """
    Schema per il risultato dell'analisi di un singolo testo.
    
    Usato sia per l'endpoint singolo che per ciascun elemento del batch.
    """
    text: str = Field(..., description="Testo analizzato")
    sentiment: str = Field(..., description="Sentiment rilevato: positive, negative, neutral")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidenza della predizione (0-1)")


class BatchSentimentOutput(BaseModel):
    """
    Schema per l'output dell'API batch - contiene tutti i risultati.
    
    Usato dall'endpoint POST /predict/batch
    """
    results: List[SentimentResult] = Field(..., description="Risultati per ogni testo")
    total: int = Field(..., description="Numero totale di testi analizzati")


class HealthResponse(BaseModel):
    """
    Schema per la risposta dell'endpoint di health check.
    """
    status: str = Field(..., description="Stato del servizio")
    model_loaded: bool = Field(..., description="Se il modello è caricato correttamente")
    model_name: str = Field(..., description="Nome del modello utilizzato")
