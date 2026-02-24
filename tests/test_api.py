"""
Test per l'API FastAPI.
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_endpoint():
    """Test endpoint di health check."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] == True

def test_predict_positive():
    """Test predizione positiva."""
    payload = {
        "text": "I absolutely love this! It's fantastic!"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "positive" in data["sentiment"].lower()

def test_predict_negative():
    """Test predizione negativa."""
    payload = {
        "text": "This is terrible! I hate it!"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "negative" in data["sentiment"].lower()

def test_predict_batch_endpoint():
    """Test endpoint di predizione batch."""
    payload = {
        "texts": [
            "Great service!",
            "Not satisfied.",
            "It's okay."
        ]
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    assert "results" in data
    assert "total" in data
    assert data["total"] == 3
    assert len(data["results"]) == 3
    
    for result in data["results"]:
        assert "text" in result
        assert "sentiment" in result
        assert "confidence" in result

def test_invalid_input_too_long():
    """Test con input troppo lungo."""
    payload = {
        "text": "a" * 1000  # Supera il limite di 512 caratteri
    }
    response = client.post("/predict", json=payload)
    # Dovrebbe gestire l'errore appropriatamente
    assert response.status_code in [422, 500]

def test_empty_batch():
    """Test con batch vuoto."""
    payload = {
        "texts": []
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 422  # Validation error
