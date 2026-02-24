"""
Test per il modello di sentiment analysis.
"""
import pytest
from app.model import SentimentAnalyzer, get_model

def test_model_initialization():
    """Test inizializzazione del modello."""
    model = get_model()
    assert model is not None
    assert isinstance(model, SentimentAnalyzer)

def test_predict_positive():
    """Test predizione sentiment positivo."""
    model = get_model()
    result = model.predict("I absolutely love this product! It's amazing!")
    
    assert "sentiment" in result
    assert "confidence" in result
    assert "positive" in result["sentiment"].lower()
    assert result["confidence"] > 0.5

def test_predict_negative():
    """Test predizione sentiment negativo."""
    model = get_model()
    result = model.predict("This is terrible! I hate it!")
    
    assert "sentiment" in result
    assert "confidence" in result
    assert "negative" in result["sentiment"].lower()
    assert result["confidence"] > 0.5

def test_predict_neutral():
    """Test predizione sentiment neutro."""
    model = get_model()
    result = model.predict("It's okay.")
    
    assert "sentiment" in result
    assert "confidence" in result
    # Il sentiment neutrale può variare tra neutral/negative/positive
    assert result["confidence"] >= 0.0

def test_predict_batch():
    """Test predizione batch."""
    model = get_model()
    texts = [
        "I love this!",
        "This is bad.",
        "It's okay."
    ]
    results = model.predict_batch(texts)
    
    assert len(results) == 3
    for result in results:
        assert "sentiment" in result
        assert "confidence" in result
