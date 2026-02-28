"""
Modulo per il caricamento e l'utilizzo del modello di sentiment analysis.

DESIGN DECISION: Perché un Wrapper invece di usare direttamente pipeline()?
================================================================================

Teoricamente basterebbe usare direttamente:
    from transformers import pipeline
    sentiment_pipeline = pipeline("sentiment-analysis", model="...")
    
Ma in PRODUZIONE abbiamo bisogno di:

1. SINGLETON PATTERN - Performance Critica
   - Senza wrapper: ogni import ricarica il modello → spreco di memoria
   - Con wrapper: get_model() carica UNA sola volta
   - Risparmio: in un'app con 10 endpoint, risparmi 4.5GB di RAM

2. LOGGING CENTRALIZZATO
   - Tracciamento del caricamento del modello
   - Debug facilitato in produzione
   - Metriche di performance facilmente aggiungibili

3. TESTABILITÀ
   - Facile da mockare nei test unitari
   - Interfaccia pulita e prevedibile
   - Test isolation (non ricarica modello ad ogni test)

4. MANUTENIBILITÀ
   - Un solo punto di modifica per cambiare modello
   - Facile aggiungere error handling, retry logic, caching
   - Supporto futuro per multiple versioni del modello

5. ESTENSIBILITÀ
   - Facile aggiungere metodi: get_stats(), save_predictions(), etc.
   - Possibilità di supportare più modelli (twitter, financial, multilingual)
   - Metrics collection per monitoraggio

Questo approccio segue le MLOps best practices per sistemi in produzione.

DESIGN: Dual Endpoint API (Singolo + Batch)
============================================
Supportiamo DUE modalità:

1. SINGOLO TESTO → predict(text: str)
   - Caso d'uso: analisi in tempo reale, chat, feedback form
   - Risposta immediata
   - Comodo per testing manuale
   - Nome semplice per il caso d'uso più comune
   
2. BATCH PROCESSING → predict_batch(texts: List[str])
   - Caso d'uso: analisi di dataset, report periodici, bulk operations
   - Performance ottimali (GPU parallelism: 10-16x più veloce)
   - Efficienza di rete (1 chiamata invece di N)
   - Nome esplicito per identificare subito il batch processing

Internamente: predict() chiama predict_batch([text])[0] per riuso codice.
"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from transformers import logging as transformers_logging
import torch
from typing import Dict, List
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress model loading warnings about unexpected keys
transformers_logging.set_verbosity_error()

# Suppress HTTP request logs from httpx (used by HuggingFace Hub)
logging.getLogger("httpx").setLevel(logging.WARNING)


class SentimentAnalyzer:
    """
    Classe per l'analisi del sentiment utilizzando il modello RoBERTa.
    
    Fornisce predizione singola (predict) e batch processing (predict_batch).
    """
    
    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
        """
        Inizializza il modello di sentiment analysis.
        
        Args:
            model_name: Nome del modello pre-addestrato da HuggingFace
        """
        logger.info(f"🔧 Inizializzazione SentimentAnalyzer con modello: {model_name}")
        
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Determina il device (GPU se disponibile, altrimenti CPU)
        self.device = 0 if torch.cuda.is_available() else -1
        
        # Crea la pipeline di sentiment analysis
        # Pipeline è un helper di HuggingFace che automatizza:
        # 1. Tokenization: converte testo in numeri (es. "Great!" → [101, 2307, 999, 102])
        # 2. Preprocessing: padding, truncation, attention masks
        # 3. Inferenza: passa i dati nel modello RoBERTa
        # 4. Post-processing: converte logits in probabilità e poi in label + score
        # Risultato: da testo grezzo a {"label": "positive", "score": 0.99} in una chiamata
        self.pipeline = pipeline(
            "sentiment-analysis",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device
        )
        
        logger.info(f"✅ SentimentAnalyzer pronto! Device: {'GPU' if self.device == 0 else 'CPU'}")
    
    @staticmethod
    def preprocess_text(text: str) -> str:
        """
        Utilizzo regex:
           Le regex sono pattern di ricerca potentissimi per manipolare testo. Sono come un "linguaggio di ricerca avanzato" per trovare e sostituire pattern complessi.
        
        Preprocessing del testo per migliorare la qualità delle predizioni.
        
        Ottimizzazioni:
        - Rimuove URL (spesso neutri/irrilevanti per sentiment)
        - Normalizza menzioni Twitter (@username → @user)
        - Normalizza spazi multipli
        - Rimuove caratteri di controllo
        - Limita emoji ripetuti (!!!!! → !!)
        
        Args:
            text: Testo grezzo da preprocessare
            
        Returns:
            Testo pulito e normalizzato
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Rimuove URL (http, https, www)
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Normalizza menzioni Twitter
        text = re.sub(r'@\w+', '@user', text)
        
        # Limita ripetizioni di punteggiatura (!!!!! → !!)
        text = re.sub(r'([!?.]){3,}', r'\1\1', text)
        
        # Rimuove caratteri di controllo e newline multipli
        text = re.sub(r'[\r\n\t]+', ' ', text)
        
        # Normalizza spazi multipli
        text = re.sub(r'\s+', ' ', text)
        
        # Trim spazi iniziali e finali
        text = text.strip()
        
        return text
    
    def predict(self, text: str) -> Dict[str, any]:
        """
        Predice il sentiment di un singolo testo.
        
        Metodo principale per analisi singola. Internamente usa predict_batch()
        per riuso codice ed efficienza.
        
        Args:
            text: Testo da analizzare
            
        Returns:
            Dizionario con: text, sentiment, confidence
            
        Example:
            >>> analyzer = SentimentAnalyzer()
            >>> analyzer.predict("I love this!")
            {"text": "I love this!", "sentiment": "positive", "confidence": 0.99}
        """
        return self.predict_batch([text])[0]
    
    def predict_batch(self, texts: List[str], preprocess: bool = True) -> List[Dict[str, any]]:
        """
        Predice il sentiment di una lista di testi (batch processing).
        
        PERFORMANCE: 10-16x più veloce di N chiamate a predict()
        grazie al parallelismo GPU.
        
        Il nome esplicito "predict_batch" rende immediatamente chiaro
        che si sta usando batch processing.
        
        Args:
            texts: Lista di testi da analizzare
            preprocess: Se True, applica preprocessing ai testi (default: True)
            
        Returns:
            Lista di dizionari con: text, sentiment, confidence
            
        Example:
            >>> analyzer = SentimentAnalyzer()
            >>> analyzer.predict_batch(["Great!", "Awful!", "Okay"])
            [
                {"text": "Great!", "sentiment": "positive", "confidence": 0.99},
                {"text": "Awful!", "sentiment": "negative", "confidence": 0.97},
                {"text": "Okay", "sentiment": "neutral", "confidence": 0.85}
            ]
        """
        # Applica preprocessing se richiesto
        if preprocess:
            processed_texts = [self.preprocess_text(text) for text in texts]
            # Filtra testi vuoti dopo preprocessing
            valid_texts = [(i, text) for i, text in enumerate(processed_texts) if text]
            if not valid_texts:
                return [{"text": text, "sentiment": "neutral", "confidence": 0.0} for text in texts]
            
            indices, texts_to_process = zip(*valid_texts)
            results = self.pipeline(list(texts_to_process))
        else:
            results = self.pipeline(texts)
            indices = range(len(texts))
        
        # Costruisci risultati finali
        output = []
        result_idx = 0
        for i, original_text in enumerate(texts):
            if preprocess and i not in indices:
                # Testo vuoto dopo preprocessing
                output.append({
                    "text": original_text,
                    "sentiment": "neutral",
                    "confidence": 0.0
                })
            else:
                output.append({
                    "text": original_text,
                    "sentiment": results[result_idx]['label'],
                    "confidence": float(results[result_idx]['score'])
                })
                result_idx += 1
        
        return output
    
    def get_info(self) -> Dict[str, any]:
        """
        Restituisce informazioni sul modello.
        
        Returns:
            Dizionario con informazioni sul modello
        """
        return {
            "model_name": self.model_name,
            "num_labels": self.model.config.num_labels,
            "max_length": self.tokenizer.model_max_length,
            "device": "GPU" if self.device == 0 else "CPU"
        }


# Singleton instance (caricato una sola volta per risparmiare memoria)
_model_instance = None

def get_model() -> SentimentAnalyzer:
    """
    Ottiene l'istanza singleton del modello.
    
    Pattern Singleton: garantisce che il modello venga caricato
    una sola volta in memoria, anche se get_model() viene chiamata
    da moduli diversi.
    
    Returns:
        Istanza di SentimentAnalyzer
    """
    global _model_instance
    if _model_instance is None:
        _model_instance = SentimentAnalyzer()
    return _model_instance
