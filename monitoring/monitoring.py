"""
Script di monitoraggio per valutare le performance del modello.
"""
import os
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend non-interattivo
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

# Aggiungi la directory parent al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.model import get_model
from datasets import load_dataset

def load_test_data():
    """Carica il dataset di test TweetEval."""
    print("📥 Caricamento dataset TweetEval...")
    dataset = load_dataset("tweet_eval", "sentiment")
    test_data = dataset["test"]
    
    # Converti il dataset
    texts = test_data["text"]
    labels = test_data["label"]
    
    # Mappa le label (0: negative, 1: neutral, 2: positive)
    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    true_labels = [label_map[label] for label in labels]
    
    return texts, true_labels

def evaluate_model():
    """Valuta il modello sul dataset di test."""
    print("🚀 Avvio monitoraggio del modello...")
    
    # Carica il modello
    model = get_model()
    
    # Carica i dati di test
    texts, true_labels = load_test_data()
    
    # Limita a 500 campioni per velocizzare
    texts = texts[:500]
    true_labels = true_labels[:500]
    
    print(f"📊 Valutazione su {len(texts)} campioni...")
    
    # Predizioni
    predictions = []
    for i, text in enumerate(texts):
        if (i + 1) % 100 == 0:
            print(f"   Processati {i + 1}/{len(texts)} campioni...")
        
        result = model.predict(text)
        # Estrai solo la label (rimuovi eventuali prefissi come "LABEL_")
        sentiment = result["sentiment"].lower()
        if "positive" in sentiment:
            predictions.append("positive")
        elif "negative" in sentiment:
            predictions.append("negative")
        else:
            predictions.append("neutral")
    
    # Calcola le metriche
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average='weighted', zero_division=0
    )
    
    print("\n" + "="*60)
    print("📊 RISULTATI DEL MONITORAGGIO")
    print("="*60)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("="*60)
    
    # Confusion Matrix
    cm = confusion_matrix(true_labels, predictions, labels=["negative", "neutral", "positive"])
    
    # Salva report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = Path("monitoring/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # Salva metriche
    metrics_df = pd.DataFrame({
        "timestamp": [timestamp],
        "accuracy": [accuracy],
        "precision": [precision],
        "recall": [recall],
        "f1_score": [f1],
        "num_samples": [len(texts)]
    })
    metrics_df.to_csv(report_dir / f"metrics_{timestamp}.csv", index=False)
    
    # Salva predizioni
    results_df = pd.DataFrame({
        "text": texts,
        "true_label": true_labels,
        "predicted_label": predictions
    })
    results_df.to_csv(report_dir / f"predictions_{timestamp}.csv", index=False)
    
    # Genera confusion matrix plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Negative", "Neutral", "Positive"],
                yticklabels=["Negative", "Neutral", "Positive"])
    plt.title(f'Confusion Matrix - {timestamp}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(report_dir / f"confusion_matrix_{timestamp}.png", dpi=150)
    plt.close()
    
    # Classification report
    report = classification_report(true_labels, predictions)
    with open(report_dir / f"classification_report_{timestamp}.txt", "w") as f:
        f.write(report)
    
    print(f"\n✅ Report salvati in: {report_dir}")
    
    # Controlla se c'è degradazione
    alert_threshold = 0.70
    if accuracy < alert_threshold:
        print(f"\n⚠️  ALERT: L'accuracy ({accuracy:.4f}) è sotto la soglia ({alert_threshold})!")
        print("    Considera il retraining del modello.")
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

if __name__ == "__main__":
    evaluate_model()
