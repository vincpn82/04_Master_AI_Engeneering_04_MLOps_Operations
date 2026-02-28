"""
Gradio app per HuggingFace Spaces.
"""
import sys
from pathlib import Path

# Aggiungi la directory parent al path per importare il modulo app
sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr
from app.model import get_model

# Carica il modello
model = get_model()

def analyze_sentiment(text):
    """Analizza il sentiment del testo."""
    if not text:
        return "Inserisci un testo da analizzare"
    
    result = model.predict(text)
    
    # Formatta il risultato
    sentiment = result["sentiment"]
    confidence = result["confidence"]
    
    emoji = "😊" if "positive" in sentiment.lower() else "😞" if "negative" in sentiment.lower() else "😐"
    
    return f"{emoji} **Sentiment**: {sentiment}\n\n**Confidence**: {confidence:.2%}"

# Crea l'interfaccia Gradio
demo = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(
        lines=5,
        placeholder="Inserisci un testo da analizzare...",
        label="Testo"
    ),
    outputs=gr.Markdown(label="Risultato"),
    title="🏢 MachineInnovators - Sentiment Analysis",
    description="""
    ### Monitoraggio della Reputazione Online
    
    Questo strumento utilizza un modello di AI per analizzare il sentiment di testi sui social media.
    Perfetto per monitorare la reputazione aziendale in tempo reale!
    
    **Modello**: cardiffnlp/twitter-roberta-base-sentiment-latest
    """,
    examples=[
        ["I absolutely love this new product! It's amazing!"],
        ["This is the worst experience I've ever had."],
        ["The product is okay, nothing special."],
        ["MachineInnovators Inc. has excellent customer service!"],
    ],
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch()
