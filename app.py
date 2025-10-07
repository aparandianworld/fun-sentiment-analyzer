from fastapi import FastAPI, HTTPException
from transformers import pipeline
from pydantic import BaseModel
from typing import List
import gradio as gr
import uvicorn
import torch
import plotly.express as px
import time
import plotly.io as pio

app = FastAPI(title = "Fun Sentiment Analyzer API")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

sentiment_pipeline = pipeline(
    "sentiment-analysis", 
    model = "cardiffnlp/twitter-roberta-base-sentiment-latest", 
    device = device,
    truncation = True,
    max_length = 512,
)

# Temporary test
test_text = "This sentiment analyzer API is awesome!"
start = time.time()
result = sentiment_pipeline(test_text)
print(f"Inference time: {time.time() - start}")
print(result)

# Pydantic models API inputs
class TextInput(BaseModel):
    text: str

class BatchTextInput(BaseModel):
    texts: List[str]

# API endpoint for single text sentiment analysis
@app.post("/analyze")
async def analyze_sentiment(input: TextInput):
    try:
        result = sentiment_pipeline(input.text)
        return {"label": result[0]["label"], "confidence": result[0]["score"]}
    except Exception as e:
        raise HTTPException(status_code = 500, detail = str(e))

# API endpoint for batch text sentiment analysis
@app.post("/batch-analyze")
async def batch_analyze_sentiment(input: BatchTextInput):
    try:
        results = sentiment_pipeline(input.texts)
        labels = [result["label"] for result in results]
        fig = px.histogram(
            x=labels,
            title="Sentiment Distribution",
            labels={"x": "Sentiment", "y": "Count"}
        )
        return {
            "results": [
                { "label": result["label"], "confidence": result["score"] } for result in results
            ],
            "plot": fig.to_json()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def visualize_plot(plot_json: str):
    try:
        fix = pio.from_json(plot_json)
        fig.show()
    except Exception as e:
        print(f"Error visualizing plot: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host = "0.0.0.0", port = 8000)