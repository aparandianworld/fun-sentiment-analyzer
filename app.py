from fastapi import FastAPI, HTTPException
from transformers import pipeline
from pydantic import BaseModel
from typing import List
import gradio as gr
import uvicorn
import torch
import plotly.express as px

app = FastAPI(title = "Fun Sentiment Analyzer API")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

sentiment_pipeline = pipeline(
    "sentiment-analysis", 
    model = "cardiffnlp/twitter-roberta-base-sentiment-latest", 
    device = device,
)

# Temporary test
# test_text = "This sentiment analyzer API is awesome!"
# print(sentiment_pipeline(test_text))

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

if __name__ == "__main__":
    uvicorn.run(app, host = "0.0.0.0", port = 8000)