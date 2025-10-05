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

sentiment_pipeline = pipeline("sentiment-analysis", model = "cardiffnlp/twitter-roberta-base-sentiment-latest", device = device)

# Temporary test
test_text = "This sentiment analyzer API is awesome!"
print(sentiment_pipeline(test_text))