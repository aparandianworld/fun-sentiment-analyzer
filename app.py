from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel
import gradio as gr
import uvicorn
import plotly.express as px
from typing import List

app = FastAPI(title = "Fun Sentiment Analyzer API")