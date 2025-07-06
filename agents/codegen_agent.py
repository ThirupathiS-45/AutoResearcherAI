# agents/codegen_agent.py

import google.generativeai as genai
from config import GEMINI_API_KEY
from dotenv import load_dotenv
import os

# Load .env (optional if already loaded)
load_dotenv()

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Use the correct working model
model = genai.GenerativeModel("models/gemini-1.5-flash")

def generate_code_from_paper(summary, instructions=None):
    prompt = f"""
You are an AI research engineer.

Generate clean, working Python code based on this research summary:

Research Summary:
{summary}

Instructions (optional):
{instructions or "Use PyTorch and make it simple."}

Return only code.
    """

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print("❌ Error from Gemini:", e)
        return None
