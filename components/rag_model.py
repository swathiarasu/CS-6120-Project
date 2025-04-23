import google.generativeai as genai
import os

class RAGModel:
    def __init__(self, model_name="models/gemini-1.5-flash-latest"):
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel(model_name)

    def generate_answer(self, question, context):
        prompt = f"""You are a helpful research assistant.

Context:
{context}

Question:
{question}

Answer:"""
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f" Gemini API error: {e}"
