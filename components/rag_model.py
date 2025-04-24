import google.generativeai as genai
import os

class RAGModel:
    def __init__(self, model_name="models/gemini-2.5-flash-preview-04-17"):
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel(model_name)

    def generate_answer(self, question, context):
        prompt = f"""You are a research assistant. You are provided with context whenever any question is asked. This context can be used for answering the questions if relevant otherwise it can be ignored. 
{context}
Given above is the context. The context is to help you and the answer should not be totally dependent on the context but if the context is useful in generating the answer then use it. Don't mention explicitly that the context is provided to you. 
Answer the following Question in detail:
{question}
"""
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f" Gemini API error: {e}"
