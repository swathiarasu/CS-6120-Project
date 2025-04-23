# from transformers import AutoTokenizer, AutoModelForCausalLM

# class RAGModel:
#     def __init__(self):
#         # Gemma is a decoder-only model, so use CausalLM
#         self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it", use_auth_token=True)
#         self.model = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-it", use_auth_token=True)

#     def generate_answer(self, question, context):
#         # Format the prompt for decoder-only model
#         prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
#         inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
#         outputs = self.model.generate(**inputs, max_new_tokens=200)
#         return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


# from transformers import AutoTokenizer, AutoModelForCausalLM

# class RAGModel:
#     def __init__(self, device="cpu"):
#         self.device = device
#         self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token=True)
#         self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", token=True).to(device)

#     def generate_answer(self, question, context):
#         prompt = f"Question: {question}\n\nAnswer:"
#         inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
#         outputs = self.model.generate(**inputs, max_new_tokens=200)
#         return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch

# class RAGModel:
#     def __init__(self, model_id="allenai/sciphi-llama-2-7b", device="cuda" if torch.cuda.is_available() else "cpu"):
#         self.device = device
#         self.tokenizer = AutoTokenizer.from_pretrained(model_id)
#         self.model = AutoModelForCausalLM.from_pretrained(
#             model_id,
#             device_map="auto",
#             torch_dtype=torch.float16 if "cuda" in device else torch.float32
#         )
#         self.model.eval()

#     def generate_answer(self, question, context):
#         prompt = f"""You are a scientific research assistant.

# Context:
# {context}

# Question:
# {question}

# Answer:"""

#         inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
#         with torch.no_grad():
#             outputs = self.model.generate(
#                 **inputs,
#                 max_new_tokens=300,
#                 do_sample=True,
#                 temperature=0.7,
#                 top_p=0.95,
#                 eos_token_id=self.tokenizer.eos_token_id,
#                 pad_token_id=self.tokenizer.eos_token_id
#             )

#         return self.tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()




# from openai import OpenAI
# import os

# class RAGModel:
#     def __init__(self, model_name="gpt-3.5-turbo"):
#         self.model_name = model_name
#         self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

#     def generate_answer(self, question, context):
#         system_prompt = "You are an expert AI assistant helping answer research questions based on documents."
#         user_prompt = f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"

#         response = self.client.chat.completions.create(
#             model=self.model_name,
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": user_prompt}
#             ],
#             temperature=0.3,
#             max_tokens=500
#         )
#         return response.choices[0].message.content.strip()

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
