"""
Mi primer chatbot GemUzi
Uso: python scripts/prueba_gemini.py [--local]
"""

import sys
import os
from dotenv import load_dotenv

USE_LOCAL = "--local" in sys.argv

load_dotenv()

if USE_LOCAL:
    import ollama
else:
    from google import genai
    from tenacity import retry, stop_after_attempt, wait_exponential
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("No se encontró GEMINI_API_KEY en .env")
    client = genai.Client(api_key=api_key)

historial = []

print(f"Modo: {'Ollama local' if USE_LOCAL else 'Gemini API'}")
print("Escribe 'exit' para terminar.\n")

while True:
    usuario = input("|Usuario|: ")

    if usuario.lower() == "exit":
        print("¡Hasta luego!")
        break

    print("|GemUzi|: ", end="", flush=True)
    respuesta = ""

    if USE_LOCAL:
        historial.append({"role": "user", "content": usuario})
        stream = ollama.chat(model="gemma4:e4b-it-qat", messages=historial, stream=True)
        for chunk in stream:
            token = chunk["message"]["content"]
            print(token, end="", flush=True)
            respuesta += token
        historial.append({"role": "model", "content": respuesta})
    else:
        historial.append({"role": "user", "parts": [{"text": usuario}]})

        @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
        def _llamar_ia():
            return client.models.generate_content(
                model="gemma-4-31b-it",
                contents=historial,
                stream=True, # Apply streaming to get tokens as they arrive and not wait for the entire response to be generated
            )
        for chunk in _llamar_ia():
            token = chunk.text
            print(token, end="", flush=True)
            respuesta += token
        historial.append({"role": "model", "parts": [{"text": respuesta}]})
    print("\n")