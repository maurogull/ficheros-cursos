import requests
import google.generativeai as genai

def enviar_prompt(api_key, p):
    """Envía mensajes a Gemini API"""

    genai.configure(api_key=api_key)               

    print(f"Enviando peticion a Gemini" )
    
    try:
        response = genai.GenerativeModel('gemini-2.5-flash').generate_content(p)
    except Exception as e:
        print(f"Gemini API error: {e}")
        exit(1)

    return response.text

API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

prompt = "¿Cual es la capital de Francia?"
resp = enviar_prompt(API_KEY, prompt)

print(f"Respuesta del modelo: {resp}")
