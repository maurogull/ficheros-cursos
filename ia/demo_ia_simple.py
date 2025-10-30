import requests

def enviar_prompt(ip, modelo, p):
    """Envía mensajes estilo chat a LM Studio y devuelve la respuesta del LLM"""
    
    url = f"{ip}/v1/chat/completions"

    datosenviar = {
        "model": modelo,
        "messages": [
            {"role": "system", "content": "Contesta en español y se conciso en tus respuestas"},
            {"role": "user", "content": p}
            ]
    }

    print(f"Enviando peticion al modelo: {modelo}" )
    resultado = requests.post(url, json=datosenviar)
    resultado_decodificado = resultado.json()
    return resultado_decodificado

    # return resultado_decodificado["choices"][0]["message"]["content"]


prompt = "¿Cual es la capital de Francia?"
resp = enviar_prompt("http://127.0.0.1:12345", "deepseek-r1-0528-qwen3-8b", prompt)
print(f"Respuesta del modelo: {resp}")

