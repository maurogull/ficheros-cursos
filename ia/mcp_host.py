import asyncio
import json
import re
from time import sleep
import requests
from fastmcp import Client
import logging
from rich.console import Console
from rich.logging import RichHandler
import time

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
handler = RichHandler(
    console=Console(stderr=True),
    rich_tracebacks=True
)
formatter = logging.Formatter("%(message)s")
handler.setFormatter(formatter)
logger.handlers.clear()
logger.addHandler(handler)
logger.propagate = False


# --- wrapper para LM Studio ---
class LLM:
    #def __init__(self, model="meta-llama-3.1-8b-instruct", host="http://192.168.1.38:12345"):
    #def __init__(self, model="deepseek/deepseek-r1-0528-qwen3-8b", host="http://192.168.1.38:12345"):
    def __init__(self, model="deepseek-r1-0528-qwen3-8b", host="http://127.0.0.1:12345"):
        self.model = model
        self.host = host

    def complete(self, messages):
        """Envía mensajes estilo chat a LM Studio y devuelve la inferencia del LLM"""
        logger.info(f"Enviando peticion al modelo: {self.model}" )
        url = f"{self.host}/v1/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.8,
        }
        res = requests.post(url, json=payload, timeout=600)  # notar el timeout
        res.raise_for_status()
        data = res.json()
        return data["choices"][0]["message"]["content"]


# --- Llamar a un MCP server local ---
async def call_local_tool(tool_name: str, args: dict):
    client = Client("local_server.py")

    async with client:
        await client.ping()

        result = await client.call_tool(tool_name, args)
        return result.data

    
async def main():
    start_time = time.time()
    llm = LLM()

    tools_info = """
    Las herramientas disponibles son:
    - mauristica(number: float): Devuelve el resultado de aplicar la función Mauristica al numero dado.
    """

    system_msg = {
        "role": "system",
        "content": (
            "Eres un modelo que forma parte de un sistema con varias herramientas disponibles.\n"
            "Cuando necesites usar una herramienta responde solamente un diccionario JSON de la siguiente forma y sin agregar nada más antes ni después:\n"
            "{ \"action\": \"call_tool\", \"name\": \"tool_name\", \"tool_call_id\": \"call_id\", \"arguments\": { ... } }\n"
            f"{tools_info}"
            "No envies mas de un diccionario a la vez, hazlo en varias iteraciones, esperando la respuesta de uno antes de pedir la siguiente.\n"
            "Nunca mezcles lenguaje natural con JSON, si necesitas usar una herramienta responde solo con el JSON.\n"
            "Si no necesitas más herramientas para continuar, responde normalmente en lenguaje natural en español."
        )
    }

    user_msg = {"role": "user", "content": "Aplica la funcion mauristica al numero 335 y dime el resultado."}
    #user_msg = {"role": "user", "content": "Dime si el resultado de aplicar la funcion mauristica al numero 100 es mayor al del numero 300"}
    
    messages_big_history = [system_msg, user_msg]

    response = llm.complete(messages_big_history)
    logger.info(f"Respuesta del modelo: {response}")
    messages_big_history.append({"role": "assistant", "content": response})

    no_reasoning = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)  # limpiamos razonamiento (deepseek)

    iteraciones = 1

    while True:
        # el modelo nos está pidiendo usar una tool?
        try:
            data = json.loads(no_reasoning)
        except Exception as e:
            logger.info(f"Respuesta final obtenida, terminando." )
            break

        if iteraciones >= 5:
            logger.error("Iteraciones máximas alcanzadas, abortando.")
            break

        if data.get("action") != "call_tool":
            logger.error("Solicitud del modelo no entendible, abortando.")
            break
            
        tool_name = data["name"]
        args = data.get("arguments", {})
        id = data.get("tool_call_id", 1)
        logger.info(f"Modelo solicita usar tool: {tool_name} con args {args} e id {id}")

        tool_output = await call_local_tool(tool_name, args)
        logger.info(f"Resultado de la tool: {tool_output}")

        messages_big_history.append({"role": "tool", "tool_call_id": f"{id}", "content": f"{tool_output}"})

        # Devolvemos el resultado de la tool al modelo e iteramos de nuevo
        response = llm.complete(messages_big_history)

        iteraciones += 1

        logger.info(f"Respuesta del modelo: {response}")
        messages_big_history.append({"role": "assistant", "content": response})
        no_reasoning = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)

    end_time = time.time()
    logger.info(f"Segundos transcurridos: {end_time - start_time:.2f}")
    

if __name__ == "__main__":
    asyncio.run(main())
