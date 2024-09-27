import os
import base64
from io import BytesIO
import json

from langchain_openai import ChatOpenAI
from PIL import Image


def get_llm(model_name="gpt-4o", use_groq=False, temperature=0.5, max_tokens=4096):
    groq_api_key = os.environ.get("GROQ_API_KEY", "dummy")
    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        openai_api_base="https://api.groq.com/openai/v1" if use_groq else None,
        openai_api_key=groq_api_key if use_groq else None,
        max_tokens=max_tokens
    )
    return llm


def pil_image_to_base64(image: Image.Image, image_format="PNG"):
    buffered = BytesIO()
    image.save(buffered, format=image_format)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def convert_response_output(response):
    try:
        output = json.loads(response['output'])
    except json.JSONDecodeError:
        output = response['output']

    return {"response": output}
