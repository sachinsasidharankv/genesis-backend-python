import base64
from io import BytesIO

from langchain_openai import ChatOpenAI
from PIL import Image


def get_llm(model_name="gpt-4o", temperature=0.5, max_tokens=4096):
    return ChatOpenAI(model_name=model_name, temperature=temperature, max_tokens=max_tokens)


def pil_image_to_base64(image: Image.Image, image_format="PNG"):
    buffered = BytesIO()
    image.save(buffered, format=image_format)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')
