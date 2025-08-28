from google import genai
from google.genai import types
from PIL import Image
import io
import os
from dotenv import load_dotenv

# Load env vars from .env
load_dotenv()

gemini_api_key = os.environ.get("GEMINI_API_KEY")
if not gemini_api_key:
    raise RuntimeError("Set GEMINI_API_KEY in your environment or .env")

gemini_client = genai.Client(api_key=gemini_api_key)

# Caption the given image using Gemini API, model 2.5-flash
def get_llm_response(image_data: bytes) -> str:
    image = Image.open(io.BytesIO(image_data))
    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Part.from_bytes(
                data=image_data,
                mime_type='image/jpeg',
            ),
            'Return exactly one short, vivid caption. No lists, no alternatives.'
        ]
    )
    return response.text.strip()