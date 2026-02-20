import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

key = os.getenv("GOOGLE_API_KEY")
print("API KEY:", key[:10], "...")  # Just to confirm it's loading

genai.configure(api_key=key)

try:
    models = genai.list_models()
    for m in models:
        print(m.name)
except Exception as e:
    print("ERROR:", e)
