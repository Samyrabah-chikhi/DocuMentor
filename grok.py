from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("API_KEY")

client = Groq(api_key=API_KEY)
completion = client.chat.completions.create(
    model="compound-beta-kimi",
    messages=[
      {
        "role": "user",
        "content": "Hello there, tell me more about yourself"
      }
    ],
    temperature=0.6,
    max_completion_tokens=4096,
    top_p=1,
    stream=True,
    stop=None,
)

for chunk in completion:
    print(chunk.choices[0].delta.content or "", end="")
