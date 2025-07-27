import httpx
import json

url = "http://localhost:11434/api/generate"

question = "Tell me 2 sentences"

data = {
    "model": "gemma3:1b",
    "prompt": question,
    "stream": True
}

full_response = ""
with httpx.stream("POST", url, json=data, timeout=None) as response:
    print("Status:", response.status_code)
    for line in response.iter_lines():
        if line:
            chunk = json.loads(line)
            text_part = chunk.get("response", "")
            print(chunk.get("response", ""), end="", flush=True)
            full_response += text_part
    print("\n\n")         
print(full_response)