import asyncio
import httpx
import os
import yaml

CONFIG_PATH = "config.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# Read API Key
_env_file = ".env"
api_key = ""
if os.path.exists(_env_file):
    with open(_env_file, "r") as f:
        for line in f:
            if line.startswith("GEMINI_API_KEY="):
                api_key = line.split("=", 1)[1].strip()

async def test():
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # We will test a system prompt and a user prompt
    payload = {
        "model": "gemini-3.5-flash",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello."}
        ]
    }
    
    url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, headers=headers, json=payload)
        print("Status Code:", resp.status_code)
        print("Response Body:", resp.text)

asyncio.run(test())
