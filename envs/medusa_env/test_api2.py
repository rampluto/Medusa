import os
import requests

token = os.environ.get("HF_TOKEN")
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
data = {
    "model": "Qwen/Qwen2.5-72B-Instruct",
    "messages": [{"role": "user", "content": "Hi"}],
    "max_tokens": 5
}

urls = [
    "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-72B-Instruct/v1/chat/completions",
    "https://router.huggingface.co/hf-inference/models/Qwen/Qwen2.5-72B-Instruct/v1/chat/completions",
    "https://router.huggingface.co/hf-inference/v1/chat/completions"
]

for u in urls:
    print(f"\n--- {u} ---")
    try:
        r = requests.post(u, headers=headers, json=data)
        print(r.status_code)
        print(r.text[:200])
    except Exception as e:
        print(e)
