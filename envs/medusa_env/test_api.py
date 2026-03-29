import os
from openai import OpenAI

for base in [
    "https://huggingface.co/api/inference-proxy/together/v1",
    "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-72B-Instruct/v1",
    "https://api-inference.huggingface.co/v1"
]:
    print(f"\nTesting {base}...")
    try:
        client = OpenAI(base_url=base, api_key=os.environ["HF_TOKEN"])
        res = client.chat.completions.create(
            model="Qwen/Qwen2.5-72B-Instruct",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5
        )
        print("SUCCESS!")
        break
    except Exception as e:
        print(f"FAILED: {e}")
