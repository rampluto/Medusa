import os
import requests

HF_TOKEN = os.environ["HF_TOKEN"]
BASE_URL="https://rampluto-medusa-env.hf.space"

session = requests.Session()
session.headers.update({
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json",
})

session.verify = False


r = session.post(f"{BASE_URL}/reset", timeout=30)
print("reset:", r.status_code, r.text)
