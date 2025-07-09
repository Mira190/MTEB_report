import requests

url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
headers = {"Authorization": "sk-e7d4ad3c14aa46bd8a99689470bad257"}

data = {
    "model": "qwen3-32b",
    "input": {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ]
    },
    "parameters": {
        "enable_thinking": False
    }
}

response = requests.post(url, headers=headers, json=data)
print(response.json())
