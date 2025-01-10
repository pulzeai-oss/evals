import json
import requests

# Set up your API key (replace with your actual key)
api_key = "your-api-key"
url = "https://api.pulze.ai/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}",
    "Pulze-Feature-Flags": '{ "auto_tools": "true" }'
}
data = {
    "plugins": ["web-search"],
    "model": "openai/gpt-4o",
    "messages": [
            {"role": "user", "content": f"Tell me a joke."}
        ]
    # "temperature": temperature,
    # "max_tokens": max_tokens
}
response = requests.post(url, headers=headers, json=data)
response.raise_for_status()

response = requests.post(url, headers=headers, json=data, stream=True)
response.raise_for_status()

# Parse the JSON response
json_response = response.json()

# Print the JSON response
print(json.dumps(json_response, indent=4))
