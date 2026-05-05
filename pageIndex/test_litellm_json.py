import litellm
import json

response = litellm.completion(
    model="ollama_chat/gemma4:latest",
    messages=[
        {
            "role": "system",
            "content": "Return only valid JSON. No markdown. No explanation.",
        },
        {
            "role": "user",
            "content": 'Return JSON exactly like this schema: {"toc_detected": true}',
        },
    ],
    api_base="http://localhost:11434",
    temperature=0,
    format="json",
)

content = response["choices"][0]["message"]["content"]

print("RAW OUTPUT:")
print(repr(content))

print("\nJSON PARSE:")
print(json.loads(content))
