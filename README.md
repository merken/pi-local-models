# pi-local-models
Local model discovery for PI Coding Agent

This supports any openai-api spec provider
- llama.cpp
- Ollama
- oMLX
- llama-swap
- MTPLX

## Installation:

```
pi install npm:pi-local-models
```

## Configuration

Create a minimal config for your local LLM harness:
```
{
    "url": "http://127.0.0.1:8080",
    "apiKey": "1234"
}
```

See a full example [here](./local-models.json)

Store the file at user level: ~/.pi/agent/local-models.json
Or at project level: ./.pi/agent/local-models.json

Run pi and perform the /models command to select any discovered [local-model].