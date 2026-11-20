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

## Model discovery mapping

Each engine reports models slightly differently. The extension reads both `data[]` and
`models[]`, then maps the fields to the pi `ProviderModelConfig`:

| Field | splash | oMLX | llama.cpp |
| --- | --- | --- | --- |
| `id` | `data[].id` | `data[].id` | `data[].id` (GGUF path) |
| `name` | `data[].id` | `data[].id` | `models[].name`, path trimmed |
| `contextWindow` | `context_length` / `max_model_len` | `max_model_len` | `meta.n_ctx` |
| `input` | `input_modalities`, `vision` | `model_type` (`vlm` = image) | `architecture.input_modalities` |
| `reasoning` | `reasoning.thinks` | config value | config value |
| `inputLimits` | `limits.request_bytes`, `limits.images`, `limits.image_bytes` | - | - |

Fields the engine does not report fall back to `local-models.json` (`contextWindow`,
`maxTokens`, `reasoning`, `input`).

## Tests

```
npm test           # node --test tests/
npm run typecheck  # tsc --noEmit on the extension and the tests
```

`tests/fixtures/` holds one captured `/v1/models` response per engine: splash, oMLX,
llama.cpp, ollama, llama-swap, and mtp-lx. Each provider is tested twice: once on the
mapping functions, and once end to end with a stubbed `fetch` and a temp config file.