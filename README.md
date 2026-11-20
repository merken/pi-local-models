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
    "url": "http://127.0.0.1:8080"
}
```

If the engine needs a key, you must also say how the key is sent:
```
{
    "url": "http://127.0.0.1:8080",
    "apiKey": "1234",
    "apiKeyHeader": "bearer"
}
```

See a full example [here](./local-models.json)

Store the file at user level: ~/.pi/agent/local-models.json
Or at project level: ./.pi/agent/local-models.json

Run pi and use the `/model` command to select any discovered [local-model].

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

Ollama and llama-swap report no context length in `/v1/models`. The extension then
reads `context_length` from `POST /api/show`. llama-swap entries with
`capabilities.embeddings`, `reranker`, or `decisions` are dropped: they cannot answer a
chat request.

Fields the engine does not report fall back to `local-models.json` (`contextWindow`,
`maxTokens`, `reasoning`, `input`).

## Optional config keys

| Key | Default | Meaning |
| --- | --- | --- |
| `apiKeyHeader` | required with `apiKey` | `"bearer"` sends `Authorization: Bearer`, `"x-api-key"` sends `x-api-key`, `"both"` sends both. Ignored when there is no `apiKey`. |
| `timeoutMs` | `10000` | How long the engine may take to answer. The timer stops as soon as a response arrives, so a slow body does not count. |
| `retry` | `{ "attempts": 3, "delayMs": 1500 }` | Retry count for a refused connection or a 5xx. Timeouts and 4xx are not retried. |
| `contextProbe` | `true` | Call `POST /api/show` for models that report no context length (Ollama). Set `false` to skip it. |
| `maxTokensField` | unset | Set `"max_completion_tokens"` if your engine rejects `max_tokens`. |

`url` and `apiKey` support `$VAR` and `${VAR}` from the environment:

```json
{
    "url": "$LOCAL_LLM_URL",
    "apiKey": "$LOCAL_LLM_KEY",
    "retry": { "attempts": 4, "delayMs": 2000 }
}
```

llama.cpp reports the GGUF file path as the model id. When the server also lists a
clean alias (no `/`, not ending in `.gguf`), the extension uses that alias as the id.
When splash reports `reasoning.efforts`, those efforts become pi thinking levels.

## Config validation

`readConfig` checks the file before any request. It stops with a message that names the
file when:

- the file has no `url`, or the `url` does not start with `http://` or `https://`;
- the file has `apiKey` but no `apiKeyHeader`;
- `apiKeyHeader` is not `bearer`, `x-api-key`, or `both`;
- the file is not valid JSON.

An unset environment variable is not expanded, so `"url": "$LOCAL_LLM_URL"` fails with
an invalid url instead of calling a nonsense endpoint.

## Show the current catalog

```
/local-models
```

This prints the engine url, the config file in use, and for each model: id, context
window, max tokens, input modalities, thinking support, and quantization.

## Refresh the model list

The engine often starts after Pi. Run this command in the TUI:

```
/local-models-refresh
```

The command re-reads `local-models.json`, fetches `/v1/models` again, and replaces the
`[local-model]` entries. Then run `/model` to select one. Pi also calls the provider's
`refreshModels` during its normal model refresh, so `/model` picks up newly loaded
models without a restart.

## Tests

```
npm test           # node --test tests/
npm run typecheck  # tsc --noEmit on the extension and the tests
```

`tests/fixtures/` holds one captured `/v1/models` response per engine: splash, oMLX,
llama.cpp, ollama, llama-swap, and mtp-lx. Each provider is tested twice: once on the
mapping functions, and once end to end with a stubbed `fetch` and a temp config file.