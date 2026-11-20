# Refresh the local provider model list in PI coding agent

Currently, when the `url` or `apiKey` fron the config `/Users/merken/code/pi-local-models/local-models.json` do not result in a successfull call to '/v1/models' in function `listModels` from the extension `/Users/merken/code/pi-local-models/extensions/local-models.ts`, it results in the error: `No local models were discovered`.

Add a Pi command to refresh to local models list and try again, sometimes the local inference enging (ollama, llamacpp, omlx) is not yet up and running.
We want to avoid having to quit PI and restart it again and be able to refresh within the TUI.


# Fix the PI upgrade

The current extension file at `/Users/merken/code/pi-local-models/extensions/local-models.ts` fails to compile at `line 167`, the `models` array is not matching the typescript interface `ProviderModelConfig` from `/Users/merken/code/pi-local-models/node_modules/@earendil-works/pi-coding-agent/dist/core/extensions/types.d.ts`.

The `splash` local inference engine provides the following response for `/v1/models`:
```json
{"object":"list","data":[{"id":"unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M","object":"model","created":0,"owned_by":"splash","max_model_len":196608,"context_length":196608,"vision":true,"input_modalities":["text","image","pdf"],"limits":{"request_bytes":134217728,"images":64,"image_bytes":33554432,"pdf_bytes":67108864,"pdf_pages":64},"reasoning":{"thinks":true,"efforts":[],"default":null}}],"models":[{"name":"unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M","description":"Splash resident model","release_date":""}]}
```

`oMLX` provides:
```json
{"object":"list","data":[{"id":"Jundot--Qwen3.8-27B-oQ4e-fp16-mtp","object":"model","created":1795159475,"owned_by":"omlx","model_type":"vlm","max_model_len":131072},{"id":"Jundot--Qwen3.8-27B-oQ4e-mtp","object":"model","created":1795159475,"owned_by":"omlx","model_type":"vlm","max_model_len":131072},{"id":"Jundot--Qwen3.8-27B-oQ4h-mtp","object":"model","created":1795159475,"owned_by":"omlx","model_type":"vlm","max_model_len":131072},{"id":"Jundot--gemma-4-26B-A4B-it-oQ4e-mtp","object":"model","created":1795159475,"owned_by":"omlx","model_type":"vlm","max_model_len":131072},{"id":"Qwen3.6-35B-A3B-UD-Q4_K_M","object":"model","created":1795159476,"owned_by":"omlx","model_type":"vlm","max_model_len":131072},{"id":"incoai--Qwen3.6-35B-A3B-DFlash2","object":"model","created":1795159476,"owned_by":"omlx","model_type":"llm","max_model_len":131072},{"id":"incoai--Qwen3.8-27B-DFlash2","object":"model","created":1795159476,"owned_by":"omlx","model_type":"llm","max_model_len":131072},{"id":"mlx-community--Qwen3.8-27B-4bit","object":"model","created":1795159476,"owned_by":"omlx","model_type":"vlm","max_model_len":163840}]}
```

llamacpp provides:
```json
{"models":[{"name":"/Users/merken/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-GGUF/snapshots/a483e9e6cbd595906af30beda3187c2663a1118c/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf","model":"/Users/merken/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-GGUF/snapshots/a483e9e6cbd595906af30beda3187c2663a1118c/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf","modified_at":"","size":"","digest":"","type":"model","description":"","tags":[""],"capabilities":["completion"],"parameters":"","details":{"parent_model":"","format":"gguf","family":"","families":[""],"parameter_size":"","quantization_level":""}}],"object":"list","data":[{"id":"/Users/merken/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-GGUF/snapshots/a483e9e6cbd595906af30beda3187c2663a1118c/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf","aliases":["/Users/merken/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-GGUF/snapshots/a483e9e6cbd595906af30beda3187c2663a1118c/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"],"tags":[],"object":"model","architecture":{"input_modalities":["text"],"output_modalities":["text"]},"created":1795159678,"owned_by":"llamacpp","meta":{"vocab_type":2,"n_vocab":248320,"n_ctx":262144,"n_ctx_train":262144,"n_embd":2048,"n_params":34660610688,"size":22123538944,"ftype":"Q4_K - Medium"}}]}
```

Map the JSON response to the `ProviderModelConfig` and make sure all provider responses are supported `llamacpp`, `oMLX`, `splash`.