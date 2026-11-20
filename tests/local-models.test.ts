// Unit tests for the local-models extension.
// Run with: npm test

import assert from "node:assert";
import { test } from "node:test";
import { mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { fileURLToPath } from "node:url";
import type { ExtensionAPI, ProviderConfig, ProviderModelConfig } from "@earendil-works/pi-coding-agent";
import extension, {
    isChatModel,
    listModels,
    modelInput,
    modelInputLimits,
    modelKey,
    modelReasoning,
    normalizeModels,
    readConfig,
    toProviderModelConfig,
    type LocalModelsConfig,
    type ServerModel,
} from "../extensions/local-models.ts";

type ChatModel = Extract<ProviderModelConfig, { type?: "chat" }>;

const fixturesDir = join(fileURLToPath(import.meta.url), "..", "fixtures");

function fixture(name: string): unknown {
    return JSON.parse(readFileSync(join(fixturesDir, name), "utf-8"));
}

// Config that mirrors local-models.json in the repo.
function config(overrides: Partial<LocalModelsConfig> = {}): LocalModelsConfig {
    return {
        url: "http://127.0.0.1:8000",
        apiKey: "1234",
        contextWindow: 190000,
        maxTokens: 65000,
        reasoning: true,
        input: ["text", "image"],
        ...overrides,
    };
}

// Normalized and mapped models for one provider response.
function registeredModels(name: string, cfg: LocalModelsConfig = config()): ChatModel[] {
    return normalizeModels(fixture(name))
        .map((model) => toProviderModelConfig(model, cfg))
        .filter((model): model is ProviderModelConfig => model !== undefined)
        .map((model) => model as ChatModel);
}

function byId(models: ChatModel[], id: string): ChatModel {
    const model = models.find((m) => m.id === id);
    assert.ok(model, `no model with id ${id}`);
    return model;
}

// Every field pi requires on a chat model entry.
function assertChatModel(model: ChatModel): void {
    assert.equal(typeof model.id, "string", "id must be a string");
    assert.ok(model.id.length > 0, "id must not be empty");
    assert.equal(typeof model.name, "string", "name must be a string");
    assert.ok(model.name.length > 0, "name must not be empty");
    assert.equal(typeof model.reasoning, "boolean", "reasoning must be a boolean");
    assert.ok(Array.isArray(model.input), "input must be an array");
    for (const modality of model.input) {
        assert.ok(modality === "text" || modality === "image", `unsupported input modality: ${modality}`);
    }
    assert.ok(model.contextWindow > 0, "contextWindow must be positive");
    assert.ok(model.maxTokens > 0, "maxTokens must be positive");
    assert.ok(model.maxTokens <= model.contextWindow, "maxTokens must fit inside contextWindow");
    assert.deepEqual(model.cost, { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 });
    const compat = model.compat as { supportsDeveloperRole?: boolean; supportsReasoningEffort?: boolean } | undefined;
    assert.equal(compat?.supportsDeveloperRole, false);
    assert.equal(compat?.supportsReasoningEffort, false);
    if (model.inputLimits) {
        assert.ok(typeof model.inputLimits.maxRequestBytes === "number" || typeof model.inputLimits.images === "object", "inputLimits must carry a limit");
    }
}

function writeConfig(dir: string, cfg: Record<string, unknown>): void {
    const agentDir = join(dir, ".pi", "agent");
    mkdirSync(agentDir, { recursive: true });
    writeFileSync(join(agentDir, "local-models.json"), JSON.stringify(cfg));
}

// Run the extension with a stubbed /v1/models response and a temp config file.
async function registerProvider(payload: unknown, cfg: Record<string, unknown>): Promise<ProviderConfig | undefined> {
    const dir = mkdtempSync(join(tmpdir(), "pi-local-models-"));
    writeConfig(dir, cfg);

    const originalCwd = process.cwd();
    const originalFetch = globalThis.fetch;
    let registered: ProviderConfig | undefined;

    globalThis.fetch = async () =>
        ({ ok: true, json: async () => payload, text: async () => JSON.stringify(payload) }) as unknown as Response;
    process.chdir(dir);

    try {
        const api = {
            registerProvider: (name: string, providerConfig: ProviderConfig) => {
                registered = providerConfig;
            },
        } as unknown as ExtensionAPI;
        await extension(api);
    } finally {
        process.chdir(originalCwd);
        globalThis.fetch = originalFetch;
    }
    return registered;
}

test("splash: data[] and models[] are merged into one entry", () => {
    const entries = normalizeModels(fixture("splash.json"));
    assert.equal(entries.length, 1);
    assert.equal(modelKey(entries[0]), "unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M");
});

test("splash: context, vision, reasoning, and limits are mapped", () => {
    const models = registeredModels("splash.json");
    assert.equal(models.length, 1);
    const model = models[0];
    assertChatModel(model);
    assert.equal(model.id, "unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M");
    assert.equal(model.name, "unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M");
    assert.equal(model.contextWindow, 196608);
    assert.equal(model.maxTokens, 65000);
    assert.deepEqual(model.input, ["text", "image"]);
    assert.equal(model.reasoning, true);
    assert.deepEqual(model.inputLimits, {
        maxRequestBytes: 134217728,
        images: { maxPerRequest: 64, resize: { maxBytes: 33554432 } },
    });
});

test("splash: reasoning.thinks decides thinking support", () => {
    assert.equal(modelReasoning({ reasoning: { thinks: true, efforts: [], default: null } }, false), true);
    assert.equal(modelReasoning({ reasoning: { thinks: false } }, true), false);
    assert.equal(modelReasoning({ reasoning: true }, false), true);
    assert.equal(modelReasoning({}, true), true);
});

test("splash: pdf input is dropped, image input is kept", () => {
    const model = normalizeModels(fixture("splash.json"))[0];
    assert.deepEqual(modelInput(model, ["text"]), ["text", "image"]);
    assert.equal(modelInputLimits(model)?.maxRequestBytes, 134217728);
});

test("oMLX: all eight models are discovered", () => {
    const models = registeredModels("omlx.json");
    assert.equal(models.length, 8);
    models.forEach(assertChatModel);
});

test("oMLX: vlm models accept images, llm models do not", () => {
    const models = registeredModels("omlx.json");
    assert.deepEqual(byId(models, "Jundot--Qwen3.8-27B-oQ4e-fp16-mtp").input, ["text", "image"]);
    assert.deepEqual(byId(models, "incoai--Qwen3.6-35B-A3B-DFlash2").input, ["text"]);
    assert.deepEqual(byId(models, "incoai--Qwen3.8-27B-DFlash2").input, ["text"]);
});

test("oMLX: contextWindow comes from max_model_len", () => {
    const models = registeredModels("omlx.json");
    assert.equal(byId(models, "Jundot--Qwen3.8-27B-oQ4e-mtp").contextWindow, 131072);
    assert.equal(byId(models, "mlx-community--Qwen3.8-27B-4bit").contextWindow, 163840);
});

test("oMLX: reasoning falls back to the config value", () => {
    const on = registeredModels("omlx.json", config({ reasoning: true }));
    const off = registeredModels("omlx.json", config({ reasoning: false }));
    assert.equal(byId(on, "incoai--Qwen3.8-27B-DFlash2").reasoning, true);
    assert.equal(byId(off, "incoai--Qwen3.8-27B-DFlash2").reasoning, false);
});

test("llama.cpp: data[] and models[] describe one model", () => {
    const entries = normalizeModels(fixture("llamacpp.json"));
    assert.equal(entries.length, 1);
    assert.equal(isChatModel(entries[0]), true);
});

test("llama.cpp: the GGUF path is the id and the file name is the display name", () => {
    const model = registeredModels("llamacpp.json")[0];
    assert.ok(model.id.endsWith("Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"), "id keeps the full GGUF path");
    assert.equal(model.name, "Qwen3.6-35B-A3B-UD-Q4_K_M");
});

test("llama.cpp: context comes from meta.n_ctx and input from architecture", () => {
    const model = registeredModels("llamacpp.json")[0];
    assertChatModel(model);
    assert.equal(model.contextWindow, 262144);
    assert.deepEqual(model.input, ["text"]);
    assert.equal(model.inputLimits, undefined);
});

test("llama.cpp: maxTokens is clamped to the context window", () => {
    const model = registeredModels("llamacpp.json", config({ maxTokens: 300000 }))[0];
    assert.equal(model.maxTokens, 262144);
});

test("ollama: models[] entries use the name as the id", () => {
    const models = registeredModels("ollama.json");
    assert.equal(models.length, 2);
    models.forEach(assertChatModel);
    assert.equal(models[0].id, "llama3.2:latest");
    assert.equal(models[0].name, "llama3.2:latest");
});

test("ollama: context and modalities fall back to the config", () => {
    const models = registeredModels("ollama.json", config({ contextWindow: 8192, maxTokens: 4096, input: ["text"] }));
    const plain = byId(models, "llama3.2:latest");
    assert.equal(plain.contextWindow, 8192);
    assert.equal(plain.maxTokens, 4096);
    assert.deepEqual(plain.input, ["text"]);

    // "vision" in the capabilities list means the model takes images.
    const vision = byId(models, "llama3.2-vision:latest");
    assert.deepEqual(vision.input, ["text", "image"]);
});

test("llama-swap: capabilities.in sets modalities and context_length sets the window", () => {
    const models = registeredModels("llama-swap.json");
    assert.equal(models.length, 1);
    const chat = models[0];
    assertChatModel(chat);
    assert.equal(chat.id, "qwen3-8b-vision");
    assert.equal(chat.contextWindow, 60160);
    assert.deepEqual(chat.input, ["text", "image"]);
});

test("llama-swap: embedding and reranker models are not chat models", () => {
    const entries = normalizeModels(fixture("llama-swap.json"));
    const ids = entries.map((entry) => modelKey(entry));
    assert.deepEqual(ids, ["qwen3-8b-vision", "gte-modernbert", "gte-reranker"]);
    assert.deepEqual(entries.map(isChatModel), [true, false, false]);
    assert.equal(registeredModels("llama-swap.json").length, 1);
});

test("mtp-lx: vlm models are kept and embedding models are skipped", () => {
    const entries = normalizeModels(fixture("mtp-lx.json"));
    assert.deepEqual(entries.map(isChatModel), [true, false]);
    const models = registeredModels("mtp-lx.json");
    assert.equal(models.length, 1);
    assert.equal(models[0].contextWindow, 131072);
    assert.deepEqual(models[0].input, ["text", "image"]);
});

test("unknown engine: a bare entry uses the config values", () => {
    const entries = normalizeModels({ data: [{ id: "my-model", object: "model", created: 0 }] });
    const [model] = entries
        .map((entry) => toProviderModelConfig(entry, config({ contextWindow: 32768, maxTokens: 8192, reasoning: false, input: ["text"] })))
        .filter((model): model is ProviderModelConfig => model !== undefined)
        .map((model) => model as ChatModel);
    assertChatModel(model);
    assert.equal(model!.contextWindow, 32768);
    assert.equal(model!.maxTokens, 8192);
    assert.equal(model!.reasoning, false);
    assert.deepEqual(model!.input, ["text"]);
    assert.equal(model!.name, "my-model");
});

test("unknown engine: defaults apply when the config has no contextWindow", () => {
    const [model] = normalizeModels({ data: [{ id: "tiny" }] })
        .map((entry) => toProviderModelConfig(entry, config({ contextWindow: undefined, maxTokens: 999999 })))
        .filter((model): model is ProviderModelConfig => model !== undefined)
        .map((model) => model as ChatModel);
    assert.equal(model.contextWindow, 128000);
    assert.equal(model.maxTokens, 128000);
});

test("edge cases: entries without an id or with a failed load are skipped", () => {
    const entries = normalizeModels({
        data: [
            { object: "model" },
            { id: "ok-model" },
            { id: "broken-model", status: { value: "failed", failed: true, exit_code: 1 } },
        ],
    });
    const models = entries
        .map((entry) => toProviderModelConfig(entry, config()))
        .filter((model): model is ProviderModelConfig => model !== undefined)
        .map((model) => model as ChatModel);
    assert.equal(models.length, 1);
    assert.equal(models[0].id, "ok-model");
});

test("edge cases: duplicate ids in data[] and models[] are merged", () => {
    const entries = normalizeModels({
        data: [{ id: "dup", max_model_len: 4096 }],
        models: [{ name: "dup", description: "same model" }],
    });
    assert.equal(entries.length, 1);
    assert.equal(entries[0].max_model_len, 4096);
    assert.equal(entries[0].description, "same model");
});

test("edge cases: an empty response yields no models", () => {
    assert.deepEqual(normalizeModels({ object: "list", data: [] }), []);
    assert.deepEqual(normalizeModels({}), []);
});

test("readConfig: the trailing /v1 is removed and defaults are applied", () => {
    const dir = mkdtempSync(join(tmpdir(), "pi-local-models-config-"));
    writeConfig(dir, { url: "http://127.0.0.1:8000/v1" });
    const cfg = readConfig(dir);
    assert.equal(cfg.url, "http://127.0.0.1:8000");
    assert.equal(cfg.maxTokens, 16384);
    assert.equal(cfg.reasoning, false);
    assert.deepEqual(cfg.input, ["text"]);
    assert.equal(cfg.contextWindow, undefined);
    assert.equal(cfg.apiKey, undefined);
});

test("readConfig: input keeps only text and image", () => {
    const dir = mkdtempSync(join(tmpdir(), "pi-local-models-config-"));
    writeConfig(dir, { url: "http://ai.local:8700", apiKey: "1234", contextWindow: 190000, maxTokens: 65000, reasoning: true, input: ["text", "image", "pdf", "audio"] });
    const cfg = readConfig(dir);
    assert.deepEqual(cfg.input, ["text", "image"]);
    assert.equal(cfg.contextWindow, 190000);
    assert.equal(cfg.maxTokens, 65000);
    assert.equal(cfg.reasoning, true);
});

test("readConfig: the config file is read from the home directory when the project has none", () => {
    const dir = mkdtempSync(join(tmpdir(), "pi-local-models-home-"));
    const home = mkdtempSync(join(tmpdir(), "pi-local-models-home-"));
    writeConfig(home, { url: "http://127.0.0.1:8080/v1", maxTokens: 2048 });
    const originalHome = process.env.HOME;
    process.env.HOME = home;
    try {
        const cfg = readConfig(dir);
        assert.equal(cfg.url, "http://127.0.0.1:8080");
        assert.equal(cfg.maxTokens, 2048);
    } finally {
        process.env.HOME = originalHome;
    }
});

test("listModels: GET /v1/models is called with the api key header", async () => {
    const originalFetch = globalThis.fetch;
    let request: { url: string; method: string; headers: Record<string, string> } | undefined;
    globalThis.fetch = ((url: string, options: { method?: string; headers?: HeadersInit }) => {
        request = { url, method: options.method ?? "GET", headers: (options.headers ?? {}) as Record<string, string> };
        return Promise.resolve({ ok: true, json: () => fixture("omlx.json"), text: () => "" } as unknown as Response);
    }) as unknown as typeof fetch;
    try {
        const models = await listModels(config());
        assert.equal(request?.url, "http://127.0.0.1:8000/v1/models");
        assert.equal(request?.method, "GET");
        assert.equal(request?.headers["x-api-key"], "1234");
        assert.equal(models.length, 8);
    } finally {
        globalThis.fetch = originalFetch;
    }
});

test("listModels: an unavailable server throws with the status code", async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = (() => Promise.resolve({ ok: false, status: 503, text: async () => "model not loaded" })) as unknown as typeof fetch;
    try {
        await assert.rejects(() => listModels(config()), /503: model not loaded/);
    } finally {
        globalThis.fetch = originalFetch;
    }
});

const providers: Array<{ name: string; fixture: string }> = [
    { name: "splash", fixture: "splash.json" },
    { name: "oMLX", fixture: "omlx.json" },
    { name: "llama.cpp", fixture: "llamacpp.json" },
    { name: "ollama", fixture: "ollama.json" },
    { name: "llama-swap", fixture: "llama-swap.json" },
    { name: "mtp-lx", fixture: "mtp-lx.json" },
];

for (const provider of providers) {
    test(`${provider.name}: the provider is registered with valid model entries`, async () => {
        const registered = await registerProvider(fixture(provider.fixture), {
            url: "http://127.0.0.1:8000",
            apiKey: "1234",
            contextWindow: 190000,
            maxTokens: 65000,
            reasoning: true,
            input: ["text", "image"],
        });
        assert.ok(registered, "registerProvider was called");
        assert.equal(registered.baseUrl, "http://127.0.0.1:8000/v1");
        assert.equal(registered.api, "openai-completions");
        assert.equal(registered.apiKey, "1234");
        assert.ok(registered.models && registered.models.length > 0, "at least one model is registered");
        (registered.models as ProviderModelConfig[]).forEach((model) => assertChatModel(model as ChatModel));
    });
}

test("a server with no models does not register a provider", async () => {
    const registered = await registerProvider({ object: "list", data: [] }, { url: "http://127.0.0.1:8000" });
    assert.equal(registered, undefined);
});
