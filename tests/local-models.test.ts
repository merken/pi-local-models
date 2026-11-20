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
    PROVIDER_NAME,
    discoverModels,
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

// A chat model entry: the ProviderModelConfig variant pi uses for text models.
type ChatModel = Extract<ProviderModelConfig, { type?: "chat" }>;
type CommandHandler = (args: string, ctx: unknown) => Promise<void>;

interface FakePi {
    providers: Map<string, ProviderConfig>;
    commands: Map<string, { description?: string; handler: CommandHandler }>;
}

interface RecordedRequest {
    url: string;
    method: string;
    headers: Record<string, string>;
    signal?: AbortSignal;
}

const fixturesDir = join(fileURLToPath(import.meta.url), "..", "fixtures");
let requests: RecordedRequest[] = [];

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

const testConfig: LocalModelsConfig = {
    url: "http://127.0.0.1:8000",
    apiKey: "1234",
    contextWindow: 190000,
    maxTokens: 65000,
    reasoning: true,
    input: ["text", "image"],
};

// Normalized and mapped models for one provider response.
function registeredModels(name: string, cfg: LocalModelsConfig = config()): ChatModel[] {
    return normalizeModels(fixture(name))
        .map((model) => toProviderModelConfig(model, cfg))
        .filter((model): model is ProviderModelConfig => model !== undefined)
        .map((model) => model as ChatModel);
}

function byId(models: ChatModel[], id: string): ChatModel {
    const model = models.find((model) => model.id === id);
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
        assert.ok(
            typeof model.inputLimits.maxRequestBytes === "number" || typeof model.inputLimits.images === "object",
            "inputLimits must carry a limit",
        );
    }
}

function writeConfig(dir: string, cfg: object): void {
    const agentDir = join(dir, ".pi", "agent");
    mkdirSync(agentDir, { recursive: true });
    writeFileSync(join(agentDir, "local-models.json"), JSON.stringify(cfg));
}

function stubFetch(payload: unknown): typeof fetch {
    return ((url: string, options: { method?: string; headers?: HeadersInit; signal?: AbortSignal }) => {
        requests.push({
            url,
            method: options.method ?? "GET",
            headers: (options.headers ?? {}) as Record<string, string>,
            signal: options.signal,
        });
        return Promise.resolve({ ok: true, json: () => payload, text: () => JSON.stringify(payload) } as unknown as Response);
    }) as unknown as typeof fetch;
}

function stubErrorResponse(status: number, text: string): typeof fetch {
    return (() =>
        Promise.resolve({ ok: false, status, text: () => text, json: () => ({}) } as unknown as Response)) as unknown as typeof fetch;
}

// Run fn with a stubbed fetch and a temp working directory.
async function runWithStub<T>(dir: string, fetchImpl: typeof fetch, fn: () => Promise<T>): Promise<T> {
    const originalCwd = process.cwd();
    const originalFetch = globalThis.fetch;
    globalThis.fetch = fetchImpl;
    process.chdir(dir);
    try {
        return await fn();
    } finally {
        process.chdir(originalCwd);
        globalThis.fetch = originalFetch;
    }
}

function fakePi(): FakePi {
    return { providers: new Map(), commands: new Map() };
}

function fakeApi(pi: FakePi): ExtensionAPI {
    return {
        registerProvider: (name: string, providerConfig: ProviderConfig) => pi.providers.set(name, providerConfig),
        registerCommand: (name: string, command: { description?: string; handler: CommandHandler }) => pi.commands.set(name, command),
    } as unknown as ExtensionAPI;
}

// Load the extension with a stubbed /v1/models response and a temp config file.
async function loadExtension(configFile: object, payload: unknown, dir?: string): Promise<{ pi: FakePi; dir: string }> {
    const workDir = dir ?? mkdtempSync(join(tmpdir(), "pi-local-models-"));
    writeConfig(workDir, configFile);
    const pi = fakePi();
    requests = [];
    await runWithStub(workDir, stubFetch(payload), async () => {
        await extension(fakeApi(pi));
    });
    return { pi, dir: workDir };
}

// A command context with a model registry that runs the provider refreshModels callback.
function commandContext(pi: FakePi, hasUI: boolean = true): { notices: string[]; ctx: unknown } {
    const notices: string[] = [];
    const ctx = {
        mode: hasUI ? "tui" : "print",
        hasUI,
        ui: {
            notify: (message: string, type: string = "info") => notices.push(`${type}: ${message}`),
        },
        modelRegistry: {
            getProvider: (name: string) => (pi.providers.has(name) ? { id: name } : undefined),
            getAll: () => (pi.providers.get(PROVIDER_NAME)?.models ?? []).map((model) => ({ ...model, provider: PROVIDER_NAME })),
            refresh: async () => {
                const errors = new Map<string, Error>();
                const provider = pi.providers.get(PROVIDER_NAME);
                if (provider?.refreshModels) {
                    try {
                        const models = await provider.refreshModels(refreshContext(new AbortController().signal));
                        pi.providers.set(PROVIDER_NAME, { ...provider, models });
                    } catch (error) {
                        errors.set(PROVIDER_NAME, error as Error);
                    }
                }
                return { aborted: false, errors };
            },
        },
    };
    return { notices, ctx };
}

function refreshContext(signal: AbortSignal): { allowNetwork: boolean; force: boolean; signal: AbortSignal; publish: () => Promise<boolean> } {
    return { allowNetwork: true, force: true, signal, publish: async () => true };
}

function refreshModels(pi: FakePi, signal: AbortSignal): Promise<ChatModel[]> {
    const provider = pi.providers.get(PROVIDER_NAME);
    assert.ok(provider?.refreshModels, "the provider defines refreshModels");
    return provider.refreshModels(refreshContext(signal)) as Promise<ChatModel[]>;
}

const providers: Array<{ name: string; fixture: string }> = [
    { name: "splash", fixture: "splash.json" },
    { name: "oMLX", fixture: "omlx.json" },
    { name: "llama.cpp", fixture: "llamacpp.json" },
    { name: "ollama", fixture: "ollama.json" },
    { name: "llama-swap", fixture: "llama-swap.json" },
    { name: "mtp-lx", fixture: "mtp-lx.json" },
];

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
    assert.deepEqual(entries.map((entry) => modelKey(entry)), ["qwen3-8b-vision", "gte-modernbert", "gte-reranker"]);
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
    const models = normalizeModels({ data: [{ id: "my-model", object: "model", created: 0 }] })
        .map((model) => toProviderModelConfig(model, config({ contextWindow: 32768, maxTokens: 8192, reasoning: false, input: ["text"] })))
        .filter((model): model is ProviderModelConfig => model !== undefined)
        .map((model) => model as ChatModel);
    assertChatModel(models[0]);
    assert.equal(models[0].contextWindow, 32768);
    assert.equal(models[0].maxTokens, 8192);
    assert.equal(models[0].reasoning, false);
    assert.deepEqual(models[0].input, ["text"]);
    assert.equal(models[0].name, "my-model");
});

test("unknown engine: defaults apply when the config has no contextWindow", () => {
    const models = normalizeModels({ data: [{ id: "tiny" }] })
        .map((model) => toProviderModelConfig(model, config({ contextWindow: undefined, maxTokens: 999999 })))
        .filter((model): model is ProviderModelConfig => model !== undefined)
        .map((model) => model as ChatModel);
    assert.equal(models[0].contextWindow, 128000);
    assert.equal(models[0].maxTokens, 128000);
});

test("edge cases: entries without an id or with a failed load are skipped", () => {
    const models = normalizeModels({
        data: [
            { object: "model" },
            { id: "ok-model" },
            { id: "broken-model", status: { value: "failed", failed: true, exit_code: 1 } },
        ],
    })
        .map((model) => toProviderModelConfig(model, config()))
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
    const dir = mkdtempSync(join(tmpdir(), "pi-local-models-project-"));
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
    const dir = mkdtempSync(join(tmpdir(), "pi-local-models-"));
    writeConfig(dir, testConfig);
    requests = [];
    const models = await runWithStub(dir, stubFetch(fixture("omlx.json")), () => listModels(config(testConfig as LocalModelsConfig)));
    assert.equal(requests.length, 1);
    assert.equal(requests[0].url, "http://127.0.0.1:8000/v1/models");
    assert.equal(requests[0].method, "GET");
    assert.equal(requests[0].headers["x-api-key"], "1234");
    assert.equal(models.length, 8);
});

test("listModels: an unavailable server throws with the status code", async () => {
    const dir = mkdtempSync(join(tmpdir(), "pi-local-models-"));
    writeConfig(dir, testConfig);
    await runWithStub(dir, stubErrorResponse(503, "model not loaded"), async () => {
        await assert.rejects(() => listModels(config(testConfig as LocalModelsConfig)), /503: model not loaded/);
    });
});

test("discoverModels: the same response maps to model entries", async () => {
    const dir = mkdtempSync(join(tmpdir(), "pi-local-models-"));
    writeConfig(dir, testConfig);
    const models = await runWithStub(dir, stubFetch(fixture("splash.json")), () =>
        discoverModels(config(testConfig)),
    );
    assert.equal(models.length, 1);
    assert.equal(models[0].id, "unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M");
});

for (const provider of providers) {
    test(`${provider.name}: the provider is registered with valid model entries`, async () => {
        const { pi } = await loadExtension(testConfig, fixture(provider.fixture));
        const registered = pi.providers.get(PROVIDER_NAME);
        assert.ok(registered, "registerProvider was called");
        assert.equal(registered.baseUrl, "http://127.0.0.1:8000/v1");
        assert.equal(registered.api, "openai-completions");
        assert.equal(registered.apiKey, "1234");
        assert.ok(registered.models && registered.models.length > 0, "at least one model is registered");
        assert.ok(registered.refreshModels, "the provider can refresh its model list");
        (registered.models as ProviderModelConfig[]).forEach((model) => assertChatModel(model as ChatModel));
    });

    test(`${provider.name}: refreshModels returns the same list as startup discovery`, async () => {
        const { pi, dir } = await loadExtension(testConfig, fixture(provider.fixture));
        const models = await runWithStub(dir, stubFetch(fixture(provider.fixture)), () =>
            refreshModels(pi, new AbortController().signal),
        );
        const expected = registeredModels(provider.fixture, config(testConfig as LocalModelsConfig));
        assert.deepEqual(models.map((model) => model.id), expected.map((model) => model.id));
        models.forEach(assertChatModel);
    });
}

test("refreshModels: the abort signal is passed to fetch", async () => {
    const { pi, dir } = await loadExtension(testConfig, fixture("omlx.json"));
    const controller = new AbortController();
    requests = [];
    await runWithStub(dir, stubFetch(fixture("omlx.json")), () => refreshModels(pi, controller.signal));
    assert.equal(requests[requests.length - 1].url, "http://127.0.0.1:8000/v1/models");
    assert.equal(requests[requests.length - 1].signal, controller.signal);
});

test("refreshModels: an unreachable engine reports the error", async () => {
    const { pi, dir } = await loadExtension(testConfig, fixture("omlx.json"));
    await runWithStub(dir, stubErrorResponse(503, "model not loaded"), async () => {
        await assert.rejects(() => refreshModels(pi, new AbortController().signal), /503: model not loaded/);
    });
});

test("refreshModels: a config file edited after startup is used", async () => {
    const { pi, dir } = await loadExtension(testConfig, fixture("omlx.json"));
    writeConfig(dir, { ...testConfig, url: "http://127.0.0.1:9000", maxTokens: 4096 });
    const models = await runWithStub(dir, stubFetch(fixture("splash.json")), () => refreshModels(pi, new AbortController().signal));
    assert.equal(models.length, 1);
    assert.equal(models[0].maxTokens, 4096);
    assert.equal(requests[requests.length - 1].url, "http://127.0.0.1:9000/v1/models");
});

test("/local-models-refresh: the command is registered", async () => {
    const { pi } = await loadExtension(testConfig, fixture("omlx.json"));
    const command = pi.commands.get("local-models-refresh");
    assert.ok(command, "the command is registered");
    assert.ok(command.description, "the command has a description");
});

test("/local-models-refresh: models discovered after startup are added", async () => {
    const { pi, dir } = await loadExtension(testConfig, { object: "list", data: [] });
    assert.equal(pi.providers.get(PROVIDER_NAME)!.models!.length, 0, "no models at startup");

    const { notices, ctx } = commandContext(pi);
    await runWithStub(dir, stubFetch(fixture("omlx.json")), () => pi.commands.get("local-models-refresh")!.handler("", ctx));

    const models = pi.providers.get(PROVIDER_NAME)!.models! as ChatModel[];
    assert.equal(models.length, 8);
    assert.equal(notices.length, 1);
    assert.ok(notices[0].startsWith("info: local-models: refreshed 8 model(s)"), notices[0]);
    assert.ok(notices[0].includes("/model"), "the notice points the user to /model");
});

test("/local-models-refresh: a replaced catalog is reported", async () => {
    const { pi, dir } = await loadExtension(testConfig, fixture("omlx.json"));
    const { notices, ctx } = commandContext(pi);
    await runWithStub(dir, stubFetch(fixture("llamacpp.json")), () => pi.commands.get("local-models-refresh")!.handler("", ctx));
    const models = pi.providers.get(PROVIDER_NAME)!.models! as ChatModel[];
    assert.equal(models.length, 1);
    assert.equal(models[0].name, "Qwen3.6-35B-A3B-UD-Q4_K_M");
    assert.ok(notices[0].includes("refreshed 1 model(s)"), notices[0]);
});

test("/local-models-refresh: an unreachable engine is reported as an error", async () => {
    const { pi, dir } = await loadExtension(testConfig, fixture("omlx.json"));
    const { notices, ctx } = commandContext(pi);
    await runWithStub(dir, stubErrorResponse(503, "model not loaded"), () => pi.commands.get("local-models-refresh")!.handler("", ctx));
    assert.equal(notices.length, 1);
    assert.ok(notices[0].startsWith("error: local-models: refresh failed"), notices[0]);
    assert.ok(notices[0].includes("http://127.0.0.1:8000"), "the notice names the server url");
});

test("/local-models-refresh: the result is printed when there is no TUI", async () => {
    const { pi, dir } = await loadExtension(testConfig, fixture("omlx.json"));
    const { ctx } = commandContext(pi, false);
    const originalLog = console.log;
    const lines: string[] = [];
    console.log = (message: unknown) => lines.push(String(message));
    try {
        await runWithStub(dir, stubFetch(fixture("llamacpp.json")), () => pi.commands.get("local-models-refresh")!.handler("", ctx));
    } finally {
        console.log = originalLog;
    }
    assert.equal(lines.length, 1);
    assert.ok(lines[0].includes("local-models: refreshed 1 model(s)"), lines[0]);
});

test("/local-models-refresh: a server with no chat models is reported as a warning", async () => {
    const { pi, dir } = await loadExtension(testConfig, fixture("omlx.json"));
    const { notices, ctx } = commandContext(pi);
    await runWithStub(dir, stubFetch({ object: "list", data: [] }), () => pi.commands.get("local-models-refresh")!.handler("", ctx));
    assert.equal(notices.length, 1);
    assert.ok(notices[0].startsWith("warning: local-models: the server returned no models"), notices[0]);
});

test("/local-models-refresh: the provider is created when the config appeared after startup", async () => {
    const dir = mkdtempSync(join(tmpdir(), "pi-local-models-project-"));
    const emptyHome = mkdtempSync(join(tmpdir(), "pi-local-models-home-"));
    const originalHome = process.env.HOME;
    process.env.HOME = emptyHome;

    const pi = fakePi();
    const originalCwd = process.cwd();
    process.chdir(dir);
    try {
        await extension(fakeApi(pi));
        assert.equal(pi.providers.size, 0, "no provider without a config file");
        assert.ok(pi.commands.has("local-models-refresh"), "the command is still registered");

        writeConfig(dir, testConfig);
        const { notices, ctx } = commandContext(pi);
        await runWithStub(dir, stubFetch(fixture("llamacpp.json")), () => pi.commands.get("local-models-refresh")!.handler("", ctx));

        const models = pi.providers.get(PROVIDER_NAME)!.models! as ChatModel[];
        assert.equal(models.length, 1);
        assert.equal(models[0].name, "Qwen3.6-35B-A3B-UD-Q4_K_M");
        assert.ok(notices[0].includes("refreshed 1 model(s)"), notices[0]);
    } finally {
        process.chdir(originalCwd);
        process.env.HOME = originalHome;
    }
});
