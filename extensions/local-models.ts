// ~/.pi/agent/extensions/local-models.ts
// Pi extension for local models integration
//
// Configure per-project via .pi/agent/local-models.json:
//   { "url": "http://127.0.0.1:8000", "apiKey": "1234" }
// Or globally: {homedir}/.pi/agent/local-models.json
// - Windows: %USERPROFILE%/.pi/agent/local-models.json
// - macOs/Linux: ~/.pi/agent/local-models.json

import type { ExtensionAPI, ProviderModelConfig } from "@earendil-works/pi-coding-agent";
import { existsSync, readFileSync } from "node:fs";
import { homedir } from "os";
import { join } from "node:path";

const DEFAULT_CTX_WINDOW = 128000;
const DEFAULT_MAX_TOKENS = 16384;
const DEFAULT_INPUT: ModelInput[] = ["text"];

export type ModelInput = "text" | "image";
export type InputLimits = NonNullable<ProviderModelConfig["inputLimits"]>;

export interface LocalModelsConfig {
    url: string;
    apiKey?: string;
    contextWindow?: number;
    maxTokens: number;
    reasoning: boolean;
    input: ModelInput[];
}

interface LoadingPayload {
    current?: string;
    stage?: string;
    value?: number;
}

/**
 * One entry from a local inference engine /v1/models response.
 * The engines report the same model in different shapes:
 * - splash:     data[] with max_model_len, context_length, vision, input_modalities, limits, reasoning.thinks
 * - oMLX:       data[] with max_model_len, model_type ("vlm" | "llm")
 * - llama.cpp:  data[] with meta.n_ctx and architecture.input_modalities, plus a second models[] array
 *               with name, description, and capabilities for the same models
 * - llama-swap: data[] with context_length, context_window, and capabilities.in / capabilities.out
 * - ollama:     models[] with name, model, and details only
 */
export interface ServerModel {
    id?: string;
    name?: string;
    model?: string;
    aliases?: string[];
    object?: string;
    owned_by?: string;
    created?: number;
    description?: string;

    // Context window
    context_length?: number;
    context_window?: number;
    max_model_len?: number;
    n_ctx?: number;
    meta?: {
        n_ctx?: number;
        n_ctx_train?: number;
        ftype?: string;
        llamaswap?: {
            state?: string;
            ttl?: number;
            context_window?: number;
        };
    };

    // Input modalities
    vision?: boolean;
    input_modalities?: string[];
    architecture?: {
        input_modalities?: string[];
        output_modalities?: string[];
    };
    model_type?: string;
    // llama.cpp reports capabilities as a string list, llama-swap as an object.
    capabilities?: string[] | {
        in?: string[];
        out?: string[];
        tools?: boolean;
        embeddings?: boolean;
        reranker?: boolean;
        decisions?: boolean;
    };

    // Reasoning support
    reasoning?: boolean | {
        thinks?: boolean;
        efforts?: string[];
        default?: string | null;
    };

    // Request size limits
    limits?: {
        request_bytes?: number;
        images?: number;
        image_bytes?: number;
        pdf_bytes?: number;
        pdf_pages?: number;
    };

    // Loader state (ollama, splash, llama-swap)
    state?: string;
    payload?: LoadingPayload;
    status?: {
        value?: string;
        failed?: boolean;
        exit_code?: number;
    };
}

export function readConfig(cwd: string): LocalModelsConfig {
    // First check home directory
    let file = join(cwd, ".pi", "agent", "local-models.json");
    if (!existsSync(file)) {
        file = join(homedir(), ".pi", "agent", "local-models.json");
    }
    if (!existsSync(file)) {
        throw new Error(`no local-models.json found in ${file}`);
    }

    const raw = readFileSync(file, "utf-8");
    const cfg = JSON.parse(raw);
    let url: string = '';
    let apiKey: string | undefined = undefined;
    let contextWindow: number | undefined = undefined;
    let maxTokens: number = DEFAULT_MAX_TOKENS;
    let input: ModelInput[] = [...DEFAULT_INPUT];
    let reasoning: boolean = false;
    if (cfg.url) {
        // Remove trailing v1
        url = cfg.url;
        if (url.endsWith('/v1')) {
            url = url.slice(0, -3);
        } else if (url.endsWith('v1')) {
            url = url.slice(0, -2);
        }
    }

    if (cfg.apiKey)
        apiKey = cfg.apiKey;

    if (cfg.contextWindow)
        contextWindow = cfg.contextWindow;

    if (cfg.maxTokens)
        maxTokens = cfg.maxTokens;

    if (cfg.reasoning)
        reasoning = cfg.reasoning;

    if (Array.isArray(cfg.input)) {
        // pi chat models accept only "text" and "image". Other modalities
        // (for example "pdf" or "audio") are handled by the harness, not the model.
        input = (cfg.input as unknown[]).filter((m): m is ModelInput => m === "text" || m === "image");
    }

    return {
        url: url,
        apiKey: apiKey,
        contextWindow: contextWindow,
        maxTokens: maxTokens,
        reasoning: reasoning,
        input: input
    };
}

function openAiApiCall(config: LocalModelsConfig, endpoint: string, method: string, body?: Record<string, unknown>) {
    const hasApiKey = config.apiKey ? true : false;
    const postHeaders = { "Content-Type": "application/json" } as HeadersInit
    const postHeadersWithApiKey = { "Content-Type": "application/json", "x-api-key": config.apiKey } as HeadersInit
    const getHeaders = {} as HeadersInit
    const getHeadersWithApiKey = { "x-api-key": config.apiKey } as HeadersInit

    let headers: HeadersInit;
    headers = getHeaders;
    if (hasApiKey) {
        headers = getHeadersWithApiKey;
    }
    if (method === 'POST') {
        if (hasApiKey)
            headers = postHeadersWithApiKey;
        else
            headers = postHeaders
    }

    return fetch(`${config.url}${endpoint}`, {
        method: method,
        headers: headers,
        body: method === 'POST' ? JSON.stringify(body) : undefined,
    }).then(async (res) => {
        if (!res.ok) {
            const text = await res.text();
            throw new Error(`${res.status}: ${text.slice(0, 200)}`);
        }
        return res.json();
    });
}

export async function listModels(config: LocalModelsConfig): Promise<ServerModel[]> {
    const raw = (await openAiApiCall(config, "/v1/models", "GET")) as {
        data?: ServerModel[];
        models?: ServerModel[];
    };
    return normalizeModels(raw);
}

/** Unique key for a model entry. `data[].id` and `models[].name` are the same model. */
export function modelKey(model: ServerModel): string | undefined {
    return model.id ?? model.name ?? model.model;
}

/**
 * Merge the two lists a server can return into one deduplicated list.
 * OpenAI-style servers (splash, oMLX, ollama) use `data[]`. llama.cpp also
 * returns `models[]` with name, description, and capabilities for the same models.
 */
export function normalizeModels(raw: unknown): ServerModel[] {
    const source = (raw ?? {}) as { data?: ServerModel[]; models?: ServerModel[] };
    const byKey = new Map<string, ServerModel>();
    for (const model of source.data ?? []) {
        const key = modelKey(model);
        if (!key) continue;
        byKey.set(key, model);
    }
    for (const model of source.models ?? []) {
        const key = modelKey(model);
        if (!key) continue;
        // `data[]` wins on overlapping fields, `models[]` adds display metadata.
        byKey.set(key, { ...model, ...byKey.get(key) });
    }
    return [...byKey.values()];
}

/** Short display name. llama.cpp reports the full GGUF file path as the model id. */
export function displayName(model: ServerModel): string {
    const raw = model.name ?? model.id ?? model.model ?? "model";
    const isFilePath = raw.startsWith("/") || /\.(gguf|safetensors|bin|onnx|mlx|json)$/i.test(raw);
    if (!isFilePath) return raw;
    return (raw.split("/").pop() ?? raw).replace(/\.(gguf|safetensors|bin|onnx|mlx|json)$/i, "");
}

/** Modalities the engine reports for this model, in whichever field it uses. */
function reportedInputModalities(model: ServerModel): string[] {
    const capabilities = model.capabilities;
    const fromCapabilities = Array.isArray(capabilities)
        ? capabilities.filter((capability) => capability === "text" || capability === "image" || capability === "vision")
        : (capabilities?.in ?? []);
    return [
        ...(model.input_modalities ?? []),
        ...(model.architecture?.input_modalities ?? []),
        ...fromCapabilities,
    ];
}

/** Input modalities the model accepts, with the config value as fallback. */
export function modelInput(model: ServerModel, fallback: ModelInput[]): ModelInput[] {
    const modalities = reportedInputModalities(model);
    const hasModalityInfo = modalities.length > 0 || typeof model.vision === "boolean" || Boolean(model.model_type);
    if (!hasModalityInfo) return fallback.length > 0 ? fallback : DEFAULT_INPUT;

    // A chat model always takes text. Only add image when the engine says so.
    const input: ModelInput[] = ["text"];
    const acceptsImage = model.vision === true || model.model_type === "vlm" || modalities.includes("image") || modalities.includes("vision");
    if (acceptsImage) input.push("image");
    return input;
}

/** llama.cpp and oMLX report no reasoning flags, so the config value decides. */
export function modelReasoning(model: ServerModel, fallback: boolean): boolean {
    if (typeof model.reasoning === "boolean") return model.reasoning;
    if (model.reasoning && typeof model.reasoning.thinks === "boolean") return model.reasoning.thinks;
    return fallback;
}

/** Splash request limits map onto pi cache-safe input limits. */
export function modelInputLimits(model: ServerModel): InputLimits | undefined {
    const limits: InputLimits = {};
    const { request_bytes, images, image_bytes } = model.limits ?? {};
    if (request_bytes) limits.maxRequestBytes = request_bytes;
    if (images || image_bytes) {
        const imageLimits: NonNullable<InputLimits["images"]> = {};
        if (images) imageLimits.maxPerRequest = images;
        if (image_bytes) imageLimits.resize = { maxBytes: image_bytes };
        limits.images = imageLimits;
    }
    return Object.keys(limits).length > 0 ? limits : undefined;
}

const NON_CHAT_MODEL_TYPES = [
    "embedding",
    "embeddings",
    "rerank",
    "reranker",
    "classify",
    "classifier",
    "decisions",
    "image",
    "audio",
    "tts",
];

/** Embedding, rerank, and decision models cannot answer a chat request. */
export function isChatModel(model: ServerModel): boolean {
    const capabilityObject = model.capabilities && !Array.isArray(model.capabilities) ? model.capabilities : undefined;
    if (capabilityObject?.embeddings || capabilityObject?.reranker || capabilityObject?.decisions) return false;
    if (Array.isArray(model.capabilities) && model.capabilities.length > 0) {
        const chat = ["completion", "prediction", "generation", "text", "chat"];
        return model.capabilities.some((capability) => chat.includes(capability));
    }
    if (model.model_type && NON_CHAT_MODEL_TYPES.includes(model.model_type.toLowerCase())) return false;
    const outputs = capabilityObject?.out ?? model.architecture?.output_modalities ?? [];
    if (outputs.length > 0) return outputs.includes("text");
    return true;
}

/** Map one server model entry to the pi ProviderModelConfig shape. */
export function toProviderModelConfig(model: ServerModel, config: LocalModelsConfig): ProviderModelConfig | undefined {
    const id = modelKey(model);
    if (!id || model.status?.failed === true || !isChatModel(model)) return undefined;

    const contextWindow = model.context_length ?? model.context_window ?? model.max_model_len ?? model.meta?.n_ctx ?? model.n_ctx ?? config.contextWindow ?? DEFAULT_CTX_WINDOW;
    const maxTokens = Math.max(1, Math.min(config.maxTokens, contextWindow));

    const modelConfig: ProviderModelConfig = {
        id,
        name: displayName(model),
        reasoning: modelReasoning(model, config.reasoning),
        input: modelInput(model, config.input),
        contextWindow,
        maxTokens,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
        compat: {
            supportsDeveloperRole: false,
            supportsReasoningEffort: false,
        },
    };
    const inputLimits = modelInputLimits(model);
    if (inputLimits) modelConfig.inputLimits = inputLimits;
    return modelConfig;
}

export default async function (pi: ExtensionAPI) {
    const config = readConfig(process.cwd());
    const serverModels = await listModels(config).catch((): ServerModel[] => []);
    if (serverModels.length === 0) {
        console.log("\x1b[31m", "local-models: " + "No local models were discovered");
        return;
    }

    console.log('\x1b[36m%s\x1b[0m', "local-models: " + serverModels.length + " local models were discovered, type /model to select a [local-model]");

    const models = serverModels
        .map((model) => toProviderModelConfig(model, config))
        .filter((model): model is ProviderModelConfig => model !== undefined);

    if (models.length === 0) {
        console.log("\x1b[31m", "local-models: no usable model entry in /v1/models");
        return;
    }

    pi.registerProvider("local-model", {
        baseUrl: `${config.url}/v1`,
        api: "openai-completions",
        apiKey: config.apiKey ?? "not-needed",
        models
    });
};