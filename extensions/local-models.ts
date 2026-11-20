// ~/.pi/agent/extensions/local-models.ts
// Pi extension for local models integration
//
// Configure per-project via .pi/agent/local-models.json:
//   { "url": "http://127.0.0.1:8000", "apiKey": "1234" }
// Or globally: {homedir}/.pi/agent/local-models.json
// - Windows: %USERPROFILE%/.pi/agent/local-models.json
// - macOs/Linux: ~/.pi/agent/local-models.json

import type { ExtensionAPI, ExtensionContext, ProviderConfig, ProviderModelConfig } from "@earendil-works/pi-coding-agent";
import { existsSync, readFileSync } from "node:fs";
import { homedir } from "os";
import { join } from "node:path";

const DEFAULT_CTX_WINDOW = 128000;
const DEFAULT_MAX_TOKENS = 16384;
const DEFAULT_INPUT: ModelInput[] = ["text"];
const DEFAULT_TIMEOUT_MS = 10000;
const DEFAULT_RETRY_ATTEMPTS = 3;
const DEFAULT_RETRY_DELAY_MS = 1500;
export const PROVIDER_NAME = "local-model";
export const PROVIDER_LABEL = "Local models";

let lastDiscovered: ServerModel[] = [];

/** The raw entries from the most recent discovery, for the /local-models command. */
export function getDiscoveredModels(): ServerModel[] {
    return lastDiscovered;
}

export type ModelInput = "text" | "image";
export type InputLimits = NonNullable<ProviderModelConfig["inputLimits"]>;

export type ApiKeyHeader = "bearer" | "x-api-key" | "both";
export type MaxTokensField = "max_tokens" | "max_completion_tokens";

export interface RetryConfig {
    attempts: number;
    delayMs: number;
}

export interface LocalModelsConfig {
    url: string;
    apiKey?: string;
    apiKeyHeader: ApiKeyHeader;
    contextWindow?: number;
    maxTokens: number;
    reasoning: boolean;
    input: ModelInput[];
    timeoutMs: number;
    maxTokensField?: MaxTokensField;
    retry: RetryConfig;
    /** Ask /api/show for a context length when the engine does not report one. */
    contextProbe: boolean;
    configPath: string;
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

    // Quantization and size metadata (llama.cpp, ollama)
    details?: {
        quantization_level?: string;
        parameter_size?: string;
        format?: string;
    };
}

export function positiveNumber(value: unknown, fallback: number): number {
    const parsed = Number(value);
    return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

// Allow "$VAR" and "${VAR}" in url and apiKey, like pi's provider config.
function expandEnv(value: string): string {
    return value.replace(/\$(?:\{([A-Za-z0-9_]+)\}|([A-Za-z0-9_]+))/g, (match, braced, bare) => {
        const name = braced ?? bare;
        return process.env[name] ?? match;
    });
}

const API_KEY_HEADERS: ApiKeyHeader[] = ["bearer", "x-api-key", "both"];

/**
 * A config file is usable when it names the engine and says how to authenticate it.
 * Everything else has a default.
 */
function validateConfig(file: string, cfg: Record<string, unknown>): { url: string; apiKey?: string; apiKeyHeader: ApiKeyHeader } {
    const rawUrl = cfg.url === undefined || cfg.url === null ? "" : expandEnv(String(cfg.url)).trim();
    if (!rawUrl) {
        throw new Error(`${file} must set "url"`);
    }
    if (!/^https?:\/\/[^\s]+$/i.test(rawUrl)) {
        throw new Error(`${file} has an invalid "url": ${rawUrl}. It must start with http:// or https://`);
    }

    const apiKey = cfg.apiKey === undefined || cfg.apiKey === null || String(cfg.apiKey) === "" ? undefined : expandEnv(String(cfg.apiKey));
    const header = cfg.apiKeyHeader === undefined ? undefined : String(cfg.apiKeyHeader);
    if (apiKey && header === undefined) {
        throw new Error(`${file} sets "apiKey" but no "apiKeyHeader". Set "bearer", "x-api-key", or "both".`);
    }
    if (header !== undefined && !API_KEY_HEADERS.includes(header as ApiKeyHeader)) {
        throw new Error(`${file} has an invalid "apiKeyHeader": ${JSON.stringify(cfg.apiKeyHeader)}. Use "bearer", "x-api-key", or "both".`);
    }

    // Remove trailing v1
    let url = rawUrl;
    if (url.endsWith('/v1')) {
        url = url.slice(0, -3);
    } else if (url.endsWith('v1')) {
        url = url.slice(0, -2);
    }

    return { url, apiKey, apiKeyHeader: (cfg.apiKeyHeader as ApiKeyHeader | undefined) ?? "both" };
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
    let cfg: Record<string, unknown>;
    try {
        cfg = JSON.parse(raw) as Record<string, unknown>;
    } catch (error) {
        throw new Error(`${file} is not valid JSON: ${errorMessage(error)}`);
    }

    const { url, apiKey, apiKeyHeader } = validateConfig(file, cfg);
    const retry = cfg.retry as RetryConfig | undefined;
    const maxTokensField: MaxTokensField | undefined = ["max_tokens", "max_completion_tokens"].includes(String(cfg.maxTokensField))
        ? (cfg.maxTokensField as MaxTokensField)
        : undefined;

    return {
        url,
        apiKey: apiKey || undefined,
        apiKeyHeader,
        contextWindow: typeof cfg.contextWindow === "number" && cfg.contextWindow > 0 ? cfg.contextWindow : undefined,
        maxTokens: positiveNumber(cfg.maxTokens, DEFAULT_MAX_TOKENS),
        reasoning: Boolean(cfg.reasoning),
        input: Array.isArray(cfg.input)
            // pi chat models accept only "text" and "image". Other modalities
            // (for example "pdf" or "audio") are handled by the harness, not the model.
            ? (cfg.input as unknown[]).filter((m): m is ModelInput => m === "text" || m === "image")
            : [...DEFAULT_INPUT],
        timeoutMs: positiveNumber(cfg.timeoutMs, DEFAULT_TIMEOUT_MS),
        maxTokensField,
        retry: {
            attempts: positiveNumber(retry?.attempts, DEFAULT_RETRY_ATTEMPTS),
            delayMs: positiveNumber(retry?.delayMs, DEFAULT_RETRY_DELAY_MS),
        },
        contextProbe: cfg.contextProbe !== false,
        configPath: file,
    };
}

export function authHeaders(config: LocalModelsConfig): Record<string, string> {
    if (!config.apiKey) return {};
    const bearer = { Authorization: `Bearer ${config.apiKey}` };
    const apiKeyHeader = { "x-api-key": config.apiKey };
    // pi sends a Bearer token when it calls the model. Discovery should match.
    if (config.apiKeyHeader === "bearer") return bearer;
    if (config.apiKeyHeader === "x-api-key") return apiKeyHeader;
    return { ...bearer, ...apiKeyHeader };
}

async function openAiApiCall(config: LocalModelsConfig, endpoint: string, method: string, body?: Record<string, unknown>, signal?: AbortSignal): Promise<unknown> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort("timeout"), config.timeoutMs);
    let onAbort: (() => void) | undefined;
    if (signal) {
        if (signal.aborted) controller.abort(signal.reason);
        else {
            onAbort = () => controller.abort(signal.reason);
            signal.addEventListener("abort", onAbort, { once: true });
        }
    }

    const headers: Record<string, string> = { ...authHeaders(config) };
    if (method === "POST") headers["Content-Type"] = "application/json";
    const call = `${config.url}${endpoint}`;

    try {
        const response = await fetch(call, {
            method: method,
            headers: headers as HeadersInit,
            body: method === "POST" ? JSON.stringify(body) : undefined,
            signal: controller.signal,
        });
        // The engine answered. Stop the timeout so reading the body is not cut short.
        clearTimeout(timer);
        if (!response.ok) {
            const text = await response.text();
            throw new Error(`${response.status}: ${text.slice(0, 200)}`);
        }
        return await response.json();
    } catch (error) {
        if (controller.signal.aborted && !signal?.aborted) {
            throw new Error(`no response from ${call} within ${config.timeoutMs}ms`);
        }
        throw error;
    } finally {
        clearTimeout(timer);
        if (signal && onAbort) signal.removeEventListener("abort", onAbort);
    }
}

function hasContext(model: ServerModel): boolean {
    return typeof model.context_length === "number" || typeof model.context_window === "number" || typeof model.max_model_len === "number" || typeof model.meta?.n_ctx === "number" || typeof model.n_ctx === "number";
}

/**
 * Ollama reports no context length in /v1/models. Its /api/show endpoint does.
 * Other engines answer 404, so the probe is cheap and ignored on failure.
 */
async function probeContextWindows(config: LocalModelsConfig, models: ServerModel[], signal?: AbortSignal): Promise<void> {
    for (const model of models) {
        const id = modelKey(model);
        if (!id || hasContext(model)) continue;
        const info = (await openAiApiCall(config, "/api/show", "POST", { name: id }, signal).catch(() => undefined)) as
            { model?: { context_length?: number } } | undefined;
        const contextLength = info?.model?.context_length;
        if (typeof contextLength === "number" && contextLength > 0) model.context_length = contextLength;
    }
}

export async function listModels(config: LocalModelsConfig, signal?: AbortSignal): Promise<ServerModel[]> {
    const raw = (await openAiApiCall(config, "/v1/models", "GET", undefined, signal)) as {
        data?: ServerModel[];
        models?: ServerModel[];
    };
    const models = normalizeModels(raw);
    if (config.contextProbe) {
        await probeContextWindows(config, models, signal);
    }
    return models;
}

/**
 * Model list for one provider refresh. Retries while the engine is still booting.
 * Throws when the engine is unreachable, so pi records the failure and
 * /local-models-refresh can report it.
 */
export async function discoverModels(config: LocalModelsConfig, signal?: AbortSignal): Promise<ProviderModelConfig[]> {
    let serverModels: ServerModel[] = [];
    for (let attempt = 1; attempt <= config.retry.attempts; attempt++) {
        try {
            serverModels = await listModels(config, signal);
            break;
        } catch (error) {
            if (attempt === config.retry.attempts || signal?.aborted || !isRetryable(error)) throw error;
            await sleep(config.retry.delayMs, signal);
        }
    }
    lastDiscovered = serverModels;
    return sortModels(
        serverModels
            .map((model) => toProviderModelConfig(model, config))
            .filter((model): model is ProviderModelConfig => model !== undefined),
    );
}

/** A hung engine or a bad key is not worth retrying. A refused connection is. */
function isRetryable(error: unknown): boolean {
    const message = error instanceof Error ? error.message : String(error);
    if (message.includes("within")) return false;
    const status = Number(message.split(":")[0]);
    return !(Number.isFinite(status) && status >= 400 && status < 500);
}

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
    return new Promise((resolve, reject) => {
        const timer = setTimeout(resolve, ms);
        if (signal) {
            signal.addEventListener("abort", () => {
                clearTimeout(timer);
                reject(signal.reason instanceof Error ? signal.reason : new Error("refresh cancelled"));
            }, { once: true });
        }
    });
}

export function sortModels(models: ProviderModelConfig[]): ProviderModelConfig[] {
    return [...models].sort((a, b) => a.name.localeCompare(b.name));
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

/**
 * llama.cpp reports the GGUF file path as the model id. When the server also lists
 * a clean alias, use it: the picker shows a readable id and the API accepts it.
 */
export function pickModelId(model: ServerModel): string {
    const base = modelKey(model) ?? "";
    if (!base.includes("/")) return base;
    const alias = (model.aliases ?? []).find(
        (value) => typeof value === "string" && value.length > 0 && !value.includes("/") && !value.endsWith(".gguf"),
    );
    return alias ?? base;
}

/** Splash reports reasoning efforts. Map them to pi thinking levels. */
export function thinkingLevelMap(model: ServerModel): Record<string, string> | undefined {
    const efforts = typeof model.reasoning === "object" && model.reasoning ? model.reasoning.efforts ?? [] : [];
    const map: Record<string, string> = {};
    for (const level of ["minimal", "low", "medium", "high", "xhigh", "max"]) {
        if (efforts.includes(level)) map[level] = level;
    }
    return Object.keys(map).length > 0 ? map : undefined;
}

/** Map one server model entry to the pi ProviderModelConfig shape. */
export function toProviderModelConfig(model: ServerModel, config: LocalModelsConfig): ProviderModelConfig | undefined {
    const id = pickModelId(model);
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
            ...(config.maxTokensField ? { maxTokensField: config.maxTokensField } : {}),
        },
    };
    const levels = thinkingLevelMap(model);
    if (levels) modelConfig.thinkingLevelMap = levels;
    const inputLimits = modelInputLimits(model);
    if (inputLimits) modelConfig.inputLimits = inputLimits;
    return modelConfig;
}

function errorMessage(error: unknown): string {
    return error instanceof Error ? error.message : String(error);
}

// The TUI shows notifications. Print mode and rpc mode have no notification surface.
function report(ctx: ExtensionContext, message: string, type: "info" | "warning" | "error"): void {
    if (ctx.hasUI) {
        ctx.ui.notify(message, type);
    } else {
        console.log(message);
    }
}

/** Provider entry for pi. refreshModels re-reads the config, so edits apply without a reload. */
function providerConfig(config: LocalModelsConfig, models: ProviderModelConfig[]): ProviderConfig {
    return {
        name: PROVIDER_LABEL,
        baseUrl: `${config.url}/v1`,
        api: "openai-completions",
        apiKey: config.apiKey ?? "not-needed",
        models,
        refreshModels: async ({ signal }) => discoverModels(readConfig(process.cwd()), signal),
    };
}

export default async function (pi: ExtensionAPI) {
    let config: LocalModelsConfig | undefined;
    try {
        config = readConfig(process.cwd());
    } catch (error) {
        console.log("\x1b[31m", `local-models: ${errorMessage(error)}`);
    }

    if (config) {
        let models: ProviderModelConfig[] = [];
        try {
            models = await discoverModels(config);
        } catch (error) {
            console.log("\x1b[31m", `local-models: ${errorMessage(error)}`);
        }

        if (models.length === 0) {
            console.log("\x1b[31m", "local-models: No local models were discovered. Start the engine, then run /local-models-refresh");
        } else {

            console.log("\x1b[36m%s\x1b[0m", "local-models: " + models.length + " local models were discovered, type /model to select a [local-model]");

            pi.registerProvider(PROVIDER_NAME, providerConfig(config, models));

            console.log("\x1b[36m", "local-models: providers registered succesfully.");
        }
    }

    pi.registerCommand("local-models-refresh", {
        description: "Ask the local inference server for its model list again",
        handler: async (_args, ctx) => {
            // The config file may appear or change after pi started.
            let currentConfig: LocalModelsConfig;
            try {
                currentConfig = readConfig(process.cwd());
            } catch (error) {
                report(ctx, `local-models: ${errorMessage(error)}`, "error");
                return;
            }

            if (!ctx.modelRegistry.getProvider(PROVIDER_NAME)) {
                pi.registerProvider(PROVIDER_NAME, providerConfig(currentConfig, []));
            }

            const result = await ctx.modelRegistry.refresh({
                providers: [PROVIDER_NAME],
                allowNetwork: true,
                force: true,
            });

            if (result.aborted) {
                report(ctx, "local-models: refresh was cancelled", "warning");
                return;
            }

            const error = result.errors.get(PROVIDER_NAME);
            if (error) {
                report(ctx, `local-models: refresh failed: ${error.message}. Is the engine running at ${currentConfig.url}?`, "error");
                return;
            }

            const models = ctx.modelRegistry.getAll().filter((model) => model.provider === PROVIDER_NAME);
            if (models.length === 0) {
                report(ctx, "local-models: the server returned no models", "warning");
                return;
            }

            report(ctx, `local-models: refreshed ${models.length} model(s): ${models.map((model) => model.name).join(", ")}. Use /model to select one.`, "info");
        },
    });

    pi.registerCommand("local-models", {
        description: "Show the local models this extension registered",
        handler: async (_args, ctx) => {
            let currentConfig: LocalModelsConfig;
            try {
                currentConfig = readConfig(process.cwd());
            } catch (error) {
                report(ctx, `local-models: ${errorMessage(error)}`, "error");
                return;
            }

            const models = ctx.modelRegistry.getAll().filter((model) => model.provider === PROVIDER_NAME);
            if (models.length === 0) {
                report(ctx, `local-models: no models registered from ${currentConfig.url}. Run /local-models-refresh`, "warning");
                return;
            }

            const raw = new Map(getDiscoveredModels().map((model) => [pickModelId(model), model]));
            const lines = [`local-models: ${models.length} model(s) from ${currentConfig.url} (config: ${currentConfig.configPath})`];
            for (const model of models) {
                const entry = raw.get(model.id);
                const quant = entry?.meta?.ftype || entry?.details?.quantization_level;
                lines.push(
                    `  ${model.name} [${model.id}] ${model.contextWindow} ctx, ${model.maxTokens} max tokens, ` +
                    `input ${model.input.join("+")}, thinking ${model.reasoning ? "on" : "off"}${quant ? `, ${quant}` : ""}`,
                );
            }
            report(ctx, lines.join("\n"), "info");
        },
    });
    console.log("\x1b[36m", "local-models: extension loaded.");
}