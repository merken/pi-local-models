// ~/.pi/agent/extensions/local-models.ts
// Pi extension for local models integration
//
// Configure per-project via .pi/agent/local-models.json:
//   { "url": "http://127.0.0.1:8000", "apiKey": "1234" }
// Or globally: {homedir}/.pi/agent/local-models.json
// - Windows: %USERPROFILE%/.pi/agent/local-models.json
// - macOs/Linux: ~/.pi/agent/local-models.json

import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";
import { existsSync, readFileSync } from "node:fs";
import { homedir } from "os";
import { join } from "node:path";
import { TextDecoder } from "node:util";

const STATUS_KEY = "local-models";
const MODEL_LOAD_POLL_MS = 500;
const MODEL_LOAD_TIMEOUT_MS = 10 * 60 * 1000;
const MODEL_LOAD_BAR_WIDTH = 20;
const DEFAULT_CTX_WINDOW = 128000;
const DEFAULT_MAX_TOKENS = 16384;
const DEFAULT_INPUT = ["text"];

type TerminalLoadState = "loaded" | "failed" | "unavailable" | "timeout";

type StatusThemeColor = "accent" | "dim" | "muted" | "error";

interface LocalModelsConfig {
    url: string;
    apiKey?: string;
    contextWindow?: number,
    maxTokens?: number,
    reasoning: boolean,
    input?: string[],
}

interface StatusUI {
    readonly theme?: StatusTheme;
    setStatus(key: string, text: string | undefined): void;
}

interface StatusTheme {
    fg(color: StatusThemeColor, text: string): string;
}

interface LoadingPayload {
    current?: string;
    stage?: string;
    value?: number;
}

interface ServerModel {
    id: string;
    state?: string;
    max_model_len?: number;
    payload?: LoadingPayload;
    status?: {
        value?: string;
        failed?: boolean;
        exit_code?: number;
    };
}

interface ModelSseEvent {
    model?: string;
    event?: string;
    state?: string;
    payload?: LoadingPayload;
    data?: {
        status?: string;
        progress?: LoadingPayload;
        exit_code?: number;
    };
}

interface ModelRef {
    provider?: string;
    id?: string;
}

interface ActiveLoad {
    modelId: string;
    controller: AbortController;
    promise: Promise<void>;
}

function readConfig(cwd: string): LocalModelsConfig {
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
    let contextWindow: number = DEFAULT_CTX_WINDOW;
    let maxTokens: number = DEFAULT_MAX_TOKENS;
    let input: string[] = DEFAULT_INPUT;
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

    if (cfg.input)
        input = cfg.input;

    return {
        url: url,
        apiKey: apiKey,
        contextWindow: contextWindow,
        maxTokens: maxTokens,
        reasoning: reasoning,
        input: input
    };
}

function rpc(config: LocalModelsConfig, endpoint: string, method: string, body?: Record<string, unknown>) {

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

function isSelectableModel(id: string): boolean {
    return (
        id !== "local-model" &&
        id !== "main" &&
        !/^models:\d+$/.test(id)
    );
}

async function listModels(config: LocalModelsConfig): Promise<ServerModel[]> {
    const data = (await rpc(config, "/v1/models", "GET")) as {
        data?: ServerModel[];
    };
    return (data.data ?? []).filter((m) => m.id && isSelectableModel(m.id));
}

async function findModel(
    config: LocalModelsConfig,
    modelId: string
): Promise<ServerModel | undefined> {
    return (await listModels(config)).find((m) => m.id === modelId);
}

function sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
}

function modelState(model: ServerModel): string | undefined {
    if (model.status?.failed || (model.status?.exit_code ?? 0) !== 0) {
        return "failed";
    }
    return model.status?.value ?? model.state;
}

function isLocalModel(
    model: ModelRef | undefined
): model is ModelRef & { provider: "local-model"; id: string } {
    return model?.provider === "local-model" && typeof model.id === "string";
}

function themeFg(
    ui: StatusUI,
    color: StatusThemeColor,
    text: string
): string {
    if (!text) return "";
    return ui.theme?.fg(color, text) ?? text;
}

function loadingStatus(
    ui: StatusUI,
    _modelId: string,
    progress?: LoadingPayload
): string {
    const value = progress?.value;
    const stage = progress?.current ?? progress?.stage;
    const hasProgress = typeof value === "number" && Number.isFinite(value);
    const clamped = hasProgress ? Math.max(0, Math.min(1, value)) : 0;
    const filled = Math.round(clamped * MODEL_LOAD_BAR_WIDTH);
    const empty = MODEL_LOAD_BAR_WIDTH - filled;

    const label = themeFg(ui, "dim", "Loading ");
    const open = themeFg(ui, "dim", "[");
    const filledBar = themeFg(ui, "accent", "#".repeat(filled));
    const emptyBar = themeFg(ui, "dim", "-".repeat(empty));
    const close = themeFg(ui, "dim", "]");
    const percent = themeFg(ui, "dim", ` ${Math.round(clamped * 100)}%`);
    const suffix = stage ? themeFg(ui, "muted", ` ${stage}`) : "";

    return `${label}${open}${filledBar}${emptyBar}${close}${percent}${suffix}`;
}

function loadedStatus(ui: StatusUI, modelId: string): string {
    return `${themeFg(ui, "accent", "Loaded")}${themeFg(ui, "dim", ` ${modelId}`)}`;
}

function failedStatus(ui: StatusUI, modelId: string): string {
    return `${themeFg(ui, "error", "Failed to load")}${themeFg(ui, "dim", ` ${modelId}`)}`;
}

function parseSseMessage(raw: string): ModelSseEvent | undefined {
    const data = raw
        .split("\n")
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trimStart())
        .join("\n")
        .trim();

    if (!data || data === "[DONE]") return undefined;

    try {
        return JSON.parse(data) as ModelSseEvent;
    } catch {
        return undefined;
    }
}

function applySseEvent(
    event: ModelSseEvent | undefined,
    modelId: string,
    ui: StatusUI
): TerminalLoadState | undefined {
    if (!event || (event.model && event.model !== modelId)) return undefined;

    const status = event.data?.status ?? event.state;
    const progress = event.data?.progress ?? event.payload;
    if (status === "loading") {
        ui.setStatus(STATUS_KEY, loadingStatus(ui, modelId, progress));
    } else if (status === "loaded") {
        ui.setStatus(STATUS_KEY, loadedStatus(ui, modelId));
        return "loaded";
    } else if (status === "unloaded" || status === "failed") {
        ui.setStatus(STATUS_KEY, failedStatus(ui, modelId));
        return "failed";
    }

    return undefined;
}

async function watchModelSse(
    config: LocalModelsConfig,
    modelId: string,
    ui: StatusUI,
    signal: AbortSignal
): Promise<TerminalLoadState | undefined> {
    let res: Response;
    try {
        res = await fetch(`${config.url}/models/sse`, { signal });
    } catch {
        return signal.aborted ? undefined : "unavailable";
    }

    if (!res.ok || !res.body) return "unavailable";

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    try {
        while (!signal.aborted) {
            const { value, done } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");

            let boundary = buffer.indexOf("\n\n");
            while (boundary !== -1) {
                const raw = buffer.slice(0, boundary);
                buffer = buffer.slice(boundary + 2);

                const result = applySseEvent(parseSseMessage(raw), modelId, ui);
                if (result) return result;

                boundary = buffer.indexOf("\n\n");
            }
        }
    } catch {
        return signal.aborted ? undefined : "unavailable";
    } finally {
        await reader.cancel().catch(() => undefined);
    }

    return undefined;
}

async function pollTerminalModelState(
    config: LocalModelsConfig,
    modelId: string,
    signal: AbortSignal
): Promise<TerminalLoadState> {
    const deadline = Date.now() + MODEL_LOAD_TIMEOUT_MS;

    while (!signal.aborted && Date.now() < deadline) {
        try {
            const model = await findModel(config, modelId);
            const state = model ? modelState(model) : undefined;
            if (state === "loaded" || state === "sleeping") return "loaded";
            if (state === "failed") return "failed";
        } catch {
            // Keep waiting; SSE may still be connected and the router may be reloading.
        }

        await sleep(MODEL_LOAD_POLL_MS);
    }

    return signal.aborted ? "unavailable" : "timeout";
}

async function trackModelLoad(
    config: LocalModelsConfig,
    modelId: string,
    ui: StatusUI,
    signal: AbortSignal
) {
    ui.setStatus(STATUS_KEY, loadingStatus(ui, modelId));

    const controller = new AbortController();
    const abort = () => controller.abort();
    signal.addEventListener("abort", abort, { once: true });

    try {
        const ssePromise = watchModelSse(config, modelId, ui, controller.signal);
        const pollPromise = pollTerminalModelState(config, modelId, controller.signal);

        let result = await Promise.race([ssePromise, pollPromise]);
        if (!result || result === "unavailable") {
            result = await pollPromise;
        }

        controller.abort();

        if (signal.aborted) return;

        if (result === "loaded") {
            ui.setStatus(STATUS_KEY, loadedStatus(ui, modelId));
            await sleep(1500);
            ui.setStatus(STATUS_KEY, undefined);
        } else if (result === "failed") {
            ui.setStatus(STATUS_KEY, failedStatus(ui, modelId));
        } else {
            ui.setStatus(STATUS_KEY, undefined);
        }
    } finally {
        signal.removeEventListener("abort", abort);
        controller.abort();
    }
}

export default async function (pi: ExtensionAPI) {
    const config = readConfig(process.cwd());
    const serverModels = await listModels(config).catch((): ServerModel[] => []);
    if (serverModels.length === 0) {
        console.log("\x1b[31m", "local-models: " + "No local models were discovered");
        return;
    }

    console.log('\x1b[36m%s\x1b[0m', "local-models: " + serverModels.length + " local models were discovered, type /model to select a [local-model]");

    const models = serverModels.map((m) => ({
        id: String(m.id),
        name: String(m.id),
        reasoning: config.reasoning,
        input: config.input,
        contextWindow: config.contextWindow ?? m.max_model_len,
        maxTokens: config.maxTokens,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
        compat: {
            supportsDeveloperRole: false,
            supportsReasoningEffort: false,
        }
    }));

    pi.registerProvider("local-model", {
        baseUrl: `${config.url}/v1`,
        api: "openai-completions",
        apiKey: config.apiKey ?? "not-needed",
        models
    });

    let activeLoad: ActiveLoad | undefined;

    async function ensureModelLoaded(
        cwd: string,
        modelId: string,
        ui: StatusUI
    ): Promise<void> {
        if (
            activeLoad?.modelId === modelId &&
            !activeLoad.controller.signal.aborted
        ) {
            await activeLoad.promise;
            return;
        }

        activeLoad?.controller.abort();
        const controller = new AbortController();

        const promise = (async () => {
            const config = readConfig(cwd);
            ui.setStatus(STATUS_KEY, loadingStatus(ui, modelId));

            const current = await findModel(config, modelId).catch(() => undefined);
            const currentState = current ? modelState(current) : undefined;
            if (currentState === "loaded" || currentState === "sleeping") {
                ui.setStatus(STATUS_KEY, undefined);
                return;
            }

            const progressPromise = trackModelLoad(
                config,
                modelId,
                ui,
                controller.signal
            );

            if (currentState !== "loading") {
                await sleep(50);
                await rpc(config, "models/load", "POST", { model: modelId }).catch(async () => {
                    const latest = await findModel(config, modelId).catch(() => undefined);
                    const latestState = latest ? modelState(latest) : undefined;
                    if (latestState === "loaded" || latestState === "sleeping") {
                        ui.setStatus(STATUS_KEY, undefined);
                        controller.abort();
                    } else if (latestState !== "loading") {
                        ui.setStatus(STATUS_KEY, undefined);
                        controller.abort();
                    }
                });
            }

            await progressPromise;
        })();

        activeLoad = { modelId, controller, promise };
        try {
            await promise;
        } finally {
            if (activeLoad?.controller === controller) {
                activeLoad = undefined;
            }
        }
    }

    pi.on("model_select", (event: { model: ModelRef | undefined; }, ctx: { cwd: string; ui: StatusUI; }) => {
        if (!isLocalModel(event.model)) return;
        void ensureModelLoaded(ctx.cwd, event.model.id, ctx.ui).catch(
            () => undefined
        );
    });

    pi.on("before_provider_request", async (event: { payload: any; }, ctx: { model: ModelRef | undefined; cwd: string; ui: StatusUI; }) => {
        if (!isLocalModel(ctx.model)) return;
        await ensureModelLoaded(ctx.cwd, ctx.model.id, ctx.ui);
        return event.payload;
    });
};