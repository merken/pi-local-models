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

const DEFAULT_CTX_WINDOW = 128000;
const DEFAULT_MAX_TOKENS = 16384;
const DEFAULT_INPUT = ["text"];

interface LocalModelsConfig {
    url: string;
    apiKey?: string;
    contextWindow?: number,
    maxTokens?: number,
    reasoning: boolean,
    input?: string[],
}

interface LoadingPayload {
    current?: string;
    stage?: string;
    value?: number;
}

interface ServerModel {
    id: string;
    state?: string;
    context_length?: number;
    max_model_len?: number;
    payload?: LoadingPayload;
    status?: {
        value?: string;
        failed?: boolean;
        exit_code?: number;
    };
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
    let contextWindow: number | undefined = undefined;
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

async function listModels(config: LocalModelsConfig): Promise<ServerModel[]> {
    const data = (await openAiApiCall(config, "/v1/models", "GET")) as {
        data?: ServerModel[];
    };
    return (data.data ?? []);
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
        contextWindow: m.context_length ?? m.max_model_len ?? config.contextWindow ?? DEFAULT_CTX_WINDOW,
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
};