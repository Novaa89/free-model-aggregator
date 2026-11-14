// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// The server reads the trial brief (provider directory, quality table, workload manifest),
// probes every provider's catalog route, builds one catalog, registers with the env, and then
// answers the client's 20 requests: POST /v1/chat/completions and POST /v1/messages, JSON and
// server-sent events, with GET /v1/models in the wire its auth header names.
//
// Routing: for each request, every free model that meets the request's requirements is a
// candidate, ranked by the quality table's rating (best first). A provider call that is refused
// (402, 429 or 5xx) is retried once on the same model — a refusal on the provider's failure
// schedule is transient, and a retry that lands keeps the best model — and then the request
// fails over to the next candidate. A provider that fails two calls in a row is deprioritised
// for the rest of the trial: its budget is spent or it is in outage, and hammering it only
// spends calls. Every choice is a function of the brief, the requests and the providers'
// answers: no wall clock, no unseeded randomness, so a replay leaves the same state.
//
// Wires: an OpenAI client request is passed to an openai-wire provider untouched (its system
// prompt, images, tools, stop sequences, max_tokens and JSON mode); an Anthropic client
// request is rewritten to Chat Completions and back; a Gemini provider is spoken to with
// generateContent / streamGenerateContent. The provider echoes what it parsed, and the client
// compares that echo with what it sent, so every translation keeps the system text, the user
// text, the image bytes, the tool names, max_tokens, the stop sequences and the JSON-mode flag
// exactly. A JSON-mode OpenAI request is only ever routed to a wire that can carry JSON mode
// (openai or gemini), never to an anthropic-wire provider, which would echo it absent.
// Streaming answers are consumed whole from the provider and re-emitted in the client wire's
// event order, with the provider's text whole, its usage counts and its stop reason mapped.
//
// `node main.ts --list` prints the catalog document and exits 0. Run outside a trial, with no
// MAKEFUN_PROMPT_FILE and no MAKEFUN_ENV_URL set, it prints {"providers":[]} and exits 0.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { readFileSync } from 'node:fs';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider inference call may take, in milliseconds. */
const CALL_TIMEOUT_MS = 15000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** A fixed epoch second for every listed model's `created`: nothing in a reply reads the clock. */
const FIXED_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in a reply reads the clock. */
const FIXED_CREATED_AT = '2026-01-01T00:00:00Z';

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

type Wire = 'openai' | 'anthropic' | 'gemini';
type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';
type Json = Record<string, unknown>;
type Stop = 'stop' | 'max_tokens' | 'tool_use' | 'refusal';

const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);
const stringOf = (v: unknown): string => (typeof v === 'string' ? v : '');
const numberOrNull = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);
const boolOrNull = (v: unknown): boolean | null => (typeof v === 'boolean' ? v : null);

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

// ---------------------------------------------------------------------------
// The brief: provider directory, quality table, workload manifest.
// ---------------------------------------------------------------------------

const PROVIDERS_SCHEMA = 'free-model-aggregator-01/providers/v1';
const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1';
const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';

interface ProviderDirectoryEntry {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  base: string;
  key_env: string;
  key: string;
}

interface RatingRow {
  model: string;
  provider: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

interface WorkloadRequest {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requires: Requirement;
}

interface Brief {
  providers: ProviderDirectoryEntry[];
  ratings: RatingRow[];
  best: { id: string; rating_milli: number }[];
  manifest: WorkloadRequest[];
  error: string | null;
}

/** The index of the `}` closing the object that opens at `start`, or -1; string contents are skipped. */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/** Every JSON object in `text` that carries a string `schema`, in the order they appear. */
function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = objectEnd(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') out.push(value);
    } catch {
      // Not JSON on its own: look on.
    }
  }
  return out;
}

function providerOf(v: unknown): ProviderDirectoryEntry | null {
  if (!isRecord(v)) return null;
  const id = stringOf(v.id);
  const wire = v.wire === 'openai' || v.wire === 'anthropic' || v.wire === 'gemini' ? v.wire : null;
  if (id === '' || wire === null) return null;
  const auth =
    v.auth === 'bearer' || v.auth === 'x-api-key' || v.auth === 'x-goog-api-key' || v.auth === 'none' ? v.auth : 'bearer';
  // The brief's directory writes `path` (/p/<id>, joined to $MAKEFUN_ENV_URL); a directory that spells an
  // absolute base instead may use base_path, base_url or base. Take the first non-empty, `path` first.
  const base =
    stringOf(v.path) !== ''
      ? stringOf(v.path)
      : stringOf(v.base_path) !== ''
        ? stringOf(v.base_path)
        : stringOf(v.base_url) !== ''
          ? stringOf(v.base_url)
          : stringOf(v.base);
  return { id, name: stringOf(v.name) !== '' ? stringOf(v.name) : id, wire, auth, base, key_env: stringOf(v.key_env), key: stringOf(v.key) };
}

function ratingOf(v: unknown): RatingRow | null {
  if (!isRecord(v)) return null;
  const model = stringOf(v.model);
  const provider = stringOf(v.provider);
  if (model === '' || provider === '') return null;
  return {
    model,
    provider,
    rating_milli: typeof v.rating_milli === 'number' && Number.isFinite(v.rating_milli) ? v.rating_milli : 0,
    context_length: numberOrNull(v.context_length),
    tools: v.tools === true,
    vision: v.vision === true,
    json: v.json === true,
    free: v.free === true,
  };
}

function workloadRequestOf(v: unknown): WorkloadRequest | null {
  if (!isRecord(v)) return null;
  const id = stringOf(v.id);
  if (id === '') return null;
  const requires = isRecord(v.requires) ? v.requires : {};
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: {
      tools: requires.tools === true,
      vision: requires.vision === true,
      json: requires.json === true,
      ...(typeof requires.min_context === 'number' && Number.isFinite(requires.min_context) ? { min_context: requires.min_context } : {}),
    },
  };
}

/** The brief at `path`, its three documents found by schema. */
function readBrief(path: string): Brief {
  const empty: Brief = { providers: [], ratings: [], best: [], manifest: [], error: null };
  if (path === '') return { ...empty, error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { ...empty, error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = documentsIn(text);
  const directoryDoc = docs.find((d) => d.schema === PROVIDERS_SCHEMA && Array.isArray(d.providers)) ?? null;
  if (directoryDoc === null || !Array.isArray(directoryDoc.providers)) {
    return { ...empty, error: `the brief at ${path} holds no ${PROVIDERS_SCHEMA} document` };
  }
  const providers = directoryDoc.providers.flatMap((v) => {
    const p = providerOf(v);
    return p === null ? [] : [p];
  });
  const ratingsDoc = docs.find((d) => d.schema === RATINGS_SCHEMA && Array.isArray(d.ratings)) ?? null;
  const ratings =
    ratingsDoc === null || !Array.isArray(ratingsDoc.ratings)
      ? []
      : ratingsDoc.ratings.flatMap((v) => {
          const r = ratingOf(v);
          return r === null ? [] : [r];
        });
  const best =
    ratingsDoc !== null && Array.isArray(ratingsDoc.best)
      ? ratingsDoc.best.flatMap((v) => {
          if (!isRecord(v) || stringOf(v.id) === '') return [];
          return [{ id: stringOf(v.id), rating_milli: numberOrNull(v.rating_milli) ?? 0 }];
        })
      : [];
  const workloadDoc = docs.find((d) => d.schema === WORKLOAD_SCHEMA && Array.isArray(d.manifest)) ?? null;
  const manifest =
    workloadDoc === null || !Array.isArray(workloadDoc.manifest)
      ? []
      : workloadDoc.manifest.flatMap((v) => {
          const w = workloadRequestOf(v);
          return w === null ? [] : [w];
        });
  return { providers, ratings, best, manifest, error: null };
}

// ---------------------------------------------------------------------------
// Catalog probe.
// ---------------------------------------------------------------------------

/** A base path joined to the env's URL; an absolute base_url is used as it stands. */
function joinBase(envUrl: string, base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, '');
  const prefix = envUrl.replace(/\/+$/, '');
  if (b === '') return prefix;
  return `${prefix}${b.startsWith('/') ? '' : '/'}${b.replace(/\/+$/, '')}`;
}

/** The headers a provider wants its key in. */
function authHeaders(auth: AuthScheme, key: string): Record<string, string> {
  if (key === '') return {};
  if (auth === 'bearer') return { authorization: `Bearer ${key}` };
  if (auth === 'x-api-key') return { 'x-api-key': key };
  if (auth === 'x-goog-api-key') return { 'x-goog-api-key': key };
  return {};
}

/** One model as a provider's catalog route listed it; null members are what the listing did not say. */
interface ProbedModel {
  id: string;
  display_name: string;
  free: boolean | null;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

interface Probe {
  provider: ProviderDirectoryEntry;
  ok: boolean;
  status: number;
  models: ProbedModel[];
  error: string | null;
}

/** A price field of a listing: 0 when every price is zero, null when the listing says nothing about price. */
function zeroPriced(v: Json): boolean | null {
  const pricing = isRecord(v.pricing) ? v.pricing : null;
  if (pricing === null) return null;
  const values = Object.values(pricing).flatMap((x) => {
    if (typeof x === 'number') return [x];
    if (typeof x === 'string' && x.trim() !== '') return [Number(x)];
    return [];
  });
  if (values.length === 0) return null;
  return values.every((n) => Number.isFinite(n) && n === 0);
}

/** One model as a listing wrote it, in whichever wire's spelling. */
function modelIn(wire: Wire, v: unknown): ProbedModel | null {
  if (!isRecord(v)) return null;
  const id = wire === 'gemini' ? stringOf(v.name).replace(/^models\//, '') : stringOf(v.id);
  if (id === '') return null;
  const context = numberOrNull(v.context_length) ?? numberOrNull(v.context_window) ?? numberOrNull(v.inputTokenLimit);
  return {
    id,
    display_name:
      stringOf(v.display_name) !== '' ? stringOf(v.display_name) : stringOf(v.displayName) !== '' ? stringOf(v.displayName) : id,
    free: boolOrNull(v.free) ?? zeroPriced(v),
    context_length: context,
    tools: boolOrNull(v.tools),
    vision: boolOrNull(v.vision),
    json: boolOrNull(v.json),
  };
}

/** The models one catalog route's answer listed, in its wire's shape. */
function modelsIn(wire: Wire, body: unknown): ProbedModel[] {
  if (!isRecord(body)) return [];
  const list = wire === 'gemini' ? body.models : body.data;
  if (!Array.isArray(list)) return [];
  return list.flatMap((v) => {
    const m = modelIn(wire, v);
    return m === null ? [] : [m];
  });
}

/** Probe one provider's catalog route, in its own wire, tolerating every failure. */
async function probeProvider(envUrl: string, provider: ProviderDirectoryEntry, timeoutMs = PROBE_TIMEOUT_MS): Promise<Probe> {
  const base = joinBase(envUrl, provider.base);
  const url = provider.wire === 'gemini' ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers: Record<string, string> = { accept: 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  try {
    const res = await fetch(url, { headers, signal: AbortSignal.timeout(timeoutMs) });
    if (!res.ok) return { provider, ok: false, status: res.status, models: [], error: `GET ${url} answered HTTP ${res.status}` };
    const body: unknown = await res.json();
    return { provider, ok: true, status: res.status, models: modelsIn(provider.wire, body), error: null };
  } catch (err) {
    return { provider, ok: false, status: 0, models: [], error: `GET ${url} failed: ${(err as Error).message}` };
  }
}

interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

/** What `--list` prints and what the entry POSTs to /register: the same document, byte for byte. */
interface CatalogReport {
  providers: CatalogProvider[];
}

function ratingOfModel(ratings: RatingRow[], providerId: string, modelId: string): RatingRow | null {
  return ratings.find((r) => r.provider === providerId && r.model === modelId) ?? null;
}

/**
 * Whether a model is free: the quality table's row where it has one (it is the sealed truth), the
 * listing's own word where it does not, and paid when neither says. A paid model is never routed.
 */
function modelFree(model: ProbedModel, row: RatingRow | null): boolean {
  if (row !== null) return row.free;
  return model.free ?? false;
}

interface Capabilities {
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

/** What a model can do: the quality table's row where it has one, the listing's own fields where it does not. */
function capabilitiesOf(model: ProbedModel, row: RatingRow | null): Capabilities {
  return {
    context_length: row?.context_length ?? model.context_length ?? null,
    tools: row?.tools ?? model.tools ?? null,
    vision: row?.vision ?? model.vision ?? null,
    json: row?.json ?? model.json ?? null,
  };
}

/** Whether a model's capabilities meet one request's requirements. An unknown capability never meets one. */
function meetsRequirements(requires: Requirement, caps: Capabilities): boolean {
  if (requires.tools === true && caps.tools !== true) return false;
  if (requires.vision === true && caps.vision !== true) return false;
  if (requires.json === true && caps.json !== true) return false;
  if (requires.min_context !== undefined && (caps.context_length === null || caps.context_length < requires.min_context)) return false;
  return true;
}

/**
 * The catalog the probes and the quality table make: every provider that answered, every model it listed.
 * Its base_url is the directory's own path (`/p/<id>`), never the path joined to `envUrl`.
 */
function buildCatalog(probes: readonly Probe[], ratings: RatingRow[]): CatalogReport {
  const providers: CatalogProvider[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const seen = new Set<string>();
    const models: CatalogModel[] = [];
    for (const model of [...probe.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))) {
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      const row = ratingOfModel(ratings, probe.provider.id, model.id);
      models.push({ id: model.id, free: modelFree(model, row), context_length: row?.context_length ?? model.context_length ?? null });
    }
    providers.push({ id: probe.provider.id, base_url: probe.provider.base, key_env: probe.provider.key_env, wire: probe.provider.wire, models });
  }
  return { providers };
}

// ---------------------------------------------------------------------------
// Routing: quality-ranked candidates, one retry, then failover.
// ---------------------------------------------------------------------------

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;
/** Per-provider trial state: successes served and consecutive failed calls. */
const providerState = new Map<string, { success: number; consecFail: number }>();

function stateOf(providerId: string): { success: number; consecFail: number } {
  let state = providerState.get(providerId);
  if (state === undefined) {
    state = { success: 0, consecFail: 0 };
    providerState.set(providerId, state);
  }
  return state;
}

/** A provider that failed two calls in a row is deprioritised: its budget is spent or it is in outage. */
function penalized(providerId: string): boolean {
  return stateOf(providerId).consecFail >= 2;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/**
 * The free models that can serve one request, best rating first. A provider that failed twice in a
 * row sorts after every healthy one. A JSON-mode OpenAI request never goes to an anthropic-wire
 * provider: that wire cannot carry response_format, and the provider would echo JSON mode absent.
 */
function candidatesFor(requires: Requirement, needsJsonEcho: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (needsJsonEcho && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort((a, b) => {
    const pa = penalized(a.probe.provider.id) ? 1 : 0;
    const pb = penalized(b.probe.provider.id) ? 1 : 0;
    if (pa !== pb) return pa - pb;
    if (b.rating !== a.rating) return b.rating - a.rating;
    if (a.probe.provider.id !== b.probe.provider.id) return a.probe.provider.id < b.probe.provider.id ? -1 : 1;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
  return out;
}

/** The free models this entry serves, one row a model id, for GET /v1/models. */
function servedModels(): { id: string; provider: string; display_name: string }[] {
  const seen = new Set<string>();
  const out: { id: string; provider: string; display_name: string }[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))) {
      if (seen.has(model.id)) continue;
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      seen.add(model.id);
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const entry = brief.manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** Whether this client request needs its provider to echo JSON mode: an OpenAI request with response_format. */
function needsJsonEcho(clientApi: 'openai' | 'anthropic', body: Json): boolean {
  if (clientApi !== 'openai') return false;
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return format === 'json_object' || format === 'json_schema';
}

// ---------------------------------------------------------------------------
// Text helpers for the wire translations.
// ---------------------------------------------------------------------------

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

// ---------------------------------------------------------------------------
// Request translation: client wire -> upstream wire.
// ---------------------------------------------------------------------------

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that names `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks become
 * tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through.
 */
function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: { name: typeof block.name === 'string' ? block.name : '', arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) },
        });
      }
      const text = textOf(content);
      messages.push({ role: 'assistant', content: text !== '' ? text : null, ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}) });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({ role: 'tool', tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '', content: toolResultText(block.content) });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** One OpenAI content part as an Anthropic content block: text or an image; null for the rest. */
function blockOf(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { type: 'text', text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
    if (data !== null) return { type: 'image', source: { type: 'base64', media_type: data[1] !== '' ? data[1] : 'image/png', data: data[2] } };
    return { type: 'image', source: { type: 'url', url } };
  }
  return null;
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request that names `model`: system
 * messages become the system prompt, image_url parts become image blocks, tool_calls become tool_use
 * blocks, tool messages become tool_result blocks, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p and stop sequences pass through. response_format is never carried:
 * the Anthropic wire cannot echo JSON mode, so a JSON-mode request is never routed here.
 */
function openAIToAnthropic(body: Json, model: string): Json {
  const systemParts: string[] = [];
  const messages: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'tool') {
      messages.push({
        role: 'user',
        content: [
          {
            type: 'tool_result',
            tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
            content: typeof raw.content === 'string' ? raw.content : textOf(raw.content),
          },
        ],
      });
      continue;
    }
    if (raw.role === 'assistant') {
      const blocks: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') blocks.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        blocks.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsOf(call.function.arguments),
        });
      }
      if (blocks.length > 0) messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: 'user', content });
      continue;
    }
    if (Array.isArray(content)) {
      const blocks = content.flatMap((part) => {
        if (!isRecord(part)) return [];
        const block = blockOf(part);
        return block === null ? [] : [block];
      });
      if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
    }
  }

  const out: Json = { model, messages };
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : undefined;
  if (maxTokens !== undefined) out.max_tokens = maxTokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stop = body.stop;
  if (typeof stop === 'string') out.stop_sequences = [stop];
  else if (Array.isArray(stop) && stop.length > 0) out.stop_sequences = stop.filter((s) => typeof s === 'string');
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required') out.tool_choice = { type: 'any' };
  else if (choice === 'none') out.tool_choice = { type: 'none' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/**
 * A Chat Completions request as a Gemini generateContent body: system messages become the system
 * instruction, user and model messages become contents (images as inlineData, tool calls as
 * functionCall parts, tool messages as functionResponse parts), tools become functionDeclarations,
 * and max_tokens, temperature, top_p, stop sequences and JSON mode become the generation config.
 */
function openAIToGemini(body: Json, model: string): Json {
  const systemParts: string[] = [];
  const contents: Json[] = [];
  const toolNames = new Map<string, string>();
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'assistant') {
      const parts: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') parts.push({ text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        const name = typeof call.function.name === 'string' ? call.function.name : '';
        if (typeof call.id === 'string' && call.id !== '') toolNames.set(call.id, name);
        parts.push({ functionCall: { name, args: argumentsOf(call.function.arguments) } });
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    if (raw.role === 'tool') {
      const name = (typeof raw.tool_call_id === 'string' ? toolNames.get(raw.tool_call_id) : undefined) ?? 'unknown';
      contents.push({
        role: 'user',
        parts: [{ functionResponse: { name, response: { result: typeof raw.content === 'string' ? raw.content : textOf(raw.content) } } }],
      });
      continue;
    }
    const content = raw.content;
    const parts: Json[] = [];
    if (typeof content === 'string') {
      if (content !== '') parts.push({ text: content });
    } else if (Array.isArray(content)) {
      for (const part of content) {
        if (!isRecord(part)) continue;
        if (part.type === 'text' && typeof part.text === 'string') parts.push({ text: part.text });
        else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
          const url = part.image_url.url;
          const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
          if (data !== null) parts.push({ inlineData: { mimeType: data[1] !== '' ? data[1] : 'image/png', data: data[2] } });
          else parts.push({ fileData: { fileUri: url } });
        }
      }
    }
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }

  const out: Json = { contents };
  if (systemParts.length > 0) out.systemInstruction = { parts: [{ text: systemParts.join('\n') }] };
  if (Array.isArray(body.tools)) {
    const declarations = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  const config: Json = {};
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : undefined;
  if (maxTokens !== undefined) config.maxOutputTokens = maxTokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  const stop = body.stop;
  if (typeof stop === 'string') config.stopSequences = [stop];
  else if (Array.isArray(stop) && stop.length > 0) config.stopSequences = stop.filter((s) => typeof s === 'string');
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  if (format === 'json_object' || format === 'json_schema') config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  void model;
  return out;
}

/** A client request as a Gemini generateContent body, via Chat Completions when the client is Anthropic. */
function toGemini(clientApi: 'openai' | 'anthropic', body: Json, model: string): Json {
  if (clientApi === 'anthropic') return openAIToGemini(anthropicToOpenAI(body, model), model);
  return openAIToGemini(body, model);
}

// ---------------------------------------------------------------------------
// Upstream answers, normalised.
// ---------------------------------------------------------------------------

interface ToolCall {
  id: string;
  name: string;
  args: string;
}

interface NormalizedAnswer {
  text: string;
  inputTokens: number;
  outputTokens: number;
  stop: Stop;
  toolCalls: ToolCall[];
}

function stopFromOpenAI(finish: unknown): Stop {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'stop';
}

function stopFromAnthropic(reason: unknown): Stop {
  if (reason === 'max_tokens') return 'max_tokens';
  if (reason === 'tool_use') return 'tool_use';
  if (reason === 'refusal') return 'refusal';
  return 'stop';
}

function stopFromGemini(reason: unknown): Stop {
  if (reason === 'MAX_TOKENS') return 'max_tokens';
  if (reason === 'SAFETY') return 'refusal';
  return 'stop';
}

/** One server-sent event frame: its event name (null when unnamed) and its data lines joined. */
interface SseFrame {
  event: string | null;
  data: string;
}

function parseSseFrames(text: string): SseFrame[] {
  const frames: SseFrame[] = [];
  for (const frame of text.replace(/\r\n/g, '\n').split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

function parseJsonFrame(data: string): Json | null {
  const payload = data.trim();
  if (payload === '' || payload === '[DONE]') return null;
  try {
    const value: unknown = JSON.parse(payload);
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

/** An OpenAI Chat Completions answer (JSON or SSE) as one normalised answer. */
function parseOpenAIAnswer(text: string, stream: boolean): NormalizedAnswer {
  const answer: NormalizedAnswer = { text: '', inputTokens: 0, outputTokens: 0, stop: 'stop', toolCalls: [] };
  if (!stream) {
    let body: Json | null = null;
    try {
      const value: unknown = JSON.parse(text);
      body = isRecord(value) ? value : null;
    } catch {
      body = null;
    }
    if (body === null) return answer;
    const choices = Array.isArray(body.choices) ? body.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : {};
    const message = isRecord(choice.message) ? choice.message : {};
    answer.text = typeof message.content === 'string' ? message.content : textOf(message.content);
    const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
    for (const call of calls) {
      if (!isRecord(call) || !isRecord(call.function)) continue;
      answer.toolCalls.push({
        id: typeof call.id === 'string' ? call.id : 'call_unknown',
        name: typeof call.function.name === 'string' ? call.function.name : '',
        args: typeof call.function.arguments === 'string' ? call.function.arguments : JSON.stringify(argumentsOf(call.function.arguments)),
      });
    }
    answer.stop = stopFromOpenAI(choice.finish_reason);
    const usage = isRecord(body.usage) ? body.usage : {};
    answer.inputTokens = numberOrZero(usage.prompt_tokens);
    answer.outputTokens = numberOrZero(usage.completion_tokens);
    return answer;
  }
  const byIndex = new Map<number, ToolCall>();
  for (const frame of parseSseFrames(text)) {
    const chunk = parseJsonFrame(frame.data);
    if (chunk === null) continue;
    if (isRecord(chunk.usage)) {
      answer.inputTokens = numberOrZero(chunk.usage.prompt_tokens);
      answer.outputTokens = numberOrZero(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') answer.text += delta.content;
    const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
    for (const call of calls) {
      if (!isRecord(call)) continue;
      const index = typeof call.index === 'number' ? call.index : byIndex.size;
      let entry = byIndex.get(index);
      if (entry === undefined) {
        entry = { id: '', name: '', args: '' };
        byIndex.set(index, entry);
      }
      if (typeof call.id === 'string' && call.id !== '') entry.id = call.id;
      const fn = isRecord(call.function) ? call.function : {};
      if (typeof fn.name === 'string' && fn.name !== '') entry.name = fn.name;
      if (typeof fn.arguments === 'string') entry.args += fn.arguments;
    }
    if (choice.finish_reason !== null && choice.finish_reason !== undefined) answer.stop = stopFromOpenAI(choice.finish_reason);
  }
  answer.toolCalls = [...byIndex.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([index, entry]) => ({ id: entry.id !== '' ? entry.id : `call_${index}`, name: entry.name, args: entry.args }));
  return answer;
}

/** An Anthropic Messages answer (JSON or SSE) as one normalised answer. */
function parseAnthropicAnswer(text: string, stream: boolean): NormalizedAnswer {
  const answer: NormalizedAnswer = { text: '', inputTokens: 0, outputTokens: 0, stop: 'stop', toolCalls: [] };
  if (!stream) {
    let body: Json | null = null;
    try {
      const value: unknown = JSON.parse(text);
      body = isRecord(value) ? value : null;
    } catch {
      body = null;
    }
    if (body === null) return answer;
    const content = Array.isArray(body.content) ? body.content : [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'text' && typeof block.text === 'string') answer.text += block.text;
      else if (block.type === 'tool_use') {
        answer.toolCalls.push({
          id: typeof block.id === 'string' ? block.id : 'toolu_unknown',
          name: typeof block.name === 'string' ? block.name : '',
          args: JSON.stringify(isRecord(block.input) ? block.input : {}),
        });
      }
    }
    answer.stop = stopFromAnthropic(body.stop_reason);
    const usage = isRecord(body.usage) ? body.usage : {};
    answer.inputTokens = numberOrZero(usage.input_tokens);
    answer.outputTokens = numberOrZero(usage.output_tokens);
    return answer;
  }
  const byIndex = new Map<number, ToolCall>();
  for (const frame of parseSseFrames(text)) {
    const data = parseJsonFrame(frame.data);
    if (data === null) continue;
    if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      answer.inputTokens = numberOrZero(data.message.usage.input_tokens);
    } else if (frame.event === 'content_block_start') {
      const index = typeof data.index === 'number' ? data.index : byIndex.size;
      const block = isRecord(data.content_block) ? data.content_block : {};
      if (block.type === 'tool_use') {
        byIndex.set(index, {
          id: typeof block.id === 'string' ? block.id : `toolu_${index}`,
          name: typeof block.name === 'string' ? block.name : '',
          args: '',
        });
      }
    } else if (frame.event === 'content_block_delta') {
      const index = typeof data.index === 'number' ? data.index : -1;
      const delta = isRecord(data.delta) ? data.delta : {};
      if (delta.type === 'text_delta' && typeof delta.text === 'string') answer.text += delta.text;
      else if (delta.type === 'input_json_delta' && typeof delta.partial_json === 'string') {
        const entry = byIndex.get(index);
        if (entry !== undefined) entry.args += delta.partial_json;
      }
    } else if (frame.event === 'message_delta') {
      const delta = isRecord(data.delta) ? data.delta : {};
      if (typeof delta.stop_reason === 'string') answer.stop = stopFromAnthropic(delta.stop_reason);
      if (isRecord(data.usage)) answer.outputTokens = numberOrZero(data.usage.output_tokens);
    }
  }
  answer.toolCalls = [...byIndex.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([, entry]) => entry);
  return answer;
}

/** A Gemini generateContent answer (JSON or SSE) as one normalised answer. */
function parseGeminiAnswer(text: string, stream: boolean): NormalizedAnswer {
  const answer: NormalizedAnswer = { text: '', inputTokens: 0, outputTokens: 0, stop: 'stop', toolCalls: [] };
  const readOne = (body: Json): void => {
    const candidates = Array.isArray(body.candidates) ? body.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : {};
    const content = isRecord(candidate.content) ? candidate.content : {};
    const parts = Array.isArray(content.parts) ? content.parts : [];
    for (const part of parts) {
      if (!isRecord(part)) continue;
      if (typeof part.text === 'string') answer.text += part.text;
      else if (isRecord(part.functionCall)) {
        answer.toolCalls.push({
          id: `call_${answer.toolCalls.length}`,
          name: typeof part.functionCall.name === 'string' ? part.functionCall.name : '',
          args: JSON.stringify(isRecord(part.functionCall.args) ? part.functionCall.args : {}),
        });
      }
    }
    if (typeof candidate.finishReason === 'string') answer.stop = stopFromGemini(candidate.finishReason);
    const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
    if (typeof usage.promptTokenCount === 'number') answer.inputTokens = usage.promptTokenCount;
    if (typeof usage.candidatesTokenCount === 'number') answer.outputTokens = usage.candidatesTokenCount;
  };
  if (!stream) {
    let body: Json | null = null;
    try {
      const value: unknown = JSON.parse(text);
      body = isRecord(value) ? value : null;
    } catch {
      body = null;
    }
    if (body !== null) readOne(body);
    return answer;
  }
  for (const frame of parseSseFrames(text)) {
    const body = parseJsonFrame(frame.data);
    if (body !== null) readOne(body);
  }
  return answer;
}

// ---------------------------------------------------------------------------
// Client answers, in the wire the client used.
// ---------------------------------------------------------------------------

function stopToOpenAI(stop: Stop): string {
  if (stop === 'max_tokens') return 'length';
  if (stop === 'tool_use') return 'tool_calls';
  if (stop === 'refusal') return 'content_filter';
  return 'stop';
}

function stopToAnthropic(stop: Stop): string {
  if (stop === 'max_tokens') return 'max_tokens';
  if (stop === 'tool_use') return 'tool_use';
  if (stop === 'refusal') return 'refusal';
  return 'end_turn';
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** Split a text into pieces of at most `size` characters, for streamed deltas. */
function splitText(text: string, size: number): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += size) pieces.push(text.slice(at, at + size));
  return pieces.length === 0 ? [''] : pieces;
}

/** A normalised answer as an OpenAI Chat Completions body, or as its SSE stream. */
function emitOpenAI(answer: NormalizedAnswer, model: string, stream: boolean): string {
  const usage = { prompt_tokens: answer.inputTokens, completion_tokens: answer.outputTokens, total_tokens: answer.inputTokens + answer.outputTokens };
  if (!stream) {
    const message: Json = { role: 'assistant', content: answer.text };
    if (answer.toolCalls.length > 0) {
      message.tool_calls = answer.toolCalls.map((call) => ({
        id: call.id,
        type: 'function',
        function: { name: call.name, arguments: call.args },
      }));
    }
    return JSON.stringify({
      id: 'chatcmpl-agg',
      object: 'chat.completion',
      created: FIXED_CREATED,
      model,
      choices: [{ index: 0, message, finish_reason: stopToOpenAI(answer.stop) }],
      usage,
    });
  }
  const base = { id: 'chatcmpl-agg', object: 'chat.completion.chunk', created: FIXED_CREATED, model };
  const frames: string[] = [
    sseEvent(null, { ...base, choices: [{ index: 0, delta: { role: 'assistant', content: '' }, finish_reason: null }] }),
  ];
  for (const piece of splitText(answer.text, 120)) {
    if (piece === '') continue;
    frames.push(sseEvent(null, { ...base, choices: [{ index: 0, delta: { content: piece }, finish_reason: null }] }));
  }
  answer.toolCalls.forEach((call, index) => {
    frames.push(
      sseEvent(null, {
        ...base,
        choices: [{ index: 0, delta: { tool_calls: [{ index, id: call.id, type: 'function', function: { name: call.name, arguments: '' } }] }, finish_reason: null }],
      }),
    );
    if (call.args !== '') {
      frames.push(
        sseEvent(null, {
          ...base,
          choices: [{ index: 0, delta: { tool_calls: [{ index, function: { arguments: call.args } }] }, finish_reason: null }],
        }),
      );
    }
  });
  frames.push(sseEvent(null, { ...base, choices: [{ index: 0, delta: {}, finish_reason: stopToOpenAI(answer.stop) }] }));
  frames.push(sseEvent(null, { ...base, choices: [], usage }));
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** A normalised answer as an Anthropic Messages body, or as its SSE stream. */
function emitAnthropic(answer: NormalizedAnswer, model: string, stream: boolean): string {
  if (!stream) {
    const content: Json[] = [];
    if (answer.text !== '') content.push({ type: 'text', text: answer.text });
    for (const call of answer.toolCalls) {
      content.push({ type: 'tool_use', id: call.id, name: call.name, input: argumentsOf(call.args) });
    }
    if (content.length === 0) content.push({ type: 'text', text: '' });
    return JSON.stringify({
      id: 'msg_agg',
      type: 'message',
      role: 'assistant',
      model,
      content,
      stop_reason: stopToAnthropic(answer.stop),
      stop_sequence: null,
      usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
    });
  }
  const frames: string[] = [
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_agg',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.inputTokens, output_tokens: 0 },
      },
    }),
  ];
  let index = 0;
  if (answer.text !== '' || answer.toolCalls.length === 0) {
    frames.push(sseEvent('content_block_start', { type: 'content_block_start', index, content_block: { type: 'text', text: '' } }));
    for (const piece of splitText(answer.text, 120)) {
      if (piece === '') continue;
      frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'text_delta', text: piece } }));
    }
    frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    index += 1;
  }
  for (const call of answer.toolCalls) {
    frames.push(
      sseEvent('content_block_start', { type: 'content_block_start', index, content_block: { type: 'tool_use', id: call.id, name: call.name, input: {} } }),
    );
    if (call.args !== '') {
      frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: call.args } }));
    }
    frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    index += 1;
  }
  frames.push(
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: stopToAnthropic(answer.stop), stop_sequence: null },
      usage: { output_tokens: answer.outputTokens },
    }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return frames.join('');
}

// ---------------------------------------------------------------------------
// One provider call, and one client request served.
// ---------------------------------------------------------------------------

interface UpstreamResult {
  ok: boolean;
  status: number;
  answer: NormalizedAnswer | null;
}

/** One inference call to one candidate, in its own wire, read whole. */
async function callCandidate(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json): Promise<UpstreamResult> {
  const { probe, model } = candidate;
  const base = joinBase(ENV_URL, probe.provider.base);
  const stream = body.stream === true;
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(probe.provider.auth, probe.provider.key) };
  let url: string;
  let upstreamBody: Json;
  if (probe.provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    upstreamBody =
      clientApi === 'openai'
        ? { ...body, model: model.id, ...(stream ? { stream: true, stream_options: { include_usage: true } } : {}) }
        : anthropicToOpenAI(body, model.id);
    if (stream) headers.accept = 'text/event-stream';
  } else if (probe.provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = '2023-06-01';
    upstreamBody = clientApi === 'anthropic' ? { ...body, model: model.id } : openAIToAnthropic(body, model.id);
    if (stream) headers.accept = 'text/event-stream';
  } else {
    const action = stream ? 'streamGenerateContent' : 'generateContent';
    url = `${base}/v1beta/models/${encodeURIComponent(model.id)}:${action}${stream ? '?alt=sse' : ''}`;
    upstreamBody = toGemini(clientApi, body, model.id);
    if (stream) headers.accept = 'text/event-stream';
  }
  let res: Response;
  try {
    res = await fetch(url, { method: 'POST', headers, body: JSON.stringify(upstreamBody), signal: AbortSignal.timeout(CALL_TIMEOUT_MS) });
  } catch (err) {
    log(`call to ${probe.provider.id}/${model.id} failed: ${(err as Error).message}`);
    return { ok: false, status: 0, answer: null };
  }
  const text = await res.text();
  if (!res.ok) {
    log(`call to ${probe.provider.id}/${model.id} answered HTTP ${res.status}`);
    return { ok: false, status: res.status, answer: null };
  }
  const answer =
    probe.provider.wire === 'openai'
      ? parseOpenAIAnswer(text, stream)
      : probe.provider.wire === 'anthropic'
        ? parseAnthropicAnswer(text, stream)
        : parseGeminiAnswer(text, stream);
  return { ok: true, status: res.status, answer };
}

interface ServedResponse {
  status: number;
  contentType: string;
  text: string;
}

/** Serve one client request: the best free model that answers, failing over down the ranking. */
async function serve(clientApi: 'openai' | 'anthropic', body: Json): Promise<ServedResponse> {
  const requires = requirementsFor(body);
  const candidates = candidatesFor(requires, needsJsonEcho(clientApi, body));
  const stream = body.stream === true;
  for (const candidate of candidates) {
    // One retry on the same model: a refusal on the provider's failure schedule is transient,
    // and a retry that lands keeps the best model for this request.
    for (let attempt = 0; attempt < 2; attempt++) {
      const result = await callCandidate(candidate, clientApi, body);
      if (result.ok && result.answer !== null) {
        const state = stateOf(candidate.probe.provider.id);
        state.success += 1;
        state.consecFail = 0;
        const text = clientApi === 'openai' ? emitOpenAI(result.answer, candidate.model.id, stream) : emitAnthropic(result.answer, candidate.model.id, stream);
        return { status: 200, contentType: stream ? 'text/event-stream' : 'application/json', text };
      }
      stateOf(candidate.probe.provider.id).consecFail += 1;
      // A refusal that is not a budget or rate refusal (a 400, a 404) is the request's own fault:
      // retrying the same model cannot fix it, so fail over at once.
      if (result.status !== 0 && result.status !== 402 && result.status !== 429 && result.status < 500) break;
    }
  }
  const message = 'no free model on the directory meets this request';
  if (clientApi === 'anthropic') {
    return { status: 503, contentType: 'application/json', text: JSON.stringify({ type: 'error', error: { type: 'overloaded_error', message } }) };
  }
  return { status: 503, contentType: 'application/json', text: JSON.stringify({ error: { message, type: 'no_free_model', code: null } }) };
}

// ---------------------------------------------------------------------------
// HTTP server: the two client APIs and the model list.
// ---------------------------------------------------------------------------

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendServed(res: ServerResponse, served: ServedResponse): void {
  res.writeHead(served.status, { 'content-type': served.contentType, 'content-length': Buffer.byteLength(served.text) });
  res.end(served.text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, { type: 'error', error: { type, message } });
  else sendJson(res, status, { error: { message, type, code: null } });
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') {
    sendJson(res, 200, {
      data: list.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_CREATED_AT })),
      has_more: false,
    });
  } else {
    sendJson(res, 200, {
      object: 'list',
      data: list.map((m) => ({ id: m.id, object: 'model', created: FIXED_CREATED, owned_by: m.provider })),
    });
  }
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    const body = await readJson(req);
    if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
    sendServed(res, await serve('openai', body));
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    const body = await readJson(req);
    if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
    sendServed(res, await serve('anthropic', body));
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(probes, brief.ratings);
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
