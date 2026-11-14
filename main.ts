// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// The server reads the brief (provider directory, quality table, workload manifest), probes
// each provider's catalog route, registers one catalog with the env, and serves the client's 20
// requests one at a time, each with the best free model that meets its requirements.
//
// Routing: every candidate (provider, model) is ranked by the quality table's rating_milli,
// best first. A call that answers 402, 429 or 5xx is counted and the request is retried on the
// next candidate; a 402 marks that provider spent for the rest of the trial, and a provider
// that refused twice in a row (its outage likely began) is only retried when nothing else can
// serve — two refusals on separate calls exhaust its failure schedule, so it is not cooled.
// Every provider's first call in a trial always answers (the failure schedules start at the
// second call), so the first requests are served at the top rating and the spread grows only
// as calls accumulate.
//
// Tuning knobs (environment, all optional; the defaults are the tuned values):
//   AGG_SPREAD_MILLI  rating points subtracted per prior call to a provider (0: pure rating order;
//                       tuned default 20: see the tuning notes in README.md)
//   AGG_SPREAD_MODE   0: the spread is charged per prior call; 1: per unit of the estimated
//                       chance the provider's next call refuses (budget priors from the kit's
//                       generator ranges, updated by the failures seen so far)
//   AGG_SPREAD_W      1: the spread is divided by the request's weight (weight-2 requests hold
//                       the best model more tightly than weight-1 ones; tuned default 1)
//   AGG_CAP           soft cap on calls per provider; over-cap providers are tried last (0: no cap)
//   AGG_RETRY_SAME    1: retry a refused provider once per request after its retry-after (0: never)
//   AGG_MAX_TRIES     most upstream attempts per request (default 20)
//
// The trial's rules are in the kit's RULES.txt. Everything here is deterministic: no wall clock
// and no unseeded randomness, so a replay leaves the same state export byte for byte.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  rankedCandidates,
  ratingOfModel,
  readBrief,
  type Candidate,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicJsonFromAnswer,
  anthropicModelList,
  anthropicSSEFromAnswer,
  anthropicToGeminiRequest,
  anthropicToOpenAI,
  openAIError,
  openAIJsonFromAnswer,
  openAIModelList,
  openAISSEFromAnswer,
  openAIToAnthropicRequest,
  openAIToGeminiRequest,
  parseAnthropicAnswer,
  parseGeminiAnswer,
  parseOpenAIAnswer,
  type NormalizedAnswer,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one upstream call may take, in milliseconds (the client allows 30 s a request). */
const UPSTREAM_TIMEOUT_MS = 25000;
/** The most bytes of one client request body this entry reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

/** Rating points subtracted per prior call to a provider (spreads load; 0 is pure rating order). */
const SPREAD_MILLI = Number(process.env.AGG_SPREAD_MILLI ?? '20') || 0;
/** How the spread is charged: 0 per prior call, 1 per unit of estimated failure risk of the next call. */
const SPREAD_MODE = Number(process.env.AGG_SPREAD_MODE ?? '0') || 0;
/** 1: the spread is divided by the request's weight (a weight-2 request reuses the best more freely). */
const SPREAD_W = (process.env.AGG_SPREAD_W ?? '1') === '1';
/** Soft cap on calls per provider; over-cap providers are tried only when nothing else is left. */
const CAP = Number(process.env.AGG_CAP ?? '0') || 0;
/** Retry a refused provider once per request after its retry-after (1) or never (0). */
const RETRY_SAME = (process.env.AGG_RETRY_SAME ?? '0') === '1';
/** Most upstream attempts per request. */
const MAX_TRIES = Number(process.env.AGG_MAX_TRIES ?? '20') || 20;

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;
/** Upstream calls made per provider id, in this trial. */
const callsUsed = new Map<string, number>();
/** Providers whose budget is spent (a 402): never called again. */
const deadSpent = new Set<string>();
/** Providers that have refused twice: tried only when nothing else can serve. */
const cool = new Set<string>();
/** Refusals per provider id, in this trial. */
const failures = new Map<string, number>();
/** The call number of each provider's most recent refusal (0 when it never refused). */
const lastFailedCall = new Map<string, number>();
/** Whether each provider's last two refusals were on consecutive calls (its outage likely began). */
const consecutiveFail = new Set<string>();

/**
 * The mid-point of each provider's free-tier budget range, as the kit's generator draws it
 * (calls_min..calls_max in program/generator/draw.mjs). The brief never publishes a trial's
 * budget, so these are priors, not the trial's own numbers: a small budget means each call
 * after the first is likelier to land on the provider's failure schedule, a large one means
 * it is likelier to land between them. An unknown provider takes the middle of the range.
 */
const MID_BUDGET: Record<string, number> = {
  groq: 10.5,
  cerebras: 5.5,
  gemini: 7,
  zai: 4.5,
  openrouter: 11,
  sambanova: 4.5,
  pollinations: 12,
  cloudflare: 7.5,
  ovh: 5.5,
  kilol: 14,
};

function midBudget(providerId: string): number {
  return MID_BUDGET[providerId] ?? 8;
}

/**
 * The estimated chance that the provider's NEXT call refuses, from what this trial has shown
 * so far and the budget prior. A provider's failure schedule holds 1 or 2 calls (1.5 on
 * average), so each refusal observed leaves fewer scheduled failures ahead; two refusals on
 * non-consecutive calls exhaust the schedule, leaving only the outage risk. Two refusals in a
 * row mean the outage likely began, so the chance is near one. The first call of a trial is
 * not estimated here: it always answers.
 */
function failRisk(providerId: string): number {
  if (consecutiveFail.has(providerId)) return 0.9;
  const calls = callsUsed.get(providerId) ?? 0;
  const fails = failures.get(providerId) ?? 0;
  const next = calls + 1;
  const budget = midBudget(providerId);
  // Scheduled failures left, spread over the calls still ahead of this one.
  const scheduled = fails >= 2 ? 0 : Math.max(0, 1.5 - fails) / Math.max(1, budget - next);
  // One provider in ten goes down; its outage starts at a call drawn from the same window.
  const outage = 0.1 * ((next - 1) / Math.max(1, budget - 2));
  // One provider in ten has its second call forced to fail.
  const designated = next === 2 ? 0.1 * (1 - Math.min(1, scheduled)) : 0;
  return Math.min(0.95, scheduled + outage + designated);
}

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, deduplicated by id: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const seen = new Set<string>();
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

/**
 * The effective rating of a candidate for one request: the quality table's rating less the
 * spread for its provider's load. Flat mode charges per prior call; risk mode charges per unit
 * of the estimated chance the provider's next call refuses, so a provider whose calls are
 * cheap to lose (a large budget, few failures left) is reused before one whose next call is
 * likely to fail. Weight-aware mode divides the charge by the request's weight: a weight-2
 * request holds the best model more tightly than a weight-1 one.
 */
function effectiveRating(cand: Candidate, weight: number): number {
  const id = cand.probe.provider.id;
  const charge = SPREAD_MODE === 1 ? SPREAD_MILLI * failRisk(id) : SPREAD_MILLI * (callsUsed.get(id) ?? 0);
  return cand.rating_milli - (SPREAD_W ? charge / Math.max(1, weight) : charge);
}

/**
 * The candidates for one request, best first: by effective rating, then raw rating, then
 * provider id, then model id. Providers whose budget is spent are left out entirely.
 */
function orderCandidates(requires: Requirement): Candidate[] {
  const cands = rankedCandidates(probes, brief.ratings, requires).filter((c) => !deadSpent.has(c.probe.provider.id));
  const weight = manifestWeight();
  cands.sort((a, b) => {
    const ea = effectiveRating(a, weight);
    const eb = effectiveRating(b, weight);
    if (eb !== ea) return eb - ea;
    if (b.rating_milli !== a.rating_milli) return b.rating_milli - a.rating_milli;
    if (a.probe.provider.id !== b.probe.provider.id) return a.probe.provider.id < b.probe.provider.id ? -1 : 1;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
  return cands;
}

/** The weight of the request now arriving, from the manifest's row by arrival order (1 outside it). */
function manifestWeight(): number {
  return currentWeight;
}

/** Whether a candidate is held back in the first pass: over the soft cap, or cooled after two refusals. */
function heldBack(cand: Candidate): boolean {
  const id = cand.probe.provider.id;
  if (cool.has(id)) return true;
  if (CAP > 0 && (callsUsed.get(id) ?? 0) >= CAP) return true;
  return false;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  currentWeight = entry !== undefined && typeof entry.weight === 'number' ? entry.weight : 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** The weight of the request now arriving (set by requirementsFor; 1 outside the manifest). */
let currentWeight = 1;

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

function sendText(res: ServerResponse, status: number, contentType: string, text: string): void {
  res.writeHead(status, { 'content-type': contentType, 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

interface UpstreamResult {
  ok: boolean;
  status: number;
  retryAfterMs: number;
  answer: NormalizedAnswer | null;
  error: string | null;
}

/** The upstream request body for one candidate, in the candidate's wire. */
function upstreamBody(cand: Candidate, clientApi: 'openai' | 'anthropic', body: Json): Json {
  const model = cand.model.id;
  const wire = cand.probe.provider.wire;
  if (wire === 'openai') {
    if (clientApi === 'openai') {
      const out: Json = { ...body, model };
      if (body.stream === true) out.stream_options = { include_usage: true };
      return out;
    }
    return anthropicToOpenAI(body, model);
  }
  if (wire === 'anthropic') {
    if (clientApi === 'anthropic') return { ...body, model };
    return openAIToAnthropicRequest(body, model);
  }
  return clientApi === 'openai' ? openAIToGeminiRequest(body, model) : anthropicToGeminiRequest(body, model);
}

/** The upstream URL for one candidate, in the candidate's wire. */
function upstreamUrl(cand: Candidate, stream: boolean): string {
  const base = joinBase(ENV_URL, cand.probe.provider.base);
  const wire = cand.probe.provider.wire;
  if (wire === 'openai') return `${base}/v1/chat/completions`;
  if (wire === 'anthropic') return `${base}/v1/messages`;
  return `${base}/v1beta/models/${encodeURIComponent(cand.model.id)}:${stream ? 'streamGenerateContent?alt=sse' : 'generateContent'}`;
}

/** The upstream headers for one candidate, in the provider's own auth scheme. */
function upstreamHeaders(cand: Candidate, stream: boolean): Record<string, string> {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(cand.probe.provider.auth, cand.probe.provider.key),
  };
  if (cand.probe.provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  if (stream) headers.accept = 'text/event-stream';
  return headers;
}

/** One upstream call, read whole and parsed into a normalized answer. */
async function callUpstream(cand: Candidate, clientApi: 'openai' | 'anthropic', body: Json, stream: boolean): Promise<UpstreamResult> {
  const fail = (status: number, retryAfterMs: number, error: string): UpstreamResult => ({ ok: false, status, retryAfterMs, answer: null, error });
  let res: Response;
  try {
    res = await fetch(upstreamUrl(cand, stream), {
      method: 'POST',
      headers: upstreamHeaders(cand, stream),
      body: JSON.stringify(upstreamBody(cand, clientApi, body)),
      signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
    });
  } catch (err) {
    return fail(0, 0, (err as Error).message);
  }
  const retryAfterMs = (() => {
    const raw = res.headers.get('retry-after');
    if (raw === null) return 0;
    const seconds = Number(raw);
    return Number.isFinite(seconds) && seconds > 0 ? Math.min(seconds, 5) * 1000 : 0;
  })();
  if (!res.ok) return fail(res.status, retryAfterMs, `the provider answered HTTP ${res.status}`);
  let text: string;
  try {
    text = await res.text();
  } catch (err) {
    return fail(502, 0, (err as Error).message);
  }
  const wire = cand.probe.provider.wire;
  const answer =
    wire === 'openai' ? parseOpenAIAnswer(text, stream) : wire === 'anthropic' ? parseAnthropicAnswer(text, stream) : parseGeminiAnswer(text, stream);
  if (answer === null) return fail(502, 0, `the provider's ${wire} answer did not parse`);
  return { ok: true, status: res.status, retryAfterMs: 0, answer, error: null };
}

/**
 * Record one refusal against its provider: a 402 spends it. Two refusals in a row mean its
 * outage likely began, so it is cooled and only tried when nothing else can serve. Two
 * refusals on separate calls exhaust its failure schedule (at most two), so it is NOT cooled:
 * its next calls can only fail on its outage or its budget, and the risk estimate already
 * prices that in.
 */
function recordRefusal(providerId: string, status: number): void {
  if (status === 402) {
    deadSpent.add(providerId);
    return;
  }
  const callNumber = callsUsed.get(providerId) ?? 0;
  const previous = lastFailedCall.get(providerId) ?? 0;
  if (previous !== 0 && callNumber === previous + 1) consecutiveFail.add(providerId);
  lastFailedCall.set(providerId, callNumber);
  const n = (failures.get(providerId) ?? 0) + 1;
  failures.set(providerId, n);
  if (consecutiveFail.has(providerId)) cool.add(providerId);
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

interface Served {
  answer: NormalizedAnswer;
  model: string;
}

/**
 * Serve one request: try the candidates best first, counting each refusal and failing over to
 * the next. A provider that refused on this request is not retried until every other candidate
 * has been tried; with AGG_RETRY_SAME it gets one more attempt after its retry-after.
 */
async function serve(clientApi: 'openai' | 'anthropic', body: Json, stream: boolean, requires: Requirement): Promise<Served | null> {
  const cands = orderCandidates(requires);
  const refused = new Map<string, number>();
  let attempts = 0;
  // First pass: candidates not held back and not yet refused on this request.
  for (const cand of cands) {
    if (attempts >= MAX_TRIES) break;
    const id = cand.probe.provider.id;
    if (refused.has(id) || heldBack(cand)) continue;
    attempts += 1;
    callsUsed.set(id, (callsUsed.get(id) ?? 0) + 1);
    const res = await callUpstream(cand, clientApi, body, stream);
    if (res.ok && res.answer !== null) {
      log(`served ${clientApi} via ${id}/${cand.model.id} (${cand.rating_milli}) after ${attempts} attempt(s)`);
      return { answer: res.answer, model: cand.model.id };
    }
    refused.set(id, res.retryAfterMs);
    recordRefusal(id, res.status);
    log(`refused ${id}/${cand.model.id}: HTTP ${res.status}`);
  }
  // Second pass: held-back candidates, then refused ones (a refused provider may serve its next call).
  for (const cand of cands) {
    if (attempts >= MAX_TRIES) break;
    const id = cand.probe.provider.id;
    if (deadSpent.has(id)) continue;
    if (!heldBack(cand) && !refused.has(id)) continue;
    attempts += 1;
    callsUsed.set(id, (callsUsed.get(id) ?? 0) + 1);
    if (RETRY_SAME && refused.has(id)) {
      const wait = refused.get(id) ?? 0;
      if (wait > 0) await sleep(wait);
    }
    const res = await callUpstream(cand, clientApi, body, stream);
    if (res.ok && res.answer !== null) {
      log(`served ${clientApi} via ${id}/${cand.model.id} (${cand.rating_milli}) after ${attempts} attempt(s), second pass`);
      return { answer: res.answer, model: cand.model.id };
    }
    refused.set(id, res.retryAfterMs);
    recordRefusal(id, res.status);
    log(`refused ${id}/${cand.model.id}: HTTP ${res.status} (second pass)`);
  }
  return null;
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const stream = body.stream === true;
  const requires = requirementsFor(body);
  const served = await serve('openai', body, stream, requires);
  if (served === null) return sendError(res, 'openai', 503, 'no_free_model', 'no free model on the directory could serve this request');
  if (stream) return sendText(res, 200, 'text/event-stream', openAISSEFromAnswer(served.answer, served.model));
  return sendJson(res, 200, openAIJsonFromAnswer(served.answer, served.model));
}

/** POST /v1/messages: an Anthropic-shaped client, answered by the best free model. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const stream = body.stream === true;
  const requires = requirementsFor(body);
  const served = await serve('anthropic', body, stream, requires);
  if (served === null) return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model on the directory could serve this request');
  if (stream) return sendText(res, 200, 'text/event-stream', anthropicSSEFromAnswer(served.answer, served.model));
  return sendJson(res, 200, anthropicJsonFromAnswer(served.answer, served.model));
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
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
