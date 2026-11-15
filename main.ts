// free-model-aggregator-02: one server, two client wires, a real free model
// behind every answer.
//
// The trial runs this server as `node /bot/main.ts` from the repository's root,
// with Node 22's standard library alone (TypeScript runs directly, and a
// relative import needs its .ts extension). The env is at $MAKEFUN_ENV_URL;
// model calls go through the grader's gateway at $OPENAI_BASE_URL, an
// OpenAI-compatible endpoint that reaches OpenRouter on the key pushed to the
// competition. The brief at $MAKEFUN_PROMPT_FILE carries two documents:
//   - the free model list: each free OpenRouter model's gateway id
//     (`openrouter/<vendor>/<model>:free`), its published rating in
//     milli-points, what it can do (tools, vision, JSON mode, context length)
//     and its free flag;
//   - the six prompts: each one's id, its api (openai or anthropic), whether it
//     streams, its weight (1 or 2) and the requirements a model must meet to
//     serve it. Each prompt carries a unique token the answer must contain.
// The server:
//   1. listens on 0.0.0.0 at a free port (never 8080: the env itself is there);
//   2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>,
//      "catalog": <catalog>};
//   3. serves GET /v1/models (in the shape of the auth header it got: x-api-key
//      means Anthropic, Authorization means OpenAI), POST /v1/chat/completions
//      and POST /v1/messages, each with a JSON body or with server-sent events;
//   4. answers each prompt by dialling ONE free model through the gateway.
//      OPENAI_API_KEY in the sandbox is a dummy the gateway swaps for the
//      OpenRouter key pushed to the competition; the real key never enters the
//      sandbox;
//   5. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env
//      says {"done": true}. A process still running when the wall ends scores
//      zero, so never wait for anything else.
// `node main.ts --list` prints the same catalog JSON and exits 0 before any
// server starts; the register body's catalog is that same document, byte for
// byte.
//
// ROUTING (quality is 60% of the score)
//   The brief's ratings are the quality column's currency, so every prompt goes
//   to the highest-rated free model that meets its requirements, read off the
//   brief at runtime — never a hardcoded id, so a later brief's table still
//   routes right. On this brief that is inkling-2-small (1475) for an
//   unconstrained or tools prompt and nemotron-3.5-super-vl (1303) for a
//   JSON-mode prompt, the best JSON-capable model.
//
// FAILOVER (10% of the score)
//   A gateway call answered with a 4xx or a 5xx is retried on the next-best free
//   model that meets the prompt, up to 8 attempts a prompt, and every refusal is
//   reported on the answer in `x-makefun-upstream-failures`, a JSON array of at
//   most 8 rows {"status", "code", "model"} in the order the calls refused:
//   the header is the only way the failover column can see a refusal the server
//   recovered from. A model that refused is cooled for the rest of the trial —
//   a rate-limited model takes every later prompt down with it — and is only
//   tried again once every other candidate has refused too. Cooldowns are
//   call-count based, never wall-clock based, so a live pass and its replay
//   make the same choices.
//
// THE MODEL RULE
//   Whatever this server reports in a response's `model` field must be the model
//   it actually dialled. It sends the brief's model id in the gateway request
//   and puts that same id back on every answer and every streamed chunk (the
//   provider's own answer spells it without the gateway's `openrouter/`
//   prefix), so its claim is the call it made, never another.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  capabilitiesOf,
  freeModels,
  isRecord,
  meetsRequirements,
  readBrief,
  type CatalogReport,
  type Json,
  type ModelRow,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIJsonWithModel,
  openAIModelList,
  openAIStreamWithModel,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** The gateway refuses a call with no output cap, or one over this. */
const MAX_OUTPUT_TOKENS = 8192;
/** The output cap a gateway call gets when the client named none. */
const DEFAULT_OUTPUT_TOKENS = 1024;
/** The most dial attempts one prompt costs, so the failures trail stays within 8 rows. */
const MAX_ATTEMPTS = 8;
/**
 * How long one gateway dial may take before it counts as a refusal: a hung
 * upstream used to hang the whole trial (the poll loop keeps running, the
 * prompt never answers, the wall ends, the score is zero). A timed-out dial
 * is recorded in the trail as `upstream_timeout`, cools that model, and the
 * failover moves on to the next candidate. The timeout is on the dial only —
 * it never picks a model and never changes a replay, where the cassette
 * answers at once.
 */
const GATEWAY_DIAL_TIMEOUT_MS = 30_000;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key for the entry's gateway lane; the gateway swaps it for the entrant's own key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief = readBrief(PROMPT_FILE);
const catalog: CatalogReport = buildCatalog(brief);
/** Which prompt row the request now arriving is: the brief's prompts are in the client's arrival order. */
let manifestIndex = 0;
/** Models that refused a gateway call this trial: cooled until every other candidate has refused too. */
const cooled = new Set<string>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/**
 * The gateway's Chat Completions route. OPENAI_BASE_URL is the OpenAI-compatible
 * root (`…/v1` on the trial network); a base that does not already end in `/v1`
 * gets one, so both spellings reach the route.
 */
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

/** The free models the brief lists, as GET /v1/models names them. */
function servedModels(): ServedModel[] {
  return freeModels(brief).map((model) => ({ id: model.id, provider: model.provider, display_name: model.display_name }));
}

/**
 * The free candidates that meet `requires`, best first: rating_milli
 * descending, the brief's order breaking ties (the sort is stable). A cooled
 * model sorts after every uncooled one, so a refusal never costs a prompt
 * its best healthy model.
 */
function rankedCandidates(requires: Requirement): ModelRow[] {
  const candidates = freeModels(brief).filter((model) => meetsRequirements(requires, capabilitiesOf(model)));
  candidates.sort((a, b) => b.rating_milli - a.rating_milli);
  const fresh = candidates.filter((m) => !cooled.has(m.id));
  const stale = candidates.filter((m) => cooled.has(m.id));
  return [...fresh, ...stale];
}

/** The requirements of the prompt now arriving: the brief's row by arrival order, plus the body's own. */
function requirementsFor(body: Json): Requirement {
  const entry = brief.prompts[manifestIndex];
  manifestIndex += 1;
  const fromBody = requirementsFromBody(body);
  if (entry === undefined) return fromBody;
  // The union: a model must meet the manifest's row and whatever the body itself asks for.
  return {
    ...(entry.requires.tools === true || fromBody.tools === true ? { tools: true } : {}),
    ...(entry.requires.vision === true || fromBody.vision === true ? { vision: true } : {}),
    ...(entry.requires.json === true || fromBody.json === true ? { json: true } : {}),
    ...(entry.requires.min_context !== undefined ? { min_context: entry.requires.min_context } : {}),
  };
}

/** Requirements read off a request body, when the brief has no row for it. */
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

function sendJson(res: ServerResponse, status: number, body: unknown, failures?: FailureTrail): void {
  const text = JSON.stringify(body);
  const headers: Record<string, string> = { 'content-type': 'application/json', 'content-length': String(Buffer.byteLength(text)) };
  if (failures !== undefined && failures.length > 0) headers['x-makefun-upstream-failures'] = JSON.stringify(failures);
  res.writeHead(status, headers);
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

/** One row of the `x-makefun-upstream-failures` trail. */
interface FailureRow {
  status: number;
  code: string;
  model: string | null;
}
type FailureTrail = FailureRow[];

/** The error code of a gateway refusal's body, or `http_<status>` when it names none. */
function failureCode(status: number, text: string): string {
  try {
    const parsed: unknown = JSON.parse(text);
    if (isRecord(parsed) && isRecord(parsed.error)) {
      if (typeof parsed.error.code === 'string' && parsed.error.code !== '') return parsed.error.code;
      if (typeof parsed.error.type === 'string' && parsed.error.type !== '') return parsed.error.type;
    }
  } catch {
    // Not JSON: fall through.
  }
  return `http_${status}`;
}

/** The gateway body for one dial: the output cap the gateway requires, added when the client named none. */
function gatewayBody(body: Json, model: string): Json {
  const out: Json = { ...body, model };
  for (const key of ['max_tokens', 'max_completion_tokens'] as const) {
    const value = out[key];
    if (typeof value === 'string' && value.trim() !== '' && Number.isFinite(Number(value))) out[key] = Math.floor(Number(value));
  }
  const hasCap =
    (typeof out.max_tokens === 'number' && Number.isFinite(out.max_tokens)) ||
    (typeof out.max_completion_tokens === 'number' && Number.isFinite(out.max_completion_tokens));
  if (!hasCap) out.max_tokens = DEFAULT_OUTPUT_TOKENS;
  if (typeof out.max_tokens === 'number') out.max_tokens = Math.min(Math.max(1, Math.floor(out.max_tokens)), MAX_OUTPUT_TOKENS);
  if (typeof out.max_completion_tokens === 'number') {
    out.max_completion_tokens = Math.min(Math.max(1, Math.floor(out.max_completion_tokens)), MAX_OUTPUT_TOKENS);
  }
  return out;
}

/** POST one request to the gateway's Chat Completions route, naming the model this entry dialled. */
async function dial(model: string, body: Json): Promise<Response> {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    authorization: `Bearer ${GATEWAY_KEY}`,
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  return fetch(chatCompletionsUrl(), {
    method: 'POST',
    headers,
    body: JSON.stringify(gatewayBody(body, model)),
    signal: AbortSignal.timeout(GATEWAY_DIAL_TIMEOUT_MS),
  });
}

/**
 * Dial the candidates in rating order until one answers, recording every
 * refusal in the trail and cooling the model that refused it. Returns the
 * successful response and the model that served it, or a null response with
 * the trail holding the whole story when every candidate refused.
 */
async function dialWithFailover(
  candidates: ModelRow[],
  body: Json,
): Promise<{ response: Response | null; model: string | null; trail: FailureTrail }> {
  const trail: FailureTrail = [];
  const attempts = candidates.slice(0, MAX_ATTEMPTS);
  for (const candidate of attempts) {
    let upstream: Response;
    try {
      upstream = await dial(candidate.id, body);
    } catch (err) {
      const timedOut = (err as Error).name === 'TimeoutError';
      trail.push({ status: timedOut ? 504 : 502, code: timedOut ? 'upstream_timeout' : 'upstream_unreachable', model: candidate.id });
      cooled.add(candidate.id);
      log(`dial ${candidate.id}: ${timedOut ? `no answer in ${GATEWAY_DIAL_TIMEOUT_MS} ms` : `transport error: ${(err as Error).message}`}`);
      continue;
    }
    if (!upstream.ok) {
      const text = await upstream.text();
      trail.push({ status: upstream.status, code: failureCode(upstream.status, text), model: candidate.id });
      cooled.add(candidate.id);
      log(`dial ${candidate.id}: HTTP ${upstream.status} ${failureCode(upstream.status, text)}`);
      continue;
    }
    if (upstream.body === null) {
      trail.push({ status: 502, code: 'empty_upstream_body', model: candidate.id });
      cooled.add(candidate.id);
      continue;
    }
    return { response: upstream, model: candidate.id, trail };
  }
  return { response: null, model: null, trail };
}

/** A prompt every candidate refused, as a 502 in the client's wire. */
function refused(res: ServerResponse, wire: 'openai' | 'anthropic', trail: FailureTrail): void {
  const last = trail[trail.length - 1];
  const detail = last !== undefined ? `the gateway answered HTTP ${last.status} (${last.code}) on ${last.model ?? 'an unknown model'}` : 'no free model answered';
  if (wire === 'anthropic') sendError(res, wire, 502, 'api_error', detail);
  else sendError(res, wire, 502, 'upstream_error', detail);
}

/**
 * Serve one prompt on either wire: pick the candidates, dial with failover,
 * and answer in the wire the client used. The two handlers used to be two
 * copies of this same flow — one place where a prompt can go unanswered is
 * easier to keep right than two. The wires differ only at the two ends:
 * the Anthropic request is rewritten to Chat Completions before the dial
 * and the answer is rewritten back after it; the OpenAI request passes
 * through, and each wire names its own error when no model meets the prompt.
 */
async function serve(req: IncomingMessage, res: ServerResponse, wire: 'openai' | 'anthropic'): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, wire, 400, 'invalid_request_error', 'the request body is not one JSON object');
  const candidates = rankedCandidates(requirementsFor(body));
  if (candidates.length === 0) {
    return sendError(res, wire, 503, wire === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model in the brief meets this request');
  }
  const upstreamBody = wire === 'anthropic' ? anthropicToOpenAI(body, 'auto') : { ...body, model: 'auto' };
  const served = await dialWithFailover(candidates, upstreamBody);
  if (served.response === null || served.model === null) return refused(res, wire, served.trail);
  const { response: upstream, model, trail } = served;
  if (body.stream === true) {
    const headers: Record<string, string> = { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' };
    if (trail.length > 0) headers['x-makefun-upstream-failures'] = JSON.stringify(trail);
    res.writeHead(200, headers);
    try {
      const relay =
        wire === 'anthropic'
          ? anthropicEventsFromOpenAI(upstream.body as ReadableStream<Uint8Array>, model)
          : openAIStreamWithModel(upstream.body as ReadableStream<Uint8Array>, model);
      for await (const frame of relay) res.write(frame);
    } finally {
      res.end();
    }
    return;
  }
  const answer: unknown = await upstream.json();
  const shaped =
    wire === 'anthropic'
      ? openAIToAnthropic(isRecord(answer) ? answer : {}, model)
      : openAIJsonWithModel(isRecord(answer) ? answer : {}, model);
  sendJson(res, 200, shaped, trail);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await serve(req, res, 'openai');
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await serve(req, res, 'anthropic');
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

/** POST /register with the port and the one catalog; the env answers with the trial's prompt count. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
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
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
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
  log(`listening on 0.0.0.0:${port} with ${servedModels().length} free models`);
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
