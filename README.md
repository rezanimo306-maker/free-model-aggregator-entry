# Free-model aggregator

One server, two client wires, ten free providers. It serves the OpenAI Chat Completions API
(`POST /v1/chat/completions`) and the Anthropic Messages API (`POST /v1/messages`), in JSON and
in server-sent events, routing each request to the best free model that meets its requirements,
and failing over when a provider refuses.

## How it works

- Reads the brief (`$MAKEFUN_PROMPT_FILE`): the provider directory, the sealed quality table
  and the workload manifest. Probes every provider's catalog route in the provider's own wire
  (OpenAI, Anthropic or Gemini) and builds one catalog, which `node main.ts --list` prints and
  the register body carries byte for byte.
- Routing ranks every free model meeting a request's requirements by the quality table's
  `rating_milli`, best first. A 402 marks that provider's budget spent; a provider that has
  refused twice is only retried when nothing else can serve. Each provider's first call in a
  trial always answers, so the heavy requests are served at the top rating.
- Upstream wires: OpenAI (`/v1/chat/completions`), Anthropic (`/v1/messages`) and Gemini
  (`/v1beta/models/<model>:generateContent` / `:streamGenerateContent?alt=sse`). Every
  translation is lossless on the fields the providers echo (system prompt, user text, tool
  names, `max_tokens`, stop sequences, image bytes, JSON mode), so the client's echo checks
  pass whichever wire serves the request.
- Responses are read whole into one normalized answer and rebuilt in the client's wire, so the
  streaming event order, stop reasons and usage counts are always correct. An OpenAI streaming
  call always sets `stream_options.include_usage`, which the client needs for its usage check.

## Tuning knobs (environment, all optional)

- `AGG_SPREAD_MILLI` — rating points subtracted per prior call to a provider (0: pure rating order;
  tuned default 400 in risk mode)
- `AGG_SPREAD_MODE` — 0: the spread is charged per prior call; 1: per unit of the estimated
  chance the provider's next call refuses (budget priors from the kit's generator ranges,
  updated by the failures seen so far; tuned default 1)
- `AGG_SPREAD_W` — 1: the spread is divided by the request's weight (weight-2 requests hold
  the best model more tightly than weight-1 ones; tuned default 1)
- `AGG_FIRST_BONUS` — rating points added to a provider's first call in a trial (tuned default 50).
  A first call always answers (the failure schedules start at the second call), so the bonus
  steers the router toward unused providers and away from the risky second call — trading a few
  rating points for a saved failover call.
- `AGG_SCARCITY` — rating points charged for spending a provider's first call on a request when
  that provider is one of few that can serve a LATER request (tuned default 60). The charge sums,
  over the requests still to come, each request's weight divided by how many providers can serve
  it — a request only two providers can serve weighs far more than one any of the ten can. It
  reserves scarce first calls (json, min_context, tools+vision) for the requests that need them
  instead of spending them on abundant ones. Deterministic: it reads only the manifest, the
  probes and the quality table.

## Tuning (local, 10 instances)

The shipped defaults (`AGG_SPREAD_MODE=1`, `AGG_SPREAD_MILLI=400`, `AGG_FIRST_BONUS=50`,
`AGG_SCARCITY=60`, `AGG_SPREAD_W=1`) were chosen by sweeping routing variants against the 3 shipped
practice trials plus 7 extra instances drawn from the kit's own generator with beacon-derived seeds
(see `../extra-trials/`). Score 99 is the practical ceiling: quality 100 needs zero rating loss, but
the best model (OpenRouter `z-ai/glm-5.3-flash:free`, 1475) has a budget of 10–12 calls and 17
requests want it, so some requests must fall back to 1470 or lower. Round 2 changed the routing
from a flat per-call spread to a risk-aware one: the old flat spread (mode 0, 20 points a call)
spread load evenly but walked into the providers' scheduled second-call failures — on trial-0
three providers' second calls (OpenRouter, Cloudflare, Z.ai) all failed on one request, costing
4 calls. The risk-aware spread charges per unit of the estimated chance the next call refuses,
scaled up so a risky second call is genuinely expensive, and the first-call bonus (a first call
always answers) steers the router to unused providers first. Result on the 10 instances: mean
23.2 provider calls (round 1: 27.4), max 29 (round 1: 33), trial-0 22 calls (round 1: 24) at
quality 99 — inside the dev top-20 cutoff of 22 calls with quality intact. Quality is the binding
constraint: a single trial at 98 still scores 99, but a trial at 97 scores 98, so the defaults
keep quality 99 on 7 of 10 instances (round 1: also 7 of 10) rather than chasing the last call
with a larger bonus or spread, both of which dropped quality to 98 on most instances.

Round 3 pushed the same levers harder. Two changes: the spread was raised 200 → 400 (a risky
second call must now be better by 400 rating points to be taken over an unused provider), the
`failRisk` estimator was fixed to charge a first call 0 (it always answers — the old code charged
it the scheduled-failure density), and a scarcity charge was added that reserves scarce first
calls for the later requests that need them. Sweeps over spread 200–2000, first bonus 30–150 and
scarcity 0–150 all plateau at 21 calls on trial-0, never 20: the last failure is always the
second call of the top-rated provider (OpenRouter on trial-0) taken for a JSON request once every
JSON-capable provider's first call is spent — and no legitimate rule can avoid it, because the
failure schedule (which call numbers refuse) is drawn per trial and never published: the brief,
the `/instance` endpoint and the catalog all omit budgets and failure schedules, and the trial
seed is a one-way hash of the instance seed, so the entry cannot recompute the draw. A 20-call
trial-0 needs the router to know that specific call refuses, which is trial-0 knowledge, not
routing. Result on the 10 instances: mean 22.7 provider calls (round 2: 23.2), max 25 (round 2:
29), trial-0 21 calls (round 2: 22) at quality 98 — one call off the 20-call floor, score 99 on
all 10 instances. The trade is real but small: the stronger spread wins 5 of 10 trials, ties 2,
loses 3, and quality drops 99 → 98 on 7 instances (exact quality 98.4–98.9, still safely above
the 97.5 line where score would fall to 98), so the defaults trade a little quality headroom
for fewer calls, which is what the rank key rewards (score first, then calls).
- `AGG_CAP` — soft cap on calls per provider; over-cap providers are tried last (0: no cap)
- `AGG_RETRY_SAME` — 1: retry a refused provider once per request after its retry-after (0: never)
- `AGG_MAX_TRIES` — most upstream attempts per request (default 20)

## Files

- `main.ts` — the server: brief, catalog, routes, routing and failover, register and poll.
- `lib/providers.ts` — the brief's types and parser, the auth headers, the catalog probe.
- `lib/wire.ts` — the client wires, the three upstream wires, the response builders.
- `LICENSE` — MIT.
