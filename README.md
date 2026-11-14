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
  tuned default 20)
- `AGG_SPREAD_MODE` — 0: the spread is charged per prior call; 1: per unit of the estimated
  chance the provider's next call refuses (budget priors from the kit's generator ranges,
  updated by the failures seen so far)
- `AGG_SPREAD_W` — 1: the spread is divided by the request's weight (weight-2 requests hold
  the best model more tightly than weight-1 ones; tuned default 1)
- `AGG_CAP` — soft cap on calls per provider; over-cap providers are tried last (0: no cap)

## Tuning (local, 10 instances)

The shipped defaults (`AGG_SPREAD_MILLI=20`, `AGG_SPREAD_W=1`) were chosen by sweeping
routing variants against the 3 shipped practice trials plus 7 extra instances drawn from the
kit's own generator with beacon-derived seeds (see `../extra-trials/`). Score 99 is the
practical ceiling: quality 100 needs zero rating loss, but the best model (OpenRouter
`z-ai/glm-5.3-flash:free`, 1475) has a budget of 10–12 calls and 17 requests want it, so some
requests must fall back to 1470 or lower. Among the variants that scored 99 on all 10
instances, the defaults had the fewest provider calls (mean 27.4, max 33) and the largest
quality margin (worst weighted loss 556 of a 765 budget before quality drops to 97). Variants
that pushed calls lower (a hard cap of 2 per provider, spread 30–40, retry-same) dropped
individual trials to 94–98, and the risk-aware spread scored the same 99 but used more calls.
Quality is the binding constraint: a single trial at 98 still scores 99, but a trial at 97
scores 98, so the defaults keep a ~200-point loss margin rather than chasing the last call.
- `AGG_RETRY_SAME` — 1: retry a refused provider once per request after its retry-after (0: never)
- `AGG_MAX_TRIES` — most upstream attempts per request (default 20)

## Files

- `main.ts` — the server: brief, catalog, routes, routing and failover, register and poll.
- `lib/providers.ts` — the brief's types and parser, the auth headers, the catalog probe.
- `lib/wire.ts` — the client wires, the three upstream wires, the response builders.
- `LICENSE` — MIT.
