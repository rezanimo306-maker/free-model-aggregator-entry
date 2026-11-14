// The client wires and the three upstream wires.
//
// The trial's client calls the entry as an OpenAI service (POST /v1/chat/completions) or an
// Anthropic one (POST /v1/messages), with a JSON body or with server-sent events. The entry
// forwards each request to a provider on the provider's own wire — openai, anthropic or gemini —
// and answers back in the client's wire.
//
// The providers echo what they parsed from the upstream request (system prompt, user text, tool
// names, max_tokens, stop sequences, image bytes, JSON mode) and the client compares that echo
// with what it sent, so every translation below is lossless on exactly those fields. A field
// rewritten or dropped loses the request.
//
// The providers answer with one fixed text (the echo) and usage counts, so every upstream
// response is read fully into one normalized answer — { text, input tokens, output tokens } —
// and the client's JSON or SSE answer is built from that. Buffering costs nothing here: the
// providers are local and fast, and the client reads the whole body before it checks anything.
//
// Everything here is a pure function of its arguments: no clock and no randomness, so the same
// body and the same provider answer give the same bytes on a replay. The two model listings use
// fixed timestamps for the same reason: a `created` that moved between runs would be a
// nondeterministic document.

import { isRecord, type Json } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** One provider answer, read whole: the text, the usage, and the stop family it named. */
export interface NormalizedAnswer {
  text: string;
  input: number;
  output: number;
  /** The provider's stop reason verbatim: "stop" | "end_turn" | "STOP" (all one family). */
  stop: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
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

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/** Anthropic's stop_reason as OpenAI's finish_reason. */
function finishReasonOf(stop: unknown): string {
  if (stop === 'max_tokens') return 'length';
  if (stop === 'tool_use') return 'tool_calls';
  if (stop === 'refusal') return 'content_filter';
  return 'stop';
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that names `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks become
 * tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
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
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
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

/** A Chat Completions answer as an Anthropic Messages object. */
export function openAIToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.prompt_tokens), output_tokens: numberOrZero(usage.completion_tokens) },
  };
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** Split text the way the providers do: 13 characters a delta. Any split passes the order check. */
function splitText(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += 13) pieces.push(text.slice(at, at + 13));
  return pieces.length === 0 ? [''] : pieces;
}

// ---------------------------------------------------------------------------
// OpenAI client -> Anthropic upstream
// ---------------------------------------------------------------------------

/** One OpenAI content part as an Anthropic content block: text or image; null for the rest. */
function blockOf(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { type: 'text', text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = /^data:([^,]*);base64,(.*)$/s.exec(url);
    if (data !== null) {
      return { type: 'image', source: { type: 'base64', media_type: data[1] === '' ? 'image/png' : data[1], data: data[2] } };
    }
    return { type: 'image', source: { type: 'url', url } };
  }
  return null;
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request that names `model`.
 *
 * The echo fields are preserved exactly: system messages join with "\n" into `system`; each user
 * message's text blocks and images become Anthropic blocks in the same order; tool messages become
 * tool_result blocks appended to the most recent user message (a trailing tool_result-only user
 * message would read as empty user text on the provider's side, while the client counts the last
 * real user message, so they must never stand alone at the end); tools, tool_choice, max_tokens
 * and stop sequences are rewritten, and streaming passes through.
 */
export function openAIToAnthropicRequest(body: Json, model: string): Json {
  const systemParts: string[] = [];
  const messages: Json[] = [];
  /** Index into `messages` of the last user message, for trailing tool results to join. */
  let lastUser = -1;
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'assistant') {
      const content: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') content.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        content.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsOf(call.function.arguments),
        });
      }
      if (content.length > 0) messages.push({ role: 'assistant', content });
      continue;
    }
    if (raw.role === 'tool') {
      const result: Json = {
        type: 'tool_result',
        tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
        content: typeof raw.content === 'string' ? raw.content : toolResultText(raw.content),
      };
      if (lastUser !== -1) {
        const target = messages[lastUser];
        if (isRecord(target) && Array.isArray(target.content)) (target.content as Json[]).push(result);
      } else {
        messages.push({ role: 'user', content: [result] });
        lastUser = messages.length - 1;
      }
      continue;
    }
    if (raw.role === 'user') {
      const content: Json[] = [];
      if (typeof raw.content === 'string') {
        if (raw.content !== '') content.push({ type: 'text', text: raw.content });
      } else if (Array.isArray(raw.content)) {
        for (const part of raw.content) {
          if (!isRecord(part)) continue;
          const block = blockOf(part);
          if (block !== null) content.push(block);
        }
      }
      if (content.length > 0) {
        messages.push({ role: 'user', content });
        lastUser = messages.length - 1;
      }
    }
  }

  const out: Json = { model, messages };
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : undefined;
  if (maxTokens !== undefined) out.max_tokens = maxTokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = typeof body.stop === 'string' ? [body.stop] : Array.isArray(body.stop) ? body.stop.filter((s): s is string => typeof s === 'string') : [];
  if (stops.length > 0) out.stop_sequences = stops;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [{ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } }];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required' || choice === 'any') out.tool_choice = { type: 'any' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

// ---------------------------------------------------------------------------
// Gemini upstream requests
// ---------------------------------------------------------------------------

/** A media type and base64 payload out of a data URL, or null when it is not one. */
function dataUrlParts(url: string): { mime: string; data: string } | null {
  const m = /^data:([^,]*);base64,(.*)$/s.exec(url);
  if (m === null) return null;
  return { mime: m[1] === '' ? 'image/png' : m[1], data: m[2] };
}

/** One OpenAI message's content as Gemini parts. */
function geminiPartsOfOpenAI(content: unknown): Json[] {
  const parts: Json[] = [];
  if (typeof content === 'string') {
    if (content !== '') parts.push({ text: content });
    return parts;
  }
  if (!Array.isArray(content)) return parts;
  for (const part of content) {
    if (!isRecord(part)) continue;
    if (part.type === 'text' && typeof part.text === 'string' && part.text !== '') parts.push({ text: part.text });
    else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
      const d = dataUrlParts(part.image_url.url);
      if (d !== null) parts.push({ inlineData: { mimeType: d.mime, data: d.data } });
      else parts.push({ fileData: { fileUri: part.image_url.url } });
    }
  }
  return parts;
}

/** One Anthropic message's content as Gemini parts. */
function geminiPartsOfAnthropic(content: unknown): Json[] {
  const parts: Json[] = [];
  if (typeof content === 'string') {
    if (content !== '') parts.push({ text: content });
    return parts;
  }
  if (!Array.isArray(content)) return parts;
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string' && block.text !== '') parts.push({ text: block.text });
    else if (block.type === 'image' && isRecord(block.source)) {
      if (block.source.type === 'base64' && typeof block.source.data === 'string') {
        parts.push({ inlineData: { mimeType: typeof block.source.media_type === 'string' ? block.source.media_type : 'image/png', data: block.source.data } });
      } else if (block.source.type === 'url' && typeof block.source.url === 'string') {
        parts.push({ fileData: { fileUri: block.source.url } });
      }
    }
  }
  return parts;
}

/**
 * The shared Gemini body builder: contents in message order, the system instruction, the tool
 * declarations and the generation config.
 *
 * Gemini's parser takes the LAST non-model content's text as the user text, so a trailing tool
 * result must never stand alone as the last content: tool parts are folded into the most recent
 * user content (they carry no text, so the user text it holds does not change).
 */
function geminiBody(
  systemText: string,
  contents: Json[],
  tools: Json[],
  generationConfig: Json,
): Json {
  const out: Json = { contents };
  if (systemText !== '') out.systemInstruction = { parts: [{ text: systemText }] };
  if (tools.length > 0) out.tools = [{ functionDeclarations: tools }];
  if (Object.keys(generationConfig).length > 0) out.generationConfig = generationConfig;
  return out;
}

/** The generation config for a request: output cap, stops, JSON mode, temperature, top_p. */
function geminiConfig(opts: { maxTokens?: number; stops?: string[]; json?: boolean; temperature?: number; topP?: number }): Json {
  const config: Json = {};
  if (opts.maxTokens !== undefined) config.maxOutputTokens = opts.maxTokens;
  if (opts.stops !== undefined && opts.stops.length > 0) config.stopSequences = opts.stops;
  if (opts.json === true) config.responseMimeType = 'application/json';
  if (opts.temperature !== undefined) config.temperature = opts.temperature;
  if (opts.topP !== undefined) config.topP = opts.topP;
  return config;
}

/** An OpenAI Chat Completions request as a Gemini generateContent body. */
export function openAIToGeminiRequest(body: Json, _model: string): Json {
  const systemParts: string[] = [];
  const contents: Json[] = [];
  let lastUser = -1;
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
      for (const call of Array.isArray(raw.tool_calls) ? raw.tool_calls : []) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        parts.push({ functionCall: { name: typeof call.function.name === 'string' ? call.function.name : '', args: argumentsOf(call.function.arguments) } });
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    if (raw.role === 'tool') {
      const part: Json = { functionResponse: { name: '', response: { content: typeof raw.content === 'string' ? raw.content : toolResultText(raw.content) } } };
      if (lastUser !== -1) {
        const target = contents[lastUser];
        if (isRecord(target) && Array.isArray(target.parts)) (target.parts as Json[]).push(part);
      } else {
        contents.push({ role: 'user', parts: [part] });
        lastUser = contents.length - 1;
      }
      continue;
    }
    if (raw.role === 'user') {
      const parts = geminiPartsOfOpenAI(raw.content);
      if (parts.length > 0) {
        contents.push({ role: 'user', parts });
        lastUser = contents.length - 1;
      }
    }
  }
  const tools: Json[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') continue;
      tools.push({ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } });
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : undefined;
  const stops = typeof body.stop === 'string' ? [body.stop] : Array.isArray(body.stop) ? body.stop.filter((s): s is string => typeof s === 'string') : [];
  return geminiBody(systemParts.join('\n'), contents, tools, geminiConfig({
    maxTokens,
    stops,
    json: format === 'json_object' || format === 'json_schema',
    temperature: typeof body.temperature === 'number' ? body.temperature : undefined,
    topP: typeof body.top_p === 'number' ? body.top_p : undefined,
  }));
}

/** An Anthropic Messages request as a Gemini generateContent body. */
export function anthropicToGeminiRequest(body: Json, _model: string): Json {
  const contents: Json[] = [];
  let lastUser = -1;
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'assistant') {
      const parts: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') parts.push({ text });
      if (Array.isArray(raw.content)) {
        for (const block of raw.content) {
          if (!isRecord(block) || block.type !== 'tool_use') continue;
          parts.push({ functionCall: { name: typeof block.name === 'string' ? block.name : '', args: isRecord(block.input) ? block.input : {} } });
        }
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    if (raw.role === 'user') {
      const parts: Json[] = [];
      if (Array.isArray(raw.content)) {
        for (const block of raw.content) {
          if (!isRecord(block)) continue;
          if (block.type === 'tool_result') {
            parts.push({ functionResponse: { name: '', response: { content: toolResultText(block.content) } } });
            continue;
          }
          parts.push(...geminiPartsOfAnthropic([block]));
        }
      } else if (typeof raw.content === 'string' && raw.content !== '') {
        parts.push({ text: raw.content });
      }
      if (parts.length > 0) {
        // A user message of tool results only must not stand alone at the end: fold its parts into
        // the most recent user content when it holds no text.
        const hasText = parts.some((p) => typeof p.text === 'string');
        if (!hasText && lastUser !== -1) {
          const target = contents[lastUser];
          if (isRecord(target) && Array.isArray(target.parts)) (target.parts as Json[]).push(...parts);
        } else {
          contents.push({ role: 'user', parts });
          lastUser = contents.length - 1;
        }
      }
    }
  }
  const tools: Json[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || typeof raw.name !== 'string') continue;
      tools.push({ name: raw.name, description: typeof raw.description === 'string' ? raw.description : '', parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} } });
    }
  }
  const stops = Array.isArray(body.stop_sequences) ? body.stop_sequences.filter((s): s is string => typeof s === 'string') : [];
  return geminiBody(textOf(body.system), contents, tools, geminiConfig({
    maxTokens: typeof body.max_tokens === 'number' ? body.max_tokens : undefined,
    stops,
    json: false,
    temperature: typeof body.temperature === 'number' ? body.temperature : undefined,
    topP: typeof body.top_p === 'number' ? body.top_p : undefined,
  }));
}

// ---------------------------------------------------------------------------
// Upstream response parsers: one normalized answer per wire.
// ---------------------------------------------------------------------------

/** The text of a Gemini candidate's parts, joined as the provider wrote them. */
function geminiCandidateText(candidate: Json): string {
  const content = isRecord(candidate.content) ? candidate.content : null;
  if (content === null || !Array.isArray(content.parts)) return '';
  return content.parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
}

/** Split a raw SSE body into its frames' data payloads. */
function ssePayloads(text: string): string[] {
  const out: string[] = [];
  for (const frame of text.replace(/\r\n/g, '\n').split('\n\n')) {
    if (frame.trim() === '') continue;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('data:')) data.push(line.slice(5).replace(/^ /, ''));
    }
    if (data.length > 0) out.push(data.join('\n'));
  }
  return out;
}

/** An OpenAI-wire answer (JSON or SSE) as one normalized answer, or null when unusable. */
export function parseOpenAIAnswer(bodyText: string, stream: boolean): NormalizedAnswer | null {
  if (!stream) {
    let value: unknown;
    try {
      value = JSON.parse(bodyText);
    } catch {
      return null;
    }
    if (!isRecord(value)) return null;
    const choices = Array.isArray(value.choices) ? value.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null || !isRecord(choice.message)) return null;
    const message = choice.message;
    const text = typeof message.content === 'string' ? message.content : textOf(message.content);
    const usage = isRecord(value.usage) ? value.usage : {};
    return { text, input: numberOrZero(usage.prompt_tokens), output: numberOrZero(usage.completion_tokens), stop: typeof choice.finish_reason === 'string' ? choice.finish_reason : 'stop' };
  }
  let text = '';
  let input = 0;
  let output = 0;
  let stop = 'stop';
  for (const payload of ssePayloads(bodyText)) {
    if (payload.trim() === '[DONE]') continue;
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    if (isRecord(chunk.usage)) {
      input = numberOrZero(chunk.usage.prompt_tokens);
      output = numberOrZero(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    if (isRecord(choice.delta) && typeof choice.delta.content === 'string') text += choice.delta.content;
    if (typeof choice.finish_reason === 'string') stop = choice.finish_reason;
  }
  return { text, input, output, stop };
}

/** An Anthropic-wire answer (JSON or SSE) as one normalized answer, or null when unusable. */
export function parseAnthropicAnswer(bodyText: string, stream: boolean): NormalizedAnswer | null {
  if (!stream) {
    let value: unknown;
    try {
      value = JSON.parse(bodyText);
    } catch {
      return null;
    }
    if (!isRecord(value) || !Array.isArray(value.content)) return null;
    const text = textOf(value.content);
    const usage = isRecord(value.usage) ? value.usage : {};
    return { text, input: numberOrZero(usage.input_tokens), output: numberOrZero(usage.output_tokens), stop: typeof value.stop_reason === 'string' ? value.stop_reason : 'end_turn' };
  }
  let text = '';
  let input = 0;
  let output = 0;
  let stop = 'end_turn';
  for (const payload of ssePayloads(bodyText)) {
    let event: unknown;
    try {
      event = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(event)) continue;
    if (event.type === 'message_start' && isRecord(event.message) && isRecord(event.message.usage)) {
      input = numberOrZero(event.message.usage.input_tokens);
    } else if (event.type === 'content_block_delta' && isRecord(event.delta) && event.delta.type === 'text_delta' && typeof event.delta.text === 'string') {
      text += event.delta.text;
    } else if (event.type === 'message_delta') {
      if (isRecord(event.delta) && typeof event.delta.stop_reason === 'string') stop = event.delta.stop_reason;
      if (isRecord(event.usage)) output = numberOrZero(event.usage.output_tokens);
    }
  }
  return { text, input, output, stop };
}

/** A Gemini-wire answer (JSON or SSE) as one normalized answer, or null when unusable. */
export function parseGeminiAnswer(bodyText: string, stream: boolean): NormalizedAnswer | null {
  if (!stream) {
    let value: unknown;
    try {
      value = JSON.parse(bodyText);
    } catch {
      return null;
    }
    if (!isRecord(value) || !Array.isArray(value.candidates)) return null;
    const candidate = isRecord(value.candidates[0]) ? value.candidates[0] : null;
    if (candidate === null) return null;
    const meta = isRecord(value.usageMetadata) ? value.usageMetadata : {};
    return {
      text: geminiCandidateText(candidate),
      input: numberOrZero(meta.promptTokenCount),
      output: numberOrZero(meta.candidatesTokenCount),
      stop: typeof candidate.finishReason === 'string' ? candidate.finishReason : 'STOP',
    };
  }
  let text = '';
  let input = 0;
  let output = 0;
  let stop = 'STOP';
  for (const payload of ssePayloads(bodyText)) {
    let frame: unknown;
    try {
      frame = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(frame) || !Array.isArray(frame.candidates)) continue;
    const candidate = isRecord(frame.candidates[0]) ? frame.candidates[0] : null;
    if (candidate === null) continue;
    text += geminiCandidateText(candidate);
    if (typeof candidate.finishReason === 'string') stop = candidate.finishReason;
    if (isRecord(frame.usageMetadata)) {
      input = numberOrZero(frame.usageMetadata.promptTokenCount);
      output = numberOrZero(frame.usageMetadata.candidatesTokenCount);
    }
  }
  return { text, input, output, stop };
}

// ---------------------------------------------------------------------------
// Client answers, built from one normalized answer.
// ---------------------------------------------------------------------------

/** A Chat Completions JSON answer for `model`, carrying the provider's text and usage. */
export function openAIJsonFromAnswer(answer: NormalizedAnswer, model: string): Json {
  return {
    id: 'chatcmpl-agg',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: answer.text }, finish_reason: 'stop', logprobs: null }],
    usage: { prompt_tokens: answer.input, completion_tokens: answer.output, total_tokens: answer.input + answer.output },
  };
}

/** A Chat Completions SSE stream for `model`, in order: role, deltas, finish, usage, [DONE]. */
export function openAISSEFromAnswer(answer: NormalizedAnswer, model: string): string {
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-agg',
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames: string[] = [sseEvent(null, chunk({ role: 'assistant', content: '' }, null))];
  for (const piece of splitText(answer.text)) frames.push(sseEvent(null, chunk({ content: piece }, null)));
  frames.push(sseEvent(null, chunk({}, 'stop')));
  frames.push(
    sseEvent(null, {
      id: 'chatcmpl-agg',
      object: 'chat.completion.chunk',
      created: FIXED_OPENAI_CREATED,
      model,
      choices: [],
      usage: { prompt_tokens: answer.input, completion_tokens: answer.output, total_tokens: answer.input + answer.output },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** A Messages JSON answer for `model`, carrying the provider's text and usage. */
export function anthropicJsonFromAnswer(answer: NormalizedAnswer, model: string): Json {
  return {
    id: 'msg_agg',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: 'end_turn',
    stop_sequence: null,
    usage: { input_tokens: answer.input, output_tokens: answer.output },
  };
}

/** A Messages SSE stream for `model`, in Anthropic's event order. */
export function anthropicSSEFromAnswer(answer: NormalizedAnswer, model: string): string {
  const frames: string[] = [
    sseEvent('message_start', {
      type: 'message_start',
      message: { id: 'msg_agg', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: answer.input, output_tokens: 0 } },
    }),
    sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
  ];
  for (const piece of splitText(answer.text)) {
    frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }));
  }
  frames.push(
    sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: 'end_turn', stop_sequence: null }, usage: { output_tokens: answer.output } }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return frames.join('');
}
