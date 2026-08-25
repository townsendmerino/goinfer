# Scoping: goinfer × DeepSeek Harness (dsh)

> **Status:** scoping only — nothing here is committed work. Facts below were read from dsh's
> repo + docs on **2026-08-21**; dsh is a **developer preview** that promises
> compatibility-breaking changes, so re-verify everything at pickup. **Box:** mac for the
> interactive run-through (dsh is a Node/pnpm app, `npx @deepseek-ai/dsh web` → 127.0.0.1:3080);
> serve on either box.
>
> What dsh is: DeepSeek's MIT agent harness on their Cordis framework — "everything is a plugin"
> (models, tools, skills, sessions, sandboxes, loops, UI). Providers: DeepSeek native,
> Anthropic/OpenAI/Bedrock/Vertex/Azure catalog entries, and — the seam that matters here —
> **custom OpenAI-compatible endpoints** in `$DSH_HOME/settings.yaml`
> (`api: openai-completions`, `baseURL`, `apiKeyEnv`, `models: [{id}]`). There is a `dsh-plugin`
> GitHub topic for community plugins.

## Why this pairing is interesting at all

dsh + goinfer is a **fully local agent stack with zero cloud**: their web UI at :3080, our
single static binary at :8080, no Python, no API key that goes anywhere. That story writes
itself — but only after the plain integration actually works. Hence the tiers.

## Tier 0 — the recipe (do this, now-ish; no code beyond doc 1)

A documented, *tested* "Use goinfer with DeepSeek Harness" section (serve docs or README's
integrations area) containing:

- `cmd/serve` invocation + the `settings.yaml` provider block (baseURL
  `http://127.0.0.1:8080/v1`, `api: openai-completions`, any apiKeyEnv — loopback needs no key;
  non-loopback hard-fails without `-api-key`, say so).
- Model guidance for agent traffic, stated honestly: a ≥32k-context model (Qwen3.8-27B,
  gpt-oss-20b, DeepSeek-lite class); `-ctx` sized to the transcript; and the two known
  trade-offs — GPU-resident models re-prefill each turn (no prefix-KV reuse on the resident
  path), while the staged/CPU path keeps warm-KV session reuse that agent loops benefit from;
  deep-context decode slows (benchmarks §B7). Set expectations, don't let the harness discover
  them.
- The `role: "developer"` note — **not a blocker** (dsh's
  `compat.supportsDeveloperRole: false` makes it send `system`, so the run works today), but
  the serve alias (`docs/prompts/serve-developer-role.md`) lands **first** anyway: without it,
  a forgotten flag silently demotes the whole agent scaffold to a user message
  (`messagesToTurns`' default arm, verified at `dc8355e`), and the run's findings would read as
  model quality rather than request mangling. Silent-wrong before characterization.

**Gate:** the recipe is written only from a real end-to-end run — dsh web driving goinfer
through at least one multi-turn, tool-using agent task, with every friction point either fixed,
worked around in the recipe, or filed. A recipe nobody ran is a claim nobody can reproduce
(claim-discipline rule 7). Record the run's findings in `docs/measurements/` — the developer-role
demotion is already characterized (see the prompt doc), so the run starts past it; expect others
(tool-call schema shape, streaming event details, context stuffing).

## Tier 1 — a dsh provider plugin (gated, not default)

A small TypeScript `dsh-plugin` package (`goinfer-local` or similar) that makes goinfer a
first-class provider rather than a hand-edited YAML block. Plausible value over Tier 0:

- discovery (find a running serve, read `/v1/models`, populate the model list);
- lifecycle (optionally *spawn* the single-file binary — the model-in-binary story inside the
  harness: the plugin points at one file and the whole backend exists);
- correct compat defaults baked in (developer-role flag pre-set until the alias ships, text-only
  vs vision declaration per model, streaming settings).

**Precondition, both halves required:** (a) Tier 0's run shows friction a plugin would actually
remove — if the YAML block just works, a plugin is maintenance without a customer; and (b) the
plugin API surface looks stable enough to build on — dsh explicitly promises breaking changes,
and coupling goinfer's name to a moving preview for visibility is the kind of trade this repo
declines. Re-read the plugin docs at pickup; if the API churns, park this tier with a date and a
re-check trigger, the C3 deferral pattern.

**Rule regardless:** nothing in goinfer's repo may *depend* on dsh. The plugin lives as its own
small repo/package; goinfer's side is only the OpenAI surface it already ships. Everything must
degrade to Tier 0's plain-endpoint recipe if dsh moves.

## Tier 2 — the fully-local demo (promo pile, not engineering)

A short recorded demo (the vhs pattern): `dsh web` + one goinfer binary, an agent task end to
end, network cable conceptually unplugged. Rides the launch-materials pile (E5 rebuild), gated
only on Tier 0 being smooth. This is where the pairing pays — as a story told once the substance
is boringly true.

## Decision rule and sequencing

Tier 0 after the developer-role task lands (not because it blocks — because its absence makes
the findings silently unreliable) → its measured run
decides Tier 1 (build/park, recorded either way) → Tier 2 whenever Tier 0 is smooth, batched
with other launch materials. Budget: Tier 0 ≈ one session including the run; Tier 1 unbudgeted
until its precondition is met.

## Risks, named

dsh is days old and preview-labeled (churn); its agent traffic shape hits goinfer's measured
weak axes (deep context, resident re-prefill) — the recipe manages expectations rather than
hiding them; and the integration's value depends on dsh's own trajectory, which we don't
control — hence: smallest committed surface, everything degradable to a plain OpenAI endpoint.
