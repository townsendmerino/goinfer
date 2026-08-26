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
- The `role: "developer"` note — **RESOLVED, `4ca19e9` (2026-08-25).** `developer` is now an
  alias for `system` on the OpenAI-compatible routes, so dsh's `compat.supportsDeveloperRole`
  flag can be left at its default and the recipe does not need to mention it. It was never a
  blocker (the flag made it send `system`); it was sequenced first because the old behavior
  silently demoted the whole agent scaffold to a user message, which would have made this run's
  findings read as model quality rather than request mangling. Recipe should state the alias
  exists, so a reader who has the flag set knows they no longer need it.

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

## DECISION 2026-08-26: Tier 1 is PARKED. Both halves were evaluated; both point the same way.

**Half (b), API stability — FAILS, on dsh's own words.** rc.2's shipped READMEs state the stance
three times over:

- `dsh-llm-pi-ai`: *"the **pre-release array shape** (with per-profile `provider` fields) fails load
  with migration directions."* The provider seam a goinfer plugin would sit on **has already broken
  once** — and it is the same break this campaign walked into: the scoping doc above recorded the
  array shape on 2026-08-21 and rc.2 rejects it. Four days.
- `dsh-session`: *"`SESSION_FORMAT_VERSION` stays pinned at `0` — pre-release, no broad
  compatibility implied … older: no upgrade path ships yet."*
- `dsh-storage-json`: *"no migration, pre-release stance."*

The API itself is real and rather good (`ctx.llm`, `ctx.tools.register`, cordis 4.0.1). It is not
*absent*; it is *moving*, with the project saying so plainly. There is also no plugin-authoring
template or SDK package, and the authoring docs link to monorepo-internal paths
(`../../../.agents/notes/…`) that do not ship — so an external author is reading half a map.

**Half (a), friction — WEAKER THAN WHEN THE TIER WAS SCOPED, and this is the stronger argument.**
Tier 0 did find real friction, and the run log lists it. But writing the recipe *dissolved most of
it*: the hanging `npx` install is dsh's own and a plugin cannot fix it; the `llm-pi-ai:` namespace,
the dict-not-list shape, and the mandatory-but-unused `apiKeyEnv` are now five lines of documented
YAML that work. What a plugin would still uniquely buy is narrow:

| a plugin would add | worth it today? |
|---|---|
| discovery (find a running serve, read `/v1/models`) | saves one hand-written model entry |
| lifecycle (spawn the single binary from the harness) | genuinely nice — the model-in-one-file story *inside* dsh |
| compat defaults pre-set | **already unnecessary** — goinfer accepts dsh's three defaults as of v0.15.0 (G12) |

One of the three reasons is gone *because the Tier-0 work removed it*, which is the outcome the
tiering was designed to produce: fix the substance, and the wrapper stops being needed.

**Re-check triggers — concrete, so this is a deferral and not a shrug.** Revisit when ANY holds:

1. dsh publishes a **non-prerelease** version (no `-rc`), or states a compatibility policy for
   `llm-pi-ai`'s provider profile schema;
2. dsh ships a **plugin-authoring package or template** with its own stability statement;
3. someone asks for goinfer-in-dsh **lifecycle spawning** specifically — the one item on the list
   the recipe cannot document away.

**Standing rule, unchanged and now load-bearing:** nothing in goinfer's repo depends on dsh. Tier 0
shipped as documentation plus fixes to goinfer's own OpenAI surface — all of which (G12, G13, G18,
G19, G21) are improvements for *every* harness, not for dsh. That is why parking Tier 1 costs
nothing: none of the value already banked is contingent on it.

**Rule regardless:** nothing in goinfer's repo may *depend* on dsh. The plugin lives as its own
small repo/package; goinfer's side is only the OpenAI surface it already ships. Everything must
degrade to Tier 0's plain-endpoint recipe if dsh moves.

## Tier 2 — the fully-local demo (promo pile, not engineering)

A short recorded demo (the vhs pattern): `dsh web` + one goinfer binary, an agent task end to
end, network cable conceptually unplugged. Rides the launch-materials pile (E5 rebuild), gated
only on Tier 0 being smooth. This is where the pairing pays — as a story told once the substance
is boringly true.

## Decision rule and sequencing

Tier 0 — **now unblocked; the developer-role alias landed at `4ca19e9` (2026-08-25), which
was sequenced first not because it blocked but because its absence made the findings silently
unreliable** → its measured run
decides Tier 1 (build/park, recorded either way) → Tier 2 whenever Tier 0 is smooth, batched
with other launch materials. Budget: Tier 0 ≈ one session including the run; Tier 1 unbudgeted
until its precondition is met.

## Risks, named

dsh is days old and preview-labeled (churn); its agent traffic shape hits goinfer's measured
weak axes (deep context, resident re-prefill) — the recipe manages expectations rather than
hiding them; and the integration's value depends on dsh's own trajectory, which we don't
control — hence: smallest committed surface, everything degradable to a plain OpenAI endpoint.
