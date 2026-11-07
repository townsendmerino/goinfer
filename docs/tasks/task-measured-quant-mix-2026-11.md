# Task: a measured per-tensor int4/int8 mix (`int4mix` chosen by measurement, not by rule) — 2026-11

> **Status: SCOPED 2026-11-07, unstarted, parked for later (Francis).** Nothing here is approved to run yet. Starting
> it is Francis's call, because it spends nights of machine time on nobara.
>
> **The idea.** goinfer already resolves its quant per tensor at load time: `matmulQuant` turns `int4mix` into int8 for
> attention, embed, head and router, and int4 for the FFN bulk (`docs/code-notes/decoder.md`, "quantInt4Mix"). That
> split is a fixed rule, from an early calibration spike. This task measures how much damage each weight table takes
> at int4 on a real model, then gives int8 to the tables that suffer most, until a memory budget is spent.
>
> **Why now, and for whom.** tack runs Qwen3-8B on nobara's RTX 2070 SUPER (8 GB). On that card the window is
> limited by KV memory (about 20k positions at 144 KiB each), not by weights. A mix that matches `int4mix`'s quality in
> fewer bytes buys KV room; one that beats `int4` at the same bytes buys quality. Either helps tack directly.

---

## 0. Where the idea comes from, and how much to trust it

- **The prompt (2026-11-07):** a r/LocalLLM post and its write-up,
  <https://huggingface.co/Andy5k/Qwen3.5-0.8B-hand-quant>. An agent wrote a quantiser for Qwen3.5-0.8B from the BF16
  file alone, with no quantisation tool and no imatrix:
  - it rounded each weight to protect the input columns that carry signal, measured from a float run;
  - it measured each weight table's damage one at a time;
  - it spent a byte budget across IQ4_XS, Q5_K and Q6_K with a knapsack solver.
- **Its numbers** (held-out KL against BF16, llama.cpp, self-reported):
  - its 498.5 MB file: 0.0157;
  - Unsloth's UD-Q3_K_XL at 492.2 MB: 0.0925;
  - UD-Q4_K_XL at 558.8 MB: 0.0273 (beaten with about 11% fewer bytes);
  - UD-Q5_K_XL at 606.6 MB: 0.0104 (not reached).
- **Discounts:**
  - One 0.8B model, whose tied vocabulary table is 33.8% of its weights and most of its error. In an 8B that table is
    a far smaller share.
  - KL only: no task-level or greedy-agreement measure.
  - The method is not new. Activation-aware rounding (AWQ, imatrix) and per-tensor bit allocation (Unsloth's dynamic
    quants) are known. What is new is a clean, from-scratch run under strict rules.
- **Why a GGUF like that wouldn't help goinfer as a file:** goinfer requantizes every source to its own resident
  precisions (`docs/quantization.md`, "Reading a format is not standing behind it"). A file's per-tensor choices among
  IQ4_XS, Q5_K and Q6_K are flattened at load time. So the method has to live in goinfer's own load-time policy, which
  is what this task scopes.

## 1. What goinfer already has, and the lessons that bind this task

- **Per-tensor resolution exists.** `matmulQuant` resolves a mixed mode to int8 or int4 per tensor, and the resident
  weights and the `.giw` never carry the mixed label (`decoder/weightmat.go`). A measured mix is a different table fed
  to the same mechanism.
- **Backends.**
  - CUDA and the CPU backend have a real W8A8 GEMV.
  - Metal runs `int4mix` natively since `task-metal-int8-2026-10.md`, slice 4 (`r.w8Attn`), but only for the
    attention/FFN split.
  - An arbitrary per-tensor mix on Metal is an open question (§5).
- **The forward gate is mandatory.** The weight-space per-row IMMA scale search won 1.24× in weight space and gave ppl
  108 against 28.5 in the forward (`docs/quantization.md`, "What we refuse"). So no proxy result here counts until a
  full forward confirms it.
- **MoE is out of scope.** Routers flip top-k under tiny noise (779 of 3,200 pairs), so a dense result says nothing
  about a MoE (`docs/quantization.md`, "Why MoE changes the answer"). This task is dense models only: Qwen3-8B first.

## 2. The phases

### Q0. Per-table damage on Qwen3-8B

- **Reference:** the f32 forward of the safetensors checkpoint, on nobara's CPU (64 GB RAM; f32 8B is about 32 GB).
- **The quantity:** for each matmul table (36 layers × 7 projections, plus embed and head), the damage of that one
  table at int4 with everything else at int8.
- **Two measures, cheap then exact:**
  - **The proxy:** the activation-weighted output error of that one table, ‖(W − Ŵ)·X‖² over activations X captured
    from the reference forward on calibration text. One capture pass, then a cheap computation per table.
  - **The exact measure:** held-out KL against the f32 reference, for a sample of tables (the proxy's top, middle and
    bottom ranks), to check that the proxy ranks tables the way the forward does. If the two disagree, the proxy is
    dropped and the exact measure is used throughout, at its cost.
- **Output:** a ranked table of damage per tensor, its byte cost at int4 and int8, and the proxy-against-forward check.
- **Text:**
  - Calibration and held-out sets must be disjoint, and neither may be any E1 task's repository text. A tuned quant
    must not have seen tack's eval code.
  - Record their source and license.

### Q1. Allocation under a budget

- A knapsack over the tables: minimize total measured damage subject to a byte budget, each table at int4 or int8.
- Budgets to compare:
  - (a) `int4`'s bytes plus the cost of `int4mix`'s int8 tables, which is `int4mix`'s own size;
  - (b) halfway between `int4` and `int4mix`;
  - (c) `int4`'s bytes plus 5%.
- **The arms**, all on the same reference and held-out text: `int4`, `int4mix`, `int8int8`, and the measured mix at
  each budget.
- **Measures:**
  - held-out KL against f32;
  - greedy agreement over N steps against f32, the repo's usual parity measure;
  - perplexity;
  - resident bytes and decode tok/s on the target backend.
- **The forward gate:** a measured mix counts only if it beats `int4mix` on KL and agreement at equal or fewer bytes,
  or matches it in fewer bytes. The same rule applies against `int4` at budget (c).

### Q2. Activation-aware rounding (a separate arm)

- When requantizing a table to int4, choose group scales that minimize the activation-weighted error from Q0's
  captured X, not the plain weight error.
- This is the post's other half, and it differs from the refused weight-space search in one way: it is fit against
  real activations. It still faces Q1's forward gate, alone and combined with the measured mix.
- If it costs load time, record by how much. If it needs calibration data at load, it can't be a default; it would be
  baked into the `.giw` sidecar.

### Q3. Task-level check (only if Q1 or Q2 wins)

- tack's tier-1 live check on the winning mix against `int4`, on nobara: does the model still call tools and finish
  turns the same way?
- A full E1 or C-series screen only if the gain is big enough to be worth one. That would be registered on tack's side
  with full measurement rules, because its number would settle a direction.

### Q4. Shipping (only if Q3 holds)

- A load-time policy: a per-model mix table, chosen by measurement and checked into the repo or baked into the
  sidecar's cache key. Exposed as a `--quant` value or a profile name, with `DecodePath()` and the banner saying which
  mix is resident.
- `docs/quantization.md` gains the measured result in its "What we recommend" section, and the table's provenance.

## 3. Process

- **Machine:** nobara, the CPU reference by night through the night queue (`night.py`), holding the timing lock. The
  CUDA decode and memory checks also run on nobara. No Mac time is needed until §5's Metal question.
- **Rigor:** Q0–Q2 are exploratory measurements, recorded in `docs/measurements/` with commands and commits. A number
  that goes into `docs/quantization.md` or a release note is published, so the comparison behind it is registered
  before it runs (Francis's 2026-10-30 rule: full discipline only where a number is published or settles a direction).
- **PRs:** as every goinfer change: a PR that merges itself, Gemini review for product code.

## 4. Cost, roughly

- Q0's capture pass: one f32 forward over the calibration text on nobara's CPU. Hours, not days, at a few thousand
  tokens. The per-table proxy is minutes.
- The exact-measure sample: about 20 tables × one held-out forward each, one or two nights.
- Q1: one forward per arm and budget, about six forwards, one night.
- Q2: one night for the rounding search, then Q1's gate again.
- The agent work to write the capture and allocation code, which `cmd/` would hold, is small next to the machine time.

## 5. Open questions

- **Metal:** can the Metal resident runner take an arbitrary per-tensor int4/int8 mix, or only `int4mix`'s attention
  split (`w8AttnEligible`)? If only the split, the measured mix ships on CUDA and the CPU first.
- **Does the ranking transfer?** If a mix measured on Qwen3-8B also helps Qwen3-4B or another dense family, one
  measurement per family might do; if not, it is one measurement per model.
- **The embedding and head tables.** In the 0.8B they dominated. In an 8B, check `--embed-int4`'s known cost (about
  2.3 points of top-1, `docs/quantization.md`) inside the same budget, since the head table is a candidate like any
  other.
- **Calibration text:** which public, permissively licensed code-and-prose corpus, disjoint from E1's repositories.
