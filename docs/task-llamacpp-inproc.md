# Task: llama.cpp in-process — one timer, one token stream, full logits (2026-09-12)

> **BLUF.** The peer matrix drives llama.cpp over HTTP (`llama-server`, `scripts/bench_peer.py`),
> which is the right method for a peer row and the wrong instrument for three questions the matrix
> cannot answer: *how far are goinfer's logits from llama.cpp's on the same GGUF*, *what does
> goinfer's per-token cost do with depth that llama.cpp's does not*, and *do the two tokenizers
> even agree on the fixtures*. This doc scopes **a sixth Go module, `bench/llamacpp/`, that
> `dlopen`s a pinned prebuilt `libllama` through purego — no cgo, `CGO_ENABLED=0`, the same
> mechanism the Metal backend already uses — and runs llama.cpp and goinfer in the same process on
> the same token IDs with the same clock.** Its first deliverable is the peer matrix's still-unbuilt
> fidelity column, done at the full-vocabulary logit level rather than `n_probs`. It is an
> instrument, not a peer row: its ratios live in their own table and never beside
> `bench_peer.py`'s.
>
> **What this is not.** Not a dependency of goinfer (the product stays cgo-free; the module is
> outside `go.work`'s CI surface and behind a build tag). Not a wait on Go proposal #81450 ("cgo
> without a C toolchain") — that proposal formalises exactly this shape (annotated bindings to a
> prebuilt library, no C compiler at build time) and would replace the hand-written signatures with
> generated, type-checked ones when it ships; it is the upgrade path, not the prerequisite. Both
> boxes already have a C toolchain and already have `libllama` built for the peer matrix.
>
> **Status: SCOPED 2026-09-12, nothing built.** goinfer `ffc0b83`, aikit per `go.mod`, purego
> v0.10.1 (already indirect in `metal/` and `cuda/`). Prompted by golang/go#81450.

## 0. What must not change

- **`docs/benchmarks.md` § Methodology and `CLAUDE.md` § Benchmarking govern.** `~/models` only;
  provenance header on every results file; same-session interleaved; `n = 5`, spreads reported;
  greedy with a pinned seed; thermal note. A cell from a forbidden root is void.
- **This instrument's numbers are a separate method.** Both engines in-process, one timer, no
  HTTP — so a ratio *within* this table is sound, and a ratio *across* this table and the peer
  matrix is the "kernel throughput over end-to-end throughput" mistake `CLAUDE.md` names (the
  retired "0.5B 1.78×"). Reported in its own section, cross-checked once against the matrix (§6
  step 3), never merged with it.
- **goinfer's product surface does not learn about llama.cpp.** No import from any product
  module into `bench/llamacpp/`; the arrow points one way. `README.md`'s "no cgo, no shared
  library" claim stays true of every binary a user gets.
- **Pinned llama.cpp, per box, per run.** Tag, commit, build flags and `sha256` of each loaded
  `.dylib`/`.so` in the header record. A run whose library hash differs from the pinned one is
  void. Same rule the matrix has for a peer version changing mid-run.
- **Off is a competitor** where a knob exists on one side only (flash attention, KV type, batch
  ladder): the arm without it is a row.

## 1. Why in-process, when the matrix already has llama.cpp

| question | HTTP row (`bench_peer.py`) | in-process (this doc) |
|---|---|---|
| decode tok/s, the number readers ask for | ✓ primary, one method for every engine | ✗ not its job; cross-check only |
| per-position logits, full vocabulary | `n_probs` top-k over `/completion`, truncated, JSON-rounded | `llama_get_logits_ith` — the whole row, f32 |
| same token IDs on both sides | each engine tokenizes its own prompt text | one `[]int32`, fed to both; tokenizer agreement is itself measured (§4 I1) |
| per-token latency at depth | streaming timestamps, HTTP jitter ~ms against 12–25 ms tokens | one monotonic clock around `llama_decode` + `llama_synchronize` and around goinfer's forward |
| llama.cpp's own knobs (threads, `n_batch`/`n_ubatch`, flash-attn, KV type, `n_gpu_layers`) | server flags, one setting per server restart | per-context params, ladders in one process |
| an f16-GGUF reference for the cells HF cannot fit (M35, M26, G20, H27 — matrix §4) | `n_probs` again | full logits, teacher-forced |

The fidelity column in `docs/task-peer-benchmarks.md` §4 is "still unbuilt" as of the 2026-09-04/05
first pass. The llama.cpp half of it is cheapest to build here, and the goinfer half is
`ForwardForTest`/`PrefillLastNArgmax`, which this module can call directly.

## 2. Prior art — read before building

- **Go bindings to prebuilt llama.cpp via purego already exist**, and `docs/benchmarks.md` Table 1
  already cites them: `github.com/hybridgroup/yzma` (purego + ffi, `llama.Load(libPath)`, v1.16.1
  as of 2026-06-08) and `github.com/dianlight/gollama.cpp` (purego, v0.2.2-llamacpp.b6862). Step 0
  evaluates both **before** any signature is hand-written. Adopt one if it (a) exposes
  `llama_get_logits_ith` / `llama_get_logits`, `llama_decode` with a caller-built batch, and
  `llama_synchronize`; (b) pins a llama.cpp build we can also install as `llama-server` for the
  matrix, so both instruments share one library version; (c) works `CGO_ENABLED=0` on
  darwin/arm64 and linux/amd64. If either passes, this doc's §3 shrinks to a thin wrapper and
  the struct-layout risk below is theirs, not ours.
- **`goccy/go-llama`** (llama.cpp → wasm64 → Go, single-threaded) is the CPU-lane *peer* in the
  matrix, not an instrument; it does not expose logits at the level needed here and it is not
  llama.cpp's real kernels on real threads. Out of scope.
- **golang/go#81450** — Matloob, Mui et al.; proposal stage as of this writing; `//cgo:binding`
  annotations plus `go tool cgo -gen-binding` from headers. Its header-driven struct generation is
  precisely the thing purego makes us do by hand (§3, "struct mirrors"). Track; do not wait.
- **`docs/task-bindings.md`** § "On the cgo-free promise" — the Metal backend's purego + Obj-C
  arrangement under `CGO_ENABLED=0` is the precedent that this module is not a new class of thing
  in the tree.
- **`scripts/bench_peer.py`'s header record** is the provenance shape to reuse, not reinvent: it
  already refuses to run on a non-idle box and already stamps driver, distro, commit, tree-dirty,
  loadavg, GPU temperature.

## 3. Design

**Module.** `bench/llamacpp/` with its own `go.mod` (`github.com/townsendmerino/goinfer/bench/llamacpp`),
importing the root module for `decoder`, `tokenizer` and the test hooks. Added to the gitignored
`go.work` `use` list for local builds (mandatory for cross-module work — `CLAUDE.md`); **not**
added to any CI workflow's module list, and every file carries `//go:build llamacpp` so an untagged
build of the module is empty. `go vet -tags 'llamacpp goinfer_testhooks' ./bench/llamacpp/...`
is the local gate. `gofmt` and `staticcheck` rules apply as everywhere.

**Linking.** purego `Dlopen` of absolute paths, in dependency order — `libggml-base`, `libggml`,
the backend libraries (`ggml-metal` / `ggml-cuda` / `ggml-cpu`), then `libllama`. Release builds
of llama.cpp ship backends as dynamically loaded modules, so the harness calls
`ggml_backend_load_all_from_path(dir)` after `llama_backend_init`; `LLAMACPP_LIB=<dir>` names the
directory and the header record hashes every file it loaded from it. On Linux, purego runs
without cgo via its own fake-cgo shim (amd64/arm64) — the same `CGO_ENABLED=0` on both boxes.

**Surface — about fifteen functions, one file, one llama.cpp tag.**

| group | functions |
|---|---|
| lifecycle | `llama_backend_init`, `llama_backend_free`, `ggml_backend_load_all_from_path` |
| model | `llama_model_default_params`, `llama_model_load_from_file`, `llama_model_free`, `llama_model_desc`, `llama_model_get_vocab` |
| context | `llama_context_default_params`, `llama_init_from_model`, `llama_free`, `llama_n_ctx`, `llama_set_n_threads`, `llama_memory_clear` |
| tokens | `llama_tokenize`, `llama_token_to_piece`, `llama_vocab_n_tokens` |
| compute | `llama_batch_init`, `llama_batch_free`, `llama_decode`, `llama_synchronize`, `llama_get_logits_ith` |

No sampler API: sampling is done in Go on both engines' logits with the same greedy/seeded code,
so the comparison is of forwards, not of two samplers.

**Struct mirrors are the risk, and the guard is a canary, not a comment.**
`llama_model_params` and `llama_context_params` are passed and returned *by value*; purego
supports struct-by-value on darwin/linux amd64+arm64, but the Go mirror must match the C layout of
the pinned tag exactly, and **a mirror that drifts does not error — it returns plausible garbage**
(the guard-that-inverts class in `CLAUDE.md` § Measurement discipline). So:

- one `llamacpp_pin.go` records the tag and the `sha256` of `llama.h` the mirrors were written
  against;
- a `TestParamsCanary` calls `llama_model_default_params()` and `llama_context_default_params()`
  and asserts a set of known defaults at known offsets (`use_mmap == true`, `n_ctx == 512`,
  `n_batch == 2048`, `n_threads == GGML_DEFAULT_N_THREADS`, the `n_gpu_layers` sentinel) — a
  layout shift moves at least one of them;
- the harness refuses to run unless the canary passes against the library it just loaded.

If purego's struct-by-value fails on a box (step 0 finds out), the fallback is plain cgo behind
the same build tag for that box — the module is still outside the product, the experiment is
still worth running, and the pure-Go-instrument property is a nicety, not the point.

**Both engines, one clock.** Each cell loads the goinfer model (`decoder.Load` /
`LoadGGUFBytes`, shipped-default options) and the llama.cpp model from the **same GGUF file**
(`scripts/gguf_same_weights.py` discipline: the file, not a repack), tokenizes once with goinfer's
tokenizer, checks agreement with `llama_tokenize` (I1), and then runs the workload on each engine
with `time.Now()` around the forward — `llama_decode` followed by `llama_synchronize` on the
llama.cpp side (GPU backends return before the compute lands; the logits read forces it, but the
timer must not depend on that) and the equivalent forward + backend sync on goinfer's. Arms are
interleaved per prompt, rotating order, `n = 5`. `runtime.GC()` and a settle sleep between arms so
one engine's allocation does not land in the other's window.

**Memory.** Two models resident at once on the 16 GB Mac caps the cells at S and D7 (~1 + 4.5 GB
int4 each side plus KV); M35/M26 need the engines loaded *alternately* (load, run, free, load the
other) which breaks interleaving — those are tier 2 with a stated method change. On the RTX box
the same applies to VRAM: D7 int4 on both sides at once does not fit 8 GB, so the CUDA cells are
S resident-both, D7 alternate-load.

## 4. Workloads — the instrument's own, not the matrix's W1–W8

| id | workload | reports | why it needs to be in-process |
|---|---|---|---|
| I1 | **tokenizer agreement** on the matrix's 20 fidelity prompts + the W4 transcript | mismatched positions, first divergence per prompt | precondition for every other row; a mismatch is a finding on its own (a matrix row taken on different token streams is comparing different prompts) |
| I2 | **logit fidelity, same file** — teacher-forced over the matrix §4 fixtures (≥256-token reference continuations): per-position full-vocab KL, argmax agreement, first-disagreement position | arms: llama.cpp Q4_K_M, llama.cpp f16, goinfer W4A8 default, goinfer int8int8; reference: HF fp16 where it fits (S, D7), llama.cpp f16 elsewhere | the matrix's fidelity column, built; also answers how far goinfer's group-32 W4A8 requant sits from llama.cpp's dequant of the same Q4_K_M tensors — a number nobody has |
| I3 | **per-token decode cost vs depth** — positions 1…128 at contexts 128 / 2k / 8k | p50 / p95 per-token latency, the marginal-cost-vs-depth curve per engine | P19's claim (goinfer's marginal token cost rises with KV depth where the peers' stays flat) measured with one clock instead of inferred from wall-clock differences |
| I4 | **prefill at K = 512 / 3900** with llama.cpp's `n_batch`/`n_ubatch` ladder (512, 1024, 2048) beside goinfer's batched prefill | wall time per K per setting; the do-nothing arm is llama.cpp at its defaults | names what llama.cpp's chunking buys, against `docs/task-prefill-gap.md`'s two-term model |
| I5 | **CPU lane**, `n_gpu_layers = 0`, threads = `GOMAXPROCS` on both | I2 + I3 on S | the pure-Go story with the thread count actually matched — the HTTP CPU row cannot pin llama.cpp's threads to goinfer's |

Every cell records load time, peak RSS / VRAM, the exact parameter structs (serialised), and the
library hashes.

## 5. Pre-registered bands and decision rules

Derived before the first run, per the campaign convention; revise only with a mechanism.

- **I1.** Expect **100 % agreement** on the ASCII code/prose fixtures for the Qwen2.5 BPE (both
  tokenizers read the same GGUF vocab and merges). Any mismatch → stop, name the token, decide
  whether it is a goinfer tokenizer bug (`queue-correctness`) or a llama.cpp pre-tokenizer
  difference, and record which side the matrix's rows were taken on. Fixture prompts with
  non-ASCII are where a difference would live; include two on purpose.
- **I2, instrument validity.** goinfer-vs-HF-fp16 argmax agreement computed *through this
  harness* must reproduce the existing pin-script figure for the same prompts within **1 point**,
  or the harness is wrong before it has compared anything. This is the check that the goinfer
  side is wired to the shipped-default path and not a test seam.
- **I2, the finding.** *Ambiguous → parked* band stated up front: if llama.cpp-Q4_K_M and
  goinfer-W4A8 are within **2 points** of each other against the f16 reference, the requant
  costs nothing measurable and that is the result. Beyond 2 points in goinfer's disfavour, the
  gap is a P-entry candidate in `queue-performance.md` with the KL profile attached (which
  layers/positions it comes from is I2's per-position record). Beyond 2 points in goinfer's
  favour, re-check the reference before believing it (`CLAUDE.md`: the reference can be wrong).
- **I3.** Reproduce the matrix's W1 depth-128 goinfer/llama.cpp ratio for the same cell on the
  same day within **10 %**, or the two instruments disagree and the disagreement is investigated
  before any other I3 cell is run (§6 step 3). Then the curve: llama.cpp's per-token cost from
  depth 128 → 8k is projected to rise by *less than* goinfer's on the same cell; the number of
  interest is each engine's slope, reported separately, with the ratio of slopes as the headline.
- **Void rules.** Any arm's spread > 10 %; thermal event; forbidden model root; canary failure;
  library hash drift; `loadavg` gate tripped. Void cells re-run, never averaged in.

## 6. Steps — split before build, each with a gate

0. **Spike, half a day, Mac first.** Evaluate yzma and gollama.cpp against §2's three criteria
   (a)–(c); whichever passes, or purego by hand if neither, load the 0.5B GGUF from `~/models`,
   decode 32 greedy tokens, and match the text and first-token argmax to `llama-cli` on the same
   file. Gate: runs `CGO_ENABLED=0`; canary passes; struct-by-value works on darwin/arm64. Outcome
   recorded either way in `docs/measurements/llamacpp-inproc-spike-<date>.md` — a negative here
   (purego cannot do it, fallback to cgo) is a result, not a failure.
1. **Bindings + canary.** `bench/llamacpp/internal/llama/` — the §3 surface, `llamacpp_pin.go`,
   `TestParamsCanary`. Pin the tag to the one `llama-server` was built from for the matrix on each
   box (Mac build 10621 / `c1d0e7a00`; nobara `427291b`) so the two instruments share a library;
   if they cannot (different tags per box), pin one and rebuild `llama-server` from it before the
   next matrix pass rather than carrying two.
2. **Harness.** `bench/llamacpp/cmd/inproc` — loads both engines, I1 → I2 → I3 → I4 → I5 in that
   order (I1 gates the rest), writes provenance-stamped JSON to
   `docs/measurements/llamacpp-inproc-<date>/` in `bench_peer.py`'s header shape. Refuses a
   non-idle box and a forbidden model root, like the Python harness does.
3. **Cross-check.** One S cell, Mac Metal, same day as a `bench_peer.py` W1 run: the 10 % rule in
   §5. Nothing else runs until this holds. Write it up whichever way it goes.
4. **Tier 1 runs.** S and D7 on the Mac (Metal, and I5 CPU), then S on nobara (CUDA, and I5), D7
   alternate-load on CUDA. Two quiet half-days. Report per §7.
5. **Tier 2, each with its own reason.** M35 / M26 alternate-load with llama.cpp `-ncmoe` beside
   goinfer's expert cache (a method change — stated); G20 MXFP4 both sides; the
   `--cpu-fast-attention` and int8int8 variants as goinfer-vs-goinfer rows on this instrument (§4
   I2 already carries int8int8).

**Later, not scoped here.** (a) `llama_context_params.cb_eval` — a per-tensor observation
callback (purego `NewCallback`) that would give per-layer differencing against llama.cpp's
forward, the discipline `CLAUDE.md` prefers, and the cheapest way to settle audit-2026-09-10's
dense-Granite un-permuted q/k Critical on a family HF cannot fit. (b) Swapping the hand-written
surface for `//cgo:binding` files when #81450 lands. (c) An embeddings cell (matrix W8) if the
in-process path is ever needed there — it is not, today.

## 7. Reporting

A dated section in `docs/benchmarks.md` — "In-process llama.cpp instrument, 2026-09" — placed
*after* the peer matrix with a one-line preface that it is a different method and that no ratio
crosses the boundary. One table per workload per box: engine columns, goinfer/llama.cpp ratio with
spread, parameters in a collapsed block, library tag and hashes in the provenance line. I2 fills
the matrix's fidelity column *by reference* (the matrix section cites this one for the number; the
number lives here). Raw cells under `docs/measurements/llamacpp-inproc-<date>/`. Rows that read
badly are reported at full value with mechanism where known; retractions grep the figure with its
unit across every page. The Ollama Chase Ledger gets the I2 and I3 headlines as new chips, not
as edits to existing standings.

## 8. Effort and what it displaces

Spike ½ day; bindings + canary ~1 day if hand-written, hours if yzma/gollama.cpp pass §2; harness
~1 day; runs two quiet half-days; write-up ½ day. It displaces nothing on the Metal L1 / L2-Metal
path — it is instrumentation, and its I3 output is an input to those levers (which depth band the
Metal attention gap opens in) rather than a competitor for the same time. If time is short, steps
0–3 plus I1 and I2 on S alone still deliver the fidelity column, which is the part with a
standing IOU.

## 9. Sources

`docs/task-peer-benchmarks.md` (§0 one-method rule, §1 peers, §4 fidelity column, §7 harness);
`docs/benchmarks.md` (Methodology; Table 1 footnote ᶜ and the gollama.cpp / yzma citations;
"Peer matrix 2026-09" provenance for the pinned llama-server builds; "Not done yet");
`scripts/bench_peer.py` (header record, idle gate, `gguf_same_weights.py` discipline);
`CLAUDE.md` (§ Benchmarking — `bench_compare.sh` vs peer numbers; § Measurement discipline —
inverting guards, paired differencing, pre-registration, retractions); `docs/task-bindings.md`
(purego under `CGO_ENABLED=0` precedent); `docs/task-prefill-gap.md` (two-term prefill model, I4);
`docs/audit-2026-09-10.md` (dense Granite q/k permute Critical, §6 "Later" (a));
`queue-performance.md` P19 (attention at depth) and P20 (MoE batched prefill);
golang/go#81450 — https://github.com/golang/go/issues/81450 (proposal stage, 2026-09);
purego — https://github.com/ebitengine/purego (struct-by-value support, Linux fake-cgo);
yzma — https://github.com/hybridgroup/yzma; gollama.cpp — https://github.com/dianlight/gollama.cpp;
llama.cpp `include/llama.h` at the pinned tag (the §3 surface; `ggml_backend_load_all_from_path`
in `ggml/include/ggml-backend.h`).
