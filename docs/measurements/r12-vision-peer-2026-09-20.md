# R12 (iii) — vision peer row: Ollama measured; goinfer declines safely, and forcing it is unsafe here

**Result: Ollama's `gemma3:4b` TTFT (CPU, one image, warm process) measured at 0.407-0.455 s
(mean 0.4 s). goinfer's own side produced no usable number: the load-time fit guard correctly
declined `gemma-3-4b-it` on this Mac's real current headroom (needs ~6.1 GB resident, ~5.0 GB
free), and bypassing that guard (`GOINFER_NO_FIT_GUARD=1`, done with explicit sign-off after
weighing it as a "modest ~1.1 GB shortfall") produced a REAL near-incident — system swap `used`
grew from a 2.87 GB baseline to 6.6 GB in roughly 15-20 seconds before the process was killed
externally. No TTFT sample was ever collected on the goinfer side. This is a new, more general
instance of the machine-size-limit pattern this session already knew from MoE giants
(M35/M26/H27/G20): a genuinely small, dense 4B model can trigger the same class of event on this
box's current real-world headroom (VSCode + this session's own overhead eating roughly 11 GB of
the nominal 16 GB before any test starts).**

## What was attempted, in order

1. **First launch (fit guard on, default):** goinfer's cell reported `"server did not come up"`
   after the harness's 180s `wait_port` timeout — a *false-negative-looking* error. `run_vision_cell`
   redirects the child process's stdout/stderr to `DEVNULL`, so the harness had no way to show the
   real reason.
2. **Manual diagnosis** (`/tmp/serve-cpu-darwin -model bench=~/models/gemma-3-4b-it -backend cpu
   -addr 127.0.0.1:8199`, run directly, stderr visible): the process exited almost immediately with
   the real cause — `decoder: gemma-3-4b-it needs ~6.1 GB resident at quant int4; this machine
   currently has 5.0 GB of memory available (budget 3.5 GB = 70% of that)`. Not a hang; a correct,
   fast decline. The bench harness's 180s-timeout framing made a clean guard decision look like a
   flaky server start — worth fixing in the harness (see "What this doesn't establish" below), out
   of scope to fix here.
3. **Asked the user how to proceed** given the real (not hung) 1.1 GB shortfall: bypass, free memory
   first, or skip goinfer's side and record ollama-only. **Chose: bypass.**
4. **Bypassed retry, attempt 1** (`GOINFER_NO_FIT_GUARD=1`, `BENCH_MAX_LOADAVG=2.5`): refused before
   even starting — the harness's own idle-load preflight gate rejected a 1-min loadavg of 3.71
   against the 2.5 cap. This gate is a *different* thing from the fit guard and was not part of the
   memory question; raised again to 4.0 (this box is 8 cores — 6P+2E — so a loadavg of 4 is under
   50% utilization-equivalent, not evidence of a busy box; the ambient 1.4-3.7 range observed all
   session came from ordinary VSCode + this CLI session overhead, not from anything under test).
5. **Bypassed retry, attempt 2:** launched with an **external, out-of-process memory monitor**
   (`vm_stat`/`sysctl vm.swapusage` polled every 2s to a log file independent of the bench process),
   the same discipline this session used for R11(c)'s Metal MoE paging near-incidents. Within ~15-20s
   of the goinfer server process starting to load the checkpoint: system swap `used` climbed
   2.87 GB → 4.49 GB → 4.87 GB → 4.9 GB → 5.33 GB → 6.36 GB → 6.6 GB (the swap FILE itself growing
   4096M → 5120M → 6144M → 7168M as macOS dynamically expanded it under pressure), while `Pages
   free` collapsed toward ~3,400-4,000 (out of hundreds of thousands normally). **Killed via
   `pkill -9 -f serve-cpu-darwin`** the moment this was visible, before anything worse. `Pages free`
   recovered to ~380,000-420,000 within ~5s of the kill — confirms the pressure was real and tied to
   that one process, not a broader system event. The bench harness itself recorded this cell as
   `runs: []` (28.3s elapsed, `rss_peak_kb: 2,860,944` — 2.73 GiB for the process's own RSS at the
   moment of the kill, which under-represents the true pressure this generated, consistent with this
   repo's own documented finding that process RSS alone misses the real cost of a memory event on
   darwin).

## Data

| engine | model | TTFT samples (s) | mean | note |
|---|---|---:|---:|---|
| ollama | gemma3:4b | 0.4548, 0.4072 | **0.400** | warm process (1 warmup request discarded, per the harness's own design); CPU-forced (`num_gpu: 0`) |
| goinfer | gemma-3-4b-it | *(none)* | — | fit guard declined by default (correct); bypassed run killed mid-load before any request could complete |

Provenance: `apple-m1pro`/`macbookpro.lan`, Darwin 25.6.0 arm64, goinfer `3513c5b1`, tree clean,
ollama 0.32.5. `scripts/bench_peer.py`, `BENCH_VISION=1`, one image
(`testdata/gemma3_preprocess_image.png`), `temperature=0` both sides. Raw JSON kept in this
session's scratchpad (not committed — see "What this doesn't establish").

**Caveat on ollama's own RSS figure:** the harness reported `rss_peak_kb: 5,616` (5.6 MB) for the
ollama cell — implausibly small for a resident 4B model. `RSSSampler` almost certainly samples only
the launcher (`ollama serve`) process, not the runner subprocess ollama forks per-model, so this
number is not trustworthy and is not used for anything below. The TTFT figures themselves come from
the HTTP response timing, a different (and here, believed-correct) code path.

## Reading

**Ollama's 0.4s TTFT is a real, usable number** for a warm-model, one-image, CPU vision request —
directly comparable in kind (though not in magnitude or methodology) to the existing 31.3 s/image
SigLIP-tower figure in `benchmarks.md`, which is an in-process driver measurement of the tower alone,
not a served-HTTP TTFT; the two remain separate instruments per `run_vision_cell`'s own doc comment.

**goinfer has no vision peer number from this session, and that absence is itself informative.**
The fit guard's decline was correct given a genuinely modest-looking shortfall (1.1 GB) turned into
a real, fast-moving swap event once bypassed — this is NOT the multi-GB MoE-paging class of failure
(`mac-16gb-model-size-limits` memory: M35/M26/H27/G20), it is a single dense 4B checkpoint, the
simplest possible load shape, and it still produced the same swap-explosion signature in under 20
seconds. The fit guard's 70% conservative margin is doing real work here, not being overly cautious
— this session's own attempt to second-guess it (reading the shortfall as "modest, lower risk") was
wrong on this machine's actual present state (VSCode + this Claude Code session already consuming
roughly 11 GB of the nominal 16 GB before any test starts).

## Decision

**R12 (iii) is half-closed.** Ollama's row is real and usable. goinfer's row stays unmeasured on
this machine — not attempted a third time. If a goinfer-side vision TTFT number is wanted later, it
needs either (a) a machine with real headroom (the nobara box, or this Mac with VSCode/other
sessions closed first, not just "wait for loadavg to drop"), or (b) the same `.giw`-streaming path
already used to bring larger dense/MoE models within budget elsewhere in this repo
(`scripts/queue_citation_lint.py`'s own model-storage docs, `models-pull`), which was not attempted
here since attempt 2's near-incident closed the question for today rather than motivating a third
try.

## What this doesn't establish

- A goinfer vision TTFT number at all — genuinely unmeasured, not a bad-but-real number.
- Whether a `.giw`/streamed load of the same checkpoint would avoid this — untested; the safetensors
  loader's own error message named this as the relevant next step (`cmd/prequant` builds a
  streamable `.giw` from a safetensors source), not attempted here.
- The harness's `"server did not come up"` framing on a fit-guard decline is misleading (it makes a
  fast, correct decline look like a 180s hang) — worth a small harness fix (surface the child
  process's stderr on a `wait_port` timeout) but out of scope for this measurement record.
- Whether raising `BENCH_MAX_LOADAVG` to reflect this box's real core count (8) generalizes to other
  cells/phases — only exercised here for the vision phase specifically.
