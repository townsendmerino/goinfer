# Tier-0 findings — decisions (2026-08-25)

> **Correction, 2026-08-25:** this doc says findings 2 and 1-stage-1 "ship in v0.14.0". **v0.14.0
> was already tagged on 2026-08-19.** Both landed in **v0.15.0**, the open release (G18 `3a16a4b`,
> G19 `43a3fdb`). Left in place below rather than rewritten, because the decision was right and
> only its release label was wrong.

> Companion to `docs/measurements/dsh-tier0-run-2026-08-25.md` (`892511d`). The run's verdict
> stands as filed: gate not met, correctly not forced, all three causes goinfer's. These are the
> dispositions the run stopped to ask for. Decider: Francis; drafted by Claude (Cowork) from the
> run log + repo evidence cited inline.

## Finding 2 — prefill ignores cancellation → FIX NOW, ships in v0.14.0

Correctness/consumer-trust class, not perf: an abandoned client leaves a prefill burning CPU
(measured 47:38 after two kills), and a retrying client stacks generations. Reachable from any
client that gives up — this is precisely the class C4-soak exists to catch, found early by a
real harness instead. The seam is already right (`r.Context()` reaches `drive`); the fix is the
prefill loop honoring it (per-token or per-N-tokens check — at 25 tok/s the check is free).

Gates: an abandoned-client test bounding CPU-after-disconnect; the same for a client killed
mid-queue (the retry-storm shape — each retry must not inherit a zombie's slot); confirm decode's
existing cancellation behavior while there, so the fix's scope statement is checked, not assumed.

## Finding 1 — streaming+tools buffers everything → TWO STAGES

The buffering is correct for tool-call parsing (`tools.go`'s own comment) — the defect is
silence, not buffering.

- **Stage 1, now (v0.14.0-eligible): SSE heartbeats while the buffer holds.** Comment frames
  (`: ping`) are protocol-legal and content-free; they defeat idle timeouts (dsh's 300s, and
  every other harness's) without touching parsing. Small; test asserts frames flow during a
  slow tool-path generation and that no content delta is emitted early.
- **Stage 2, queued (the real fix): incremental tool-call-aware streaming** — emit prose deltas,
  hold back only from a potential tool-call opening, per family via the `chat` package's
  existing per-family parsers. Real work; not rushed into the release.

## Finding 3 — CPU prefill single-core/superlinear vs the "batched" label → PERF QUEUE, with a named first experiment

Filed against the benchmark/advertisement claim, as the run suggested — it outranks the
integration for the CPU-agent story and does NOT block v0.14.0. But characterization first, and
the run's own config narrows it: pass 1 ran the dense 1.5B at **`-quant int4`** on CPU.
Repo evidence that makes int4 the prime suspect rather than the batched path itself:
`demo/chat`'s docs state the CPU fast lane is `int8int8` (W8A8 SDOT) and that the int4 path pays
per-token nibble unpacking; the ARCHITECTURE dispatch table promises int4 only "the same kernel
at every M" — i.e. possibly no amortization and no parallel dispatch at M=len.

**Experiment 1 (cheap, decisive): re-run the 170/620-token prefill points at `-quant int8int8`,
same model, same box.** Two branches:
- int8int8 is multi-core and multiples faster → the item narrows to "CPU int4 batched prefill:
  unparallelized/unamortized," the `prefill path: batched` label gets qualified (it must not
  advertise a speedup the kernel doesn't deliver — C2 would flag it), and the Tier-0 recipe
  moves to `int8int8` on CPU (the demo's own default), possibly reviving the CPU pass without
  kernel work.
- int8int8 is ALSO single-core at M=len → a live regression against the v0.5.0 batched-prefill
  wins; escalate in the perf queue with both curves attached.

Either way the run log's numbers (37.7→25.4 tok/s over 170→620, ~105% CPU) go into the queue
entry verbatim as the before-state.

## Pass 2 — RE-SCOPED to GPU; the 26B is dropped for integration purposes

A 26B CPU/streamed prefill of an 8k agent prompt reproduces the timeout with a longer wait —
nothing to learn. The qualifying run becomes:

- **Primary: CUDA on `linux`** — batched `PrefillLast` puts a 2048-token TTFT at ~2.1 s and 8k
  in range; dsh stays on the mac with `baseURL` pointed across the LAN, which exercises the new
  non-loopback `-api-key` hard-fail as a bonus (document the key setup in the recipe).
- **Alternate, mac-local: Metal + `--metal-fast-prefill`** (opt-in, 3.9–4.6× TTFT; the recipe
  notes it opts out of the bit-identity default and why that's acceptable for an agent session).
- The 26B remains what it always was — a capability showcase, not an agent-latency
  configuration; it re-enters only if a fits-VRAM MoE story is wanted later.

## Sequence

Finding 2 → finding 1 stage 1 → the int8int8 experiment (may revive the CPU pass for free) →
pass-2 GPU qualifying run → finding 3's kernel work at the perf queue's own priority → finding 1
stage 2. The v0.14.0 tag takes findings 2 and 1-stage-1 and waits for nothing else here.

## Recorded

The run's retraction ("goinfer's side is not the friction" — true of the surface, false of the
engine) is accepted and closed; the stop-rather-than-pick at the judgment boundary was the right
call. Three defects that no gate had caught, surfaced by one real consumer session, is the
Tier-0 gate doing exactly what the scoping doc built it for — the findings are the deliverable,
and the recipe will be written against a server that earned it.
