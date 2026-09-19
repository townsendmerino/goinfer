# Task — CPU thread scheduling: is there a stronger lever than the P-core worker cap?

**Status:** proposed, gates before code. Filed 2026-09-19.
**Venue:** `mac` (Apple Silicon) — the P-core/E-core split this doc is about does not exist on
`linux`/x86.
**Relates to:** `decoder/scratch.go`'s `maxAttnWorkers`, `decoder/mlp.go`'s
`activationFanoutWorkers`, `docs/measurements/mac-cpu-decode-vs-ollama-2026-08-22.md`.

---

## Where this starts from

The CPU decode path's only accommodation for the P-core/E-core split today is a hardcoded
**worker-count cap**, not a placement decision. `maxAttnWorkers = 6` caps attention's head-parallel
fan-out at the measured P-core count rather than `GOMAXPROCS(0)`'s full 8
(`decoder/scratch.go:177`), grounded in `docs/measurements/mac-cpu-decode-vs-ollama-2026-08-22.md`
(6 P-cores measured marginally faster than 8 in every cell tested, §4 item 1). `mlp.go` reuses the
same constant for its own activation fan-out (`activationFanoutWorkers = maxAttnWorkers`,
`decoder/mlp.go:296–299`) rather than redefining it, so the MLP/MoE-expert activation step is
already under the same cap.

A count cap bounds *how many* goroutines run, not *where* they land — the OS scheduler is still
free to put those 6 on any core, P or E, and that assumption is more likely to slip on the CPU
path the five model families that have no GPU-resident backend anywhere run on permanently
(Llama 4, Granite-4.0-H, LFM2.5, Laguna, Ling 3.0 — `docs/positioning.md:32–34`), or under
contention from anything else on the box.

## Gate 0 — is real thread placement even reachable on macOS

**The question:** does macOS expose any API to bind a thread to a specific physical core, or to
the P-core cluster as a group, beyond the advisory QoS class?

**Working expectation: no.** Unlike Linux's `sched_setaffinity`, macOS has no public API for
arbitrary thread-to-core placement. `pthread_set_qos_class_self_np` sets a *QoS class* — a
priority/energy hint the scheduler is free to disregard under contention — not a placement
guarantee.

**Checked before designing anything, per this doc's own rule**: Nirvana Code
(github.com/niravlekinwala/nirvana-code, MIT, v0.3.0) markets "P-Core / E-Core thread pinning" as
a feature — README and its comparison table's "Apple Silicon Core Affinity" row. Read against
their source rather than the label: the entire mechanism is one call,
`pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0)` (`0x21`) — `boost_thread_qos()`,
`src/engine.rs:164–171`, called from the decode/Metal-dispatch call sites
(`src/speculative.rs:244`, `src/engine.rs:738`). There is no `thread_policy_set` /
`THREAD_AFFINITY_POLICY` call anywhere in the tree, and no explicit QoS-*lowering* call either —
the README's "E-core" half of the claim (async file I/O, web routing) is whatever QoS class Rust's
default async runtime leaves those threads at, not something the code sets. **A QoS hint and a pin
are different things, and their wording does not distinguish them** — "binding … to P-cores" reads
as a placement guarantee; the code is a scheduler hint on one thread. This settles Gate 0 the way
the working expectation above already had it: QoS class is the only lever macOS actually offers,
and even that is advisory.

## What's left open

- Whether boosting QoS the same way — on the goroutines `maxAttnWorkers`/
  `activationFanoutWorkers` actually spawn, not just a top-level decode goroutine — moves the
  needle at all beyond what the count cap already buys. Unmeasured here; Nirvana's own reported
  gain from this same mechanism is a single-machine, self-reported number and not evidence for
  goinfer's kernels.
- Whether it matters more under contention (something else running on the box) than in the
  quiet-box conditions `mac-cpu-decode-vs-ollama-2026-08-22.md` measured under.

## Constraints

- Do not use the words "honest" or "honesty".
- Leave uncommitted for review.
