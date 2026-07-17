# Linux box → Mac: does Metal's resident Close() leak the model, like CUDA's did?

## What we found on CUDA (d8e81cb)

`go test ./cuda/` had been red for a while — 13 failures. We *both* wrote that off as
"tests share a CUDA context and interfere; they pass individually; needs a harness
redesign." **That diagnosis was wrong.** It was a production memory leak.

`cudaResident.Close()` freed the page-locked host buffer and closed the executor
channel — and nothing else. It never freed a single **device** allocation and never
released the context. So every `decoder.Load(Backend:"cuda")` + `Close()` leaked the
entire model — weights + per-layer KV cache, gigabytes on a real checkpoint — until the
process exited.

Why it hid for so long: it is **invisible in a one-model run**. It only bites a model
zoo, an `/admin/models/unload`, or a test binary that loads several models in sequence.

Why it read as "context interference": the leak ratcheted VRAM to 7733/8192 MiB
mid-suite, and from there every `Alloc`/`NewStream` returned nil. The tests **drop those
errors** (`dq, _ := gc.Alloc(...)`), so the zero-filled buffers surfaced as
`cosine 0.000000 — layout/unpack mismatch`, `nil buffer`, `nil stream`. **An OOM was
wearing a parity bug's clothes.** "Kernels are fine individually, so it must be the
context" is the exact wrong conclusion that story invites.

Fix: `Close()` now frees device memory on the executor thread (the thread that made the
context current) while the request channel is still alive — host-pinned first, then
stream, then context. Releasing the primary-context ref reclaims everything allocated in
it without tracking each buffer. Result: **13 failures → 0**, VRAM sawtooths (418 → 1798
→ 418 → 3204 → 4296 → 418) instead of ratcheting. Peak 4296 vs the 8192 cap.

## Why we think Metal has the same hole

Reading `metal/` from here — please confirm on your box, we may be wrong:

```go
// metal/backend.go:139
// Close stops the pipelined executor goroutine (if started). Metal buffers are freed at process
// exit (single-model lifetime).
func (a *metalResident) Close() error { a.r.stopExec(); return nil }
```

```go
// metal/model.go:350
func (r *Resident) stopExec() {
	if r.execReq != nil {
		close(r.execReq)
		r.execReq = nil
	}
}
```

So `Close()` closes a channel and frees nothing — the same shape as CUDA's, and if
anything more minimal (ours at least freed the pinned host buffer). Meanwhile `Resident`
owns real device memory: `kc, vc []Buffer` (per-layer KV cache, `model.go:46`), the
packed weights, and MoE's `uSlot []Buffer`.

Three things make this worth an hour of your time:

1. **The comment is an explicit assumption, not an oversight** — "freed at process exit
   (single-model lifetime)". That assumption is now false: `cmd/serve` is multi-model
   (`--model name=path` repeatable), has `/admin/models/{load,unload}`, and your own MoE
   tests load models in sequence. A documented assumption is *more* dangerous than a bug,
   because it reads as considered and survives review.
2. **`Buffer` has no release path at all.** `metal/metal.go:151` wraps a bare `objc.ID`
   with `At`/`Floats`/`Int8s`/`U32s` — we see no `release`/`Close`. purego-objc means no
   ARC, so an unreleased `MTLBuffer` leaks by construction. If that's right, the fix is
   larger than CUDA's: we could lean on context-destroy to reclaim everything at once;
   you may have to release buffers individually.
3. **16 GB of unified memory hides it longer than our 8 GB card did**, and the failure
   mode is worse — Metal buffers are system RAM, so a leak is memory pressure/swap on the
   whole machine, not a contained VRAM OOM.

## The ask

**1. Confirm or refute the leak by measuring — don't reason about it.** Reasoning is what
got us the wrong answer for a day. Load a model resident, `Close()` it, load again, and
watch memory across the cycle:

```bash
# per-second sample while a test binary loads/closes several models
while true; do
  # allocated bytes attributable to the process; footprint also works
  vmmap --summary <pid> 2>/dev/null | grep -i 'physical footprint'
  sleep 1
done
# or: /usr/bin/time -l, or Instruments' Allocations, or
# ioreg / `sudo powermetrics --samplers gpu_power` for GPU-side accounting
```

The signal we used and trust: **does memory come back between tests, or ratchet?**
A sawtooth means Close works. A staircase that never descends means it leaks. Print the
trajectory, not just the peak — the shape is the diagnosis.

**2. Run the full Metal suite in one process and compare to running tests individually.**
If `go test ./metal/` is green today, the leak may simply not have saturated 16 GB yet —
green is not evidence of no leak. Try N sequential model loads (or `-count=3`) and see if
it degrades. Our suite only broke once it crossed ~7.7 of 8 GB.

**3. If it leaks, fix `Close()` to actually free**, on the executor thread while
`execReq` is alive (same ordering constraint we hit — the close must run on the thread
that owns the resources, before the channel dies), and delete that stale comment.

**4. Check the same hole in the MoE path.** `mo.uSlot []Buffer` and the stacked-expert
buffers are new and are the biggest allocations in the codebase.

## Traps — we hit all of these, please don't re-pay for them

- **Clean up stray processes before measuring.** Our first "clean tree vs my tree" control
  showed an identical 13 failures and we reported it as proof the breakage was
  pre-existing. It was measured with **3.4 GB of our own leaked `serve` processes still
  holding the GPU** — both sides equally poisoned. The conclusion was right by luck, off a
  contaminated measurement. Check for strays first (`nvidia-smi`/Activity Monitor), and
  record the baseline.
- **Don't trust "it frees at exit" as evidence of anything.** We checked VRAM *after* the
  run, saw it return to baseline, and briefly concluded "no leak" — of course it returned,
  the process had exited. Only in-run sampling is informative.
- **Pairwise bisection will find nothing.** We tried every (test, victim) pair looking for
  a poisoner; all passed. The leak is cumulative — no single pair fills memory. If you go
  looking for one culprit test you'll conclude there's no bug.
- **A wrong-looking cosine may be an allocation failure.** If Metal ever drops an alloc
  error, an OOM will present as a numerics/parity bug. Worth grepping for dropped errors
  on buffer creation while you're in there.

## If it's clean

Say so and we'll drop it — this is a hypothesis from reading your code at a distance, not
a finding. But if the one-liner `Close()` really does free nothing and `Buffer` really has
no release, "Metal is fine" would need an explanation for where the memory goes.
