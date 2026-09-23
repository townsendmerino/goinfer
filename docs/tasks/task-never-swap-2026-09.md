# Task: never swap — file-backed weights by default, a firm cap where one is possible, and a tripwire where one is not (S0–S6) — 2026-09

> **Status: S3 BUILT AND MEASURED 2026-09-22 (the watch mechanism, the serving consumer, and the
> load-time consumer's plumbing + both server-side halves of the wiring, all unit-tested and
> committed as `84c70c38`; the real gpt-oss-20b positive-control run found a genuine, documented
> LIMIT — the guard detects on time but does not hold the machine under its own +1 GB bound
> unassisted, see S3's own status note below and `docs/measurements/swap-tripwire-2026-09-22.md`).
> **S1 BUILT AND MEASURED 2026-09-22** (darwin default-sidecar for serve/chat/fit, all three
> registered gates passed on real hardware — see S1's own status note below and
> `docs/measurements/sidecar-default-2026-09-22.md`). **S2 FIVE OF SIX FAMILIES DONE 2026-09-23**
> (gpt-oss, laguna, granite, nemotron and llama4 all moved off the resident-build fallback;
> gpt-oss alone has real fixture + real-checkpoint verification — swap-used flat throughout a real
> gpt-oss-20b transcode, though the run itself did not finish, disk-limited on this machine — the
> other four are backed by a new structural (AST-based) test proving no family reads a non-per-layer
> tensor, not by fixture byte-identity, since no fixture exists for any of them; see S2's own
> status note below and `docs/measurements/transcode-streaming-2026-09-23.md`), **S4/S5/S6
> unstarted.** S2's one remaining family, gemma4, has a genuinely different obstacle (a truly
> model-level fused PLE/MoE tail) the other five did not. **Correction to an earlier version of
> this line**: S2 was never actually a
> precondition for S3's OWN positive control — S3's gpt-oss-20b run (2026-09-22) already reached
> the true historical scenario directly (a plain resident load, no `-stream-weights`, S2 has
> nothing to do with that code path) and found its real limit on its own terms. S1+S2 together are
> instead what make that dangerous DIRECT path increasingly avoidable as ordinary usage, not a
> gate on S3's own already-completed measurement. S4, S6 and S5 follow. S6 is the structural fix
> for the Metal path the R11(c) runs of
> 2026-09-20/22 hit. One owner decision is flagged in S1 (what a MoE `.gguf` that will not fit
> resident does once the sidecar path exists: refuse, or load and warn).
>
> **What this is.** On 2026-09-04/05 the MacBook ran two different failures in one night and
> `benchmarks.md` files them a few paragraphs apart: a plain-CPU load of gpt-oss-20b drove swap to
> 22.6–22.9 GB of a 23.5 GB swapfile, and an M35/M26 `-stream-weights` run sat at ~3.2 GB RSS for
> 2h10 with zero completions and was followed by a watchdogd kernel panic. The first is anonymous
> memory — a whole model re-quantized into the Go heap — and the second is file-backed memory the
> OS was reclaiming faster than the SSD could refill it. S0 states the mechanism with the numbers
> the tree already holds — including the third failure, the Metal MoE pager on M26 swap-spiralling
> at every slot count on 2026-09-20/22 (R11(c)); S1–S6 are the briefs, each written to be handed to
> a session as it stands.
>
> **Siblings.** [`red-october.md`](red-october.md) R11 (the Metal pager cells this doc is a
> precondition for) · [`task-fit-to-hardware.md`](task-fit-to-hardware.md) (owns the fit guard S4
> amends) · [`../completed/task-moe-streaming.md`](../completed/task-moe-streaming.md) (owned the pager;
> Lever 1b is S5's pool mode; archived 2026-09-22) · [`task-int4-layout-2026-09.md`](task-int4-layout-2026-09.md) (kind 5, the on-disk policy
> S1 depends on) · [`task-download-and-load.md`](task-download-and-load.md) (what a load costs) ·
> [`../giw-bundles.md`](../giw-bundles.md) · `benchmarks.md` "G20" and "M35/M26 on the Mac" (the two
> incidents) · `docs/measurements/cold-user-2026-09-18-nobara-pc.md` Scenario D (the gpt-oss load
> peak, measured) · `docs/measurements/cold-user-2026-09-07-macbook-arm64.md` (the R13 live-probe
> origin: "70% of 16 GB was never actually free") · `docs/measurements/metal-moe-autopager-m26-2026-09-20.md`
> (R11(c): three Metal pager runs on M26, N=64/32/8, all swap spirals; the 1 s external kill script).

---

## S0. The mechanism — what is anonymous, what is file-backed, per load path

**Only anonymous pages reach the swapfile.** A read-only `MAP_PRIVATE` mapping of a file
(`aikit/mmap.MapReadOnly`) is clean and file-backed: under pressure macOS drops those pages and
re-faults the identical bytes later; nothing is ever written out. Go heap slices, KV caches,
scratch, and `StorageModeShared` Metal buffers are anonymous: under pressure they go to the
compressor and then the swapfile. So the question "what is getting written to swap" is answered
per load path by what ends up on the heap:

| load path | weights end up | anonymous term | file-backed term | the record |
|---|---|---|---|---|
| `.giw` (`decoder.Load` on a `.giw`, with or without `-stream-weights`) | zero-copy aliases of the mapping (kind 3 int8, kind 4/5 int4) | KV + scratch + norms/biases + (Metal) MTLBuffers | the whole weight blob | measured 2026-09-20: a 7.8 GB `.giw` loads as **1.46 GB anonymous + 7.4 GB file-backed** (`decoder/fitguard.go`'s `srcFileBytes` comment) |
| Metal resident, `.giw` source (`metal/model.go` `int4Buf` / `int4Concat`, paged experts via pread) | dense projections **copied twice** — `int4DirectWords` reinterprets the mapped nibbles into a fresh heap slice, `NewBufferUint32s` copies that into a `StorageModeShared` MTLBuffer, f32 scales converted to f16 on the way; routed experts stream by `pread` into N fixed device slots per layer | every dense projection as an MTLBuffer (~3–4 GB on M26) + the transient heap copy until GC + N × layers × per-expert slot bytes as they fill + f16 KV (~1 GB at ctx 4096 on M26) + scratch | the mapping (touched once at build, then reclaimable) | R11(c), 2026-09-20/22: swap 2.3 → 12 GB in ~16 s during build at N=64, 3.2 → 7.9 GB right after the first token at N=32, +341 MB before the 1 s kill at N=8, on a box with 55–210 MB free — while the test's own RSS read "7 MB → 892 MB after build" |
| `.gguf` direct (`loadGGUFWeights`) | **fresh heap copies** — every tensor dequantized out of the mapping and re-quantized into goinfer's own int4/int8 layout | the whole resident weight set (+ KV, scratch) | the source file, which stays mapped and fully touched for the entire parallel build | measured 2026-09-18/19 on gpt-oss-20b: **peak RSS ~24.5 GB = 12.58 GB new weights + 12.11 GB source**, dropping to ~13.0 GB after `Load` returns |
| safetensors dir (`openCheckpointMmap`) | f32 tensors may alias; quantized/converted tensors (MXFP4 experts, bf16→int4) are heap copies | most of the model for a quantized load | the mapping | `decoder/weights.go`, `mmapAliasRisk` |
| `.gguf` → sidecar `.giw` transcode (`prequant.Transcode`) | written to disk one layer at a time (`StreamTranscodeGGUF`) | ~one layer — **except** the `needsResidentSerialize` families (gemma4, gpt-oss, laguna, granite, nemotron, llama4), which build the whole model resident and serialize it once | the source | `decoder/gguf.go:1495`; the README's own gpt-oss `-stream-weights` example runs through this fallback |

**Why the M35/M26 streaming run did not swap but still killed the machine.** With the model
mapped, the pager's RAM budget on darwin is bookkeeping only: aikit's `mmap` package (`madvise_darwin`, cross-repo — described, not cited)
documents (verified empirically) that `MADV_DONTNEED`, `MADV_FREE` and the `msync` variants leave
RSS unchanged on a read-only file mapping, so eviction is a no-op and the Unified Buffer Cache
decides what stays resident. The auto budget makes it worse than it looks: `mmap.AutoBudget()`
reads `/proc/meminfo`, which does not exist on macOS, and falls back to a fixed 8 GB
(`decoder/moepaging.go:159`, `decoder/layerpaging.go:105`) — while goinfer already has a live
darwin probe, `HostRAMAvailableBytes` (`decoder/hostram_darwin.go`, from `vm_stat`), that the pager
never sees. A 20 GB model in ~11 GB of usable RAM therefore re-faults most of its active experts
from the SSD every token, continuously — the "RSS ~3.2 GB, re-reading weights from disk per token"
signature — and a sustained page-fault storm on a nearly full SSD is the shape of I/O starvation
that trips watchdogd. The record does not attribute the panic beyond that; S5's measurement is
where it gets attributed.

**Two guards exist and neither sees the `.giw` path.** `guardFit(fitCheckFor(...))`
(`decoder/model.go:451`) prices weights + KV + `srcFileBytes` for a `.gguf`, but the `.giw` branch
(`decoder/model.go:363`) returns before it — by design, since a mapped load has no allocation
peak to price; it also therefore prices none of the anonymous remainder (KV, scratch, Metal
buffers). Metal's own guard is a static 70% of `hw.memsize` (`metal/backend.go:139`,
`residentMemFraction`, set from one measured failure), deliberately not a live query because the
UBC makes "available" report what survived rather than what can be asked for — a stated reason S4
keeps rather than overrides.

**The R11(c) runs, read against the code — this is a reading, not a measurement; S6's first step
measures it.** The write-up concludes "this model class does not fit this machine's real headroom
at any N" and does not split the footprint into anonymous and file-backed. The code says what the
anonymous part is: `int4Buf` (`metal/model.go`) takes every dense projection's canonical int4
bytes from the mapping, copies them into a heap `[]uint32` (`bytesToU32`), converts the f32 scales
to f16, and then `NewBufferUint32s` copies again into a `StorageModeShared` MTLBuffer; `int4Concat`
does the same for the fused QKV and gate/up buffers. Nothing in `metal/` calls aikit's
`NewBufferNoCopy`, the API written for "alias an mmap'd `.giw` straight into Metal instead of
uploading a second GB-scale copy". So on Metal the model is mapped from disk and then *copied* —
the M-07 finding, now with its consequence: on M26 the dense (non-expert) term alone is ~3–4 GB of
anonymous memory before a slot is filled or a token decodes, on a machine that had a few hundred
MB free. The slot arithmetic is not the villain the N ladder made it look like — at ~3 MB per
expert (11.4 GB ÷ 128 ÷ 30), N=64 is ~5.7 GB of slots, N=8 is ~0.7 GB — which is why N=8 failed
too. And the test's "RSS 7 MB → 892 MB after build" is the inverting-guard shape from `CLAUDE.md`:
darwin RSS reports what survived reclaim, and on a box with 200 MB free the MTLBuffer pages were
being compressed and swapped as fast as they were written, so RSS never saw them while the
external swap monitor did. Two consequences for the briefs: S4's guard must price the MTLBuffer
copies against a *live* figure (the static 0.70 × `hw.memsize` passed a load on a machine with
<1 GB free), and S6 removes the copies.

**The one thing no cap fixes.** A per-token working set larger than the budget still reads from
the SSD every token. M35 is A3B: ~1.7 GB of active expert bytes per token at 0.5625 B/weight. A firm
cap turns the 2h10 run into a bounded-RAM run at SSD speed, not a fast one; S4 makes the guard say
so with arithmetic rather than let the run happen.

---

## The briefs

| # | brief | size | status |
|---|---|---|---|
| S1 | Sidecar `.giw` by default for every `.gguf` load on darwin (serve, chat, fit) | S–M | scoped; one owner decision |
| S2 | Streaming transcode sink for the six resident-serialize families | M | scoped; precondition for S1 on gpt-oss/gemma4 |
| S3 | Swap tripwire in the binary — abort a load, stop admitting requests, never RSS-keyed | S | scoped |
| S4 | The fit guard on the `.giw` path, the live darwin probe into the pager budget and Metal's guard, `GOMEMLIMIT`, the working-set warning | M | scoped |
| S5 | Pool mode as the darwin default for the MoE pager (measured first); a pread ring for dense layer streaming | S (measure) + M | scoped |
| S6 | Metal aliases the mapping: `NewBufferNoCopy` over a metal-target `.giw` layout, so dense weights on Metal are file-backed too | M–L | scoped; measurement step fundable now |

Every brief has the same shape: goal, standing and the registered decision rule, what to read
first, build, gates, measure, decision, record, out of scope. Rules inherited from
[`red-october.md`](red-october.md) §6 apply — pre-register anything with a band, keep the
do-nothing arm, record machine state beside every number, archive logs under `~/goinfer-logs/`,
progress logging on anything over a couple of minutes.

---

### S1 · Sidecar `.giw` by default for every `.gguf` load on darwin

> **BUILT AND MEASURED 2026-09-22.** `internal/prequant.DefaultToSidecar` is the platform policy
> (darwin default, opt-out via `-direct-load`/`GOINFER_GGUF_DIRECT=1`; linux keeps direct as its
> own default, same opt-out flag available there too, as item 1 asked). Wired into all three
> entry points: `internal/serveapp`'s `loadDecoder` (a new branch alongside the existing
> `-stream-weights` one — `StreamWeights` itself stays false for this default path, so no pager is
> built and the LoRA-adapter refusal, which keys on `StreamWeights` not on `.giw`-ness, is
> unaffected), `internal/chatapp`'s `loadFromPath` call site (chat had no streaming path at all
> before this), and `internal/fitcmd/fit.go` (item 5: reuse an already-fresh sidecar via the new
> `prequant.SidecarPathIfFresh`, which never itself transcodes — `fit` stays cheap). `--embed-int4`
> implies `-direct-load` for this path (item 3's "pick one" — the sidecar has no representation
> for the int4 embed/head pin yet); the explicit `-stream-weights --embed-int4` combination keeps
> its prior, unrelated, documented behavior. A disk-space `statfs` guard in `EnsureCachedGIW`
> refuses before a transcode starts when free disk is under the source file's own size (a
> conservative proxy — a sidecar at any real quant is never bigger than an f32 source) rather than
> risking a half-written sidecar on a full disk (M-12's own history).
>
> **All three registered gates passed on real hardware** —
> `docs/measurements/sidecar-default-2026-09-22.md` has the full run. Anonymous footprint of the
> sidecar load was 16–21% of the direct load's (bound: ≤25%), swap-used delta was 0 MB across
> every run, and greedy output was byte-identical between direct and sidecar loads on 3 prompts ×
> 64 tokens, on two real dense models (no phi3-mini checkpoint was available locally, so a second
> model was substituted — same methodology, not a weaker one). A real confound was found and fixed
> mid-measurement: a DIFFERENT, pre-existing mechanism (`task-fit-to-hardware.md`'s CPU-placement
> auto-retry) silently routed a "direct" arm through a sidecar anyway when this machine's tight
> free RAM tripped the static fit guard — caught from the server's own log line, not assumed away,
> and the measurement re-run correctly isolated. `internal/prequant/sidecar_identity_test.go` and
> `sidecar_default_test.go` carry the permanent regression coverage (heavy-gated where a real
> tokenizer-bearing checkpoint is required, matching this file's own established pattern); the
> identity comparison was itself checked against a genuine int8int8-vs-int4 mismatch to confirm it
> is not vacuously passing.
>
> **Owner decision (item 6), still flagged, not decided — and now confirmed out of reach until
> S2:** once a MoE `.gguf` can be mapped without a resident build first, the fit guard's MoE
> exclusion needs a real answer (refuse / warn-and-load / refuse-without-explicit-flag). It cannot
> be decided yet because the precondition (S2) is not built — every `needsResidentSerialize`
> family (gpt-oss included) still requires a full resident pass to produce a sidecar at all today,
> so S1's own darwin default does not yet reach gpt-oss-20b, the model this whole task is written
> against.
>
> **What is NOT done:** the adapter-parity gate item does not apply — confirmed, not assumed:
> `decoder.(*Model).LoadAdapter` refuses any base whose `w.schema == nil` ("compute-time LoRA
> needs a safetensors base"), and a sidecar is exclusively a `.gguf`-derived artifact, so a
> sidecar-loaded base can never reach `-adapter` in the first place — the combination the brief
> asked to verify cannot occur. A third interleaved measurement run (brief names three; two ran).
> `benchmarks.md` Table 1's cold-start/footprint row does not yet carry this measurement's numbers.

**Goal.** A `.gguf` given to `goinfer-serve`, `goinfer-chat` or `goinfer-chat fit` on darwin is
transcoded once to its sidecar `.giw` and mapped, so the resident weights are file-backed and the
`.gguf` direct heap load becomes the opt-out, not the default.

**Standing and the registered rule.** Today the sidecar is built only under `-stream-weights`
(`internal/serveapp/main.go`, `ensureGIW` → `prequant.EnsureCachedGIW`,
`internal/prequant/prequant.go:203`) or by the dense fit-guard auto-retry
(`internal/serveapp/main.go:1301`); `chat` (`internal/chatapp/main.go:389`) and `fit`
(`internal/fitcmd/fit.go:97`) load direct and have no streaming flag at all. **Rule (Mac, 1.5B and
gpt-oss-20b, `footprint`/`vmmap -summary` on the serving process after the first completion):
anonymous footprint of a sidecar load ≤ 25% of the direct load's, swap-used delta across the load
= 0, and the greedy token stream byte-identical to the direct load on 3 prompts × 64 tokens. All
three or the switch does not ship.** Speed is not a criterion here (S1 is a memory item); record
cold and warm TTFT anyway so a regression is visible.

**Read first.** `internal/serveapp/main.go` from `ensureGIW` through the auto-retry (the M-30 note
on the safetensors no-op; the embed-int4 note; the `DenseStreamable` exclusion and its cited
reason — "MoE CPU weight streaming is a documented, MEASURED failure mode"); `internal/prequant/prequant.go`
`Transcode` (temp + rename, the V-01 `.tmp.giw` suffix trap, `cacheFresh`'s load-probe freshness —
M-12/M-11); `decoder/gguf.go` `StreamTranscodeGGUF` and `needsResidentSerialize`
(`decoder/gguf.go:1495` — read S2 before promising anything about those families);
`decoder/model.go` `.giw` branch (`decoder/model.go:363`) and what it skips (`decoder/model.go:451`);
`decoder/weightmat.go` `GIWTargetForBackend` and the kind-5 policy in `docs/tasks/task-int4-layout-2026-09.md`
L2 (a cpu-arm64 sidecar is row4-only — about the model's int4 size; a kind-4 dual-representation
bundle is ~2× that and is what "we shouldn't be building bigger files" refers to — check which
kind the `metal` target writes before defaulting Metal loads through this path);
`docs/giw-bundles.md`; `docs/measurements/cold-user-2026-09-18-nobara-pc.md` Scenario D; the
`chat -model-tmp` flag (`internal/chatapp/main.go`, the embed build's stream-to-temp-then-mmap
precedent for the same idea).

**Build.**
1. One shared helper (the existing `ensureGIW` shape) used by all three entry points; policy:
   on darwin, a `.gguf` source resolves to its sidecar unless `--direct-load` (or
   `GOINFER_GGUF_DIRECT=1`) is set; on linux the default stays direct for now (the Linux box has
   62 GB and the measured peak fits it; make the flag available there too, so the same command
   works on both). The served model name still derives from the original `--model` spec, exactly
   as `-stream-weights` does now.
2. Sidecar location and naming unchanged (`<base>.<quant>.<target>.giw` beside the source;
   `~/models` on the Mac is already `tmutil`-excluded). Print the one-time "transcoding … minutes
   + ~model-size on disk" line before starting, and refuse with a clear message if free disk is
   below the projected sidecar size (a `statfs` check — a half-written sidecar on a full SSD is
   the failure this repo has already had once).
3. `--embed-int4`: the sidecar keeps the int8 pin today; either bake it into the sidecar name
   (`<base>.<quant>.embedint4.<target>.giw`) or keep the current note and have `--embed-int4`
   imply `--direct-load`. Pick one; document it in `--help`.
4. Compute-time LoRA: `loadAdapters` refuses `--stream-weights` because the pager mutates
   per-layer state under a shared model. A sidecar load *without* `-stream-weights` builds no
   pager (`newExpertPager`/`newLayerPager` are only constructed under `opts.StreamWeights`), so
   the refusal should key on `StreamWeights`, not on the source being a `.giw`. Verify with the
   adapter parity test on a sidecar-loaded base.
5. `fit` on a `.gguf` that already has a fresh sidecar should measure through the sidecar (a
   mapped load, lazy faults) instead of a 32 s / 256 CPU-s resident build — the cold-user report
   called that call "not free"; it becomes nearly free.
6. **Owner decision (flag it, do not decide it in the brief):** once a MoE `.gguf` can be mapped,
   the fit guard's MoE exclusion (`DenseStreamable == false` → refuse) is no longer about a heap
   peak. Options: (a) keep refusing when the working set will not fit (S4's arithmetic), (b) load,
   warn with the predicted tok/s, and let the user decide, (c) refuse only without an explicit
   `--stream-weights`. The M35 2h10 run is the case each option has to handle.
7. Docs: `--help` for the three binaries, `README.md` "run bigger than my hardware"
   (the gpt-oss example at line ~216 currently recommends `-stream-weights` on a path that will
   not stream until S2 lands — say so or reorder), `docs/quantization.md`, `docs/env-vars.md`,
   `docs/tasks/task-fit-to-hardware.md`'s CPU placement piece (the auto-retry collapses into "the
   sidecar is the default").

**Gates.** `TestStreamTranscodeMatchesResident` (`internal/prequant/stream_test.go`) stays the
bundle-level byte identity; add a stream-identity test through the real entry point
(`TestGenerate_batchedPrefillMatchesSequential` is the shape — same prompt, greedy, sidecar vs
direct, on `testdata/`-class fixtures under CI and on the real 1.5B under `GOINFER_HEAVY_TESTS`);
the adapter parity test on a sidecar-loaded base; the existing `cacheFresh` self-check path
exercised by deleting the sidecar mid-test and by truncating it.

**Measure (Mac, both arms interleaved, three loads each).** `footprint <pid>` or `vmmap -summary
<pid>` after the first completion for the anonymous / file-backed / compressed split; `sysctl -n
vm.swapusage` before and after each load (the tripwire in S3 makes this automatic later);
`GOINFER_DECODE_TIMING=1` cold TTFT (first request after load) and warm; on 1.5B, phi3-mini and —
after S2 — gpt-oss-20b, the model that produced the 22.9 GB. Record loadavg and the disk free
figure beside every number.

**Decision.** As registered. A byte-identity miss between sidecar and direct is a defect, not a
band (the two paths quantize the same tensors with the same code; a difference means one of them
does not).

**Record.** `docs/measurements/sidecar-default-2026-MM-DD.md`; `benchmarks.md` Table 1's
cold-start/footprint row gets a Mac `.gguf` line with both arms; `docs/tasks/task-fit-to-hardware.md`
and this doc's status table.

**Out of scope.** Safetensors directories (`transcodeDir` still loads resident — a separate item
if a safetensors-only model ever matters on the Mac); Linux defaults; anything about speed.

---

### S2 · Streaming transcode sink for the six resident-serialize families

> **FIVE OF SIX FAMILIES DONE 2026-09-23 — only gemma4 remains.** `needsResidentSerialize` now
> names gemma4 alone. gpt-oss, laguna, granite (the Mamba-2+MoE hybrid, `arch.granite` — not the
> plain dense Granite family `gguf_granite_permute_test.go` covers), nemotron and llama4 all
> turned out the same way once actually read: every one of their per-layer closures was already
> self-contained (every tensor `blk.{i}.*`-named, including per-layer router/bias/sink fields this
> brief worried might be model-level tails — they were not, for any of these five), so the
> `loadQ35` streaming shape (build → write → release, 2026-08-24) applied with no new design work.
> This brief's own "Read First" concerns (gpt-oss's `RowDequantizer`, the others' "per-layer block
> kinds") turned out not to block anything once actually read — only gemma4's genuinely different
> obstacle (a fused, truly model-level PLE/MoE tail, not per-layer data in the wrong place) held.
>
> **Verification is NOT the same depth for all five, and that is stated rather than blurred.**
> gpt-oss alone has real fixture coverage: `TestGptOss_streamedMatchesResident`
> (`internal/prequant/stream_families_test.go`) proves streamed-vs-resident byte-identity across
> int4/int8int8/f32 on `decoder/testdata/gptoss_tiny.gguf`, mutation-checked, PLUS a real
> `gpt-oss-20b` transcode showing swap-used flat throughout
> (`docs/measurements/transcode-streaming-2026-09-23.md`; the run itself was disk-limited before
> finishing, a capacity issue on this machine, not a streaming-fix failure — a bug in the
> monitoring script's own cleanup, caught and fixed the same session, also recorded there). Laguna,
> granite, nemotron and llama4 have **no** comparable fixture — no small, tokenizer-bearing,
> architecture-correct GGUF for any of them exists in this repo or under `~/models`, and building
> one per family from scratch (each needs a different Mamba-2/MoE tensor set) was judged real,
> separate work this pass did not do. What DOES cover all six families (the five newly-streaming
> ones plus gpt-oss and qwen35 as controls) is a new structural test,
> `decoder/gguf_streaming_shape_test.go`'s `TestStreamableFamilyClosures_onlyReadPerLayerTensors`:
> it parses `gguf.go`'s own source (the same AST technique `stream_test.go`'s
> `TestTranscode_writesViaTempThenRenames` already uses for a property real execution can't force)
> and asserts every tensor-name argument each closure passes to `mat`/`vec`/`vnorm`/`streamMat`/
> `stackedExperts`/`stackedExpertBias`/`flat` is `p+"..."`-prefixed — never a bare, model-level
> name. Mutation-checked (injecting one bare-literal read into `loadLaguna` turns it red). This is
> a real, meaningful, automated guard against exactly the failure mode that would corrupt a
> streamed bundle — but it is a STRUCTURAL proof, not a numeric one; it cannot catch a bug that
> reads the RIGHT tensor name at the WRONG index, only a bug that reads the wrong SCOPE of tensor
> entirely. `TestParityManifest_fresh`'s staleness on `decoder/gguf.go` closed via the sanctioned
> non-numeric refresh both times (gpt-oss alone, then the other four together), each backed by
> real evidence (gpt-oss's byte-identity proof; the four others' unchanged resident-path forward-
> parity tests all still green, since this change only ADDS a new `sink != nil` branch and never
> touches the existing `sink == nil` path those goldens exercise).
>
> **What is NOT done:** gemma4 (the one family with a genuinely different, harder obstacle — not
> attempted). Real or synthetic fixture-based byte-identity proof for laguna/granite/nemotron/
> llama4 specifically (the structural test is real evidence, not a substitute for it). A COMPLETED
> real gpt-oss-20b transcode (needs more free disk than this machine currently has). The brief's
> own three-run RSS-peak averaging. `docs/giw-bundles.md` not yet updated to say five families
> stream now.

**Goal.** `StreamTranscodeGGUF` bounds peak RAM to ~one layer for gemma4, gpt-oss, laguna, granite,
nemotron and llama4, the way it already does for every other family, so S1 holds on the models
that actually exceed the Mac.

**Standing and the registered rule.** `needsResidentSerialize` (`decoder/gguf.go:1495`) routes
those six through `buildWeightsFromGGUF(..., needCanonical=true, ...)` then
`SerializeWeightsToForTarget` — the whole model resident, then one serialize. For gpt-oss-20b that
is the same ~12.6 GB heap the direct load builds, on the box where that meant 22.9 GB of swap. The
qwen35 precedent is the fix shape: "its own `loadQ35` already builds one layer at a time
internally, so `buildWeightsFromGGUF`'s `sink != nil` branch streams it like every other family —
a control-flow fix (write + release each layer instead of holding all of them until one final
serialize), not a numerics one" (`decoder/gguf.go`, the comment above the fallback). **Rule: peak
RSS of a transcode ≤ 1.5 GB above the process baseline on gpt-oss-20b and on a gemma4 checkpoint
(measured with RSS sampling every 200 ms, the `scripts/bench_peer.py` `rss_peak` method), AND the bundle
byte-identical to the resident-serialize path's output on the family fixtures (the existing
`TestStreamTranscodeMatchesResident` shape, extended per family).**

**Read first.** `decoder/gguf.go` — the qwen35 branch (`loadQ35`, the `sink != nil` streaming
loop near `decoder/gguf.go:1775`), then each of the six loaders and *why* it was excluded: gemma4's
"fused PLE/MoE tail can't stream incrementally" (a model-level tail written after the layers —
the head/tail split `writeHeadGlobals` already supports: `decoder/serialize.go`'s "streaming
transcode can emit the head, then produce-write-free each layer" note), gpt-oss's stacked experts
via `RowDequantizer` (`decoder/gguf.go` ~line 1999), granite/nemotron/llama4/laguna's per-layer
block kinds. `internal/prequant/stream_families_test.go` (`TestStreamTranscode_perFamilyBodiesCarryTheirLayers`
— the per-family body check to extend). The kind-5 emission rule in `decoder/serialize.go`
(`weightMat` vs `weightMatKind3Only`; "the writer needs canonical bytes IN RAM to choose what to
write" — per layer, that is one layer's canonical bytes, which is fine).

**Build.** Per family, in this order (Mac relevance first): gpt-oss, gemma4, then the four others.
Each loader gains a per-layer sink call after its layer is complete and releases the layer before
building the next (the `loadQ35` shape); model-level tails (gemma4 PLE, router biases, sinks) go
through the head/tail writers, not through a resident `Weights`. `needsResidentSerialize` shrinks
family by family and is deleted when empty; the M-09 lesson in the comment above it (a stale
refusal comment hid a whole class) means the comment moves with the code.

**Gates.** Byte-identical bundles vs the resident path on every family fixture in `testdata/`
(mutation-check by corrupting one layer's sink write and confirming the diff); the sidecar loads
and matches the direct load's greedy stream on the real checkpoint (S1's gate); `TestStreamTranscode_ctxCancel_M21`
still cancels at a layer boundary for the new branches.

**Measure.** RSS peak of the transcode on the Mac for gpt-oss-20b and `gemma-3-4b-it`/a gemma4
checkpoint, three runs; transcode wall time (it should not get slower — the work is the same,
reordered); disk free before/after.

**Decision.** As registered; a family that cannot be made to stream (a real tail dependency)
stays on the fallback with the reason written next to `needsResidentSerialize`, and S1's docs say
that family transcodes on the Linux box (`models-pull` afterwards — the store-remote/bench-local
rule already covers moving bundles).

**Record.** `docs/measurements/transcode-streaming-2026-MM-DD.md`; `docs/giw-bundles.md` (which
families stream); the README gpt-oss example un-caveated.

**Out of scope.** `transcodeDir` (safetensors); the bundle format (no new kind, no version bump
unless a tail needs a field — if it does, v12 with the usual reader guard).

---

### S3 · Swap tripwire in the binary

> **BUILT 2026-09-22: the watch mechanism and the serving consumer. The load-time consumer is
> explicitly deferred, not built — read why before picking this up.**
>
> `decoder.SwapWatch`/`StartSwapWatch` (`decoder/swapwatch.go`) is the portable mechanism: samples
> a pluggable `SwapReader` every `PollInterval` (default 2s), captures the FIRST successful
> reading as baseline (so a machine already deep in swap from something else is measured on what
> grows AFTER the watch starts, not its pre-existing state), calls `OnTrip` once when
> `used − baseline > Threshold` (default 512 MiB), and `OnResume` once after staying within
> threshold continuously for `ResumeAfter` (hysteresis, so it does not flap). `decoder.SwapUsedBytes()`
> is the real platform probe — `sysctl -n vm.swapusage` on darwin, `/proc/meminfo`'s
> `SwapTotal − SwapFree` on linux, always-unknown elsewhere — split from the decision logic
> (`hostram_darwin.go`'s own convention) so both are independently unit-tested against real,
> committed sample output, not just against each other.
>
> **The serving consumer is wired and tested end to end.** `internal/serveapp/swapguard.go`'s
> `startSwapGuard` arms a real watch at server start (after every startup load, so the baseline is
> "steady state," not mid-load); a trip sets `server.swapGuardTripped`, which `haltGate`
> (`halt.go`) now checks as a SECOND, independent condition alongside K2's halt state — reusing
> the exact admission chokepoint every generation route already goes through (`auth(srv.haltGate(inf(...)))`,
> 9 call sites), rather than adding a tenth wrapper at each one. Unlike a K2 halt, a swap-guard
> trip never calls `s.gens.cancelAll` — only new admissions are refused, in-flight generations run
> to completion, exactly this brief's own "lets in-flight generations finish." `GOINFER_SWAP_GUARD`
> sets the threshold in MB or disables the guard entirely (`=off`); documented in `docs/env-vars.md`.
>
> **Verified, not assumed.** `decoder/swapwatch_test.go`: baseline capture (including the
> already-deep-in-swap case), the ramp (fires once, not on every tick above threshold — confirmed
> by mutation: removing the once-guard turns `TestSwapWatch_ramp` red), hysteresis (a dip that
> doesn't hold resets the clock rather than resuming early), unavailable readings are skipped
> rather than guessed, the off switch, ctx-cancellation stopping the goroutine — and the
> INVERTING-GUARD mutation check the brief itself asked for
> (`TestSwapWatch_keyingOnRSSWouldHaveMissedR11c`): fed the R11(c) run's own real RSS trajectory
> (7 MB → 892 MB → falling, as darwin reclaimed the MTLBuffer pages under pressure) against its
> own real swap-used trajectory (2.3 → 12 GB, never recovering) through the identical watch — an
> RSS-keyed watch resumes mid-spiral (the inversion CLAUDE.md warns about, shown directly); the
> swap-keyed one never does. `internal/serveapp/swapguard_test.go` drives the trip through the
> REAL `haltGate` (503 with the swap-guard reason named, then a manual clear back to 200) and
> confirms directly that a trip never touches `s.gens` (an in-flight generation's own context is
> never cancelled). The full existing `TestServe_haltUnderLoad` integration test — real
> `newServer`, real model, real K2 halt under concurrent load — still passes unchanged through the
> edited `haltGate`. All new/touched code: `gofmt` clean, `go vet` clean (native + linux cross),
> CI's pinned `staticcheck` clean, `-race` clean.
>
> **A real bug caught while wiring this, fixed before it shipped rather than in it:**
> `startSwapGuard`'s first draft read the probe ONCE itself for a startup banner before starting
> the watch — harmless for a real time-sampled OS probe, but it silently shifted a scripted test
> reader's sequence by one entry against the watch's own index, and the shifted numbers happened
> to land the "trip" delta just under the threshold instead of over it: a test that looked
> deterministic and was actually order-dependent on an implementation detail invisible from the
> test itself. Fixed by folding the banner into the watch's own first reading rather than a
> separate out-of-band call — better for production too (one real syscall instead of two).
>
> **A second thing caught, before it could make CI flaky rather than after:** `newServer` calls
> `startSwapGuard` unconditionally, and ~17 existing test files construct a real `*server` via
> `newServer` for reasons that have nothing to do with this guard. Left unguarded, each would
> start an independent real background goroutine polling actual OS swap-used every 2s for the rest
> of that test binary's life — on a real, often-loaded development box, a genuine risk of some
> OTHER test's handler spuriously 503ing through `haltGate` if the machine's real swap happened to
> move during a run. Fixed with `testing.Testing()` (the stdlib's own Go 1.21+-sanctioned way to
> ask "is this running under `go test`" from production code, confirmed to add no side effects —
> checked directly, not assumed: the real `goinfer-serve` binary's `--help` output was diffed
> before/after and gained no `-test.*` flags): under test, the guard stays unarmed UNLESS a test
> explicitly sets `GOINFER_SWAP_GUARD` itself (which `swapguard_test.go`'s own tests do, by calling
> `armSwapGuard` — the fully-injectable core — directly rather than through `startSwapGuard`).
>
> **UPDATE 2026-09-22: the load-time consumer's plumbing and both server-side halves are now
> built and unit-tested; the real gpt-oss-20b positive-control gate is the one thing still
> outstanding.** `~/models/gpt-oss-20b-MXFP4.gguf` (12.1 GB, MXFP4) is now on this Mac, rsync'd
> byte-exact from the archive, freeing up the reason the prior pass stopped here. Built since:
> - `decoder.Options.LoadAbort <-chan struct{}` (a field, not a signature break, exactly as this
>   note originally called for) is threaded through `Load` → `loadWeights` → `loadGGUFWeights` →
>   `buildGGUFWeights` → `buildWeightsFromGGUF` → every one of `parallelLayers`' 8 call sites inside
>   `buildWeightsFromGGUF`, plus `LoadGGUFBytes`'s own entry point. `parallelLayers` itself gained
>   the `abort` parameter and checks it AT GRAB (between layers, never inside one already running):
>   closing it stops new `fn(i)` calls, lets any already-grabbed one finish, and returns
>   `errLoadAborted` (`errors.Is`-checkable as `decoder.ErrLoadAborted`) UNLESS a real build error
>   also occurred, which always wins (`decoder/weights.go`'s own doc comment on `parallelLayers`
>   states this ordering explicitly).
> - **Scope, deliberately narrow and stated rather than hidden**: only the GGUF direct-build
>   resident path checks `LoadAbort` today. The safetensors direct-build path (6 call sites,
>   `decoder/weights.go`) and `StreamTranscodeGGUF`'s own transcode path (which already has M-21's
>   ctx-cancellation at a different granularity) pass `nil` explicitly, with inline comments naming
>   this doc as the reason — matching the historical incident and this checkpoint exactly, not
>   silently short of the stated brief.
> - **Unit-tested, not just built.** `decoder/parallellayers_abort_test.go`: nil-abort no-op,
>   pre-closed-abort runs nothing (both the `n==1` fast path and the worker-pool path), mid-build
>   abort lets an in-flight layer finish and starts no new one (forced to `GOMAXPROCS(1)` for
>   determinism), a real build error wins over a concurrent abort (constructed so both flags are
>   genuinely set, not just the trivial single-worker case), and an end-to-end
>   `LoadGGUFBytes(Options{LoadAbort: ...})` plumbing check. **Mutation-checked**: breaking the
>   worker-pool's abort-observe line turns 3 of these red immediately (confirmed, then reverted).
> - **The server-side wiring is also built and unit-tested**, not just the decoder-internal half.
>   `internal/serveapp/swapguard.go` gained `armLoadSwapGuard`/`startLoadSwapGuard` (mirroring
>   `armSwapGuard`/`startSwapGuard`'s own injectable-reader / `testing.Testing()`-suppressed split):
>   armed fresh around exactly one `decoder.Load` call (never the process lifetime — a load either
>   finishes or aborts, matching `SwapWatchOptions.ResumeAfter`'s own doc comment, which anticipated
>   this consumer with `ResumeAfter: 0`), scoped to `.gguf` sources that are not already
>   `-stream-weights` (the only case `LoadAbort` does anything). A new `decoder.FitDescribe`
>   (`decoder/fitguard.go`) exposes the fit guard's own priced-terms sentence
>   (`fitCheck.arithmetic()` — "X needs ~N GB resident at quant Q + M GB reading the checkpoint =
>   T GB...") so the swap-abort wrap can name resident-weight and mapped-source bytes without
>   duplicating decoder's pricing, matching this section's own "the error names the priced terms
>   from `fitCheckFor`" requirement. `internal/serveapp/loadswapguard_test.go` drives this against a
>   scripted reader (same convention as `TestSwapGuard_tripsAndResumesThroughHaltGate`), confirming
>   the trip closes the abort channel and `wrapErr` turns `ErrLoadAborted` into a message naming the
>   swap delta — **mutation-checked**: dropping the `close(abortCh)` line turns the trip test red
>   (timeout), confirmed then reverted. The full `decoder` and `internal/serveapp` suites pass under
>   `-race`; `gofmt`, `go vet` (native + linux cross), and CI's pinned `staticcheck` are all clean.
> - **Committed and pushed** as `84c70c38`; CI and `govulncheck` both green.
> - **The positive-control gate ran 2026-09-22 — result: a genuine, documented LIMIT, not a clean
>   pass.** `docs/measurements/swap-tripwire-2026-09-22.md` has the full run. Short version: the
>   in-process guard's `OnTrip` fired at baseline+0.72–0.80 GB (inside the registered "+1 GB"
>   bound), but real swap-used kept climbing for several seconds after — a burst of ~400–550 MB/s
>   once dequant work started outran both the guard's 2s poll and an independent external
>   kill-switch's 1s poll, peaking at baseline+1.69 GB before the kill-switch had to `SIGKILL` the
>   process. The machine recovered fully and immediately (no hang, no panic) — a materially better
>   outcome than the historical 22.9 GB incident, roughly an order of magnitude less exposure — but
>   the mechanism did NOT hold the line at +1 GB unassisted. Likely cause (reasoned, not directly
>   instrumented): `parallelLayers` deliberately lets an already-grabbed layer finish rather than
>   cancelling it mid-write, and several `GOMAXPROCS` workers were most likely still mid-dequant on
>   gpt-oss-20b's large per-layer MoE tensors when the trip fired. Reaching this test at all also
>   required `GOINFER_NO_FIT_GUARD=1` — the static fit guard refuses this exact load cleanly with
>   zero swap growth by default (`-fit=off` alone does NOT bypass it, confirmed the hard way on a
>   first attempt), so the tripwire is only ever the last line of defense on this machine for this
>   model, never what a normal invocation hits. Per explicit instruction: record the limit, do not
>   chase a fix under further real-hardware risk this pass — `Options.LoadAbort` and its unit tests
>   stand as built (they are correct on the narrow, already-proven contract; the gap is what a real
>   burst can do to a coarse-grained, poll-based backstop, not a bug in the dispatch logic itself).
> - **Still not done.** The two negative controls (sidecar 1.5B, 7B Metal, 100 completions each
>   never tripping); a direct per-worker allocation trace to confirm the in-flight-drain theory
>   rather than infer it from timing; `chat` (item 3 of Build, below) has no load-time guard wired
>   at all — only `internal/serveapp`. A real fix, if picked up later, would need either bounding
>   worker concurrency once armed or a finer-grained abort check inside a huge-tensor family's own
>   layer build — neither designed here.

**Goal.** goinfer notices swap growing during its own load or serving and acts before the machine
thrashes — abort the load with a message naming the term, or stop admitting requests — instead of
leaving the kill-on-sight rule to a human with `free -m` in another terminal.

**Standing and the registered rule.** The cold-user harness applied "kill at swap-used baseline +
500 MB" by hand and it fired twice on the Linux box with 37 GB free; nothing in the binary reads
swap at all outside one test helper (`swapUsed` in `decoder/mellum2_prefill_profile_test.go`, which
already parses `sysctl -n vm.swapusage`'s `used = …M` field). **Rule: on the Mac, a direct
gpt-oss-20b load (the positive control) trips and aborts before swap-used exceeds baseline + 1 GB,
with the abort message naming the resident weight bytes and the source-file term; a sidecar 1.5B
load and a 7B Metal load (negative controls) never trip across 100 completions.**

**Read first.** `docs/measurements/metal-moe-autopager-m26-2026-09-20.md`'s third-attempt section —
the external shell script that polled `vm.swapusage` every 1 s and SIGKILLed the test on two
consecutive >80 MB ticks (or one >300 MB jump, or 85% of swap total, or free pages critically low)
cut the peak excursion 5–7× against manual monitoring; that script *is* the prototype for this
brief, moved inside the process and given a way to act short of SIGKILL. `CLAUDE.md` "A guard that
INVERTS under the condition it exists for" — RSS on
darwin reports what survived reclaim, which is exactly why this guard keys on *swap-used* (a
kernel figure that only grows when the machine is losing) and never on RSS; `decoder/fitguard.go`'s
`srcFileBytes` comment (the numbers an abort message should print); `decoder/hostram_darwin.go`
(the shell-out convention for sysctl on the pure-Go root module: `syscall.Sysctl` truncates at the
first NUL, `vm.swapusage` is a struct — shell out once per poll like `HostRAMBytes` does, or parse
`sysctl -n vm.swapusage`'s text as the test helper does); the serve admission path
(`internal/serveapp`'s fair-admission queue from `task-work-queue-2026-09.md` J1 — the tripwire
is one more reason to return 529); `docs/measurements/cold-user-2026-09-18-nobara-pc.md`'s "no
warning printed first" finding, which is the user-facing bar.

**Build.**
1. A new `memwatch_darwin` file in `decoder/` (+ a linux twin reading `SwapFree` from `/proc/meminfo`, and a
   no-op elsewhere): `StartSwapWatch(ctx, opts)` samples every 2 s, records the baseline at
   first sample, and calls back on `used − baseline > threshold` (default 512 MB, env-tunable,
   `GOINFER_SWAP_GUARD=off` disables; the value is a *swap* delta, so an idle machine already
   deep in swap from something else has a baseline that includes it — the guard measures what
   this process adds).
2. Two consumers. During `decoder.Load`: the callback cancels the load's context (the transcode
   already observes ctx per layer — M-21; the direct build needs a check between layers in
   `parallelLayers`) and the error names the priced terms from `fitCheckFor` so the message reads
   "aborted: swap grew 0.9 GB during load — resident weights 12.6 GB + mapped source 12.1 GB on a
   16 GB machine; use the sidecar path / a smaller quant". During serving: the callback flips the
   admission gate to refuse new requests with 529 and a body naming the cause, lets in-flight
   generations finish, logs once, and re-opens when swap-used returns to within threshold of the
   baseline for 30 s (hysteresis, so it does not flap).
3. The banner prints the baseline swap-used at start, so a user who is already 14 GB into swap
   sees it (the Linux box's stale 14.7 GB baseline in the cold-user run was itself news).
4. `chat` gets the load-time half only (no admission gate to flip).

**Gates.** Unit tests with injected readings (baseline, ramp, hysteresis, off switch, the
"machine already swapping at start" case); a mutation check that keying on RSS instead of swap
does *not* fire on the darwin fixture where RSS falls under reclaim (the inverting-guard shape,
encoded as a test so it cannot come back); the abort path leaves no `.tmp.giw` and no partial
sidecar (S1's temp+rename already guarantees the second).

**Measure.** The three real-machine controls named in the rule, on the Mac, swap-used sampled
externally as well (`sysctl` in a loop to a log) so the guard's own reading is checked against an
independent one; time from first swap growth to abort.

**Decision.** As registered. A positive control that trips *late* (after >1 GB) is a threshold or
poll-interval finding, not a design failure — record the number and adjust with the mechanism
named.

**Record.** `docs/measurements/swap-tripwire-2026-MM-DD.md`; `docs/env-vars.md`; `docs/server.md`
(the 529 reason); `docs/tasks/task-first-hour.md`'s protocol gains "watch the banner's swap line".

**Out of scope.** Memory pressure notifications via `dispatch_source`/`libproc` (cgo or purego
into libSystem — the root module stays pure Go; the 2 s poll is enough for a load that takes
minutes); Metal VRAM (unified — swap is the signal there too).

---

### S4 · The fit guard on the `.giw` path, live probes where they belong, `GOMEMLIMIT`, and the working-set warning

**Goal.** Every load path prices the memory it will actually make anonymous, against a live
figure on darwin where one is trustworthy, and a load whose per-token working set cannot fit says
what speed that implies before it runs.

**Standing and the registered rule.** The `.giw` branch skips `fitCheckFor` (S0); the pager's
auto budget is a Linux-only probe with an 8 GB darwin fallback (S0); Metal's guard is static by
design; nothing sets a Go memory limit. The R13 follow-on added `HostRAMAvailableBytes` because "70%
of 16 GB was never actually free" pushed a load into 9.7 GB of swap. **Rule: (a) a `.giw` load on
the Mac whose KV + scratch + (Metal) buffer projection exceeds `HostRAMAvailableBytes` minus a
1 GB margin is refused or auto-pinned to a smaller context exactly as the `.gguf` path already does
(`smallerFittingContext`), and the refusal message prints the projection and the probe; (b) the
pager's budget on darwin comes from the live probe (half of available, clamped as today), never
the 8 GB fallback; (c) the M35 case prints a predicted tok/s from active bytes/token, the budget
and a measured pread rate, and requires an explicit acknowledgement (`--stream-weights` plus
`--accept-slow`, name to taste) below 2 tok/s. Each is checked by a unit test driven with the
2026-09-04 numbers (20.6 GB model, 11.2 GB budget, 1.7 GB/token) and by one real Mac run.**

**Read first.** `decoder/fitguard.go` in full (the `fitCheck` struct, `srcFileBytes`,
`cudaBuildBytes`, `smallerFittingContext`, the R13 re-pricing); `decoder/model.go` around
`decoder/model.go:363`–`decoder/model.go:451`; `decoder/hostram_darwin.go` (the approximation it
states: free + inactive + speculative + purgeable, 16 KB pages read from `vm_stat`'s header);
`decoder/backend.go` `RegisterMemoryProbe` and `metal/backend.go` `residentMemFraction` (the
reason the Metal probe is static — keep it as the *ceiling* and add the live figure as a second
bound: `min(0.70 × hw.memsize, HostRAMAvailableBytes − margin)`, so the guard can only get stricter,
never looser, which is the direction the one measured failure allows); `decoder/moepaging.go`
`newExpertPager` (budget clamp `[one expert, total expert bytes]`, `AutoBudget` at
`decoder/moepaging.go:159`); `docs/completed/task-w4a8-neon-bandwidth.md` (the pread rate this
Mac measured — ~3.7 GB/s at concurrency 1 — as the working-set arithmetic's default until the
guard measures its own, one 64 MB pread at load); `runtime/debug.SetMemoryLimit` semantics (a
soft limit: the GC works harder under it, it does not refuse allocation — document it as such;
never set it below the projected live heap or the GC will spin).

**Build.**
1. `fitCheckFor` gains a `.giw` mode: weights priced as file-backed (0 anonymous, reported
   separately), KV at the effective context, scratch from the existing per-layer terms, Metal
   buffers when `opts.Backend == "metal"` (the resident build's own byte count — it already knows
   it to decline), heap slack at the measured ~9%; called from the `.giw` branch before the
   pager/resident build, against `HostRAMAvailableBytes` on darwin and `AvailableRAM` on linux.
2. `WeightCacheBytes == 0` resolves in goinfer (`decoder/model.go`, before `newExpertPager` /
   `newLayerPager`) from the live probe, with the same clamp; the aikit fallback stays as the
   last resort and is logged when used.
3. `metal/backend.go`'s probe returns the min of the static ceiling and the live figure; the
   resident guard consults the same function so Plan and the guard cannot disagree (the property
   the current comment protects).
4. `debug.SetMemoryLimit(available − margin)` at load when available is known, re-evaluated after
   load (the limit is about slack, and slack is what the gpt-oss build had ~9% of); off with
   `GOMEMLIMIT` already set in the environment (Go's own precedence).
5. The working-set warning: for a MoE `-stream-weights` load, `activeBytesPerToken = topK ×
   expertBytes + denseCoreBytes`, `hitRate` from the budget/total ratio using the CUDA C′ curve as
   the prior (57% at 30% residency … 82% at 40 slots — `benchmarks.md` §B4.1, the only measured
   hit-rate curve in the tree; say it is a prior), `predicted tok/s ≈ 1 / (missBytes / preadRate +
   computeMs)`; print it; require the acknowledgement below the threshold. This is the arithmetic
   the M35 run needed before it started.

**Gates.** Table-driven unit tests for each of 1–5 with the incident numbers; `TestResolveCtxCapFit_shortcuts`-class
mutation checks (revert each guard and watch its case go red); no change to any CUDA path (the
`.giw` mode is CPU/Metal — CUDA's resident build has its own VRAM accounting and `cudaBuildBytes`).

**Measure.** One real Mac run per rule item: the 7B Metal load under a browser-heavy session (the
scenario the R13 probe was born from), the M35 `-stream-weights` load with the warning, a `.giw`
1.5B load at `-ctx 32768` that should auto-pin. Swap-used before/after each (S3 makes this
automatic).

**Decision.** As registered; the GOMEMLIMIT item is kept only if the measured anonymous
footprint after load is not worse and GC CPU (from `GODEBUG=gctrace=1`, one run) does not rise
more than 10% — a soft limit that costs throughput is dropped, and the reason recorded.

**Record.** `docs/tasks/task-fit-to-hardware.md` (this is its Phase 3); `docs/env-vars.md`;
`benchmarks.md` "M35/M26 on the Mac" gets the warning's prediction next to the 2h10 record once
S5 measures the real rate.

**Out of scope.** Changing `residentMemFraction` (0.70 stays; S4 only adds a second bound); CUDA;
the pool mode's own accounting (S5).

---

### S5 · Pool mode as the darwin default for the MoE pager; a pread ring for dense layer streaming

**Goal.** On darwin, where advice cannot enforce a budget, the expert pager enforces it by
allocation — the owned-buffer pread pool already in the tree — if a measurement says the copy
cost is worth the cap; and dense layer streaming gets the same option.

**Standing and the registered rule.** (The Metal pager's own M26 runs are R11(c)'s three spirals —
those are S0's dense-copy term plus slots on a box with no headroom, and S6 is their fix; this brief
is the CPU pager.) `GOINFER_MOE_PREAD_CPU=1` (`decoder/moepaging.go:167`)
selects `newExpertBufferPool`: a fixed set of owned buffers refilled by `pread` on an independent
fd, "a firm cap on every platform at the cost of a memcpy per miss and losing `.giw` zero-copy
aliasing" — Lever 1b of `task-moe-streaming.md`, never measured on the Mac against the mmap mode
at an equal budget. The dense layer pager (`decoder/layerpaging.go`) has no pool variant.
**Rule (Mac, M35 `.giw` kind 5, budget fixed at the S4 live figure, `scripts/bench_peer.py`-style
served greedy at depth 128, both modes interleaved, three runs each, swap-used and page-fault
counts sampled): pool mode ships as the darwin default if its tok/s is ≥ 0.9× mmap mode's AND its
RSS ceiling holds within 10% of the budget while mmap mode's does not; parked if tok/s is 0.75–0.9×;
if mmap mode's RSS also holds (the UBC happened to cooperate), the measurement is repeated under
memory pressure (a 6 GB `dd`-into-tmpfs-class hog running) before deciding.** The 2h10 run is the
prior for what mmap mode does under pressure; the measurement must reproduce it in bounded form
(a 20-minute time box, `--max-tokens 32`, progress logging every token) rather than re-run it.

**Read first.** `decoder/moepaging.go` top to bottom — both modes, the two mutexes and why
(audit C-30, Lever 1b's cross-call requirement), the `MmapByteOffset` pread-staging seam
(`decoder/model.go`), the Metal pager (already pread-based) for the slot arithmetic;
aikit's `mmap` package `madvise_darwin` file (cross-repo); `docs/tasks/task-moe-streaming.md` Lever 1b and §C′;
`docs/completed/task-w4a8-neon-bandwidth.md` (the pread A/B: 3.23×, faults 98.5→0/stage — the
pread path is already the measured winner for staging on this Mac; this item asks whether that
holds as the *pager's* mode); `benchmarks.md` "M35/M26 on the Mac" (the swap/watchdog caution:
time-box, `--max-tokens`, kill on the first sign of the CPU-staged fallback, prove which path ran
from the banner and `DecodePath()`).

**Build.**
1. Nothing for the measurement beyond a `--moe-pager=mmap|pool` flag (or the env var, exposed)
   and page-fault counters in the pager's own stats line (`getrusage` minor/major faults per
   token — bit-identity is untouched, so the perf counter is the wiring check, per the "an inert
   dispatch is invisible to every gate but the speedup" lesson).
2. If the rule ships: the default flips on darwin only (`runtime.GOOS`), linux keeps mmap mode
   (DONTNEED works there and the alias is free); the banner names the mode.
3. Dense pread ring (only if a dense-bigger-than-RAM case on the Mac is wanted — today the 7B
   fits): a ring of N layer-sized owned buffers, N from the budget, filled in layer order by a
   prefetch goroutine one layer ahead (the demand signal is known — `decoder/layerpaging.go` already
   issues WILLNEED one layer ahead for the same reason); bit-identical by construction; the same
   fault counters.

**Gates.** Paged ≡ non-paged byte identity in both modes (the existing test, run per mode);
`TestMoE_declines…`-class tests unaffected; the pool's cap asserted in a unit test with a fake
fd (buffers never exceed the budget, refills never exceed slot count, the current token's top-k
cannot be evicted — the LRU invariant the mmap mode already enforces).

**Measure.** As in the rule; additionally `footprint` snapshots at token 1, 16 and 32 for both
modes and the external swap log; record the SSD free space (the near-full-SSD suspicion from the
kind-4 saga has never been tested as a variable — note it, do not chase it here).

**Decision.** As registered.

**Record.** `docs/measurements/moe-pager-mode-darwin-2026-MM-DD.md`; `docs/tasks/task-moe-streaming.md`
Lever 1b closed either way; `benchmarks.md` "M35/M26 on the Mac" replaced by a bounded row with
the mode named — and `red-october.md` R11 (c) unblocked.

**Out of scope.** Speculative expert prefetch (recorded negative, 0.1–0.3% exact-set match);
the Metal pager's command-buffer boundary (M-11, R11); Linux defaults.

---

### S6 · Metal aliases the mapping — `NewBufferNoCopy` over a metal-target `.giw` layout

**Goal.** Dense weights on Metal become file-backed views of the `.giw` mapping instead of
MTLBuffer copies, so a Metal load's anonymous footprint is KV + expert slots + scratch and nothing
else — the same property the CPU `.giw` path already has, and the one the R11(c) runs needed.

**Standing and the registered rule.** Every dense projection on Metal is copied twice at build
(S0: `int4DirectWords` → heap `[]uint32` → `NewBufferUint32s` → MTLBuffer), which is audit M-07's
three copies with its consequence measured by R11(c). aikit's `gpu` package has had
`NewBufferNoCopy` since the M-14 work — "the resident-weights lever: alias an mmap'd `.giw` straight
into Metal instead of uploading a second GB-scale copy" — and nothing in `metal/` calls it; the
audit's §5 line "`NewBufferNoCopy` inapplicable to goinfer's fused/narrowed layouts" is true of the
*current on-disk layout*, not of the API. llama.cpp's Metal backend runs its whole model this way
(mmap the GGUF, wrap the mapping), which is the existence proof for the mechanism on this OS.
**Step 0 has no band — it is the `footprint` split (anonymous / file-backed / compressed) of a
Metal 1.5B, 7B and M26 (N=8) load, today's path. The build's rule: anonymous footprint after load
≤ KV + slots + scratch + 15% (the dense term gone) AND decode at depth 128 within 3% of the copied
path (the do-nothing arm, interleaved) AND logits byte-identical to the copied path (same bytes,
same kernels — a mismatch is a defect, not a band) AND the depth bench within 3% at 2048. All four
or it does not ship.** Registered prediction: under a 6 GB memory hog the aliased arm loses tok/s
(re-faults) where the copied arm loses the *machine* (swap) — record both, because that trade is
the point of the title.

**Read first.** Audit M-07 and its partial closure (the row4-skip half shipped; "releasing host
projections after upload" was the half not done — S6 replaces both halves with aliasing);
`metal/model.go` `int4Buf`, `int4DirectWords`, `bytesToU32`, `int4Concat` (the fused QKV and
gate/up buffers are the layout obstacle: a fused GEMV wants its tensors contiguous, which the
file does not guarantee today); aikit's `gpu` package `metal` file, `NewBufferNoCopy`'s contract
(page-aligned pointer, page-multiple length — Metal drops a trailing partial page — nil
deallocator, the caller keeps the mapping alive for the buffer's whole life, per-tensor sub-views
via `Buffer.At(byteOffset)` over ONE whole-mapping buffer); `decoder/model.go` `MmapByteOffset`
(the seam the Metal expert pager already uses for pread — the same offset is the sub-view offset
here); `docs/tasks/task-int4-layout-2026-09.md` L2 (per-target kinds — kind 5 for cpu-arm64 is the
precedent: the writer emits what a target wants, the reader trusts the kind byte); `decoder/serialize.go`
(kind byte per tensor, the v11 version guard); the dispatch census in `ollama-chase.md` §A2-Metal
(unfusing QKV and gate/up costs +2 dispatches/layer, measured ~0 end to end — so *unfused but
aliased* is a valid fallback if a family's fused layout cannot be written contiguously);
`docs/completed/metal-close-leak-check.md` (the leak checker `Close` ordering must satisfy);
`docs/tasks/task-gpu-paths-2026-09.md` G10 (int8 weights are requantized to W4A8 on device — those
cannot alias unless the requant moves to transcode time).

**Build.**
1. **Step 0, measure** (no code): `footprint <pid>` on the three loads named in the rule, with
   S3's tripwire armed and R11(c)'s external kill script running for the M26 cell; the numbers
   go in the measurement file as the "before" and confirm or correct S0's reading.
2. **Writer — a `metal` kind.** For `GIWTargetMetal` (the sidecar name already carries the
   target), emit each dense int4 tensor exactly as `int4Buf` would build it: nibbles in the word
   order `int4DirectWords` produces (a byte-for-byte reinterpretation of canonical today — verify,
   then it is canonical bytes with a different header), scales stored as **f16** (the `f32ToF16`
   conversion moves into the writer), the fused groups (q‖k‖v rows, gate‖up rows) written
   contiguously in `int4Concat`'s concatenation order, every tensor start 16-byte aligned (the
   kernels' `uint4` loads), the blob's end padded to a page boundary (so `NewBufferNoCopy`'s
   dropped trailing partial page can never hold a tensor). int8 (kind 3) tensors that the Metal
   build requantizes today (G10) are requantized at transcode time into the same kind, so nothing
   is copied at load. Version bump per the format's rule (each version only adds; the guard
   refuses older readers).
3. **Reader / Metal build.** When the loaded bundle carries metal-kind tensors and the model is
   mmap-backed, `buildResident` creates ONE `NewBufferNoCopy` over the mapping (base is page
   aligned by construction; length rounded down to a page multiple) and `int4Buf`/`int4Concat`
   return `Buffer.At(MmapByteOffset(...))` views for weights and scales — no heap copy, no
   MTLBuffer copy. Any tensor that is not metal-kind keeps today's copy path, so an older bundle
   still loads (slower, fatter, correct). The mapping's lifetime is tied to the resident's `Close`
   (buffers released first, then `munmap` — the leak checker is the gate).
4. Expert slots stay as they are (pread into fixed device slots); an *unpaged* MoE that fits
   (Qwen2-MoE, Mixtral on a larger Mac) becomes file-backed by the same mechanism for free.
5. Banner: "metal resident (weights aliased from `<file>`, N MB anonymous)" — the number a user
   would otherwise never see.

**Gates.** Logits byte-identical aliased vs copied on every resident-parity fixture and
`TestMetalSnapshotGolden` unchanged (same bytes reach the same kernels — the golden must not need
a re-bake, and if it does that is the defect signal); a page-boundary fixture (a tensor ending at
the last page and one straddling it, both refused or padded by the writer, never silently
truncated); `Close` ordering under the leak checker; an old-kind bundle loading through the copy
path with a logged note; the paged-MoE tests (`TestGemma4_26B_pagedRuns`-class) unchanged.

**Measure.** `footprint` split after load and after 32 tokens, both arms, 1.5B / 7B / M26 (N=8);
`scripts/bench_peer.py` Metal at depth 128 and the depth bench at 2048, both arms interleaved;
the memory-hog arm (a 6 GB anonymous hog process) for both, tok/s and swap-used sampled
externally — the R11(c) script — with the kill rule armed. Record disk free and the sidecar size.

**Decision.** As registered. A family whose fused layout cannot be written contiguously ships
unfused-and-aliased if the +2 dispatches/layer measure within 3%, else stays copied with the
reason in the writer.

**Record.** `docs/measurements/metal-nocopy-2026-MM-DD.md`; audit M-07 closed; `benchmarks.md`
Table 1's footprint row gains a Metal line; `docs/giw-bundles.md` (the metal kind, and that a
metal-target sidecar is now a promise to one backend the way kind 5 is to one arch);
`red-october.md` R11(c) re-opened for a fourth attempt with S3 + S6 in place, which is the first
attempt that would have a reason to expect a different result.

**Out of scope.** CUDA (pinned host memory and `cudaBuildBytes` are their own accounting);
WebGPU (wgpu cannot wrap host memory as a buffer); any kernel change (the layout moves to disk so
the kernels do not move at all); the expert pager's own mode (S5).

---

## Sequencing

S3 first (a day; it is insurance for everything else, including the S2 transcodes and any further
R11(c) attempt, and it needs no decision), then S1 with S2's gpt-oss and gemma4 halves (S1 without
S2 protects the dense models and leaves the two that matter on the Mac exactly where they are),
then S4 (the live figure into Metal's guard is what would have refused all three R11(c) runs
before they started), then S6's measurement step and build (it is what makes a 26B-A4B's dense
term file-backed on Metal at all), then S5's measurement. S5's build waits on its own number. All
of it is Mac-side work except S3's linux twin; S6 touches the `.giw` writer and the Metal build,
no kernel math.

## What this doc does not claim

- A firm cap makes a working set fit. It does not; S4's warning exists so that case is a
  decision, not an incident.
- The 2h10 run's zero completions are explained. S0 gives the mechanism the record supports; S5's
  bounded reproduction with fault counters is where the number comes from.
- That file-backed weights never slow down. Under pressure the UBC will evict them ahead of
  anonymous memory, and a model that "fits" can still re-fault after a browser session grows;
  the trade is slower tokens instead of swap growth, which is the trade the title asks for.
- That the R11(c) attribution in S0 is measured. It is the code read against the write-up's
  timeline; S6's first step is the `footprint`-split measurement that confirms or corrects it.
- Anything about Linux defaults or CUDA host memory (`cudaBuildBytes` is already priced; the
  Linux box has the RAM).

<!-- doc-reviewed: 2026-09-22 -->
