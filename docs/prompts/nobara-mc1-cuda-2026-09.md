# Prompt: MC1 on CUDA — multi-slot resident KV (nobara)

For a Claude Code session on `nobara-pc` (RTX 2070 SUPER 8 GB, driver 595.91.07 as of 2026-09-27), repo
`~/mycode/goinfer`. `git pull` first.

## The job

Give the CUDA resident N independent KV slots, as MC1 did for Metal, so interleaved conversations keep their own
prefix resident instead of re-prefilling their whole history every turn. It is still one generation at a time, with
no batching. Then grade it on W7.

Why: CUDA keeps one KV slot, so W7's interleaved clients evict each other. On Metal, MC1 alone gave **1.22–1.24×** at
4 clients (57.0 → 70.9 tok/s), and held the 2- and 4-client aggregate at 0.99× / 1.01× of 1 client. It is also the
prerequisite for any CUDA batching (an MC3 port). That work starts only on its own measurement, and is **not** part
of this job.

## Read first

- `docs/tasks/task-concurrency-2026-09.md`, the **MC1** section: design, gates, and the clamp fix.
- `docs/measurements/concurrency-mc1-2026-09-26.md`: the Metal grading this repeats.
- Three commits:
  - `c2f1532e`, the design. The decoder half is generic and already done: `Options.ResidentKVSlots`,
    `ResidentKVSlotter`, `residentAcquire`, `pickResidentSlot`, and serve's `-kv-sessions` → slots plus the banner.
  - `6807ab95`: the slots must be priced **before** the build allocates, or the weights are counted twice.
  - `1b5b8b5e`: the clamp is tested and logged.
- The Metal half to mirror:
  - `metal/backend.go`: `metalKVSlots`, and `KVSlots` / `UseKVSlot` on `metalResident`;
  - `metal/kv_slots_test.go`: `TestMetalKVSlots_interleavedMatchesAlone`, `TestKVSlotsWithin`,
    `TestMetalKVSlots_pricedBeforeTheBuild`.

So the CUDA work is the backend half: implement `decoder.ResidentKVSlotter` on the CUDA resident.

## CUDA specifics to settle (check each in the code; do not assume)

- **Storage.** The per-layer `kc, vc []Buffer` in `cuda/resident.go` are always f32, and an MLA layer holds one
  latent buffer. A slot is a full copy of that set.
  - The price per slot is `kvBytesForCap` / `decoder.(*Model).cudaKVBytes` at the resident's `ctxCap`.
  - `TestResidentKVBytes_matchesCUDAAllocation` should extend to N slots.
- **The fit guard runs on VRAM, and must price the slots before the build.** That is the Metal double-count bug.
  - An 8 GB card will clamp the 7B: int4 weights plus 4 f32 slots at a 4k context is ~4.5 + 4 × 0.47 GB.
  - Metal never ran a clamping load, so exercise one here. The banner must name the clamp.
- **CUDA graphs** (`GOINFER_CUDA_GRAPHS`, opt-in, off by default) capture each layer's segments with their buffer
  arguments baked in. That is the same hazard as Metal's pre-encoded executor, which `useKVSlot` had to stop.
  - Either re-capture on a slot switch, or keep one slot while graphs are on.
  - Test whichever you choose.
- **Every host-side KV writer must write the bound slot.** For example, the `r.kc[layer].At(byteOff)` upload.
- **Audit the resident's other per-sequence state.** Anything kept across tokens for *a sequence* (device-side
  positions, spec/drafter state, MoE caches keyed to history) must be per-slot, or reset on a switch.
  - The decoder already clears `resDrafterSynced` on a switch.
  - Recurrent and hybrid families already get 1 slot from the decoder. Confirm nothing on CUDA bypasses that.

## Before any timing

These tests are the identity gate, and hard: a difference is a bug and stops the job.
- A CUDA twin of `TestMetalKVSlots_interleavedMatchesAlone`, on the tiny fixture and on qwen2.5-coder-1.5b: two
  conversations interleaved on 2 slots emit exactly the ids they emit alone, and the one-slot control thrashes.
- A clamp test, and a priced-before-the-build test.
- Suites:
  - `go test -tags 'cuda goinfer_testhooks' ./cuda/`, plus the decoder and serveapp suites;
  - `go vet -tags 'cuda goinfer_testhooks' ./cuda/`;
  - CI's pinned staticcheck with the cuda tags. Prove it can go red, per CLAUDE.md;
  - `gofmt -l` across every module you touched.
- If a `decoder/` core file changes, run `scripts/refresh_parity_hashes.sh`.

## The grading: pre-register it in the MC1 section, and commit that, before any W7 timing

- *old* = `serve-cuda` built from the base commit (one slot); *new* = the MC1-CUDA commit.
  - Build each once: `CGO_ENABLED=0 go -C cuda build -tags cuda -o <dir>/serve-cuda-<hash> ./cmd/serve`.
  - Name the binaries by hash; the binary is the anchor.
- **Workload:**
  - `scripts/bench_w7_plain.py OUT --clients N --engines goinfer --backend cuda --key K --fixed-nonce --server-log LOG`;
  - `GOINFER_SERVE_CPU=<serve-cuda binary>` (the variable names the serve binary, whatever the backend);
  - `BENCH_W7_MODEL=~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf`. It is on NVMe; never read from `/srv/models`;
  - 6 turns × 128 greedy tokens per client, a fresh server per cell.
- **Cells:** 1, 2 and 4 clients, each old/new × 3 pairs, in the order old new new old old new.
- **Idle-gate every cell.** `docs/measurements/concurrency-mc3-2026-09-26/run-w7.sh` is the template. Its gate reads
  macOS `sysctl vm.loadavg`; on Linux, read `/proc/loadavg`.
- **Gates.** Hard unless marked; bars as MC1's, with the owner's permissive ship bar.
  1. Identity: the tests above, plus in W7:
     - at 1 client, `content_sha` old == new, every turn (nothing switches slots);
     - at 2 and 4 clients, new's client 0 == new's 1-client run, every turn;
     - old vs new at 2 and 4 clients is reported only. Old re-prefills history through batched prefill, which need
       not match decode-written KV bit for bit.
  2. Reuse: new's prompt − `prefill_reused_tokens` at 2 and 4 clients equals its 1-client run's, turn for turn.
  3. *(Expected band, not a gate)* new's 2- and 4-client aggregate is 0.90–1.0× its 1-client aggregate. Below 0.85×
     at 4 clients is a finding to explain.
  4. Ship: 4-client aggregate new ÷ old, median of 3 ≥ 1.03×. The owner parks only couple-percent wins; a clean,
     non-regressing win ships.
  5. Solo guard: 1-client p50 and p99 turn, new ÷ old, median ≤ 1.05× each.
- **Also report** one 4-client pair on `qwen2.5-7b-instruct-q4_k_m.gguf`, where the clamp should bind: the banner's
  slot count, and the aggregate.
- **Record** the NVIDIA driver version. CUDA rows are anchored to it.

## Running

- Long runs detach with `setsid nohup … </dev/null &`, and PPID must be 1.
- Give the start **and** expected finish time, local and UTC. Put logs in a durable directory, never `/tmp`.
- Archive raw JSON and logs in `docs/measurements/concurrency-mc1-cuda-<date>/`, with a record
  `concurrency-mc1-cuda-<date>.md` shaped like the Metal MC1 record.
- Leave `_q4kgate/` alone; it is another session's.

## When it ships

Update the docs that say CUDA keeps one slot:
- `docs/server.md`;
- `docs/ARCHITECTURE.md`;
- the task doc (MC1's result and the status header's open list);
- the count in `docs/README.md`.

Then `git fetch` + rebase, push (the pre-push citation lint must be green), and check CI with `gh run list`.
