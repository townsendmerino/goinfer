# Prompt: MC1 on WebGPU — the discrete-GPU clamp on real Vulkan hardware, then slots before context (nobara)

For a Claude Code session on `nobara-pc` (RTX 2070 SUPER 8 GB, WebGPU through wgpu-native's **Vulkan** backend), repo
`~/mycode/goinfer`. `git pull` first. Check that `go.work` lists `./gpu`; it is gitignored, and the `gpu` module's
tests cannot see the root module's current code without it.

## Where this stands

MC1 (multi-slot resident KV) shipped on WebGPU on 2026-09-27, graded on the MacBook through wgpu's Metal backend.
- Code: `3926f927`, `gpu/kv_slots.go`.
- Record: `docs/measurements/concurrency-mc1-webgpu-2026-09-27.md`. At 4 clients it reads 2.805× the one-slot build,
  every gate passes, and every reply is identical.
- Design: a slot is a whole `DecodeRunner`, because `newDecodeRunner` bakes every KV buffer into bind groups. Each
  slot shares slot 0's weights and owns its KV. `UseKVSlot` swaps `rd.rm` / `rd.runner` / `rd.batch` and moves a
  bound adapter.
- WebGPU has **no free-memory query**, so the slot count is priced two ways:
  - **darwin** (unified memory): arithmetic against RAM before anything is allocated (`darwinKVSlots`);
  - **everywhere else**: allocate slot after slot. A slot whose allocation fails, or after which a 384 MiB headroom
    probe fails, ends the count (`newKVSlot`, `buildKVSlots`).
- The second path has only run under an injected failure (`TestWebGPUKVSlots_clampedBuild`). **Nobody has seen
  what wgpu-native on Vulkan does when a real buffer allocation runs out of VRAM.** It may fail cleanly, or it may
  lose the device, or it may succeed and fail later. That is this job's first question.

**Owner decision, 2026-09-27: slots before context on WebGPU too, as on CUDA (`947e06ce`), in two steps.**
- Step 1 is done: on darwin, an unpinned context gives up positions, down to 4096, until every requested slot fits
  (`slotsBeforeContext`).
  - "Unpinned" is the new `decoder.Model.ResidentContextPinned()`. It is false when the caller left the context 0,
    **including** when the load-time fit guard auto-pinned it (R13). That pin is a one-slot ceiling, not a choice.
  - An explicit `-ctx` is never shrunk.
- Step 2 is this job: the same rule on discrete GPUs, **only if** item 2 below shows a failed allocation is clean.

## Read first

- `docs/tasks/task-concurrency-2026-09.md`: "MC1 on WebGPU", and "MC1 on CUDA" for the context policy.
- `gpu/kv_slots.go` in full, and the hooks in `gpu/residency.go` (`runnerFor`, `buildKVSlots`, `slotsBeforeContext`,
  `SetAdapter`, `release`).
- `gpu/kv_slots_test.go`.
- `cuda/resident.go` `ctxForSlots` / `resolveCtxCapFit`, the CUDA rule being mirrored.

## 1. Identity on Vulkan (hard gate; a difference stops the job)

```
cd ~/mycode/goinfer/gpu
go test -tags 'gpu goinfer_testhooks' -run 'TestWebGPUKVSlots|TestCtxWhereSlotsFit|TestDarwinKVSlots' -count=1 -v -timeout 20m . 2>&1 | tee <log>
GOINFER_HEAVY_TESTS=1 GOINFER_WEBGPU_KVSLOTS_MODEL=$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf \
  go test -tags 'gpu goinfer_testhooks' -run 'TestWebGPUKVSlots_interleavedMatchesAlone' -count=1 -v -timeout 20m . 2>&1 | tee <log>
```

- **mistral-tiny-window must run here, not skip.** It skipped on the Mac, where the fixture is not built, so the
  sliding-window layout has not been through the scenario anywhere yet. Read the `--- PASS` lines, not `ok`.
- The darwin-only tests (`_pricedAgainstMemory`, `_slotsBeforeContext`) skip on Linux. That is expected; say so.
- Then run the full `gpu` package the same way. The Mac run was 132 pass / 0 fail.

## 2. The real clamp (the question this job exists for)

Write a heavy test in `gpu/kv_slots_test.go`, e.g. `TestWebGPUKVSlots_realClamp`, selected by an env var naming a
checkpoint, and commit it. It should:
1. Load the checkpoint on webgpu with `ResidentKVSlots: 4` at the default context (16384, f32 KV). Read `KVSlots()`
   and the clamp line from stderr.
2. Run a short greedy generation (≥ 32 tokens, ≥ 2 turns, as `kvSlotsScenario` does) on **every granted slot**.
   Compare the ids with a fresh one-slot load. They must be bit-identical.
3. After `Close`, check that `LiveBufferBytes()` is back to its value before the load, and that `nvidia-smi`'s
   used memory is back within a small margin of its baseline.
4. Fail loudly on any device-lost or validation error. A skip here is not a pass.

Run it on:
- **qwen2.5-7b-instruct q4_k_m** from `~/models` (NVMe). **Never `/srv/models`**: that is the SMR archive, and a
  number read from it is void. About 4.4 GB of weights plus 1.88 GB per slot at 16k on an 8 GB card: expect 1 slot,
  with slot 2 failing, and record whether it was the allocation or the probe;
- **qwen2.5-coder-1.5b** q4_k_m: ~0.94 GB per slot, so expect all 4.

Record what happened, not what was expected: the granted count, which check stopped the count, whether the device
stayed usable, and the leak check.

**Decision rule, pre-registered here:**
- **Clean** means the count stopped with a logged reason, every granted slot is bit-identical to one-slot, nothing
  leaked, and no device-lost or validation error appeared. Then go to 3.
- **Not clean** means the device is lost, a later allocation or dispatch fails, the ids differ, or memory leaked.
  Then **stop**:
  - do not build step 2;
  - write the failure up in the task doc's MC1-on-WebGPU section, with the log;
  - propose a safe default for discrete GPUs (for example, one slot unless `-ctx` is given), leaving the choice to
    the owner.

## 3. Step 2: slots before context on discrete GPUs (only if 2 was clean)

- The rule is the one `slotsBeforeContext` applies on darwin: when `ResidentKVSlotsRequest() > 1` and
  `!ResidentContextPinned()`, give up context down to 4096 until every requested slot fits; below the floor, clamp.
- With no memory query, what fits has to come from the build. A suggested shape; check it against the code, do not
  assume it:
  1. Build as today. If `k < want` slots fit at context C, the room found is at least `k × perSlot(C)`.
  2. Pick the largest C′ in [4096, C] with `want × kvBytesPerPosition × C′` within that room.
  3. Rebuild every slot's KV and runner at C′, including slot 0's. KV is linear in the context, so each buffer is
     `size × C′ / C`; check the int8 word rounding.
  4. Set `rd.ctxCap`. Log both steps.
  - Keep the 384 MiB headroom probe after the last slot.
- Identity after a rebuild is a hard gate. Run the scenario on a shrunk build, and the 7B on real hardware: it should
  come out near 4096–5000 with 4 slots. CUDA got 4,984 on the same card; WebGPU's weights may differ.
- Update `docs/server.md`, where the MC1 WebGPU bullet says the context is not shrunk, and the task doc.

## 4. CUDA: key the rule on `ResidentContextPinned()` (small)

`resolveCtxCapFit` treats any `ResidentContextRequest() > 0` as explicit. When the host-RAM fit guard auto-pins an
unrequested context, `ctxForSlots` therefore never runs. That is rare on a box with plenty of RAM, and
`decoder.Model.ResidentContextPinned()` now tells the two cases apart.
- Make CUDA use it. A guard-pinned context is the upper bound, and the slots rule may shrink below it.
- Add a test. `decoder.SetHostRAMAvailableForTest` can force the guard to pin.
- Run the tagged CUDA suite.

## 5. Optional, reported only: W7 on Vulkan

Only after 1–3 pass. Qwen2.5 has a batched prefill on Vulkan (`PrefillLast` admits q/k/v bias there), so the thrash
costs less than on the Mac, and the win should look more like CUDA's 1.25× than the Mac's 2.8×. If you run it:
- pre-register it in the task doc first, with the Mac cell's gates;
- use `run-w7.sh` in `docs/measurements/concurrency-mc1-webgpu-2026-09-27/` as the template. Swap the idle gate
  for the CUDA one in `concurrency-mc1-cuda-2026-09-27/run-w7.sh` (load1 ≤ 2.0, no extra GPU process, GPU memory
  back to baseline);
- old = the build at `68f2dbdf` (one slot), new = your HEAD, with `--backend webgpu`.

## Rules that apply

- Commit increments, and push with every outstanding file, staged by explicit path (never `git add -A`).
  `git fetch` + rebase first, because `main` moves.
- Before pushing: `gofmt -l`, `go vet -tags 'gpu goinfer_testhooks' ./gpu/`, CI's `staticcheck` for the gpu module
  (prove it can go red first), and the citation lint by exit code. Then check CI with `gh run list`.
- A decoder edit needs `scripts/refresh_parity_hashes.sh` with its proof trailer.
- Long runs: detached, with start and expected finish times in local and UTC, and logs archived out of `/tmp`.
