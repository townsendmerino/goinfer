# WebGPU multi-slot: realClamp's slots 1-3 and the fix (2026-10-30, nobara-sb01)

Gates registered before the code and any card use: `docs/tasks/task-subsystems-2026-10.md`, "The WebGPU multi-slot finding"
(12:03). Driver 595.112.07. The card held only the desktop shell's clients (528 MiB) before each run, and the timing lock was held
exclusively, in a gap mac-tack5's C8 left (12:22:01–12:22:45). `run-ms.sh` ran all three; logs `gms*.log`.

| gate | run | result |
|---|---|---|
| G-MS1 | realClamp/qwen2.5-coder-1.5b, unchanged, at b6397eee5^ | PASS, 4 of 4 slots bit-identical |
| G-MS1 | the same at b6397eee5 | **PASS**, 4 of 4 slots bit-identical |
| G-MS2 | the rewritten realClamp at this branch, both checkpoints | **PASS**: the 1.5B's 4 conversations on 4 distinct slots, each bit-identical; the 7B grants 1 slot and matches |

**G-MS1 withdraws the claim that b6397eee5 exposed the test.** By the registered rule, a pass at b6397eee5 means the claim is
withdrawn. The unchanged test still fails at 2ca872273 (`../sb2-webgpu-slots-2026-10-30/pass1-realclamp-1.5b-on-main-2ca872273.log`),
so the change that exposed it is later than b6397eee5. Twenty-one commits in b6397eee5..2ca872273 touch `decoder/resident_reuse.go`,
`decoder/kvmgr` or `gpu/kv_slots.go`. One candidate, not tested, is e59788c2d (2026-10-21: "an own rollback continues in place, on
the GPU and the CPU"). Naming it would take a bisect over that range, about 5 steps.

**G-MS2 is what stands.** The production code, driven as serving drives it, gives every granted slot bit-identical output. The
decoder binds the slot each conversation uses, under the `ResidentKVSlotter` contract, and serving binds slots no other way. The
branch differs from `main` only in the test. The defect is the old test's: it bound each slot itself (`rd.UseKVSlot`) and then sent
the same prompt through the decoder, whose own record still named the slot it had last bound.
