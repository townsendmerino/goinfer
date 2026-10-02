# C3 — Metal consumer window, evaluated against `metal/v0.20.0` (2026-10-01)

Out-of-tree consumer evaluation of goinfer's Metal backend at the published `metal/v0.20.0` tag, owed because v0.20.0
bumps aikit (v1.45.1 → v1.51.1; RELEASING.md, "C3 · Metal consumer window"). Supersedes
`c3-metal-consumer-window-v0.18.0.md` rather than editing it. Run on `macbook-arm64` (Apple M1 Pro, 16 GB), macOS
26.6.2, Go 1.27.0.

> **Status: parts 1, 3 and 5 done by day on 2026-10-01; parts 2 and 4 queued for the Mac's night queue**
> (`run-c3-night.sh` in `c3-metal-v0.20.0/`). By day the fit guard refused even the 1.5B: 3.8 GB available with the
> owner's apps open, a 2.6 GB budget against 4.7 GB priced. A bypass on this 16 GB Mac is a standing no, and part 2 is a
> timing, which belongs at night anyway.

**Which commit.** As v0.18.0's C3 found, the root tag and the submodule tag are different commits under the two-step
tag. `v0.20.0` is `890ca565`, whose `metal/go.mod` still requires root v0.19.0; `metal/v0.20.0` is `8e4fb57c`, which
requires root v0.20.0. Everything here is at `8e4fb57c`, which is what `go get .../metal@v0.20.0` resolves.

## 1. Builds with no Xcode

Xcode is installed on this machine (`/Applications/Xcode.app`), so the literal no-Xcode case cannot be tested here. Tested
the Xcode-independent form instead, two ways:

```
cd metal && CGO_ENABLED=0 GOWORK=off go build -tags metal ./cmd/serve && CGO_ENABLED=0 GOWORK=off go vet -tags metal ./...
                                                                    # in a clean worktree at 8e4fb57c: both exit 0
CGO_ENABLED=0 GOWORK=off go install github.com/townsendmerino/goinfer/metal/cmd/serve@v0.20.0
                                                                    # from an empty directory, as a consumer: exit 0
```

The consumer binary's build info pins `goinfer/metal v0.20.0` and `goinfer v0.20.0` from the module proxy, with
`CGO_ENABLED=0`. Both binaries are 19,049,810 bytes. `otool -L` lists the same four libraries as v0.18.0 did:
`libSystem.B.dylib`, `libresolv.9.dylib`, `CoreFoundation` and `Security`. There is no Metal.framework, no libobjc and
no cgo runtime artifact. **The cgo-free claim holds, by compile flag and by binary inspection.** The consumer binary's
`--version` reads `serve v0.20.0`, with no `-dirty`, unlike the release assets (`docs/tasks/task-first-hour.md` R18).

## 2. Decode tok/s against the 73.6 claim

**Queued for tonight.** `TestZZ_metalDepthBench` (resident `ForwardArgmax`, decode only, min of 5 batches, W4A8,
qwen2.5-coder-1.5b-instruct-q4_k_m), with `GOWORK=off`. By day its `decoder.Load` was refused by the fit guard, as
described above. Pre-registered in `run-c3-night.sh`: reported against 73.6 and v0.18.0's 71.6 from the same harness;
depth 128 below 70.0 (more than 5% under the claim) is a finding to investigate.

## 3. Bit-identity within machine and OS

The consumer-built `serve` (from `go install ...@v0.20.0`) loaded qwen2.5-coder-0.5b-instruct-q4_k_m on Metal
(`-backend metal -quant int4 -ctx 4096`; `decode path: metal-resident (int4)`, weights aliased from the Metal `.giw`).
The same request was sent twice through `/v1/chat/completions` (greedy, `temperature: 0`, `seed: 42`, 96 tokens):

```
run1 finish_reason=length completion_tokens=96 prefill_reused_tokens=0
run2 finish_reason=length completion_tokens=96 prefill_reused_tokens=0
content identical: True | bytes: 318
```

**Bit-identity within machine and OS holds**, byte for byte, with neither run reusing a cached prefix, so both computed
the whole request. The 0.5B stands in for v0.18.0's 1.5B, which the fit guard refused by day. The property, identical
output for an identical request on one machine, does not depend on the model's size.

## 4. The Metal device gate (§C1-M)

**Queued for tonight**, at `8e4fb57c`, through the release's own `run-metal-gate.sh` (copied beside the job,
unchanged). The job's setup was dry-run by day: 24 fixtures linked, a clean tree, build and vet OK, no `vendor/`. For
reference, the release gate passed this morning at `f2e7c87c` (10 pass, 2 skip, 0 fail), which predates the tag's
aikit v1.51.1 bump. That bump changes only an amd64 file, so arm64 and Metal compute what they did, but C3 asks for the
tag itself.

## 5. Whether the tautological-gate shape is live on Metal

**Not found in the files sampled.** The shape is a flag that toggles a path, two runs that get compared, and nothing
asserting that the flag changed which path ran. Of the 89 Metal test files changed since `metal/v0.18.0`, about 15 set
a knob or environment switch. These four compare two paths through one:

- `attn_fa_e2e_test.go` toggles `GOINFER_METAL_ATTN_FA` and asserts `r.decodeAttnFA == enableFA`. It prefills to
  depth 1600, above the kernel's 1536 floor, so the decode steps really engage it.
- `alias_fixtures_test.go` fails if the alias-off arm aliased any tensor ("the knob is not off"), and requires at least
  one fixture to alias fused groups.
- `decode_attn_r17_test.go` drives the kernel through a direct split override on the resident, not a switch.
- `prefill_gate_test.go`'s fast arm fails loud when `PrefillLast` declines. Its own comment records that a 512 floor once
  made it `Fatalf` before any comparison, and it now pins the floor off.

Sampled, not exhaustive: 4 of about 15 toggling files.

## Verdict (by day; parts 2 and 4 pending tonight)

- **Builds with no Xcode:** the `CGO_ENABLED=0` form holds, from an empty consumer module and in-tree, and the binary
  links no Metal.framework or libobjc. **The cgo-free claim is confirmed.**
- **Bit-identity within machine and OS: holds**, byte for byte, through the consumer-built binary.
- **Tautological-gate shape: not found** in the four sampled path comparisons, each of which asserts its path or fails
  loud.
- **Decode tok/s and the device gate:** queued for tonight; this file is completed from their logs in the morning.
