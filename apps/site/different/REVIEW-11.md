# Review notes: 11-one-file-model-inside.md

<!-- citations-at: b8ab20c4bfd8 -->

Build check passed (`GOWORK=off go run ./cmd/build ... -drafts .`). Body about 700 words. `reviewed:` is empty.

## Claims and sources

| claim | source |
|---|---|
| Assets `goinfer-chat-0.5b-*` / `-1.5b-*`, six targets, `checksums.txt` covers every asset, `sha256sum -- * > checksums.txt` | .github/workflows/release-assets.yml, job `publish`, step "collect + checksum" |
| Model GGUF sha256 and bytes pinned (0.5B: 491,400,064 bytes), build fails on mismatch | same file, job `embedded`, matrix and step "fetch the model + verify the pinned digest" |
| License file fetched and hash-pinned; NOTICE.txt attached when weights ship | same file, "fetch the model license", "collect + checksum" |
| Build converts the GGUF to an int8int8 prequant bundle, embeds it with `go:embed` | demo/chat/build-embed.sh (QUANT=int8int8, `cmd/prequant -target canonical`); internal/chatapp/prequant.go:19 |
| `--version` prints tag, backends, `embedded: tier=... quant=... (baked at build time; --quant has no effect)` | internal/chatapp/version.go `versionReport` |
| `--lora` and `--stream-weights` refused on the embedded binary; `--model` runs another file instead | internal/chatapp/prequant.go `loadEmbedded`; internal/chatapp/main.go switch (~line 268) |
| macOS assets built from `./metal/cmd/chat`, Linux from `./cuda/cmd/chat`, Windows from the root (no GPU backend) | workflow "cross-compile six targets" step; demo/chat/build-embed.sh per-target loop; docs/tasks/task-gpu-paths-2026-09.md status log "G1 done" |
| Default `--backend auto`: CUDA when a device answers, Metal on Apple silicon for int4 models only, so these int8int8 files stay on the CPU on macOS; values auto / cpu / webgpu / cuda / metal | internal/loadflags/loadflags.go (`fs.StringVar(&f.Backend, "backend", "auto", backendHelp)`, `backendHelp`); decoder/backend.go `AutoBackend`; decoder/residency.go `autoMetalPrecision` |
| v0.19.0 (Latest, 2026-09-19) has 27 assets; 0.5B darwin-arm64 653,131,218 B; 1.5B darwin-arm64 1,810,622,418 B; plain chat darwin-arm64 9,370,146 B | `gh release view v0.19.0 --json assets` (read-only) |
| v0.18.0 has 0 assets now; v0.6.0-v0.15.0 had none | `gh release view v0.18.0` (0 assets); workflow header comment and `assets-guard` comment |
| 0.48 s / 1.23 s, 77 MB / 87 MB | docs/benchmarks.md "Cold start & footprint" table (also docs/legacy-benchmarks.md ~line 185) |
| 2.30 s -> 0.48 s, 772 MB -> 78 MB (raw embedded GGUF vs prequant) | docs/ARCHITECTURE.md, the `.giw` section table; CHANGELOG.md ~line 3255 |
| Decode median of 5: 52.3 / 27.8 tok/s, M1 Pro, commit 3a27941, 2026-08-22, `BenchmarkDecode` | docs/measurements/demo-chat-macbook-2026-08-22.md |
| Apache-2.0, Copyright 2024 Alibaba Cloud; license text travels as `QWEN2.5-CODER-LICENSE.txt`; attribution in NOTICE | NOTICE, "Embedded model" |
| Unsigned macOS binary, `xattr -dr com.apple.quarantine`; "not a chat genius" | demo/chat/README.md |

## Conflicts between records

1. **Date and label of 0.48 s / 77 MB.** benchmarks.md says "v0.5.0-era CPU campaign, M1 Pro, `.giw` int8". The CHANGELOG entry that first prints "2.3 s -> 0.48 s" and "772 MB -> 78 MB" sits under `[v0.1.3] - 2026-06-05` and says "M-series CPU". I found no raw log or measurement file for it. The page calls it a legacy record and does not give it a single date.
2. **What 0.48 s measures.** benchmarks.md: "first token". demo/chat/README.md sample output: "loaded ... in 0.48s" (load time). The page says both.
3. **GPU.** demo/chat/README.md says "GPU doesn't help this build ... CPU by design". The workflow (G1, 2026-09-08) builds macOS assets with Metal and Linux with CUDA. I used the workflow, and said no record measures the embedded assets on a GPU. The default became `--backend auto` on 2026-10-01 (R17, after v0.20.0); it runs these files on CUDA where a device answers and keeps them on the CPU on macOS.
4. **Sizes.** README table (v0.16.0): 652 MB / 1.81 GB. v0.19.0 is 653,131,218 / 1,810,622,418 bytes. The brief's "about 623 MB" is the same 653 MB expressed in MiB (622.9). I used the bytes and named the tag. Asset sizes are in the body only, not in `figures` (API-only).
5. **"Attached to every release."** False today: v0.18.0 has 0 assets and v0.6.0-v0.15.0 had none. The page says so in the last "doesn't" item.
6. **1.5B tier "opt-in".** The workflow's header comment and the v0.19.0 release notes say it is opt-in; the workflow's own gate builds it on every tag push (only a `workflow_dispatch` with `tier_1_5b=false` skips it), and v0.19.0 has it. I followed the workflow and the release. Those two texts are stale.
7. **demo/chat README sample output** (`in 0.48s [backend=cpu quant=int8int8]`) predates the current print format (`decode=...` is now included). I did not reproduce it.

## Left out as not verified

- Any GPU speed of an embedded asset; whether `--backend metal` / `cuda` works with the baked int8int8 canonical bundle (no record). Worth noting G10 says Metal runs int8int8 as W4A8, which I did not touch.
- The actual contents of `checksums.txt` and `QWEN2.5-CODER-LICENSE.txt` (assets not downloaded). The `grep | shasum -a 256 -c` line assumes `sha256sum`'s "hash  name" format, per the workflow; I did not run it.
- The license text itself; I relied on NOTICE for what it requires (§4(a) copy of license, §4(c) attribution, §4(d) not applicable).
- Whether the Linux/Windows binaries start on a machine with no GPU driver (the workflow only runs `--version` on the linux-amd64 file on a runner).

## Questions for the owner

1. **NOTICE says the weights are unmodified ("the exact upstream artifact and not a re-quantization of it").** But build-embed.sh bakes an int8 conversion of the pinned GGUF; the GGUF bytes themselves are not in the binary. The page states the conversion plainly. Should NOTICE (and Apache §4(b) "modified files carry notices") be revisited before this page ships?
2. Keep the 0.48 s / 77 MB row at all, or re-measure on a v0.19.0 asset (by night, per the run budget) first? Right now it is a record of unknown date.
3. The 1.5B "opt-in" wording in the workflow header and v0.19.0 notes is stale; fix that separately?
