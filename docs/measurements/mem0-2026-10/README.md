# MEM0, the nobara half: where the server's memory is (2026-10-18)

The census MEM0 asks for (`docs/tasks/task-memory-campaign-2026-10.md`, §3), on nobara-pc: CPU and CUDA cells at the doc's 20K-token prompt, the Gemma 4 E2B cell, and the library-path load that is MEM1's "before".
**Nothing is graded.** The Mac's cells (Metal, CPU on arm64, a tower image turn) are the Mac's. Tier 1, by day, through the timing lock; each half took under ten minutes.

## Provenance

| | |
|---|---|
| build | main `5310f789`, `CGO_ENABLED=0 go build -tags cuda ./cuda/cmd/serve`; **sha256 `28fd105f81884715d213297560cd2129fd2df6fbc22606e4f21713b71c18825f`** (`serve.sha256`) |
| machine | nobara-pc, RTX 2070 SUPER 8 GB, driver 595.91.07, 64 GB RAM, Nobara 44; system swap 7,956 MiB in use before and after every cell (no growth); each server's own `VmSwap` read 0 throughout |
| prompt | 20,201 tokens on the Qwen tokenizer (20,966 on Gemma 4): this repository's text at `e42db592`, cut to a character budget, as G40 and KV9 used; `max_tokens` 8, temperature 0; one cold request per fresh server |
| flags | serve's defaults except `-backend` and `-ctx 32768` (the CUDA default context, 16,384, is under the prompt: a first attempt was rejected with a 400 and measured nothing) |
| instrument | `mem0_linux.py`: `/proc/<pid>/status` every 250 ms (`RssAnon` = heap and anonymous buffers, `RssFile` = mapped weights and other file pages, `VmHWM`, `VmSwap`) and, on CUDA, the process's device memory from `nvidia-smi` every second. The Go heap in use is not exposed by serve, so `RssAnon` stands for it; the library-path program reads `runtime.MemStats` |
| checkpoints | local `~/models`: `qwen2.5-coder-1.5b-instruct-q4_k_m`, `qwen2.5-coder-7b-instruct-q4_k_m`, `gemma-4-E2B_q4_0-it` (all through their `.giw` sidecars) |
| raw | `raw/cell-*.json`, `raw/serve-*.log`, `raw/libload-out.txt`; `mem0_linux.py`, `testdata/libload.go`, `run-libload.sh` |

## 1. The server, one 20K cold prompt (MiB)

| cell | load | prefill of 20K | `RssAnon` after load → peak → end → +10 s | `RssFile` after load → peak | `VmHWM` | device memory after load → peak |
|---|---|---|---|---|---|---|
| CUDA 1.5B int4 | 4 s | 12.3 s | 91 → 309 → 301 → 309 | 933 → 994 | 1,940 (at load) | 4,792 → 4,866 (KV plan 4 × 32,768) |
| CUDA 7B int4 | 11 s | 27.6 s | 84 → 560 → 552 → 560 | 3,899 → 4,004 | 8,094 (at load) | 6,280 → 6,436 (KV plan 1 × 32,768) |
| CUDA Gemma 4 E2B int4 | 6 s | 320.6 s | 140 → 1,977 → 364 → 364 | 1,321 → 1,521 | 2,798 at load, 3,510 peak | 3,418 → 3,540 (KV plan 4 × 32,768, f32 KV) |
| CPU 1.5B int4 (f32 KV) | 2 s | 558.4 s | 105 → 1,739 → 1,168 → 1,175 | 38 → 924 | 2,104 at load, 2,662 peak | — |

## 2. The library path: `decoder.Load` on a `.gguf`, no sidecar (CPU, amd64, MiB, after `runtime.GC` and `FreeOSMemory`)

| model, `Options` | heap in use | `RssAnon` | `VmHWM` |
|---|---|---|---|
| 0.5B, `Options{}` | 2,405 | 2,412 | 2,905 |
| 0.5B, `Quant: "int4"` | 454 | 459 | 951 |
| 1.5B, `Quant: "int4"` | 1,150 | 1,158 | 2,246 |
| 1.5B, `Options{}` | 6,779 | 6,792 | 7,882 |

The app path with the sidecar holds 84 to 140 MiB of anonymous memory after load on every cell above (the weights are mapped file pages).

## 3. What the readings say about the campaign's items (observations, not causes)

- **MEM1 is library users only, and its size is confirmed.** The same 1.5B holds 105 MiB anonymous through the sidecar and 1,158 MiB (int4) or 6,792 MiB (`Options{}`, f32) through `decoder.Load`: 11× and 65×. Tack goes through the sidecar (`task-memory-campaign-2026-10.md`, MEM0).
- **MEM6 (resident prefill embeds): the doc's arithmetic reads low by about 2×.** It predicts 117 to 273 MiB of host embeddings at 20K on a dense CUDA model. The anonymous memory a 20K prompt added and still held 10 s later was **+218 MiB on the 1.5B and +476 MiB on the 7B**. On E2B the peak was +1,837 MiB over the baseline (the doc's E4B figure is about 1.0 GiB), of which 224 MiB were still held at the end of the request and 10 s later. What else is in those numbers (the prompt text, token ids, per-request buffers) was not separated.
- **A CUDA load has a transient of about twice the file on the 7B.** `VmHWM` at load is 8,094 MiB against 3,899 MiB of mapped weights and 84 MiB anonymous afterwards: the gap implies roughly 4 GB of anonymous memory during the load, gone after it (the 1.5B: 1,940 against 1,037). The sampler starts when the server is ready, so the transient itself was not captured. This is MEM10's territory (`cuda/resident.go:packWeight` copies each tensor through the host); a sampled load would size it.
- **The CPU path keeps the session's KV and returns the scratch.** After the 20K prompt `RssAnon` is 1,168 MiB: 105 baseline plus about 1,105 of f32 KV by geometry (28 layers × 2 heads × 128 × 2 × 4 B × 20,201 tokens). The peak, 1,739, is 571 MiB over that: G40's chunked prefill scratch, returned afterwards. Nothing returned in the 10 s after the request beyond that.
- **The CPU prefill is 36 tokens per second** here (20,201 tokens in 558 s), single-threaded; a 7B cell at this length would not fit a day run, and was not measured.
- **The first CPU load wrote its sidecar.** `VmHWM` 2,104 MiB at load on the CPU 1.5B is the one-time `.giw` transcode (959 MB written in 1 s), not a steady-state figure.
- **Default KV plans differ by model on CUDA.** The 1.5B and E2B get 4 conversations × 32,768 positions (3.6 GB of KV slots: 4.8 GB of device memory for a 0.9 GB model); the 7B gets 1 conversation.
- **Mapped weights stay resident on CUDA.** `RssFile` equals the checkpoint size after the upload (3.9 GiB for the 7B): clean pages, reclaimable, but counted in `VmRSS`.
- **E2B's 20K prefill is slow** (320.6 s, 65 tokens per second: exact kernels, passes of up to 512 rows), much slower than the other CUDA cells (12.3 s and 27.6 s).

## Not done here

A 7B on the CPU or Metal, a sampled CUDA load for the transient above, and Go heap readings from a running server (serve does not expose them). The Mac half is below.

## 4. The Mac half (2026-10-21)

**Nothing is graded.** By day, each cell under the timing lock (`mem0_darwin.py`, the macOS port of `mem0_linux.py` on
`g40_mem.py`'s readings).

| | |
|---|---|
| build | main `fc34efd8`, `go build ./metal/cmd/serve` (Go 1.27.2, darwin/arm64); **sha256 `4e9b37f7f1a780e71062a7183885781d9679275f9554a9d73754c87d0fbd3cc4`** |
| machine | MacBook Pro, M1 Pro, 16 GB, macOS 26.6; the owner's desktop and editor open (system swap 2.0 to 3.1 GB in use, available memory 5.7 to 8.8 GB at the cells' starts) |
| prompt | the same 20K preamble: 20,189 tokens on the Qwen tokenizer, 20,975 on Gemma 4; `max_tokens` 8, temperature 0; one cold request per fresh server. The tower cell sends one 1920×1080 PNG (gradients and a grid, made by the script) and a one-line question |
| flags | serve's defaults except `-backend` and `-ctx 32768` (the tower cell: `-vision`, default context 16,384) |
| instrument | `footprint -p` (phys_footprint and its categories, dirty bytes) and RSS every 250 ms: after load, the peak during the request, at its end, 10 s after. Available memory and system swap before and after are context only |
| checkpoints | local `~/models`: `qwen2.5-coder-1.5b-instruct-q4_k_m` (Metal v17 `.giw`; the CPU cell through its CPU sidecar), `gemma-4-E2B_q4_0-it` (Metal v17 `.giw`), `qwen3-vl-2b-instruct` (its Metal v17 sidecar, written by a warm start before the cell) |
| raw | `raw-mac/cell-*.json`, `raw-mac/serve-*.log`, `raw-mac/libload-out.txt`; `mem0_darwin.py` |

### 4.1 The server, one cold request (MiB)

| cell | load | request | footprint after load → peak → end → +10 s | at +10 s: `IOAccelerator (graphics)` / anonymous (`VM_ALLOCATE`) | peak RSS | system swap before → after |
|---|---|---|---|---|---|---|
| Metal 1.5B int4, 20K | 1 s | 35.8 s | 62 → 2,447 → 1,936 → 1,936 | 1,798 / 124 | 227 | 2,998 → 2,982 |
| Metal Gemma 4 E2B int4, 20K | 1 s | 585.5 s | 120 → 5,911 → 2,365 → 2,365 | 1,157 / 1,191 | 2,993 | 2,982 → 1,962 |
| Metal Qwen3-VL-2B, one 1920×1080 image (1,028 prompt tokens) | 2 s | 7.7 s (the tower 5.3 s) | 54 → 7,261 → 7,208 → 7,207 | 4,414 / 2,776 | 3,529 | 1,962 → 2,301 |
| CPU 1.5B int4 (f32 KV), 20K | 1 s | 310.8 s | 70 → 1,708 → 1,174 → 1,174 | — / 1,169 | 2,648 | 2,293 → 2,261 |

### 4.2 The library path on arm64: `decoder.Load` on a `.gguf`, no sidecar (Go heap in use after `runtime.GC` and `FreeOSMemory`, MiB)

| model, `Options` | arm64 (this Mac) | amd64 (§2) |
|---|---|---|
| 0.5B, `Options{}` | 2,405 | 2,405 |
| 0.5B, `Quant: "int4"` | 647 | 454 |
| 1.5B, `Quant: "int4"` | 1,854 | 1,150 |
| 1.5B, `Options{}` | not run: nobara's peak was 7.9 GB and this Mac had about 8.8 GB available | 6,779 |

(`libload.go` reads `/proc/self/status`, which macOS has not got, so only its Go heap figures are recorded here.)

### 4.3 What the Mac readings say (observations, not causes)

- **Metal's KV is the held memory on a text model.** The 1.5B holds 1,798 MiB of `IOAccelerator` after a 20K prompt, the
  size of its two 32,768-position f16 KV slots (the banner: 0.9 GB per slot), touched by the request though at load it
  read 4 MiB. Everything else it holds is 124 MiB.
- **MEM7's block is the 1.5B's transient.** Peak less end is 511 MiB: the 512 MiB prefill block, returned after the pass.
- **The tower holds 7.2 GB after one image (MEM2, MEM4).** Footprint 54 MiB after load (the tower loads on the first
  image), 7,261 at the peak and 7,207 10 s later: 4.4 GB of device buffers and 2.8 GB of anonymous host memory, with
  system swap up 339 MB. Read on main before MEM2's change (branch `mac-mem-2026-10-21`), which releases the scratch a
  grow replaces and at unload, but not between requests. What part is the tower's f32 weights on the device, its host
  copy and its scratch was not separated.
- **E2B's 20K prefill takes 585 s on Metal** (36 tokens per second; CUDA 320.6 s, §1) and peaks 3.5 GB over what it
  holds after: the E-model layer-major route is not blocked by MEM7's block (`PrefillBlockRows` is 0 there). It holds
  1.2 GB anonymous after the request, against 140 MiB after load.
- **arm64's int4 library load holds 42–61% more heap than amd64's** (647 against 454 MiB on the 0.5B, 1,854 against
  1,150 on the 1.5B); `Options{}` is the same. MEM13's heap-quant row names the arm64 dual copy, which fits; not
  confirmed by reading the heap.
- **The arm64 CPU cell matches nobara's on what it holds.** 1,174 MiB after the request (nobara: 1,168), the f32 KV by
  geometry plus the baseline; the peak, 1,708, is 534 MiB over that (nobara: 571), G40's chunked-prefill scratch, returned.
  This is the tree after G40's allocate-once fix, so G40-2's Mac readings (at `db2bca58`, before it) do not stand for it.
  The prefill ran at 65 tokens per second (nobara: 36).
- **No cell grew system swap by more than 339 MB**, and the E2B cell's start shrank it; with the owner's desktop open the
  system figure is context, not the test's.
