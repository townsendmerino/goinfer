---
title: "One file, model inside"
area: "Distribution"
order: 11
summary: "One file holds the runtime and a fixed Qwen2.5-Coder model, 0.5B or 1.5B. It is large, runs on the CPU by default, and its speed figures are old."
stand: "A release asset that is the program and a model in one executable. You download it, check its hash and run it. There is nothing else to install and no model to fetch."
measured: 2026-09-19
reviewed: 2026-09-29
facts:
  - {label: "size on macOS arm64, v0.19.0", value: "653 MB (0.5B), 1.81 GB (1.5B)"}
  - {label: "what is inside", value: "Qwen2.5-Coder-Instruct, int8, mapped from the binary"}
  - {label: "start and memory (old record)", value: "0.48 s and 77 MB, 0.5B, M1 Pro"}
  - {label: "checks", value: "model sha256 pinned; checksums.txt covers every asset"}
doesnt:
  - title: "It isn't small."
    text: "The v0.19.0 file for the 0.5B on macOS arm64 is 653,131,218 bytes, and the 1.5B file is 1,810,622,418 bytes. The 0.5B model file the release build starts from is 491,400,064 bytes. The binary holds that model converted to 8-bit (int8) weights, which take more space than the 4-bit original. Every new release is a new download of the whole thing."
  - title: "It is one model, and you cannot swap the weights inside."
    text: "The model size is fixed when the file is built, and so is the quantization: int8 weights and int8 activations, shown as int8int8. `--quant` has no effect on it. `--lora` (an adapter) and `--stream-weights` are refused. You can pass `--model` to run a different file, but then the embedded weights sit unused, and the plain runtime does the same job in about 9 MB."
  - title: "The model is not goinfer's, and it is small."
    text: "It is Qwen2.5-Coder-Instruct, copyright Alibaba Cloud, under Apache-2.0. Redistributing the binary is redistributing those weights, so the release carries the license text and a NOTICE with the attribution. The 0.5B is good at short code tasks, and the project's own demo README says it is not a chat genius."
  - title: "GPU support is compiled in on two platforms, and not shown to be used."
    text: "The macOS files are built with the Metal backend and the Linux files with CUDA. The Windows ones have no GPU backend. The default is `--backend cpu`, and every speed figure here is a CPU run. No record measures these files on a GPU. The demo README still says the build is CPU by design, a line written before the release build added the GPU backends."
  - title: "Its numbers are old, and not every release has the files."
    text: "The start-up and memory figures come from an old CPU record, first printed in the changelog for v0.1.3 (2026-06-05), not from a v0.19.0 file. The release build failed for v0.18.0, which has no files attached today, and v0.6.0 to v0.15.0 had none. The macOS binaries are unsigned, so Gatekeeper blocks the first run."
figures:
  - {text: "0.48 s", source: "docs/benchmarks.md"}
  - {text: "77 MB", source: "docs/benchmarks.md"}
  - {text: "1.23 s", source: "docs/benchmarks.md"}
  - {text: "87 MB", source: "docs/benchmarks.md"}
  - {text: "2.30 s", source: "docs/ARCHITECTURE.md"}
  - {text: "772 MB", source: "docs/ARCHITECTURE.md"}
  - {text: "78 MB", source: "docs/ARCHITECTURE.md"}
  - {text: "52.3", source: "docs/measurements/demo-chat-macbook-2026-08-22.md"}
  - {text: "27.8", source: "docs/measurements/demo-chat-macbook-2026-08-22.md"}
sources:
  - .github/workflows/release-assets.yml
  - demo/chat/build-embed.sh
  - internal/chatapp/prequant.go
  - NOTICE
  - docs/benchmarks.md
  - docs/ARCHITECTURE.md
  - docs/measurements/demo-chat-macbook-2026-08-22.md
  - demo/chat/README.md
---

## The problem

Running a local model normally takes three steps. You install a runtime, you find a model file, and you download it, often several gigabytes. Each step can fail on a machine that is offline, locked down or read-only. To hand a model to someone else, you hand over a runtime, a file, and instructions.

## What goinfer does

Each release carries a `goinfer-chat-0.5b-<os>-<arch>` file with the runtime and a small coder model in it (0.5B, about half a billion parameters), and a `goinfer-chat-1.5b-<os>-<arch>` file with a 1.5B one. The latest release is v0.19.0, from 2026-09-19. Its 0.5B file for macOS arm64 is 653,131,218 bytes (about 653 MB), and its 1.5B file is 1,810,622,418 bytes (about 1.81 GB). One `checksums.txt` covers all 27 files in the release.

```sh
curl -LO https://github.com/townsendmerino/goinfer/releases/download/v0.19.0/goinfer-chat-0.5b-darwin-arm64
curl -LO https://github.com/townsendmerino/goinfer/releases/download/v0.19.0/checksums.txt
grep ' goinfer-chat-0.5b-darwin-arm64$' checksums.txt | shasum -a 256 -c
chmod +x goinfer-chat-0.5b-darwin-arm64
xattr -dr com.apple.quarantine ./goinfer-chat-0.5b-darwin-arm64   # macOS only: the binary is unsigned
./goinfer-chat-0.5b-darwin-arm64
```

`--version` says what the file is: the release tag, the backends compiled in, and a line of the form `embedded: tier=0.5b quant=int8int8`.

## How it works

The [release workflow](https://github.com/townsendmerino/goinfer/blob/main/.github/workflows/release-assets.yml) downloads the model's GGUF file (a common model file format) from Hugging Face. It checks the file against a pinned sha256 and a pinned byte count. If either differs, the build stops, so a re-upload upstream cannot change the weights inside a tag that has already shipped. It also fetches the model's license and checks that against a pinned hash.

Then the build script, [`demo/chat/build-embed.sh`](https://github.com/townsendmerino/goinfer/blob/main/demo/chat/build-embed.sh), converts the GGUF once, at build time, into a prequantized bundle. Quantization stores each weight as a small integer plus a shared scale factor; here each weight is 8 bits (int8), already laid out the way goinfer's decoder reads them. The bundle is compiled into the binary with `go:embed`. At launch the program does not decompress or convert anything. It uses the weights where they sit in the executable's own image, which is why memory stays low. The bundle is the same `.giw` format goinfer writes beside any `.gguf` it serves (see [Starts in a hundredth of a second](/different/18-starts-in-a-hundredth/)). It is bigger than the 4-bit file it came from, and that is why the asset is larger than the model.

The workflow runs the same script for six targets: macOS, Linux and Windows, each for amd64 and arm64. The script builds each one with `CGO_ENABLED=0`, so no C compiler is involved ([No toolchain of any kind](/different/12-no-toolchain-of-any-kind/) covers that). A final job lists the files and writes `sha256sum -- * > checksums.txt`, and it attaches the model's license text and a `NOTICE.txt` when weights are included.

## What was measured

Every row but the last was measured on a MacBook Pro with an M1 Pro, on the CPU.

| what | when | result | where it is printed |
|---|---|---|---|
| Start, 0.5B and 1.5B (prequantized int8) | an old record, see below | 0.48 s and 1.23 s | [benchmarks page](https://github.com/townsendmerino/goinfer/blob/main/docs/benchmarks.md), "Cold start & footprint" |
| Memory the process holds (macOS `phys_footprint`) | same | 77 MB and 87 MB | same |
| Same 0.5B, with the raw GGUF embedded instead | same | 2.30 s and 772 MB, against 0.48 s and 78 MB | [architecture notes](https://github.com/townsendmerino/goinfer/blob/main/docs/ARCHITECTURE.md), the `.giw` section |
| Generation speed, 0.5B and 1.5B, in tokens per second (tok/s) | 2026-08-22, commit [`3a27941`](https://github.com/townsendmerino/goinfer/commit/3a27941) | median of 5 runs: 52.3 and 27.8 | [the 2026-08-22 record](https://github.com/townsendmerino/goinfer/blob/main/docs/measurements/demo-chat-macbook-2026-08-22.md) |
| File sizes, macOS arm64 | read from release v0.19.0 on 2026-09-19 | 653,131,218 and 1,810,622,418 bytes | [the release's file list](https://github.com/townsendmerino/goinfer/releases/tag/v0.19.0) |

Read the first three rows as history. The benchmarks page labels them "v0.5.0-era" (v0.5.0 is dated 2026-06-11) and "M1 Pro, `.giw` int8", and says nothing has replaced them. The changelog first prints 0.48 s under v0.1.3 (2026-06-05) and calls the machine "M-series CPU". The repo has no raw log for these rows. The demo README shows the same 0.48 s as the time to load the model, while the benchmarks page calls it the time to the first token. None of it was re-measured on a v0.19.0 file, or with the Metal backend that the macOS files now include.

The speed row is a Go benchmark run in-process, not the shipped binary. It loaded the same two models from their 4-bit GGUF files at the int8 setting the release build bakes in. It is the one speed record in the repo for this quantization and these two model sizes.

## Use it

- Pick the file for your platform and model size from the [release page](https://github.com/townsendmerino/goinfer/releases/latest). Names end in `darwin-arm64`, `darwin-amd64`, `linux-amd64`, `linux-arm64`, `windows-amd64.exe` or `windows-arm64.exe`.
- Run it with no arguments for a chat. `--help` lists the flags, including `--backend`, whose values are `cpu`, `webgpu`, `cuda` and `metal`.
- To use your own model with the same program, take the plain `goinfer-chat-<os>-<arch>` file (9,370,146 bytes for macOS arm64 in v0.19.0) and pass `--model` a GGUF path.
- To build one yourself, from a clone of the repo: `./demo/chat/build-embed.sh --name goinfer-chat-0.5b <model.gguf> darwin/arm64`. The output goes to `demo/chat/dist/`.
