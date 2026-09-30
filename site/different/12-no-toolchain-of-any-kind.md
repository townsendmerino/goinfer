---
title: "No toolchain of any kind"
area: "Distribution"
order: 12
summary: "goinfer builds with CGO_ENABLED=0: no C compiler, CUDA toolkit or Python to build or run it, and CUDA and Metal are reached without cgo."
stand: "The default build, the CUDA build and the Metal build are all pure Go. The GPU is reached at run time through the driver already on the machine, so the only build tool you need is Go."
measured: 2026-09-18
reviewed:
facts:
  - {label: "cgo", value: "none in the default, CUDA and Metal builds"}
  - {label: "CUDA toolkit", value: "not needed to build or run"}
  - {label: "needed at run time", value: "the GPU driver (libcuda.so.1 for CUDA)"}
  - {label: "WebGPU", value: "the one backend that uses cgo, behind the gpu build tag"}
doesnt:
  - title: "It doesn't need nothing."
    text: "Building from source needs Go 1.27 or newer. Running on a GPU needs the vendor's driver: on Linux the CUDA backend opens libcuda.so.1 when it starts, and if that fails it says so on stderr and runs on the CPU."
  - title: "WebGPU is not toolkit-free."
    text: "The WebGPU backend is the one place cgo is used, through a Go binding over wgpu-native. It lives in its own module behind the gpu build tag, and it is not in the release binaries because those are built with CGO_ENABLED=0. cgo needs a C compiler on the build machine, and at run time the binding drives whichever of Vulkan, Metal or DX12 the platform has."
  - title: "The CUDA binary is not fully static."
    text: "It has no cgo, but it links libc, libdl and libpthread and loads libcuda.so.1 at run time. The project's own spike notes call this \"cgo-free plus driver-only\" and say it is not the pure static-binary story."
  - title: "Changing a CUDA kernel does need a toolchain."
    text: "The kernels are written in CUDA C and compiled to PTX with NVRTC ahead of time, by cuda/build_ptx.sh, then embedded in the binary. The audited files are pinned to NVRTC 12.6.85. Users of the binary never run that step; a contributor editing a kernel does. The repo's benchmark scripts are also Python."
  - title: "The cold-install number does not prove a bare machine."
    text: "The record was taken on the developer's own box, which already had the CUDA 13.2 toolkit installed; the record says the toolkit went unused, but it did not test a machine without one. Because the driver compiles the PTX, a driver upgrade can change the generated code."
figures:
  - {text: "12.97", source: "docs/measurements/cold-user-2026-09-18-nobara-pc.md"}
  - {text: "6.19", source: "docs/measurements/cold-user-2026-09-18-nobara-pc.md"}
  - {text: "2.05", source: "docs/measurements/cold-user-2026-09-18-nobara-pc.md"}
  - {text: "595.91.07", source: "docs/measurements/cold-user-2026-09-18-nobara-pc.md"}
  - {text: "932 KB", source: "docs/cuda-backend.md"}
  - {text: "14.6 MB", source: "docs/cuda-backend.md"}
  - {text: "4.94", source: "docs/cuda-backend.md"}
  - {text: "4.09", source: "docs/cuda-backend.md"}
  - {text: "595.58.03", source: "docs/cuda-backend.md"}
sources:
  - "docs/measurements/cold-user-2026-09-18-nobara-pc.md"
  - "docs/ARCHITECTURE.md"
  - "docs/cuda-backend.md"
  - "docs/webgpu-primer.md"
  - "docs/completed/task-cuda-cgofree-spike.md"
  - "cuda/build_ptx.sh"
  - ".github/workflows/ci.yml"
  - ".github/workflows/release-assets.yml"
  - "README.md"
---

## The problem

The usual way for a Go program to call GPU code is cgo, which compiles C alongside the Go. That means a C compiler on the build machine, often a vendor toolkit as well, and a harder cross-compile, because a C compiler for the target has to be there.

Each of those is one more thing that can be the wrong version on someone else's machine. goinfer tries to leave all of them out.

## What goinfer does

The default build is pure Go, built with `CGO_ENABLED=0`. Building the CPU server needs Go and nothing else:

```sh
go install github.com/townsendmerino/goinfer/cmd/serve@latest
```

For a GPU you build the backend's own entry point. These are also pure Go, and the README lists them:

```sh
go install github.com/townsendmerino/goinfer/metal/cmd/serve@latest              # macOS
go install -tags cuda  github.com/townsendmerino/goinfer/cuda/cmd/serve@latest    # Linux + NVIDIA
```

Because there is no C code, cross-compiling is an ordinary Go cross-compile. The README's example for a Raspberry Pi is `GOOS=linux GOARCH=arm64 CGO_ENABLED=0 go build -o goinfer-serve ./cmd/serve`.

## How it works

**CUDA.** The `cuda/` module uses a Go library, `gocudrv`, that calls the NVIDIA driver without cgo. It uses `purego`, which opens `libcuda.so.1` with `dlopen` at run time. The GPU kernels were compiled ahead of time to PTX, NVIDIA's portable intermediate form. That PTX (932 KB) is embedded in the binary, and the driver compiles it for whatever card is installed. So the CUDA toolkit is used only by whoever regenerates the PTX, never to build or run goinfer.

**Metal.** The `metal/` module calls the Objective-C runtime through `purego` and compiles its Metal Shading Language kernels when the program runs. Every file is marked `//go:build darwin`, so a Mac build includes it without a tag.

**Keeping it out of the default build.** The backends are separate modules that register themselves into the `decoder` package. The root module does not import them. CI checks this on every push: neither the compiled package list nor the module graph of the root may mention a WebGPU binding, `gocudrv`, `purego`, `aikit/gpu` or a backend module.

**WebGPU** is the exception. It uses cgo through `oliverbestmann/webgpu`, in the `gpu/` module behind `-tags gpu`. The prebuilt `wgpu-native` libraries arrive as Go modules, so you do not build any Rust. It works over Vulkan, Metal or DX12.

## What was measured

A cold-user run on 2026-09-18 timed the install of v0.18.0 with the module cache redirected to an empty directory, so every dependency came over the network.

| Step | Time | Note |
|---|---|---|
| `go install` of `cmd/serve@v0.18.0`, empty cache | 12.97 s | fetched goinfer, aikit, x/text and the Go 1.27.1 toolchain |
| `go install -tags cuda` of `cuda/cmd/serve@v0.18.0`, straight after | 6.19 s | only the CUDA dependencies were new: gocudrv, aikit/gpu, purego |
| `go install` of `demo/chat@v0.18.0` | 2.05 s | dependencies already present |

The machine was a Ryzen 7 3700X with an RTX 2070 SUPER, Nobara Linux, NVIDIA driver 595.91.07, Go 1.27.0. That run timed the builds, not a CUDA decode: it shows the CUDA build needing no extra compiler step, not what the card did.

Two figures from `docs/cuda-backend.md`, both older. The CUDA server is a 14.6 MB binary. On an earlier driver (595.58.03), with a 0.5B model, start to ready took 4.94 s with the driver's JIT cache cleared and 4.09 s with it warm. The doc says that table must be re-measured after the driver change.

## Use it

- CPU server: `go install github.com/townsendmerino/goinfer/cmd/serve@latest`.
- CUDA, from a module path: `CGO_ENABLED=0 go build -tags cuda github.com/townsendmerino/goinfer/cuda/cmd/serve`, then run with `--backend cuda`.
- WebGPU (needs cgo): `go build -tags gpu github.com/townsendmerino/goinfer/gpu/cmd/serve`.
- To see what a binary carries: `goinfer-serve --version` prints its backends.
- `-tags cuda` on the root `cmd/serve` does not work. It fails the build and points you to the module path above.
