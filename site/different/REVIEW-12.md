# REVIEW-12: "No toolchain of any kind" (draft, not reviewed)

## Claims and sources

| Claim | Source |
|---|---|
| Default build pure Go, CGO_ENABLED=0; root requires only aikit and x/text | docs/ARCHITECTURE.md section 6 (first paragraph); go.mod |
| No C compiler / CUDA toolkit / Python to build or run | README lines 5 and 364-365; docs/cuda-backend.md lines 1-5, 25 |
| Five modules, backends separate, WebGPU is the only cgo one | docs/ARCHITECTURE.md "Modules" table |
| CI cleanliness guard: `go list -deps` and `go list -m all` must not mention webgpu, gocudrv, purego, aikit/gpu, backend modules | .github/workflows/ci.yml lines ~213-241; ARCHITECTURE section 6 |
| CUDA: gocudrv + purego dlopen libcuda.so.1; embedded PTX, driver JIT; NVRTC only offline | ARCHITECTURE "CUDA" section; cuda/doc.go; cuda/build_ptx.sh header |
| 932 KB embedded PTX, 14.6 MB binary, JIT 4.94 s cold / 4.09 s warm on driver 595.58.03 | docs/cuda-backend.md "What ships" (pre-2026-08-25 driver, the doc says re-measure) |
| Metal: purego + Obj-C, MSL compiled at run time, every file `//go:build darwin`, no tag | ARCHITECTURE "Metal"; metal/backend.go line 1 |
| WebGPU: oliverbestmann/webgpu, cgo, -tags gpu, own module, not in release assets, Vulkan/Metal/DX12; prebuilt libs come as Go modules | ARCHITECTURE "WebGPU"; docs/webgpu-primer.md layer 4; gpu/go.mod (libs-* entries); release-assets.yml comment "needs cgo, so it cannot be in a static release asset" |
| Cold-user timings 12.97 / 6.19 / 2.05 s, machine, driver 595.91.07, Go 1.27.0, v0.18.0, empty GOMODCACHE | docs/measurements/cold-user-2026-09-18-nobara-pc.md, Scenario A Numbers table and contamination item 3 |
| That run had CUDA 13.2 toolkit installed but unused | same record, contamination item 5 |
| CUDA binary is dynamically linked to libc/libdl/libpthread, "cgo-free + driver-only" | docs/completed/task-cuda-cgofree-spike.md "Static-binary nuance" (line ~244) |
| Missing driver: falls back to CPU with a one-line stderr note | docs/cuda-backend.md "No NVIDIA driver / dlopen fails" |
| Go 1.27+ to build from source | README line 27 |
| Commands: cmd/serve, metal/cmd/serve, `-tags cuda` cuda/cmd/serve, Pi cross-compile, `--version` prints backends, `-tags gpu` gpu/cmd/serve, root -tags cuda fails the build | README Install and Small devices; docs/cuda-backend.md lines 11-14, 265-290 |
| NVRTC pinned 12.6.85 for audited PTX; regen uses a venv with NVRTC wheels | cuda/doc.go; cuda/testdata/REGEN.md header |
| Repo benchmark scripts are Python | scripts/bench_peer.py exists |

## Conflicts and how I resolved them

- **CUDA platform.** docs/cuda-backend.md says Linux or Windows x86-64; aikit/gpu's own doc.go says CUDA on linux only, and release-assets.yml builds CUDA only for linux amd64/arm64 (Windows gets the CPU root build). I claimed nothing about Windows.
- **Root go.mod.** ARCHITECTURE says the root requires only aikit and x/text; go.mod also lists golang.org/x/sys as `// indirect`. Not material to the claim (x/sys is not a backend); I said "no backend" rather than repeat the "only" wording. The CI guard does not list x/sys.
- **CUDA scope in docs/cuda-backend.md** ("dense residency only") is stale against ARCHITECTURE (MoE, MLA etc.). Not relevant here; I did not use it.
- **README "static binary" vs CUDA binary**: README's headline says one cgo-free static binary; the spike notes say the CUDA build is dynamically linked to libc/libdl/libpthread. I used the spike's wording in "doesnt". The default CPU build's linkage was not checked.
- **Cold-user record vs the claim.** The 6.19 s figure is a build time with deps partly cached from the 12.97 s install before it. The record's "no toolkit needed" is a declaration that the toolkit (13.2) was present and unused, not a test on a bare machine. The same record's headline finding is that v0.18.0's GitHub Release had zero binary assets (the README's download path 404ed); v0.19.0 is now Latest per `gh release list`. I did not assert anything about release assets in the page and did not check whether v0.19.0 has them.

## Left out as not verified

- `ldd` output of a current CUDA binary (docs and spike quote it; I did not run it).
- That a machine with no C compiler and no toolkit builds `-tags cuda` successfully (no record tests it; CI's ubuntu runner has a stock image).
- That WebGPU needs a C compiler: stated in the page as a general property of cgo, not a repo measurement.
- Binary sizes for the default CPU build, and any comparison with other runners (left out on purpose).
- Whether the Metal MSL compile needs Xcode: docs/cuda-backend.md says no, spike says the compiler ships in the OS; no measurement cited, so the page only says it compiles at run time.
- The M-19 audit note itself: I cited the CI comments and ARCHITECTURE that reference it, not the audit doc under docs/completed.

## Questions for the owner

1. Is it fair to headline "no C compiler" when the only timed install was on a box with the CUDA toolkit present? I kept the claim in the summary but put the limit in "doesnt". Do you want a bare-VM run (no toolkit, no gcc) queued for the night?
2. The JIT start-up figures (4.94 / 4.09 s) are on an older driver. Keep them with the caveat, or drop until re-measured?
3. Should the page say plainly that Go 1.27+ is a hard requirement (it does, in "doesnt"), given the README's "no toolchain of any kind" wording?
