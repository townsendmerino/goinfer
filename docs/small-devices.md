<!-- Moved verbatim from the README on 2026-09-30 (links re-pointed; the README keeps the cross-compile example). -->

# Small devices — a Raspberry Pi, not a microcontroller

Because the whole build is `CGO_ENABLED=0`, a 64-bit ARM Linux board is an ordinary
cross-compile and a copy; nothing on the board needs installing:

```bash
GOOS=linux GOARCH=arm64 CGO_ENABLED=0 go build -o goinfer-serve ./cmd/serve
scp goinfer-serve pi@raspberrypi.local:
```

The published `goinfer-serve-linux-arm64` asset works too — it carries the CUDA backend, which
looks for `libcuda.so.1` at startup and declines to the CPU path when it is absent, so on a board
with no NVIDIA driver it is simply the CPU server.

What to expect, and what to check before quoting a number:

- **Memory is the floor.** The smallest checkpoints this project vets are the 270M–0.5B
  class; at int4 they want a few hundred MB for weights plus the KV cache, so a 512 MB board
  (Pi Zero 2 W) is the low end, running one small model with a short context. Weights load
  from a `.giw` bundle mapped from disk rather than copied into the heap, which is what makes
  the low end reachable at all — see [`docs/giw-bundles.md`](giw-bundles.md).
- **The fast arm64 kernels need the DotProd extension (ARMv8.2 `SDOT`).** goinfer reads
  the kernel's HWCAP at startup and uses `SDOT` where it exists — Cortex-A76 and newer, so a
  Pi 5 — and the base NEON kernel where it does not (Cortex-A53/A72: Pi Zero 2 W, Pi 3, Pi 4).
  Both are correct; the base kernel is several times slower.
- **No board row is published yet.** The numbers in [`docs/benchmarks.md`](benchmarks.md)
  come from a MacBook and a desktop GPU box, and a Pi figure will appear there only once it has
  been measured under the same rules (local disk, pinned versions, thermal note). Until then,
  treat "a small model decodes at a usable rate on a Pi 5" as the expectation, not a claim.

**Out of scope: microcontrollers and TinyGo.** An ESP32- or RP2040-class part has kilobytes to a
few megabytes of RAM against a model that needs hundreds, so the gap is not one a compiler can
close. TinyGo would also drop the hand-written `.s` kernels and most of `os`/`net/http`, which
is the serving path. 32-bit ARM (`GOARCH=arm`) is not a target this project builds or tests.

Back to the [README](../README.md).
