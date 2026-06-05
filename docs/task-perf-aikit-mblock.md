# Task (aikit): re-block the W8A8 matmul to reuse weights across M rows

> **For:** the aikit Claude, in `~/tmcode/aikit` (`aikit/linalg`).
> **Why:** goinfer added greedy speculative decoding (0.5B drafts for the 1.5B,
> token-identical to plain greedy — gate green). It verifies K draft tokens with
> one M=K target forward (`forwardN`). The win *requires* the M=K matmul to stream
> each weight from memory **once** and reuse it across the K activation rows. The
> current kernel doesn't — it re-reads every weight per row — so speculative is
> ~0.7× (slower) despite 85% acceptance. This kernel re-block is the unlock, and
> it **also speeds up prefill and the encoder** (any M>1 matmul). **Parity stays
> bit-identical** (it's a loop reorder — same per-element arithmetic).

## The bottleneck (goinfer measured it)

`linalg/quant.go` → `w8a8Span` is **rows-outer, columns-inner**:

```go
for i := range M {                       // each activation row
    aqi := aq[i*K : i*K+K]
    for j := j0; j < j1; j++ {            // each weight column
        drow[j] = float32(dotI8(aqi, bQ[j*K:j*K+K])) * as * bScales[j]
        //                          ^^^^^^^^^^^^^^^^ weight row j re-read for EVERY i
    }
}
```

At M=K the weight matrix `bQ` (the dominant memory traffic — decode is
bandwidth-bound, ~18% of roofline per `goinfer/docs/perf-campaign.md`) is streamed
**M times**, not once. Measured on goinfer:

- **End-to-end:** speculative (1.5B target + 0.5B draft, K=4, 85% accept, 4.3
  tokens/pass) runs **16 tok/s vs plain 1.5B greedy 23** — a *loss*.
- **`forwardN` per-position cost** drops 13→8 ms as K grows (K=2→8): the M=K pass
  amortizes the **fork/join dispatch** across rows, but **not** the weight
  stream — so it plateaus far above the ~1-weight-stream it needs.

The arithmetic the win needs: `forwardN(M=K)` should cost ≈ **one** weight stream
(+ K× the cheap SDOT compute), i.e. per-position cost should fall toward
`decode_cost / K`, not plateau.

## The fix — column-outer with M-register-tiling

Reuse each weight row across the M activation rows so it's read from RAM once and
served from L1/L2 for the rest:

```go
// columns outer; each weight row bj read once, reused across the M rows.
for j := j0; j < j1; j++ {
    bj := bQ[j*K : j*K+K]
    bs := bScales[j]
    for i := range M {
        if aScales[i] == 0 { dst[i*N+j] = 0; continue }
        dst[i*N+j] = float32(dotI8(aq[i*K:i*K+K], bj)) * aScales[i] * bs
    }
}
```

That alone gets the bandwidth win (the inner `i` loop keeps `bj` hot). For more,
**register-tile the M dimension**: process a small tile (e.g. mt=4) of rows
against `bj`, accumulating mt partial dots per weight load — the classic GEMM
micro-kernel, amortizing both the weight load *and* the SDOT setup. Pick the
tiling; the loop-swap is the core.

Notes:
- **M=1 is unaffected** (the decode hot path): with M=1 both orders are identical,
  so there's **zero regression risk** for single-token decode — the win is pure
  upside for M>1.
- The **parallelization is unchanged**: `parallelCols` still splits the `N`
  columns across workers `[j0,j1)`; the re-block is inside each worker's span.
- Write locality: column-outer writes `dst[i*N+j]` strided by N across the M
  rows. M is small (K≈4–8) and weights dominate, so the weight-bandwidth saving
  wins — but register-tiling (accumulate then write a tile) also fixes the write
  pattern if it matters.

## Scope

- **Priority: the W8A8 path** — `w8a8Span` (and `w8a8BatchSpan` if it shares the
  M-loop; note batched q/k/v is M=1 so it's a no-op there, but keep it correct).
  This is the speculative-verify, prefill, and reranker path.
- **Follow-ups (same pattern, lower priority):** the int8-weight (`MatmulBTQ8`),
  int4 (`MatmulBTQ4`), and f32 (`MatmulBT`) kernels also re-read weights per row;
  re-blocking them speeds prefill/encoder for those quant modes too. Do W8A8
  first; the rest can be a separate pass.

## Parity (state it, then verify)

Each output `dst[i,j]` is still exactly `dotI8(aq[i], bQ[j]) * scales` — the
re-block only changes the *order* in which the M×N outputs are computed, never the
reduction within any one of them. So it's **bit-identical** for any M. Verify:

- aikit: `go test ./...` (incl. `*_forward_golden`, the W8A8 closeness tests),
  `-race`, `gofmt`/`vet`.
- goinfer (the arbiter): `TestDecodeParity` (M=1, unchanged),
  `TestForwardN_matchesSequential` (M=K bit-identical to sequential),
  `TestSpeculativeGreedyParity` (token-identical to plain greedy).

## The arbiter — goinfer measures the win end-to-end

As with the whole campaign, a linalg microbench will mislead; goinfer is the
arbiter. After the re-block, goinfer re-runs:

1. `forwardN` per-position cost vs K — should now **fall toward decode/K** (weight
   stream amortized), not plateau.
2. End-to-end speculative vs plain 1.5B greedy on code prompts — target **≥1.3×**
   (the ship gate; the model predicts ~1.7–1.9× at 85% accept / K=4 once the
   weight stream is reused).
3. Prefill latency (a bonus check — prefill is M=prompt-len, so it should drop).

Tell goinfer the new aikit version; goinfer bumps `go.mod`, re-measures, and
records ship/park in `perf-campaign.md`. The speculative code is already merged
and gated, so it starts winning the moment this lands — no goinfer code change
needed beyond the version bump (optionally switching `forwardN` to the zero-alloc
`MatmulBTW8A8Into` once it's threaded through).

## Done

- [ ] `w8a8Span` re-blocked column-outer (weights reused across M); optional
      M-register-tiling. M=1 path unchanged.
- [ ] Bit-identical: aikit suite + `-race` green; goinfer `TestDecodeParity`,
      `TestForwardN_matchesSequential`, `TestSpeculativeGreedyParity` green.
- [ ] goinfer arbiter: `forwardN` per-position cost falls with K; end-to-end
      speculative ≥1.3× on code prompts (ship) or recorded as parked with numbers.
- [ ] (Optional follow-up) same re-block for Q8 / Q4 / f32 kernels — prefill +
      encoder speedup.
- [ ] Cut an aikit release; tell goinfer the version.
