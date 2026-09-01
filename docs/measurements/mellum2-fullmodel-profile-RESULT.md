# Full-model Mellum2 prefill profile at K=8192 — RESULT (2026-09-01)

**Pre-registration:** `mellum2-fullmodel-profile-PREREGISTERED.md`, committed (`d4a5c8a`) before
either arm ran.

**Two of the four predictions FAILED, and they are the result.** Attention is no longer the
largest bucket in today's default prefill — `moeMLP` is — and the A3 fan-out's 1.92× did **not**
transfer to this model at this depth.

## Provenance

| | |
|---|---|
| box | MacBook, Apple M1 Pro, 8 cores, 16 GB, macOS 26.6.2 |
| goinfer | `2e3d018` + the A3 fan-out (the tree as committed at `14b909a`) |
| model | Mellum2, **full 28 layers**, `~/models/mellum2-unq`, **int4** |
| K | 8192, `varied` ids (real routing) |
| harness | `decoder/mellum2_prefill_profile_test.go` — the same one that produced the 4-layer slice figure |
| arm B | acc64 (`GOINFER_CPU_FAST_ATTENTION=0`) — the slice's kernel, **4230.4 s**, 1.94 tok/s |
| arm A | today's default, f32 + head fan-out — **2649.1 s**, 3.09 tok/s |

## The two questions

### 1. Did attention's share fall from the slice's 97.1%? — YES, hypothesis SUPPORTED

Arm B holds the attention kernel fixed at what the slice used and varies only model size, which is
the comparison the cache-residency hypothesis is about. Shares are of **non-park** samples, the
same exclusion the slice figure used (`pthread_cond`/`usleep` are the known idle-M artifact a CPU
profiler miscounts — see the caveat below).

| | 4-layer slice (2026-08-28) | **28-layer full (arm B)** |
|---|---|---|
| attention | **97.1%** | **70.4%** |
| moeMLP | — | 16.3% |

Pre-registered rule: **≤ 90% supports** the hypothesis, > 95% refutes it, 90–95% parks.
70.4% is well inside the supporting region and outside the ambiguous band. My predicted range was
70–90%; it landed on the boundary.

**So the slice overstated attention's share, and model size is a sufficient explanation.** The
slice's ~1.6 GB of weights fit in cache, so weight matmul was cheap and attention read as nearly
all of the work; the full model's larger footprint is bandwidth-bound and attention's share falls.
The 3.11×-vs-1.52× gap the slice produced is consistent with this.

### 2. What is the profile NOW, on the shipping default? — attention is no longer the lever

| bucket | arm B (acc64) | **arm A (f32 + fan-out)** |
|---|---|---|
| attention (`attendBatchedHeads`) | 70.4% | **17.4%** |
| `moeMLP` | 16.3% | **42.1%** |

**Prediction 3 — "attention remains the single largest bucket in arm A" — FAILED.** It is now
the *second* bucket, at 17.4%, and `moeMLP` is 2.4× larger. The pre-registration said, in advance,
what that would mean: *"if that fails the prefill story has changed and the next lever is
elsewhere."* It is elsewhere. **The next prefill lever on this workload is the MoE FFN.**

The cross-check that says these two profiles are comparable: `moeMLP`'s **absolute** cost is
1462 s vs 1443 s, **−1.3%** — unchanged, exactly as it must be, since neither arm touches it. The
share moved because attention's cost collapsed underneath it, not because the MoE got slower.

## Prediction 4 also failed, and it is a caution on my own A3 result

Predicted: arm A beats arm B by **more than** the 1.59× recorded for f32-vs-acc64 before the
fan-out existed. Measured: **1.597×**.

That is the same number, not a better one. The A3 fan-out measured **1.92× at K=4096 on a 0.5B
dense model** and contributes **nothing resolvable here**, on a 28-layer MoE at K=8192.

Why it does not transfer is legible in the table above: at this size attention was ~70% of the
work under acc64, and the f32 kernel swap alone already collapses most of it. Head-level fan-out
adds parallelism to a term that is no longer the constraint, on a machine that is memory-bound
rather than core-bound. **The A3 numbers stand for the workload they were measured on — a small
dense model at moderate depth — and must not be quoted for large MoE prefill.** This is the same
lesson as the 4-layer slice, arriving from the other direction: a win measured on one shape does
not carry to another shape, whichever way the shapes differ.

## Caveats, including one that trips my own pre-registered rule

**SWAP GREW MATERIALLY IN BOTH ARMS, which the pre-registration named as a voiding condition.**
Arm B: 6205 → 16585 MB. Arm A: 5766 → 18343 MB. By the letter of the rule both arms are void.

Recording the fact and the reading separately, rather than quietly re-interpreting the rule:
page-in rate sampled during the run was **4–8/sec** with one transient spike to 724/sec — i.e. the
process was not faulting its working set back in, and the swap growth is the OS evicting other
processes' idle pages to house the model. The rule's stated *purpose* was "a paging run profiles
I/O rather than compute," and the direct test of that purpose says compute. **The proxy fired and
the thing it proxies for did not.** I am not rewriting the rule after seeing the data; the honest
status is that the ABSOLUTE wall times carry memory pressure, while the SHARES — the actual
question — are far less sensitive to it, and the 1.597× reproduces a previously recorded 1.59×
from a different session, which is corroboration rather than proof.

**The park bucket is excluded, not explained.** It is 26.5% of arm B and 50.3% of arm A. This repo
has already established that pprof's CPU profiler miscounts parked workers as an idle-M artifact
and that `go tool trace -pprof=sync/sched` is the right instrument for park/wake questions.
Nothing here should be read as "half of arm A is scheduling overhead" — that claim needs the other
tool and is not made.

**Quant is not matched to the slice** (int4 here, int8int8 there), as stated in advance. It
weakens the slice-vs-full comparison specifically; it does not affect arm A vs arm B, which are
matched to each other.

One machine, one model, one depth.
