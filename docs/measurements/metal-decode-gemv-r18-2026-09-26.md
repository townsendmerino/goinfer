# R18 — Metal decode GEMV, an MLX-shaped bit-identical int4 GEMV: band PARK at 1.17×, SHIPPED by owner decision (2026-09-26)

Pre-registration: `docs/tasks/red-october.md` § R18 (`d16a04a5`). The measurement note and the confirmation's fixed
parameters were committed in `afd77749`, before the graded run. S0:
[`metal-decode-gemv-s0-2026-09-26.md`](metal-decode-gemv-s0-2026-09-26.md).

**Owner decision, 2026-09-26: ship.** In the owner's words, "i'm happy to park stuff if it's a couple percent above
current code, but giving up gains seems stupid. ship it". Future bands are to be much more permissive than 1.35×.
The band's verdict below is kept as measured. Wiring is step 3, and its gates are recorded in § Production wiring.

**Verdict by the registered band: park.** The selected candidate, `i2244`, is bit-identical to the shipped kernels at every decode position
tested and faster at every depth on both models. Its in-sequence int4-GEMV speedup at depth 128 is:
- **1.171× on the 1.5B** (7 paired reps, 1.151–1.191), which is the grade, since the band is graded on the weaker model;
- **1.317× on the 7B** (1.305–1.329).

The registered band is ship ≥ 1.35× / park 1.15–1.35× / kill < 1.15×, so this is **park**, 0.02 above the kill line.
By the band, nothing is wired.

What it would buy is measured here as GPU time, not end to end: the full decode token goes from 12.98 to 11.36 ms on
the 1.5B (−12.5%) and from 45.62 to 35.94 ms on the 7B (−21%) at depth 128. The registration's own arithmetic turns
that into roughly 0.97× Ollama on the 1.5B and 1.10× on the 7B; that is a projection, not a `bench_peer.py` row.

On gate/up, about 1.25–1.28× is as far as any kernel shape tried here goes, whatever its arithmetic or activation
source. The 1.5B's qkv and o are too small to gain. That combination is why the 1.5B cannot reach the ship line with
this design.

## Setup

- M1 Pro 16 GB, macOS 26.6.2. W4A8 decode path, `attention_fa` on, with the resident context pinned to 4096.
- Models from `~/models`:
  - qwen2.5-coder-1.5b-instruct q4_k_m (`.gguf`, via its sidecar);
  - qwen2.5-7b-instruct q4_k_m (`.int4.metal.giw` sidecar, aliased; its `.gguf` is refused by the fit guard on this
    machine, and the guard was not bypassed).
- GPU time comes from each decode token's command-buffer timestamps, as in `TestMetalDecodeDecomp`.
- Instruments: `metal/gemv_r18_test.go` (standalone prototypes, exploratory) and `metal/gemv_r18_seq_test.go`
  (`TestR18InSequence`, the grading instrument). The in-sequence hook is `resident.r18Rows`
  (`metal/model.go`): rows per simdgroup at the four dense decode GEMV sites. It is zero in production, so every grid
  is unchanged; `TestMetalSnapshotGolden` passed 10/10 byte-identical with it in place.
- Raw logs: [`metal-decode-gemv-r18-2026-09-26/`](metal-decode-gemv-r18-2026-09-26/).

## What was tried

All of these are bit-identical by construction except `fm`. Each row's lane-to-group (SA) or lane-to-word (coal) order
is the shipped kernel's, and so is the `float(sum)·scale` accumulation. The integer sums are exact, and so are the f32
forms, which use exact power-of-two constants.

| step | variant | idea |
|---|---|---|
| 0 | `sa_r{1,2,4}` | R rows per simdgroup. The group's 32 staged activations are read once into registers and reused for R rows; the integer math (`UNP8V`) is unchanged. |
| 1 | `sa_ib`, `sa_fr` | `ib`: integer, with the −8 folded per group (−8·Σa). `fr`: MLX's shift-free masked f32 FMA against activations pre-scaled in registers by exact 16⁻ʲ. |
| 1 | `sa_fm` | The same, staged as pre-scaled floats via `exp2`. **Not bit-identical** (fast-math `exp2` is inexact: 1,245–31,249 outputs differ per shape); dropped. |
| — | `sa_ib8`, `sa_iu{2,4}` | R = 8; the group loop unrolled by 2 to put more loads in flight. |
| — | `sa_dv{i,f,s}` | MLX's activation path: no threadgroup staging. Each lane reads its group's int8 activations from device memory (L1-hot) as two `uint4`, in integer form, f32 form, or MLX's 16-bit masks. |
| 2 | `resid_st{1,2,4,8}` | The down projection (the coal family): activations staged once as int8, and R rows per simdgroup, keeping coal's per-word order. |

## Exploratory results (select; they do not grade)

### Standalone, speedup over the shipped kernel per dispatch (median of 7, SLC-defeating rotation)

Load was 1.8–3.2 during these runs, above the 2.0 idle gate, which is one reason they only select.

| model | GEMV | r2 | r4 | ib4 | fr4 | dvf4 | ib8 | iu4 | st2 | st4 | st8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.5B | qkv | 1.08–1.14 | 1.03–1.12 | 1.08–1.11 | 1.07–1.15 | 1.18 | 0.89 | 0.81 | | | |
| 1.5B | o | 1.07–1.28 | 1.01–1.19 | 1.01–1.12 | 1.07–1.17 | 1.13 | 0.77 | 0.76 | | | |
| 1.5B | gate/up | 1.17–1.18 | 1.20–1.24 | 1.19–1.23 | **1.28** | 1.25 | 1.15 | 0.82 | | | |
| 1.5B | down | | | | | | | | 1.20–1.22 | **1.27–1.29** | 0.98 |
| 7B | qkv | 1.16–1.22 | 1.14–1.20 | 1.16–1.21 | 1.08–1.12 | 1.22 | 1.05 | 0.88 | | | |
| 7B | o | 1.14–1.18 | 1.19–1.22 | 1.21–1.25 | 1.16–1.20 | 1.33 | 0.99 | 0.93 | | | |
| 7B | gate/up | 1.19–1.20 | 1.26–1.27 | 1.26–1.27 | 1.24–1.27 | 1.25 | 1.24 | 0.95 | | | |
| 7B | down | | | | | | | | 1.21–1.25 | **1.50–1.54** | 1.34–1.38 |

`dvi` (integer math on `char4` activations) collapsed to 0.32–0.35×, and `fm8` to 0.15–0.17×; both look like register
spills.

**gate/up is capped at ~1.24–1.28× whatever the arithmetic, the activation source, R or the unrolling.** At R = 4 that
is 115–140 GB/s, against 176–187 GB/s for S0's loads-only twin on the same shapes. Step 1's float form, the thesis's
main lever, is **not** what moves it: at R = 4 every bit-identical variant ties within noise.

### In sequence, `TestR18InSequence` (bit-identical through the executor at depth 128 for every arm)

A candidate is written `<F><qkvR><oR><guR><downR>`. F is `i` (integer) or `f` (f32); each digit is rows per simdgroup,
and a downR of 0 keeps the shipped coal kernel. `i1110` is the harness control, the candidate template at the shipped
grid.

**Block timing failed its own control on the 1.5B.** With 20 steps per side, as `TestMetalDecodeDecomp` does it, the
control's per-rep ratio ran 0.47–1.25, and one rep's summed work read 4.2 ms against 8–9 in the others. On the 7B the
same block timing was clean (control 0.99–1.04). Step-paired timing (a full token, then the no-op'd one, adjacent, with
the median of 20 pair differences) brought the 1.5B control to 0.984–1.002. It was adopted before any grading run
(`afd77749`, R18 measurement note).

| arm | 1.5B metric (step-paired, 5 reps) | 1.5B full token | 7B metric (block, 5 reps) | 7B full token |
|---|---:|---:|---:|---:|
| current | 1.000 (9.10 ms) | 12.99 ms | 1.000 (38.66 ms) | 45.72 ms |
| `i1110` (control) | 0.991 | 13.04 | 1.006 | 45.76 |
| `f4444` | 1.192 | 11.27 | 1.242 | 38.03 |
| `f2244` | 1.175 | 11.31 | 1.225 | 38.26 |
| `i2244` | **1.194** | 11.38 | **1.334** | 35.93 |
| `i4444` | 1.153 | 11.57 | — | — |
| `f1144` | — | — | 1.200 | 38.68 |
| `f2242` | — | — | 1.115 | 40.69 |

On the 1.5B, qkv and o gain nothing at any R. They are 1.3–1.8 MB GEMVs; at R = 4 the o projection's 1,536 rows make
only 48 threadgroups, and it is **slower** (0.80×). On the 7B the f32 form's gate/up is *worse* in sequence than the
integer form (17.2 against 15.3 ms), though they tie standalone. `i2244` is the best uniform candidate on both models and
the simplest (step 0's integer math unchanged), so it went to the confirmation run.

## Confirmation run (grades)

A fresh run of `i2244` alone against the shipped kernels, one process per model, with the parameters fixed in
`afd77749`: 7 paired reps, 20 step pairs per category, 16 identity positions, and 3 after-idle samples. The 1.5B ran
13:26:46–13:29:50 PDT and the 7B 13:29:50–13:38:01 PDT, with load1 1.55 and 1.68 at start. Logs:
[`confirm-1.5b.log`](metal-decode-gemv-r18-2026-09-26/confirm-1.5b.log) and
[`confirm-7b.log`](metal-decode-gemv-r18-2026-09-26/confirm-7b.log).

**Precondition 1, bit-identity.** At every depth (128, 2048 and 3900) on both models, 0 of 16 teacher-forced positions
differ, and 0 of 151,936 / 152,064 logits differ at any position, through the production executor
(`ForwardEmbPipe`). `TestMetalSnapshotGolden` passed 10/10 with the hook in place. That is expected: the hook is zero
in production.

**The metric**, int4-GEMV work per token, current ÷ `i2244`:

| model | depth | current work | `i2244` work | **metric** (7 reps, range) | qkv | o | gate/up | down |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.5B | **128** | 9.303 ms | 7.945 ms | **1.171×** (1.151–1.191) | 1.07× | 0.95× | 1.19× | 1.20× |
| 1.5B | 2048 | 9.423 | 7.714 | 1.218× (1.189–1.299) | 1.33× | 1.24× | 1.21× | 1.23× |
| 1.5B | 3900 | 9.274 | 7.609 | 1.221× (1.211–1.238) | 1.22× | 1.23× | 1.19× | 1.27× |
| 7B | **128** | 38.924 | 29.466 | **1.317×** (1.305–1.329) | 1.15× | 1.08× | 1.28× | 1.47× |
| 7B | 2048 | 38.917 | 29.141 | 1.337× (1.322–1.343) | 1.15× | 1.24× | 1.29× | 1.48× |
| 7B | 3900 | 38.553 | 29.212 | 1.332× (1.300–1.383) | 1.12× | 1.10× | 1.28× | 1.51× |

The all-four-no-op'd reading, a single difference per rep, agrees:
- 1.5B at 128: 9.139 → 7.504 ms, 1.22×;
- 7B at 128: 38.722 → 29.396 ms, 1.32×.

The 1.5B's small categories (0.5–0.8 ms summed over 28 layers) move by ±0.1 ms between depths; that is noise at their
size, and the GEMVs do not depend on depth.

**Precondition 2, no regression: the full token (GPU ms, median).**

| model | depth 128 | 2048 | 3900 |
|---|---|---|---|
| 1.5B | 12.984 → **11.361** (−1.62) | 14.124 → **12.420** (−1.70) | 14.886 → **13.156** (−1.73) |
| 7B | 45.624 → **35.943** (−9.68) | 49.402 → **39.616** (−9.79) | 52.014 → **42.250** (−9.76) |

**Precondition 3, after 2 s idle** (the first token after sleeping; 3 samples per arm, alternating):

| model | depth | current | `i2244` |
|---|---:|---|---|
| 1.5B | 128 | 16.5 / 24.9 / 16.9 | 21.5 / 14.0 / 15.9 |
| 1.5B | 2048 | 17.7 / 23.3 / 16.3 | 22.3 / 19.6 / 16.3 |
| 1.5B | 3900 | 17.3 / 20.8 / 18.5 | 20.1 / 20.0 / 23.2 |
| 7B | 128 | 52.2 / 47.7 / 47.7 | 44.1 / 40.6 / 41.5 |
| 7B | 2048 | 52.4 / 51.1 / 51.8 | 45.1 / 42.3 / 44.2 |
| 7B | 3900 | 58.2 / 55.5 / 55.9 | 44.5 / 46.4 / 46.0 |

On the 7B the win holds after idle. On the 1.5B, the first token after idle is dominated by clock ramp-up (16–25 ms
against a 13 ms sustained token), and the two arms are indistinguishable within three samples. That is not a win that
exists only after idle.

## Reading

- **The grade is park, and the weaker model sets it.** The 7B is within 0.03 of ship at every depth. The 1.5B is
  held down by two things this design cannot fix:
  - its qkv and o GEMVs (1.3–1.8 MB each) gain nothing at any R;
  - gate/up stops at ~1.2× in sequence, as it does standalone.
- **Rows per simdgroup is the whole win.** Step 0's integer kernel at R = 2/4 and step 2's staged down projection
  carry the confirmed result, which uses no float arithmetic at all. In sequence on the 7B, step 1's shift-free f32
  form was *worse* on gate/up (17.2 against 15.3 ms); standalone it ties.
- **What limits gate/up is not established.** It is not the integer multiply: the float forms remove it and gain
  nothing. It is not the activation reads either: device-read activations and 4× register reuse gain nothing beyond
  R = 4. The loads-only twin still runs 1.3–1.5× faster. The twin feeds its loads into an XOR, so it carries neither
  the dependent arithmetic chain nor that chain's register footprint. Occupancy and latency hiding are therefore the
  natural next suspect, but nothing here measured them. Candidates for a later look are a per-shape R rule, TG sizes
  other than 256, and more rows per threadgroup at fewer simdgroups, to bring the 1.5B's gate/up, qkv and o along.
- **Wiring was an owner decision, not a band outcome (and the owner took it: ship).** The band says park. The candidate carries no fidelity risk
  (bit-identical, so no fidelity gate applies) and no speed risk (faster at every depth on both models). What wiring
  it would cost:
  - two production kernel families (a templated SA family with the three epilogues, and a staged coal down projection);
  - a bounds guard for row counts that do not divide into 8·R rows per threadgroup, which the prototypes lack;
  - a dispatch rule (R = 2 for qkv/o, 4 for gate/up and down);
  - the per-family coverage that production wiring needs (MoE shared experts, DeltaNet, qGate and the other SA call
    sites keep the shipped kernels unless extended).

## Prior art this answers

- R18's thesis, MLX's `qmv` shape, has now been tried piece by piece:
  - register activations reused across rows per simdgroup;
  - the shift-free masked f32 FMA with −8 folded per group;
  - activations read from device instead of staged;
  - MLX's 16-bit masks.

  Only the rows-per-simdgroup reuse pays. The float arithmetic does not, which matches R1's float FMA alone
  (+2–4%).
- July's "int-MAC wall" (`completed/task-metal-cgofree-spike.md`) is not the binding limit either. Removing every
  integer multiply (`fr`, `dvf`, `dvs`) leaves gate/up where it was. What remains between these kernels and the
  loads-only twin is not arithmetic form.

## Production wiring

In progress (step 3). This section records the wiring commit and its gates.
