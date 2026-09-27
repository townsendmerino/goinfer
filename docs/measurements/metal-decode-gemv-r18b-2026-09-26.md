# R18b — Metal decode GEMV in MLX's masked, half-staged form: SHIP at 1.079× over R18 (1.5B) and 1.102× (7B), bit-identical (2026-09-26)

Pre-registration: `docs/tasks/red-october.md` § R18b (`e1f2c8a4`). The candidate and confirmation parameters were fixed
in `9bafd1f3`, before the graded run. The origin is MC3's S0 probe
([`concurrency-mc2-2026-09-26.md`](concurrency-mc2-2026-09-26.md), MC3 S0). Its masked form beat R18's shipped rows
kernel at M = 1 in standalone timing, a side finding of a probe that was about batching.

**Verdict: ship.** In-sequence int4-GEMV work at depth 128, R18's production kernels ÷ the candidate `h4244`, 7 paired
reps:
- **1.079× on the 1.5B** (1.057–1.107), which is the grade, as the weaker model;
- **1.102× on the 7B** (1.039–1.106).

Every rep on both models is above 1.0, against a ship line of ≥ 1.03×. The output is **bit-identical** at every
decode position at depths 128, 2048 and 3900 on both models. There is no regression at any depth:
- the 1.5B full token is 0.75–0.88 ms faster;
- the 7B full token is 2.84–2.91 ms faster.

## The kernel

It is the SA rows kernels (`gemv_w4a8_sa_rows` / `_bias_rows` / `_resid_rows<R>`) with a different inner arithmetic.
Lane-to-group order, rows per simdgroup and the threadgroup size are unchanged, and so is the staging size (K × 2 bytes):
- **Activations are staged as half, pre-scaled by 16^-(k mod 4).** That is exact: |a| ≤ 127, and 127 · 2⁻¹² is a
  normal half.
- **Weight nibbles are masked in place per 16-bit half-word**, with no shifts, converted once per (row, half-word) and
  dotted with the four pre-scaled activations: `float(u & 0xF0) · (a / 16) = n · a`.
- **The −8 fold** uses the group's Σa, taken from the same staged halves with one exact dot per half-word.

Every product and partial sum is an integer below 2²⁴. So each group sum equals the shipped integer `gi` exactly, in
any order, and `float(gi) · scale` accumulates per row in the shipped lane order. That makes it bit-identical by
construction, and every run below checked it.

## Why the first cut lost, and the second won

| variant (in sequence, depth 128) | 1.5B: production ÷ it | 7B | gate/up, 1.5B |
|---|---:|---:|---:|
| first cut: Σa re-read from device per lane | 0.939× | 0.977× | 4.18 ms (production 3.92) |
| **second cut: Σa from the staged halves** | **1.12×** | **1.11×** | **3.16–3.19 ms** |

Standalone, the first cut had matched the probe's gain. In sequence it lost, because each lane re-read its group's 32
activations from device memory: eight simdgroups per threadgroup, each re-reading the whole activation vector. The
standalone bench, with one kernel per dispatch and a cold weight rotation, could not see that cost.

It is the lesson R18 recorded (grade in sequence), met again. The fix removed the re-read: the lane already reads the
staged halves for its dot products, and Σa is one more exact dot on data in registers.
Logs: [`explore-*.log`](metal-decode-gemv-r18b-2026-09-26/) (first cut) and
[`explore2-*.log`](metal-decode-gemv-r18b-2026-09-26/) (second).

## Confirmation (grades)

`TestR18InSequence`, with `h4244` as the only prototype arm:
- H form, 4 rows per simdgroup for qkv and gate/up, 2 for o; the down projection unchanged at R18's staged 4;
- one process per model: the 1.5B `.gguf`, then the 7B `.int4.metal.giw`;
- 7 paired reps and 20 step pairs per category;
- M1 Pro, 2026-09-26 18:12–18:29 PDT, idle-gated (load1 1.70 at start).

Logs: [`confirm-1.5b.log`](metal-decode-gemv-r18b-2026-09-26/confirm-1.5b.log),
[`confirm-7b.log`](metal-decode-gemv-r18b-2026-09-26/confirm-7b.log).

| model | depth | production work | `h4244` work | **production ÷ h4244** (7 reps) | gate/up | full token vs production |
|---|---:|---:|---:|---:|---:|---:|
| 1.5B | **128** | 7.964 ms | 7.388 ms | **1.079×** (1.057–1.107) | 4.064 → 3.291 ms | −0.749 ms |
| 1.5B | 2048 | 7.682 | 6.897 | 1.121× (1.103–1.158) | 3.987 → 3.243 | −0.823 |
| 1.5B | 3900 | 7.725 | 6.855 | 1.136× (1.110–1.155) | 4.011 → 3.252 | −0.875 |
| 7B | **128** | 29.484 | 26.839 | **1.102×** (1.039–1.106) | 15.336 → 12.849 | −2.840 |
| 7B | 2048 | 29.312 | 26.520 | 1.106× (1.044–1.124) | 15.279 → 12.668 | −2.910 |
| 7B | 3900 | 29.201 | 26.205 | 1.112× (1.101–1.121) | 15.270 → 12.650 | −2.890 |

Against the shipped pre-R18 kernels, R18 plus R18b reads 1.25× / 1.36× / 1.37× (1.5B) and 1.45× / 1.48× / 1.49× (7B)
on the same metric. At depth 128 the full token falls from 12.98 → 10.58 ms on the 1.5B and from 45.7 → 33.2 ms on the
7B.

**Precondition 3, after idle:** the first token after 2 s idle is dominated by clock ramp-up, as in R18, and is not
worse for the candidate at any depth (see the logs). This is not a win that exists only after idle.

## Production wiring

The kernel form replaces the inner loop of production's `sa_rows_acc` (`metal/kernels.go`). Kernel names, dispatch
sites and threadgroup memory are unchanged. `buildResident`'s rule moves qkv from 2 to 4 rows per simdgroup
(`gemvRowsFor`, which falls back where rows do not tile). Production now reads `gemvRows {qkv:4 o:2 gu:4 down:4}` on
both models.

Gates, all on the wired tree:

| gate | result |
|---|---|
| bit-identity against the pre-R18 shipped kernels through the executor (`TestR18InSequence`, 16 teacher-forced positions) | **0 differ** at 128 / 2048 / 3900 on both models ([`wired-1.5b.log`](metal-decode-gemv-r18b-2026-09-26/wired-1.5b.log), [`wired-7b.log`](metal-decode-gemv-r18b-2026-09-26/wired-7b.log)) |
| `TestMetalSnapshotGolden` | 10/10 byte-identical |
| tagged Metal suite | 180 pass, 0 fail ([`metal-suite-tagged.log`](metal-decode-gemv-r18b-2026-09-26/metal-suite-tagged.log)); untagged `ok` |
| wired production vs pre-R18 shipped, GEMV work (5 reps) | 1.5B **1.380× / 1.365× / 1.365×**; 7B **1.460× / 1.486× / 1.491×** at 128 / 2048 / 3900 |
| full token (GPU ms) | 1.5B 12.93 → **10.46** at 128, 13.89 → 11.30, 14.89 → 12.27; 7B 45.78 → **33.12**, 49.67 → 36.82, 52.33 → 39.42 |

R18 alone had given 1.17× / 1.32× at depth 128. R18 plus R18b reads 1.38× / 1.46×.

### End to end against Ollama (reported, not deciding)

The setup was `scripts/bench_peer.py`, R17's and R18's protocol:
- three engines interleaved cell by cell in one session, with a server restart between cells;
- greedy decoding, 3 runs × 8 requests × 64 tokens, same weights;
- idle-gated per cell;
- the R18b build `serve-metal-c5d7e310` against its parent `serve-metal-9bafd1f3` (R18's integer rows kernels, the same
  tree otherwise; `strings` finds the half-staged form in one binary and not the other) against Ollama v0.32.5.

The run was 2026-09-26 18:57–19:06 PDT. Raw: [`e2e/`](metal-decode-gemv-r18b-2026-09-26/e2e/). Decode tok/s is the mean
of 3 runs, and every spread is ≤ 1.0.

| model | depth | R18b | R18 | Ollama | R18b ÷ R18 | **R18b ÷ Ollama** | R18 ÷ Ollama |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1.5B | 128 | 89.3 | 83.2 | 84.3 | 1.073× | **1.059×** | 0.987× |
| 1.5B | 2048 | 83.2 | 77.7 | 80.1 | 1.071× | **1.039×** | 0.970× |
| 1.5B | 3900 | 78.4 | 73.7 | 76.0 | 1.064× | **1.032×** | 0.970× |
| 7B | 128 | 29.4 | 27.1 | 24.8 | 1.085× | **1.185×** | 1.093× |
| 7B | 2048 | 26.6 | 24.7 | 23.9 | 1.077× | **1.113×** | 1.033× |
| 7B | 3900 | 24.9 | 23.3 | 23.4 | 1.069× | **1.064×** | 0.996× |

- **goinfer's Metal decode is now ahead of Ollama in every cell.** The 1.5B, which was 0.97–0.99× with R18, reads
  1.03–1.06×.
- The R18 build reproduces R18's own end-to-end row to within ~2% (83.2 / 77.7 / 73.7 / 27.1 / 24.7 here, against
  83.9 / 77.7 / 73.2 / 27.7 / 25.3 on 2026-09-26 15:06–15:35 PDT).
- The end-to-end gain, 1.06–1.09×, is what the GPU-time token predicted (10.58 vs 11.33 ms on the 1.5B at depth 128 is
  1.07×).
