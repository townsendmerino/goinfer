# TE2(b): where the variance of a served cell lives, 2026-09-28

> This is item TE2(b) of [`task-test-efficiency-2026-09.md`](../../tasks/task-test-efficiency-2026-09.md), done by
> analysing the records already on disk. **Nothing was run, timed or served.** The script is
> [`te2b_variance.py`](te2b_variance.py). It uses only the standard library, reads only, and is deterministic: two
> runs diff identical. Every number below is pasted from its output, and the full output is at the end.
> Re-run: `python3 docs/measurements/test-efficiency-2026-09/te2b_variance.py`.

## Verdict

**For the task doc's "−20–40% of cells" band, this is a documented null: restarts stay the unit.** The kill line
splits by machine, and the reason is different on each.

- **The doc's own reading** is the restart share of one recorded run mean's variance, from the 3-level model. It is
  **nobara 0%** (90% cell-bootstrap 0–66; per-cell median 25%), so the kill line is not met there. It is **Mac 67%**
  (30–85), so the kill line is met on the Mac.
- **Taking session drift out** (the 4-level model), the restart component inside one session is **nobara 0%** (0–33)
  and **Mac 31%** (0–66). For Mac goinfer alone it is 0% (0–77, 19 df). The 11 same-pass restart pairs, all on the
  Mac and 9 of them goinfer Metal, differ by a median 0.09% (max 1.41%). What the doc's reading calls "between
  restarts" is mostly **between sessions**: 69% of a Mac run mean's variance and 20% of a nobara one.
- **Where within-server variance dominates (nobara), trading restarts for completions barely pays.** A nobara CUDA
  restart costs at most 8.0–31.4 s. At goinfer's in-session restart term (0.35% sd), the cheapest design that matches
  today's 2 restarts x 3 runs is 2 x 3 itself (0% saving). Only if that term is zero does 1 restart x 6 runs match; it
  then saves 29–35% of a dense CUDA cell's wall when restarts are priced dear, and 3–11% when they are priced cheap.
  The records cannot tell those two cases apart for 1.5B or 7B CUDA: no in-session goinfer replicate of either
  exists. The break-even is tighter than the doc's kill line. One restart can match two only if the restart share of
  a run mean is **< 25%**, not < 50%.
- **On the Mac CPU, more completions per restart is the wrong lever outright.** Rates fall within one server
  lifetime. In 4 of 7 goinfer 7B restarts they fall more than 2%, the worst by −26.7%. Ollama 7B falls in 3 of its 4
  restarts, with a median of −14.2%. The decline is smooth and grows with time under sustained all-core load, which
  looks like thermal throttling. More completions per restart would pull the cell mean toward the throttled rate; a
  restart, with its teardown and idle wait, is what resets it.

So TE2(b) frees no wall by itself. TE2(a)'s "keep the server up across consecutive runs" is safe only where §5b
shows no within-server trend: nobara's dense models on CUDA and CPU, and Metal on the Mac. It is not safe on the Mac
CPU, and not for the MoE cells: one M35 lifetime rose 29.6%, one M26 lifetime fell 22.1%.

## 1. What bench_peer.py restarts, and when (from its code)

- `main()` gates, stamps, then measures one cell:
  `gate_cell_idle()` / `t0 = time.time()` / `rates, err, counts = run_cell(engine, mk, depth, cfg, backend)`. So a
  cell's `secs` excludes the idle-gate wait.
- `run_cell()` has the docstring *"Restart the server, do NRUNS runs of NCOMP completions, return per-run rates."*
  It does `proc = subprocess.Popen(...)`, then one discarded warm-up,
  `warm = post_stream(url, mk(), parse)`, then `for _ in range(nruns):` / `for _ in range(ncomp):` /
  `intervals, tf, tl, chunks, reported = post_stream(url, mk(), parse)`, then `run_rates.append(statistics.mean(rates))`
  and `comp_rates.extend(rates)`. In `finally` it runs `os.killpg(os.getpgid(proc.pid), signal.SIGTERM)` and
  `time.sleep(3)  # let VRAM settle before the next engine loads`.
- The constants: `NCOMP = 8          # completions per run  (>= 8 required)` and
  `NRUNS = 2          # runs per cell        (>= 2 required, spread reported)`. `gen_params()` sets
  `ngen, ncomp, nruns = (128, 2, 2) if DEEP_CTX else (NGEN, NCOMP, NRUNS)`, and `BENCH_RUNS` overrides nruns (the
  L1 and peer-claim gates use 3).
- The resume key is `(r["phase"], r["engine"], r.get("backend"), r["model"], r["depth"], r["config"])`, so one file
  holds one record per cell per phase.

**So the hierarchy is restart > run > completion.** A restart is one cell record, which is one server lifetime:
start, load, one warm-up, NRUNS x NCOMP timed completions, teardown, 3 s. A run is 8 back-to-back completions on
that same server. Nothing separates two runs, so the run level measures slow drift inside a lifetime and is not a
unit anyone pays for separately. A second restart of the same cell exists only in another file (another pass), or
in the same file when both phase A and phase B plan the depth-128 cell. `counts.completion_rates` holds every timed
completion in order. The script checks that each block of NCOMP reproduces its `runs[]` entry to 1e-9. All 820 used
cells pass, so no completion was silently dropped by `if intervals < 2 ...: continue`.

## 2. Method

- **Data.** Every `*.json` / `*.jsonl` (and `.gz`) under `docs/measurements/`, parsed whole or as JSON-lines. A file
  is bench_peer-shaped when it is a list whose element 0 has `"kind": "provenance"`.
- **A replicated cell** is the same host, serve binary, model, backend, depth, config, prompt, ctx pin, GPU driver and
  NGEN x NCOMP, seen in two or more restarts.
  - goinfer builds are identified by provenance `serve_binaries` / `serve_binaries_old` path + mtime. The identity
    is **engine-agnostic**: the L1 pass 2 swapped the roles, and `serve-cpu-3cd62e6d` is one build whether it ran
    as `goinfer` or as `goinfer_old`. In the example from the brief, 3cd62e6d appears in all three Mac passes of
    2026-09-28 (`arm64-speed-served.json`, `-pass2.json`, `arm64-fix-served.json`) as 3 restarts per model with 3
    same-session pairs, and 5c85f7c0 appears in two (§2 of the output).
  - Ollama is identified by bin + version. llama-server is identified by its version string, which carries the build
    commit.
  - A `*.sh` wrapper (91 cells) or MLX (4 cells) leaves the build unidentified. Those cells can replicate only inside
    their own file.
  - **The prompt is in the key.** On 2026-09-25 the harness moved to `prompt_format: essay-v2` together with the
    engine-reported token numerator, at the same `prompt_tokens` of 129. The same Ollama build reads 195.1 before and
    183.0 after at 1.5B CUDA d128, and 74.2 → 72.1 at 7B. llama-server 7B reads 80.4 → 78.2 (§2 of the output).
    Pooled, that shift would read as session drift (§7).
- **Curation** (listed verbatim in output §1): 2 void files and 30 arm files, each with the reason read from the
  record's own writeup or run script.
  - The void files are `void-attempt1-a-e-dense.json` ("nothing from them is used") and the r13 CPU depth row, which
    declares itself thermally contaminated.
  - In the arm files, a treatment is set by something the record does not carry. Examples: `GOINFER_NO_OPTFWD` in
    g26–g28, `BENCH_QUANT_OVERRIDE`, a q4k vs int4 checkpoint, and `OLLAMA_KV_CACHE_TYPE=q8_0` in the R2 Metal
    depth run. The affected engine's cells stay file-local.
- **Model.** y = ln(tok/s), so 100·sd reads as a CV in %, and components add for a ratio, which is how the gates
  read. Each replicated cell gets a nested random-effects model, estimated by method-of-moments: a general
  unbalanced nested ANOVA, with the expected-mean-square coefficients computed from the unit sizes. The estimator
  was checked against synthetic data with known components before use (a scratch check, not part of the script).
  Strata pool by summing SS and coefficients, and uncertainty comes from resampling cells (2,000 reps, 90%
  interval). Pooled SS is dominated by a few heavy-tailed cells, so per-cell medians are printed beside it. One
  example: Ollama 1.5B d512 on nobara falls from about 265 to about 195 tok/s inside every lifetime.
  - **3-level** (restart > run > completion) is the doc's framing. Its "restart" carries any drift between the
    sessions the restarts ran in.
  - **4-level** (session > restart > run > completion). A session is a chain of this host's files whose start times
    are less than 3 h apart. This separates between-session drift. Within-session drift, such as the Mac's ~2%
    pass-to-pass order effect named in the brief, still sits inside the restart term.
  - **Pairwise semivariance by separation** is a model-free cross-check. The separation is same pass, same session
    or other session, and each pair is net of its cell's own within-server noise.
- **The kill-line basis** is the doc's "split the recorded runs[]": restart share = s2_R / (s2_R + s2_B + s2_C/n_c),
  the share of one run mean's variance. The per-completion basis (more lenient) and the 3-run cell-mean basis
  (stricter) are printed too.
- **Cost.** The records do not split a cell's non-decode time into per-start work and per-completion prefill, so a
  restart's cost is bracketed:
  - the low end, a_lo, is the 3 s settle plus one warm-up decode;
  - the high end, a_hi, is all non-decode time;
  - completions get the complement.

  The idle-gate wait is excluded from both. It is paid once per restart, so leaving it out understates what a
  restart costs.

## 3. Data coverage

```
JSON / JSON-lines files under docs/measurements: 276
  bench_peer-shaped (list, element 0 kind=provenance): 94 (94 whole-file JSON, 0 JSON-lines)
  skipped, UNPARSEABLE: not JSON (Expecting value: line 1 column 1 (char 0)) nor JSON-lines (Expecting value: line 1 column 1 (char 0)): 2
      peer-claim-2026-09-25-mac/grade.jsonl
      peer-claim-2026-09-25/grade.jsonl
  skipped, not bench_peer-shaped: a dict with its own provenance/cells (another harness): 11
      attn-decode-fa-default-2026-09-23-15b.json
      [... 10 more; all 11 are named in the full output below]
  skipped, not bench_peer-shaped: a list without a provenance header: 15
  skipped, not bench_peer-shaped: other JSON: 154

cell records in the bench_peer-shaped files:
   1063  records
     22  dropped: errored cell (no runs)
    182  dropped: no completion_rates (harness before per-completion rates were kept)
      2  dropped: no engine field (not a run_cell record)
     37  dropped: void file (curated)
    820  used: cell with consistent completion_rates
         of the used cells, file-local because arm file (curated): 91
         of the used cells, file-local because unidentified: mlx: no version recorded: 4
         of the used cells, file-local because unidentified: serve path is a wrapper script (the build it execs is unrecorded): 91

replicated groups (>= 2 restarts of one cell): 108 groups, 362 restarts, 911 runs, 7072 completions, from 45 files in 25 sessions
```

What was skipped, and why:

- **The two unparseable files** are `grade.jsonl` outputs of the peer-claim grader. They are derived verdicts with
  `## <file>` header lines, not raw cells, and the raw cells they grade are in the bench_peer JSONs that were read.
- **The 11 dict-shaped files** (`attn-decode-fa-*`, `b6-splitkv-*`) come from a different harness, with its own
  `{cells, config, provenance}` layout and `blocks` per arm. They are outside bench_peer's restart model, so they
  were not converted.
- **The 182 cells without `completion_rates`** predate the harness keeping them (the g26 n15 anchors among them), so
  they cannot give the completion level.

The replicated cells split into **Mac: 33 cells, 84 restarts, 2,056 completions** and **nobara: 75, 278, 5,016**.

## 4. Results

### 4.1 The kill line on the doc's reading (3-level, restart vs within-server)

```
  stratum                 grp  rst  runs  comps     sdR%   sdB%   sdC%    R/B/C % of 1 comp   R % run mean  90% boot   R % cell mean   per-group R % run: median, % >=25, % >=50
  mac                      33   84   257   2056     5.39   3.61   3.27      55 /  25 /  20             67    30- 85              86                         59   70   64 (n=33)
  nobara                   75  278   654   5016    0.00*   0.98   1.03       0 /  48 /  52              0     0- 66               0                         25   49   36 (n=75)

  mac goinfer              15   34    99    792    0.00*   3.68   1.80       0 /  81 /  19              0     0- 77               0                         48   53   47 (n=15)
  mac peer                 18   50   158   1264     6.79   3.56   3.92      62 /  17 /  21             76    50- 90              90                         80   83   78 (n=18)
  nobara goinfer            8   22    48    384     0.35   0.29   1.46       5 /   4 /  91             26     0- 47              51                           0   25   12 (n=8)
  nobara peer              67  256   606   4632    0.00*   1.01   0.99       0 /  51 /  49              0     0- 72               0                         31   52   39 (n=67)

  mac cpu                  11   28    88    704     8.20   6.10   5.45      50 /  28 /  22             62    13- 82              83                         50   64   55 (n=11)
  mac metal                22   56   169   1352     3.08   0.38   0.91      91 /   1 /   8             98    67- 99              99                         63   73   68 (n=22)
  nobara cpu               12   35    88    704     0.40   0.12   0.43      44 /   4 /  51             81    36- 93              93                         68   58   58 (n=12)
  nobara cuda              63  243   566   4312    0.00*   1.06   1.10       0 /  48 /  52              0     0- 66               0                         24   48   32 (n=63)
```

The pooled nobara figure (0%) and its per-cell median (25%) disagree. The pool is dominated by a few cells with very
large within-server variance, and the median is the more typical cell. Both are below half. The Mac is above half on
every row except goinfer, which is 0%, but with 19 df and an upper bound of 77%.

### 4.2 Session drift separated (4-level)

```
  stratum                 grp  sess  rst  runs     sdS%   sdR%   sdB%   sdC%   share of one run mean: S / R / within   R % in-session  90% boot   (S+R) %  90% boot   in-session restart pairs
  mac                      33    57   84   257     6.74   2.54   3.61   3.27                69 /   10 /    22                     31     0- 66        78    41- 91                   27 df
  nobara                   75   218  278   654     0.52  0.00*   0.98   1.03                20 /    0 /    80                      0     0- 33        20     8- 71                   60 df

  mac goinfer              15    15   34    99      n/a  0.00*   3.68   1.80                 0 /    0 /   100                      0     0- 77         0     0- 77                   19 df
  mac peer                 18    42   50   158     5.04   5.13   3.56   3.92                38 /   40 /    22                     64    29- 91        78    61- 92                    8 df
  nobara goinfer            8     8   22    48      n/a   0.35   0.29   1.46                 0 /   26 /    74                     26     0- 47        26     0- 47                   14 df
  nobara peer              67   210  256   606     0.56  0.00*   1.01   0.99                21 /    0 /    79                      0     0- 10        21     9- 75                   46 df

  mac cpu                  11    13   28    88    20.72   2.43   6.10   5.45                90 /    1 /     9                     13     0- 45        91    13- 96                   15 df
  mac metal                22    44   56   169     3.73   0.16   0.38   0.91                98 /    0 /     2                     10     0- 53        98    73- 99                   12 df
  nobara cpu               12    29   35    88     0.43  0.00*   0.12   0.43                84 /    0 /    16                      0     0-  9        84    48- 94                    6 df
  nobara cuda              63   189  243   566     0.52  0.00*   1.06   1.10                18 /    0 /    82                      0     0- 32        18     6- 71                   54 df
```

`n/a` means no build of that class was measured in two sessions (Mac goinfer), or that no cell had two restarts
inside one session. Every goinfer build on the Mac and on nobara lived in one session only, so goinfer's session
term cannot be estimated from these records.

### 4.3 Restart-to-restart differences by separation (model-free)

```
  stratum          separation        pairs  cells   mean (sd%)  median (sd%)   |restart-mean diff| median / p90 / max %
  mac              O other session      37     13         6.02          0.31                0.52 / 23.35 / 44.24
  mac              P same pass          11     11         0.36          0.06                0.09 /  0.71 /  1.41
  mac              S same session       23     10         2.83          0.49                3.45 / 11.59 / 14.51
  mac cpu          O other session       2      2        20.78         25.36               44.24 / 44.24 / 44.24
  mac cpu          S same session       21      9         2.96          0.49                4.06 / 11.59 / 14.51
  mac goinfer      P same pass           9      9         0.23          0.06                0.09 /  0.71 /  0.71
  mac goinfer      S same session       14      7        0.00*          0.42                0.77 /  4.24 /  9.39
  mac metal        O other session      35     11         3.69          0.31                0.52 /  2.67 / 24.53
  mac metal        P same pass          11     11         0.36          0.06                0.09 /  0.71 /  1.41
  nobara           O other session     466     54        0.00*          0.04                0.15 /  0.99 /  5.03
  nobara           S same session       87     37         0.19         0.00*                0.17 /  0.84 /  1.39
  nobara cuda      O other session     426     45        0.00*          0.04                0.14 /  0.96 /  5.03
  nobara cuda      S same session       81     31         0.20          0.02                0.19 /  0.84 /  1.39
  nobara goinfer   S same session       23      8         0.27         0.00*                0.61 /  1.35 /  1.39
```

(excerpt; all rows are in the full output)

- **On nobara, separation does not matter in the typical cell.** Two restarts of one cell differ by a median 0.17%
  inside a session and 0.15% across sessions, with p90s of 0.84% and 0.99%. The pooled session term in §4.2 comes
  from a few cells (the max pair is 5.03%).
- **On the Mac, the restart itself is quiet, and what varies is the state of the machine.** Same-pass pairs (Metal)
  differ by a median 0.09%. Same-session pairs differ by 3.45%, and 21 of those 23 pairs are the CPU cells of the
  2026-09-28 L1 session, run by day at `BENCH_MAX_LOADAVG=2.5`. Other-session Metal pairs differ by a median 0.52%
  (p90 2.67%). The largest Metal pair, 24.53%, is the Ollama 7B Metal d2048 cell, whose §2 row has sdR 12.38%.

### 4.4 Within-server trend: does "more completions per restart" measure the same thing?

```
  stratum                  restarts   mean d% median d%      t   % d<-2% % d>+2%
  mac cpu goinfer                24     -2.05      0.17   -1.4       17%      4%
  mac cpu peer                   17     -6.08     -4.03   -2.5       53%     12%
  mac metal goinfer              87      0.04     -0.04    0.3        2%      5%
  mac metal peer                 64      0.11     -0.01    1.0        2%      3%
  nobara cpu goinfer             40     -0.18     -0.04   -1.3        5%      0%
  nobara cpu peer                35     -0.07     -0.04   -1.6        0%      0%
  nobara cuda goinfer           273      0.16     -0.01    1.1        0%      3%
  nobara cuda peer              275     -0.14     -0.05   -1.5        1%      1%
  nobara webgpu goinfer           5      0.93      0.75    3.0        0%     20%

  per-model strata flagged (>= 25% of restarts move > 2% first run -> last run):
    mac cpu goinfer 7B               restarts   7  moved >2%:   4  median d   -3.5%  min  -26.7%  max    0.0%
    mac cpu peer 0.5B                restarts   6  moved >2%:   3  median d   -0.8%  min  -26.2%  max    1.4%
    mac cpu peer 1.5B                restarts   6  moved >2%:   4  median d    2.0%  min  -17.6%  max    9.9%
    mac cpu peer 7B                  restarts   4  moved >2%:   3  median d  -14.2%  min  -21.8%  max    0.8%
    mac cpu peer phi3-mini           restarts   1  moved >2%:   1  median d  -10.8%  min  -10.8%  max  -10.8%
    mac metal goinfer 0.5B           restarts   6  moved >2%:   2  median d   -0.4%  min   -2.9%  max    7.9%
    mac metal goinfer phi3-mini      restarts   1  moved >2%:   1  median d   -2.6%  min   -2.6%  max   -2.6%
    nobara cuda goinfer M35          restarts  10  moved >2%:   3  median d    0.6%  min   -0.4%  max   29.6%
    nobara webgpu goinfer 1.5B       restarts   1  moved >2%:   1  median d    2.0%  min    2.0%  max    2.0%
```

nobara CUDA and CPU, and Mac Metal, show no trend: the mean d is within ±0.2%, on 35–275 restarts per stratum.
There, the run level is noise that more runs average away. The Mac CPU and the MoE cells are non-stationary inside one lifetime.

### 4.5 What a restart costs, and the design comparison

The cost bracket for the goinfer rows, per cell of that stratum, taken from the output's §6 table:

```
  machine engine   model     backend cells   T med s D med s O med s    a_lo   a_hi    b_lo  b_hi   O share of T
  mac     goinfer  1.5B      metal      21      24.7    18.6     6.5     3.8    6.5    0.77  0.89            26%
  mac     goinfer  7B        metal      19      81.8    68.9    12.8     5.9   12.8    2.87  3.16            16%
  nobara  goinfer  0.5B      cuda       18      11.0     3.0     8.0     3.2    8.0    0.19  0.49            73%
  nobara  goinfer  1.5B      cuda       19      16.4     4.4    12.2     3.3   12.2    0.28  0.83            74%
  nobara  goinfer  7B        cuda       18      44.8    13.8    31.4     3.9   31.4    0.86  2.58            70%
  nobara  goinfer  7B        cpu        12     330.8   223.3    45.4    17.0   45.4   13.96 15.73            14%
  nobara  goinfer  M35       cuda        4     496.5    55.6   440.9     6.5  440.9    3.47 30.63            89%
```

The design comparison scores one goinfer cell mean inside a session. D0 is today's two-pass gate, 2 restarts x 3
runs. "Best" is the cheapest design with Var ≤ Var(D0), priced first with dear restarts and then with cheap ones:

```
  stratum                 sdW% trend  s2_R   sdR% R%run R=1 ok  sd(D0)% sd(1x6)%           best, dear restart          best, cheap restart
  mac 1.5B metal          0.48     -  point  0.00     0    yes     0.20     0.20    1 x 6     44s/  50s   13%    1 x 6     46s/  50s    8%
                                      p95    1.28    88     no     0.93     1.30    3 x 1     38s/  50s   24%    3 x 1     33s/  50s   35%
  mac 7B cpu              8.32   YES  point  0.00     0    yes     3.40     3.40    1 x 6    201s/ 229s   12%    1 x 6    223s/ 229s    3%
  mac 7B metal            0.28     -  point  0.00     0    yes     0.11     0.11    1 x 6    151s/ 163s    8%    1 x 6    158s/ 163s    4%
                                      p95    1.28    95     no     0.91     1.29    3 x 1    107s/ 163s   34%    3 x 1     93s/ 163s   43%
  nobara 0.5B cuda        0.66     -  point  0.35    22    yes     0.37     0.44    2 x 3     25s/  25s    0%    2 x 3     30s/  30s    0%
                                      zero   0.00     0    yes     0.27     0.27    1 x 6     17s/  25s   32%    1 x 6     27s/  30s   11%
  nobara 1.5B cuda        0.78     -  point  0.35    17    yes     0.40     0.47    2 x 3     38s/  38s    0%    2 x 3     47s/  47s    0%
                                      zero   0.00     0    yes     0.32     0.32    1 x 6     25s/  38s   32%    1 x 6     43s/  47s    7%
  nobara 7B cuda          0.61     -  point  0.35    25    yes     0.35     0.43    2 x 3    104s/ 104s    0%    2 x 3    132s/ 132s    0%
                                      zero   0.00     0    yes     0.25     0.25    1 x 6     73s/ 104s   30%    1 x 6    128s/ 132s    3%
  nobara 7B cpu           0.43     -  point  0.35    39     no     0.30     0.39    4 x 1    628s/ 761s   17%    4 x 1    571s/ 789s   28%
  nobara M35 cuda         6.76   YES  point  0.35     0    yes     2.77     2.78    1 x 7    635s/1049s   39%    2 x 3   1483s/1483s    0%
```

(excerpt; every stratum and every s2_R scenario is in the full output)

What the table says:

- **nobara CUDA, dense models.** At goinfer's in-session restart term (0.35% sd), today's 2 x 3 is already the
  cheapest design for its variance. 1 x 6 widens the cell-mean sd: 0.37 → 0.44 at 0.5B, 0.40 → 0.47 at 1.5B and
  0.35 → 0.43 at 7B. Only at zero does 1 x 6 match D0, saving 29–35% of the cell wall with dear restarts and 3–11%
  with cheap ones (0.5B, 1.5B, 7B, gemma3-1b and phi3-mini CUDA). The real split of a restart's cost sits between
  the two, and TE0 will measure it.
- **The absolute precision is already fine on nobara:** D0's cell-mean sd is 0.33–0.43% on dense CUDA at the point
  estimate, well under the ±1–3% bars the gates register. So the restart question affects only wall time on nobara,
  never whether a gate can resolve its bar.
- **Mac Metal.** At the point estimate (0) 1 x 6 saves 8–13% with dear restarts (7B, 1.5B). At the p95 (1.28% sd), fewer runs
  and more restarts would be better (3 x 1). The Mac goinfer in-session term is too loosely bounded (0–77%) to pick.
- **Rows flagged `trend`** (Mac 7B CPU, Mac 0.5B Metal, M35) are the ones where more runs per restart changes the
  estimand, so their arithmetic does not apply.

### 4.6 Sensitivity: the curation is what makes the answer

```
  as analysed (curated, prompt in key)                 mac     groups  33   3-level R % run mean   67   4-level: R in-session   31%  (S+R)   78%
  as analysed (curated, prompt in key)                 nobara  groups  75   3-level R % run mean    0   4-level: R in-session    0%  (S+R)   20%
  no curation (void + arm files pooled)                mac     groups  41   3-level R % run mean   90   4-level: R in-session   93%  (S+R)   93%
  no curation (void + arm files pooled)                nobara  groups 115   3-level R % run mean   90   4-level: R in-session   94%  (S+R)   94%
  no prompt key (essay-v2 pooled with older prompts)   mac     groups  29   3-level R % run mean   68   4-level: R in-session   33%  (S+R)   76%
  no prompt key (essay-v2 pooled with older prompts)   nobara  groups  65   3-level R % run mean   26   4-level: R in-session    0%  (S+R)   44%
  neither                                              mac     groups  35   3-level R % run mean   90   4-level: R in-session   93%  (S+R)   93%
  neither                                              nobara  groups  98   3-level R % run mean   90   4-level: R in-session   95%  (S+R)   95%
```

Pooled by the automatic key alone (binary, model, depth, config), both machines read 90%+ between-restart: a false
kill. The inflation is made of unrecorded treatment arms (`GOINFER_NO_OPTFWD`, quant overrides, checkpoints,
`OLLAMA_KV_CACHE_TYPE`) masquerading as restarts. Leaving the prompt era unkeyed roughly doubles nobara's S+R share
(20% → 44%).

## 5. Verdict against the kill line

The task doc's kill for (b) is **"between-restart variance is ≥ half the total"**.

| reading | nobara | Mac | kill? |
|---|---|---|---|
| doc's basis (3-level, share of one run mean) | 0% (0–66); per-cell median 25% | 67% (30–85) | nobara no, Mac yes |
| restart inside a session (4-level) | 0% (0–33) | 31% (0–66); goinfer 0% (0–77) | no on both |
| session drift, share of one run mean (4-level) | 20% (S+R 8–71) | 69% (S+R 41–91) | — |
| one restart can replace two only if the restart share is < 25% | met at the point estimate for 0.5B/1.5B/7B CUDA (17–25%), not for phi3-mini (30%), not at the p95 | met at the point estimate, not at the p95 | — |

**Result: a documented null for the band, not the win.**

- **nobara does not trip the kill line**, but the win it would permit is at most one restart's overhead per cell
  (8.0–31.4 s on dense CUDA). It exists only if goinfer's in-session restart term is zero, which no record establishes for
  1.5B or 7B CUDA.
- **On the Mac the kill line is met,** and the reason is not the restart. It is session drift on Metal, and on the
  CPU the daytime and thermal state, which also makes "more completions per restart" bias the mean.

Restarts stay the unit. The wall is in TE1 (the idle gate), TE4 (stopping) and TE5(a) (arms).

## 6. Caveats

- **Drift vs restart.** Only same-pass pairs measure a restart alone: 11 of them, all on the Mac, 9 goinfer Metal.
  The in-session term also carries within-session drift, including the Mac's ~2% pass-to-pass order effect, and the
  records cannot separate the two. Sessions are chains of files less than 3 h apart, by file start time, because the
  records carry no per-cell timestamps.
- **Thin goinfer evidence.** nobara's in-session goinfer term rests on 14 df:
  - 0.5B CUDA (the b5-r7b repeats, and peer-claim `a-e-dense` vs `f-sampled`);
  - phi3-mini CUDA;
  - three CPU cells (`cpu-peer-reanchor`).

  1.5B and 7B CUDA goinfer have no in-session replicate, and every goinfer build lived in a single session, so
  goinfer's session term is not estimable at all. The Ollama and llama-server rows carry the cross-session evidence.
- **Unrecorded arms.** The curation comes from reading writeups and run scripts. Any unrecorded env arm still in the
  pool inflates the restart and session terms, which works against "fewer restarts", so the null is conservative in
  that direction.
- **Heavy tails.** A few cells dominate the pooled SS, as the pooled vs median gaps and the wide bootstrap intervals
  show. Read the medians and the pairwise table beside the pooled rows.
- **Cost is a bracket, not a measurement.** The idle-gate wait is excluded, and it is paid once per restart, so the
  real cost of a restart is higher than a_hi whenever the gate waits.
- **The Mac data are daytime runs at raised caps** (2.0–2.5). The owner's load is part of the Mac's variance as it is
  actually operated. These numbers do not describe a quiet Mac.
- **The trend metric compares only the first and last run.** It catches monotone drift, not oscillation.
- **Coverage gap.** 182 older cells without `completion_rates` could not contribute, so the August/early-September
  record is under-represented.

## 7. What this feeds

- **TE2(a).** Keeping the server up across consecutive runs is compatible with the data on nobara dense (CUDA, CPU)
  and on Mac Metal, where §4.4 shows no trend. It is not compatible on the Mac CPU (thermal decline) or for MoE (M35
  warm-up, M26 stall). Its equivalence check, decode inside the A/A floor, must include a trend check, not only the
  mean.
- **TE3 (noise registry).** Seed rows from §4.3:
  - same-build, same-prompt restart pairs on nobara differ by a median 0.15% across sessions (p90 0.99%, max
    5.03%);
  - Mac Metal pairs by 0.52% (p90 2.67%);
  - Mac CPU pairs by 3.45% inside one daytime session.

  `CLAUDE.md` quotes a ~3.5% between-session drift on nobara. These records do not show that for same-build,
  same-prompt cells, and TE3 should reconcile the two (it may be a different instrument or era). The registry should
  also note the 2026-09-25 harness change: the same Ollama build reads 195.1 → 183.0 at 1.5B CUDA d128 across it.
- **TE4.** Pass 2 buys precision only through its extra completions. Its real job, order-effect control, moves to
  within-pass ABBA blocks.
- **A night job, if TE2(b) is to be settled rather than parked.** Pre-register it; do not run it here. On nobara,
  1.5B and 7B CUDA goinfer, one build, 4 restarts x 6 runs in one session, riding TE1's A/A night. That measures the
  one term (goinfer in-session s2_R) this analysis could only extrapolate.

---

## Appendix: full script output

<details><summary>python3 docs/measurements/test-efficiency-2026-09/te2b_variance.py (2026-09-28)</summary>

```
## 1. data coverage

JSON / JSON-lines files under docs/measurements: 276
  bench_peer-shaped (list, element 0 kind=provenance): 94 (94 whole-file JSON, 0 JSON-lines)
  skipped, UNPARSEABLE: not JSON (Expecting value: line 1 column 1 (char 0)) nor JSON-lines (Expecting value: line 1 column 1 (char 0)): 2
      peer-claim-2026-09-25-mac/grade.jsonl
      peer-claim-2026-09-25/grade.jsonl
  skipped, not bench_peer-shaped: a dict with its own provenance/cells (another harness): 11
      attn-decode-fa-default-2026-09-23-15b.json
      attn-decode-fa-default-2026-09-23-other.json
      attn-decode-fa-served-fa-aa-floor.json
      attn-decode-fa-served-fa-gate.json
      attn-decode-fa-served-fa-ladder-cross.json
      attn-decode-fa-served-fa-ladder-d7.json
      attn-decode-fa-served-fa-ladder-small.json
      attn-decode-fa-verify-served-1p5b.json
      attn-decode-fa-verify-served-d7.json
      b6-splitkv-bb42106.json
      b6-splitkv-force-6fa487f.json
  skipped, not bench_peer-shaped: a list without a provenance header: 15
  skipped, not bench_peer-shaped: other JSON: 154

cell records in the bench_peer-shaped files:
   1063  records
     22  dropped: errored cell (no runs)
    182  dropped: no completion_rates (harness before per-completion rates were kept)
      2  dropped: no engine field (not a run_cell record)
     37  dropped: void file (curated)
    820  used: cell with consistent completion_rates
         of the used cells, file-local because arm file (curated): 91
         of the used cells, file-local because unidentified: mlx: no version recorded: 4
         of the used cells, file-local because unidentified: serve path is a wrapper script (the build it execs is unrecorded): 91

curated exclusions (reasons from each record's own writeup or script):
  VOID  peer-claim-2026-09-25/void-attempt1-a-e-dense.json: declared void by peer-claim-2026-09-25.md ('nothing from them is used'; LD_LIBRARY_PATH broke llama-server)
  VOID  r13-cpu-depth-row-2026-09-19.json: declared thermally contaminated from its midpoint by r13-cpu-depth-row-2026-09-19.md; ran at BENCH_MAX_LOADAVG=4.0
  ARM   f16kv-baseline-2026-09-26/a-int8int8-g32.json [goinfer file-local]: BENCH_QUANT_OVERRIDE phi3-mini=int8int8 (run.sh)
  ARM   f16kv-baseline-2026-09-26/b-int4.json [goinfer file-local]: BENCH_QUANT_OVERRIDE phi3-mini=int4 (run.sh)
  ARM   g26-head-nooptfwd-n15.json [goinfer file-local]: GOINFER_NO_OPTFWD on/off arm (docs/spec/10-optfwd-gate.md), set by env, not in the record
  ARM   g26-tsweep-optfwd-off.json [goinfer file-local]: GOINFER_NO_OPTFWD on/off arm (docs/spec/10-optfwd-gate.md), set by env, not in the record
  ARM   g26-tsweep-optfwd-on.json [goinfer file-local]: GOINFER_NO_OPTFWD on/off arm (docs/spec/10-optfwd-gate.md), set by env, not in the record
  ARM   g26-tsweep15-optfwd-off.json [goinfer file-local]: GOINFER_NO_OPTFWD on/off arm (docs/spec/10-optfwd-gate.md), set by env, not in the record
  ARM   g26-tsweep15-optfwd-on.json [goinfer file-local]: GOINFER_NO_OPTFWD on/off arm (docs/spec/10-optfwd-gate.md), set by env, not in the record
  ARM   g26-winvar-128-1.5B-off.json [goinfer file-local]: GOINFER_NO_OPTFWD on/off arm (docs/spec/10-optfwd-gate.md), set by env, not in the record
  ARM   g26-winvar-128-1.5B-on.json [goinfer file-local]: GOINFER_NO_OPTFWD on/off arm (docs/spec/10-optfwd-gate.md), set by env, not in the record
  ARM   g26-winvar-128-phi3-mini-off.json [goinfer file-local]: GOINFER_NO_OPTFWD on/off arm (docs/spec/10-optfwd-gate.md), set by env, not in the record
  ARM   g26-winvar-128-phi3-mini-on.json [goinfer file-local]: GOINFER_NO_OPTFWD on/off arm (docs/spec/10-optfwd-gate.md), set by env, not in the record
  ARM   g26-winvar-256-1.5B-off.json [goinfer file-local]: GOINFER_NO_OPTFWD on/off arm (docs/spec/10-optfwd-gate.md), set by env, not in the record
  ARM   g26-winvar-256-1.5B-on.json [goinfer file-local]: GOINFER_NO_OPTFWD on/off arm (docs/spec/10-optfwd-gate.md), set by env, not in the record
  ARM   g26-winvar-256-phi3-mini-off.json [goinfer file-local]: GOINFER_NO_OPTFWD on/off arm (docs/spec/10-optfwd-gate.md), set by env, not in the record
  ARM   g26-winvar-256-phi3-mini-on.json [goinfer file-local]: GOINFER_NO_OPTFWD on/off arm (docs/spec/10-optfwd-gate.md), set by env, not in the record
  ARM   g26-winvar-64-1.5B-off.json [goinfer file-local]: GOINFER_NO_OPTFWD on/off arm (docs/spec/10-optfwd-gate.md), set by env, not in the record
  ARM   g26-winvar-64-1.5B-on.json [goinfer file-local]: GOINFER_NO_OPTFWD on/off arm (docs/spec/10-optfwd-gate.md), set by env, not in the record
  ARM   g26-winvar-64-phi3-mini-off.json [goinfer file-local]: GOINFER_NO_OPTFWD on/off arm (docs/spec/10-optfwd-gate.md), set by env, not in the record
  ARM   g26-winvar-64-phi3-mini-on.json [goinfer file-local]: GOINFER_NO_OPTFWD on/off arm (docs/spec/10-optfwd-gate.md), set by env, not in the record
  ARM   g27-7b-optfwd-off.json [goinfer file-local]: GOINFER_NO_OPTFWD on/off arm (docs/spec/10-optfwd-gate.md), set by env, not in the record
  ARM   g27-7b-optfwd-on.json [goinfer file-local]: GOINFER_NO_OPTFWD on/off arm (docs/spec/10-optfwd-gate.md), set by env, not in the record
  ARM   g28-realprompt-15b-off.json [goinfer file-local]: GOINFER_NO_OPTFWD on/off arm (docs/spec/10-optfwd-gate.md), set by env, not in the record
  ARM   g28-realprompt-15b-on.json [goinfer file-local]: GOINFER_NO_OPTFWD on/off arm (docs/spec/10-optfwd-gate.md), set by env, not in the record
  ARM   g28-realprompt-phi3-off.json [goinfer file-local]: GOINFER_NO_OPTFWD on/off arm (docs/spec/10-optfwd-gate.md), set by env, not in the record
  ARM   g28-realprompt-phi3-on.json [goinfer file-local]: GOINFER_NO_OPTFWD on/off arm (docs/spec/10-optfwd-gate.md), set by env, not in the record
  ARM   metal-depth-r2-2026-09-18-raw.json [ollama file-local]: Ollama ran with OLLAMA_KV_CACHE_TYPE=q8_0 (metal-depth-r2-2026-09-18.md)
  ARM   metal-depth-r2-kvcache-isolation-2026-09-18-raw.json [ollama file-local]: Ollama isolation arm, OLLAMA_FLASH_ATTENTION=1 (r2-kvcache-isolation-run.log)
  ARM   peer-matrix-2026-09/mac-m1pro-quant-int8int8-2026-09-04.json [goinfer file-local]: a quant-override pass (int8int8)
  ARM   q4k-peer-2026-09-26/s1-q4k.json [goinfer file-local]: q4k checkpoint vs s2's int4: same binary, different weights
  ARM   q4k-peer-2026-09-26/s2-int4.json [goinfer file-local]: int4 checkpoint vs s1's q4k: same binary, different weights

replicated groups (>= 2 restarts of one cell): 108 groups, 362 restarts, 911 runs, 7072 completions, from 45 files in 25 sessions
files contributing restarts to replicated groups, by session:
  mac:s01 09-04 19:25Z       2  peer-matrix-2026-09/mac-m1pro-d7-tier1-2026-09-04.json
  mac:s02 09-05 03:49Z       2  peer-matrix-2026-09/mac-m1pro-s-tier1-2026-09-04.json
  mac:s03 09-18 02:53Z       2  peer-matrix-2026-09-17/macbook-cpu-sweep.json
  mac:s03 09-18 02:53Z       3  peer-matrix-2026-09-17/macbook-metal-sweep.json
  mac:s04 09-19 04:05Z       4  metal-depth-r2-2026-09-18-raw.json
  mac:s04 09-19 04:05Z       2  metal-depth-r2-kvcache-isolation-2026-09-18-raw.json
  mac:s04 09-19 04:05Z       3  r12-mlx-row-2026-09-18-raw.json
  mac:s06 09-20 21:42Z       1  b5-mac-r7b-561e72f6.json
  mac:s07 09-24 23:35Z       4  s6-alias-2026-09-24/gates-decode/results-7b.json
  mac:s07 09-24 23:35Z       4  s6-alias-2026-09-24/gates-decode/results.json
  mac:s07 09-24 23:35Z       8  s6-alias-2026-09-24/step2-v14/results-v14.json
  mac:s08 09-25 18:19Z       5  peer-claim-2026-09-25-mac/g-metal-decode.json
  mac:s08 09-25 18:19Z       2  peer-claim-2026-09-25-mac/i-cpu-decode.json
  mac:s09 09-26 06:11Z       6  metal-decode-attn-r17-2026-09-25/e2e/r17-e2e-metal-decode.json
  mac:s10 09-26 22:06Z       6  metal-decode-gemv-r18-2026-09-26/e2e/r18-e2e-metal-decode.json
  mac:s11 09-27 01:46Z       6  metal-decode-gemv-r18b-2026-09-26/e2e/r18b-e2e-metal-decode.json
  mac:s12 09-28 14:15Z       6  cpu-decode-peer-gap-2026-09-27/arm64-fix-served.json
  mac:s12 09-28 14:15Z       9  cpu-decode-peer-gap-2026-09-27/arm64-speed-served-pass2.json
  mac:s12 09-28 14:15Z       9  cpu-decode-peer-gap-2026-09-27/arm64-speed-served.json
  nobara:s03 09-04 19:32Z    9  peer-matrix-2026-09/nobara-w1-d7-m35-m26_2026-09-04.json
  nobara:s03 09-04 19:32Z    2  peer-matrix-2026-09/nobara-w1-llamacpp-fit-retry_2026-09-04.json
  nobara:s04 09-05 02:28Z    2  peer-matrix-2026-09/nobara-s-cpu-lane_2026-09-04.json
  nobara:s06 09-16 03:25Z   32  peer-matrix-2026-09-15/nobara-pc-cuda-sweep.json
  nobara:s07 09-17 10:57Z   32  peer-matrix-2026-09-17/nobara-pc-cuda-sweep.json
  nobara:s08 09-17 23:59Z   38  peer-matrix-2026-09-18/nobara-dense.json
  nobara:s08 09-17 23:59Z    8  peer-matrix-2026-09-18/nobara-moe.json
  nobara:s09 09-20 20:28Z   11  b5-r7b-cbf2c25d.json
  nobara:s09 09-20 20:28Z    6  b5-r7b-repeat1-cbf2c25d.json
  nobara:s09 09-20 20:28Z    6  b5-r7b-repeat2-cbf2c25d.json
  nobara:s09 09-20 20:28Z    6  b5-r7b-repeat3-cbf2c25d.json
  nobara:s10 09-21 12:53Z    9  attn-decode-fa-peer-peer-fa-run1.json
  nobara:s10 09-21 12:53Z    1  attn-decode-fa-peer-peer-fa-run2-8000.json
  nobara:s11 09-21 17:42Z    9  d7-decode-breakdown-2026-09-21/served-three-way.json
  nobara:s11 09-21 17:42Z    9  fused-rms-gu-diagnosis-2026-09-21/served-three-way.json
  nobara:s11 09-21 17:42Z    9  fused-rms-qkv-2026-09-21/served-three-way.json
  nobara:s12 09-22 17:09Z    6  cpu-peer-reanchor-2026-09-22-pass1.json
  nobara:s12 09-22 17:09Z    6  cpu-peer-reanchor-2026-09-22.json
  nobara:s14 09-24 03:37Z    3  cpu-decode-roofline-served-2026-09-23.json
  nobara:s15 09-25 17:50Z   25  peer-claim-2026-09-25/a-e-dense.json
  nobara:s15 09-25 17:50Z    9  peer-claim-2026-09-25/b-controls.json
  nobara:s15 09-25 17:50Z   10  peer-claim-2026-09-25/f-sampled.json
  nobara:s17 09-27 03:09Z   12  q4k-peer-2026-09-26/s1-q4k.json
  nobara:s17 09-27 03:09Z   12  q4k-peer-2026-09-26/s2-int4.json
  nobara:s18 09-28 02:15Z    3  cpu-decode-peer-gap-2026-09-27/baseline.json
  nobara:s19 09-28 05:29Z    3  cpu-decode-peer-gap-2026-09-27/gate5-served.json

## 2. per replicated cell: 3-level components of ln(tok/s) (restart > run > completion)

sd = 100*sqrt(component), a CV in %; * = negative estimate shown as 0. 'R%run' = restart share of one run
mean's variance (the kill-line basis); 'R%cell' = share for a 3-run cell mean. 'pairs' counts restart pairs: P same pass, S same session, O other session. 'v2' = essay-v2 prompts.

  host   binary                model     cell                        rst runs comp    mean    sdR%   sdB%   sdC%  R/B/C % 1-comp  R%run R%cell  pairs
  mac    goinfer(file-local)   1.5B      metal d128 greedy             2    6   48    73.6    0.22  0.00*   0.53    14/  0/ 86        57     80  P1
  mac    goinfer(file-local)   1.5B      metal d128 greedy             2    6   48    73.7    0.46   0.24   0.71    27/  8/ 65        63     84  P1
  mac    goinfer(file-local)   7B        metal d128 greedy             2    6   48    22.0   0.00*  0.00*   0.35     0/  0/100         0      0  P1
  mac    goinfer(file-local)   7B        metal d128 greedy             2    6   48    21.6    0.06  0.00*   0.14    14/  0/ 86        57     80  P1
  mac    goinfer_old(file-loca 1.5B      metal d128 greedy             2    6   48    73.7    0.25   0.05   0.48    22/  1/ 77        67     86  P1
  mac    goinfer_old(file-loca 1.5B      metal d128 greedy             2    6   48    74.1    0.04  0.00*   0.32     2/  0/ 98        12     29  P1
  mac    goinfer_old(file-loca 7B        metal d128 greedy             2    6   48    22.0    0.03  0.00*   0.16     2/  0/ 98        16     37  P1
  mac    goinfer_old(file-loca 7B        metal d128 greedy             2    6   48    21.9   0.00*  0.00*   0.11     0/  0/100         0      0  P1
  mac    llama-server c1d0e7a  1.5B      metal d128 greedy             2    8   64    87.1    2.04   0.57   2.20    45/  4/ 52        82     93  O1
  mac    llama-server c1d0e7a  7B        metal d128 greedy             2    8   64    26.0    0.63   0.25   0.54    53/  8/ 38        80     92  O1
  mac    ollama 0.32.5         0.5B      cpu d128 greedy               2    8   64   128.5   25.36   9.71   9.00    79/ 12/ 10        86     95  O1
  mac    ollama 0.32.5         0.5B      cpu d128 greedy v2            3    9   72   139.3    5.83   3.41   6.29    40/ 14/ 46        67     86  S3
  mac    ollama 0.32.5         0.5B      metal d128 greedy             3   10   80   144.3    1.25   0.15   1.15    54/  1/ 46        89     96  O3
  mac    ollama 0.32.5         1.5B      cpu d128 greedy               2    8   64    70.0   14.86  0.00*   4.03    93/  0/  7        99    100  O1
  mac    ollama 0.32.5         1.5B      cpu d128 greedy v2            3    9   72    67.9    6.24   5.80   6.14    35/ 30/ 34        50     75  S3
  mac    ollama 0.32.5         1.5B      metal d128 greedy             4   15  120    85.2    0.92   0.62   1.26    30/ 14/ 57        59     81  O6
  mac    ollama 0.32.5         1.5B      metal d128 greedy v2          3    9   72    84.7    0.37  0.00*   0.28    63/  0/ 37        93     98  O3
  mac    ollama 0.32.5         1.5B      metal d2048 greedy v2         3    9   72    80.1   0.00*  0.00*   1.27     0/  0/100         0      0  O3
  mac    ollama 0.32.5         1.5B      metal d3900 greedy v2         3    9   72    75.8    0.10   0.19   1.27     1/  2/ 97         4     11  O3
  mac    ollama 0.32.5         7B        cpu d128 greedy v2            3    9   72    17.2   0.00*   9.27   9.60     0/ 48/ 52         0      0  S3
  mac    ollama 0.32.5         7B        metal d128 greedy             4   12   96    25.5    0.26   0.14   0.26    44/ 13/ 43        70     88  O6
  mac    ollama 0.32.5         7B        metal d128 greedy v2          3    9   72    24.9    0.48   0.09   0.23    79/  3/ 18        94     98  O3
  mac    ollama 0.32.5         7B        metal d2048 greedy v2         3    9   72    22.4   12.38   1.23   1.21    98/  1/  1        99    100  O3
  mac    ollama 0.32.5         7B        metal d3900 greedy v2         3    9   72    23.4    0.22  0.00*   0.31    33/  0/ 67        80     92  O3
  mac    ollama(file-local)    1.5B      metal d128 greedy             2    4   32    75.3    0.03   0.02   0.12     5/  3/ 91        26     51  P1
  mac    ollama(file-local)    1.5B      metal d128 greedy             2    4   32    85.3    0.97  0.00*   1.13    43/  0/ 57        86     95  P1
  mac    serve-cpu-3cd62e6d    0.5B      cpu d128 greedy v2            3    9   72   106.8    0.32   0.33   1.33     5/  6/ 89        24     48  S3
  mac    serve-cpu-3cd62e6d    1.5B      cpu d128 greedy v2            3    9   72    50.7   0.00*   0.56   1.63     0/ 11/ 89         0      0  S3
  mac    serve-cpu-3cd62e6d    7B        cpu d128 greedy v2            3    9   72    17.4    1.90   1.86   1.78    35/ 34/ 31        48     74  S3
  mac    serve-cpu-5c85f7c0    0.5B      cpu d128 greedy v2            2    6   48    58.0    1.47   0.89   2.28    27/ 10/ 64        60     82  S1
  mac    serve-cpu-5c85f7c0    1.5B      cpu d128 greedy v2            2    6   48    28.0    2.79  0.00*   1.82    70/  0/ 30        95     98  S1
  mac    serve-cpu-5c85f7c0    7B        cpu d128 greedy v2            2    6   48     7.5   0.00*  14.60   5.63     0/ 87/ 13         0      0  S1
  mac    serve-metal           1.5B      metal d128 greedy             3    6   48    73.6    0.35   0.17   0.68    20/  5/ 76        58     80  P1 S2
  nobara llama-server 427291b  0.5B      cpu d128 greedy               2    5   40    62.4   0.00*   0.36   0.61     0/ 26/ 74         0      0  O1
  nobara llama-server 427291b  0.5B      cuda d128 greedy              5   12   96   372.3    0.97  0.00*   2.21    16/  0/ 84        60     82  O9 S1
  nobara llama-server 427291b  0.5B      cuda d2048 greedy             4    9   72   365.6    0.48   0.60   1.13    12/ 20/ 68        31     57  O6
  nobara llama-server 427291b  0.5B      cuda d3900 greedy             4    9   72   466.6    2.19   0.10   2.12    51/  0/ 48        89     96  O6
  nobara llama-server 427291b  0.5B      cuda d512 greedy              3    6   48   371.1    0.39   0.28   2.20     3/  1/ 95        18     40  O3
  nobara llama-server 427291b  1.5B      cpu d128 greedy               3   10   80    27.1    0.40   0.15   0.34    53/  8/ 39        81     93  O3
  nobara llama-server 427291b  1.5B      cuda d128 greedy              4    9   72   226.9   0.00*   0.76   1.20     0/ 28/ 72         0      0  O6
  nobara llama-server 427291b  1.5B      cuda d128 greedy v2           2    6   48   228.1   0.00*   0.56   0.95     0/ 26/ 74         0      0  S1
  nobara llama-server 427291b  1.5B      cuda d2048 greedy             4    9   72   218.3    0.36   0.19   1.10     9/  3/ 88        40     67  O6
  nobara llama-server 427291b  1.5B      cuda d2048 greedy v2          2    6   48   218.7   0.00*   0.25   0.92     0/  7/ 93         0      0  S1
  nobara llama-server 427291b  1.5B      cuda d3900 greedy             4    9   72   211.2    0.19   0.52   0.95     3/ 23/ 74         9     22  O6
  nobara llama-server 427291b  1.5B      cuda d3900 greedy v2          2    6   48   210.6   0.00*  0.00*   1.38     0/  0/100         0      0  S1
  nobara llama-server 427291b  1.5B      cuda d512 greedy              3    6   48   300.8   0.00*   2.45   2.71     0/ 45/ 55         0      0  O3
  nobara llama-server 427291b  7B        cpu d128 greedy               2    5   40     6.6   0.00*   0.06   0.29     0/  4/ 96         0      0  O1
  nobara llama-server 427291b  7B        cuda d128 greedy              5   14  112    80.4    0.12   0.14   0.43     6/  9/ 85        25     50  O10
  nobara llama-server 427291b  7B        cuda d128 greedy v2           2    6   48    78.2   0.00*   0.14   0.25     0/ 23/ 77         0      0  S1
  nobara llama-server 427291b  7B        cuda d2048 greedy             4    9   72    76.0   0.00*   0.13   0.30     0/ 15/ 85         0      0  O6
  nobara llama-server 427291b  7B        cuda d2048 greedy v2          2    6   48    76.1   0.00*   0.13   0.30     0/ 16/ 84         0      0  S1
  nobara llama-server 427291b  7B        cuda d3900 greedy             4    9   72    74.4    0.09   0.02   0.34     6/  1/ 93        35     62  O6
  nobara llama-server 427291b  7B        cuda d3900 greedy v2          2    6   48    74.3   0.00*   0.10   0.27     0/ 12/ 88         0      0  S1
  nobara llama-server 427291b  7B        cuda d512 greedy              3    6   48    77.6   0.00*   0.20   0.27     0/ 35/ 65         0      0  O3
  nobara llama-server 427291b  M26       cuda d128 greedy              4   11   88    27.4    1.65   0.23   0.56    88/  2/ 10        97     99  O6
  nobara llama-server 427291b  M26       cuda d8000 greedy ctx0 128x   4    8   16    22.4    1.00  0.00*   0.41    85/  0/ 15        92     97  O6
  nobara llama-server 427291b  M35       cuda d128 greedy              4   11   88    32.8    0.53   0.46   0.76    26/ 20/ 54        50     75  O6
  nobara llama-server 427291b  M35       cuda d8000 greedy ctx0 128x   4    8   16    30.9    0.31   0.28   0.49    23/ 19/ 58        32     59  O6
  nobara llama-server 427291b  gemma3-1b cuda d128 greedy              2    5   40   213.1   0.00*   0.87   1.24     0/ 33/ 67         0      0  O1
  nobara llama-server 427291b  gemma3-1b cuda d3900 greedy             2    5   40   203.5   0.00*   0.16   1.24     0/  2/ 98         0      0  O1
  nobara llama-server 427291b  phi3-mini cuda d128 greedy              3    8   64   127.4   0.00*   0.33   0.44     0/ 36/ 64         0      0  O2 S1
  nobara llama-server 427291b  phi3-mini cuda d3900 greedy             2    5   40    75.2    0.03   0.06   0.22     2/  6/ 92        12     29  O1
  nobara ollama 0.32.5         0.5B      cpu d128 greedy               5   11   88    57.5    0.41  0.00*   0.24    74/  0/ 26        96     99  O9 S1
  nobara ollama 0.32.5         0.5B      cpu d128 greedy v2            2    6   48    57.4    0.11  0.00*   0.25    16/  0/ 84        60     82  O1
  nobara ollama 0.32.5         0.5B      cuda d128 greedy             13   28  224   268.4    0.17  0.00*   0.38    16/  0/ 84        61     82  O68 S10
  nobara ollama 0.32.5         0.5B      cuda d128 temp0.8_topk40      4    8   64   287.2    0.05   0.11   0.34     2/  9/ 89         8     21  S6
  nobara ollama 0.32.5         0.5B      cuda d128 temp0.8_topp0.95    5   11   88   267.2    0.23   0.05   0.38    26/  1/ 73        71     88  O4 S6
  nobara ollama 0.32.5         0.5B      cuda d128 temp1.0_notrunc     2    5   40   267.7    0.36  0.00*   0.34    52/  0/ 48        90     96  O1
  nobara ollama 0.32.5         0.5B      cuda d2048 greedy             8   17  136   270.8    0.21   0.19   0.49    14/ 12/ 74        40     67  O25 S3
  nobara ollama 0.32.5         0.5B      cuda d3900 greedy             8   17  136   259.3   0.00*   0.17   0.37     0/ 17/ 83         0      0  O25 S3
  nobara ollama 0.32.5         0.5B      cuda d512 greedy              3    6   48   268.6   0.00*   0.15   0.36     0/ 14/ 86         0      0  O3
  nobara ollama 0.32.5         1.5B      cpu d128 greedy               6   16  128    24.1    0.69   0.03   0.21    91/  0/  9        99    100  O14 S1
  nobara ollama 0.32.5         1.5B      cpu d128 greedy v2            2    6   48    24.0    0.08  0.00*   0.15    21/  0/ 79        68     86  O1
  nobara ollama 0.32.5         1.5B      cuda d128 greedy              8   17  136   195.1    0.07   0.09   0.25     6/ 10/ 84        24     48  O25 S3
  nobara ollama 0.32.5         1.5B      cuda d128 greedy v2           2    6   48   183.0    0.18   0.11   0.22    35/ 13/ 52        64     84  S1
  nobara ollama 0.32.5         1.5B      cuda d2048 greedy             8   17  136   179.9    0.10   0.07   0.25    13/  6/ 81        45     71  O25 S3
  nobara ollama 0.32.5         1.5B      cuda d2048 greedy v2          2    6   48   179.9   0.00*  0.00*   0.24     0/  0/100         0      0  S1
  nobara ollama 0.32.5         1.5B      cuda d3900 greedy             8   17  136   174.9    0.10  0.00*   0.24    15/  0/ 85        58     81  O25 S3
  nobara ollama 0.32.5         1.5B      cuda d3900 greedy v2          2    6   48   176.3    0.18   0.00   0.26    32/  0/ 68        79     92  S1
  nobara ollama 0.32.5         1.5B      cuda d512 greedy              3    6   48   231.5   0.00*  10.32   6.58     0/ 71/ 29         0      0  O3
  nobara ollama 0.32.5         7B        cpu d128 greedy               5   11   88     6.0    0.25   0.07   0.26    46/  3/ 50        83     93  O9 S1
  nobara ollama 0.32.5         7B        cpu d128 greedy v2            2    6   48     6.0    0.22   0.10   0.31    32/  7/ 61        69     87  O1
  nobara ollama 0.32.5         7B        cuda d128 greedy              9   22  176    74.2    0.06   0.03   0.10    22/  8/ 70        57     80  O33 S3
  nobara ollama 0.32.5         7B        cuda d128 greedy v2           2    6   48    72.1    0.02   0.03   0.10     3/  9/ 88        15     34  S1
  nobara ollama 0.32.5         7B        cuda d2048 greedy             8   17  136    71.0   0.00*   0.05   0.09     0/ 21/ 79         0      0  O25 S3
  nobara ollama 0.32.5         7B        cuda d2048 greedy v2          2    6   48    70.9    0.04   0.05   0.08    13/ 23/ 64        30     56  S1
  nobara ollama 0.32.5         7B        cuda d3900 greedy             8   17  136    69.6    0.04   0.02   0.09    18/  3/ 79        57     80  O25 S3
  nobara ollama 0.32.5         7B        cuda d3900 greedy v2          2    6   48    69.6    0.17   0.06   0.09    71/  9/ 19        86     95  S1
  nobara ollama 0.32.5         7B        cuda d512 greedy              3    6   48    73.7   0.00*   0.05   0.14     0/ 13/ 87         0      0  O3
  nobara ollama 0.32.5         7B        cuda d8000 greedy ctx0 128x   2    4    8    56.6    0.22   0.01   0.04    96/  0/  4        98     99  O1
  nobara ollama 0.32.5         M26       cuda d128 greedy              4   11   88    22.2   0.00*   0.05   0.33     0/  2/ 98         0      0  O6
  nobara ollama 0.32.5         M26       cuda d8000 greedy ctx0 128x   4    8   16    19.4    1.52   0.02   0.07   100/  0/  0       100    100  O6
  nobara ollama 0.32.5         M35       cuda d128 greedy              4   11   88    24.0    0.65   0.10   0.14    93/  2/  4        97     99  O6
  nobara ollama 0.32.5         M35       cuda d8000 greedy ctx0 128x   4    8   16    22.9    0.91   0.03   0.11    98/  0/  1        99    100  O6
  nobara ollama 0.32.5         gemma3-1b cuda d128 greedy              3    7   56   149.5    0.18   0.15   0.24    29/ 20/ 52        52     77  O3
  nobara ollama 0.32.5         gemma3-1b cuda d3900 greedy             2    5   40   148.7   0.00*  0.00*   0.30     0/  0/100         0      0  O1
  nobara ollama 0.32.5         phi3-mini cuda d128 greedy              4   10   80   125.8    0.07   0.06   0.07    35/ 29/ 36        51     76  O5 S1
  nobara ollama 0.32.5         phi3-mini cuda d128 temp0.8_topp0.95    2    5   40   125.9   0.00*   0.04   0.09     0/ 19/ 81         0      0  O1
  nobara ollama 0.32.5         phi3-mini cuda d128 temp1.0_notrunc     2    5   40   125.7    0.07   0.08   0.07    27/ 44/ 30        36     63  O1
  nobara ollama 0.32.5         phi3-mini cuda d3900 greedy             2    5   40    74.7    0.01   0.05   0.08     1/ 27/ 72         2      5  O1
  nobara serve-cpu-old         0.5B      cpu d128 greedy               2    4   32    38.5   0.00*  0.00*   1.28     0/  0/100         0      0  S1
  nobara serve-cpu-old         1.5B      cpu d128 greedy               2    4   32    13.2   0.00*   0.55   0.78     0/ 33/ 67         0      0  S1
  nobara serve-cpu-old         7B        cpu d128 greedy               2    4   32     4.5   0.00*  0.00*   0.25     0/  0/100         0      0  S1
  nobara serve-cuda            0.5B      cuda d128 greedy              4    8   64   334.6   0.00*   0.76   1.79     0/ 15/ 85         0      0  S6
  nobara serve-cuda            0.5B      cuda d128 temp0.8_topk40      4    8   64   323.0    0.54  0.00*   1.76     9/  0/ 91        43     69  S6
  nobara serve-cuda            0.5B      cuda d128 temp0.8_topp0.95    4    8   64   315.4    0.27   0.20   1.57     3/  1/ 96        17     39  S6
  nobara serve-cuda-411e7fc4   0.5B      cuda d128 greedy              2    6   48   338.1    0.88   0.26   1.94    17/  1/ 82        59     81  S1
  nobara serve-cuda-411e7fc4   phi3-mini cuda d128 greedy              2    6   48   143.2   0.00*   0.10   0.15     0/ 30/ 70         0      0  S1

  groups whose restart share of a run mean is >= 50%: 48 of 108. A single group has 2-13 restarts, so one group's estimate is very noisy; read the pooled tables.

## 3. pooled 3-level, per machine (restart here includes any drift between the restarts' sessions)

  stratum                 grp  rst  runs  comps     sdR%   sdB%   sdC%    R/B/C % of 1 comp   R % run mean  90% boot   R % cell mean   per-group R % run: median, % >=25, % >=50
  mac                      33   84   257   2056     5.39   3.61   3.27      55 /  25 /  20             67    30- 85              86                         59   70   64 (n=33)
  nobara                   75  278   654   5016    0.00*   0.98   1.03       0 /  48 /  52              0     0- 66               0                         25   49   36 (n=75)

## 3a. pooled 3-level, per machine x model

  stratum                 grp  rst  runs  comps     sdR%   sdB%   sdC%    R/B/C % of 1 comp   R % run mean  90% boot   R % cell mean   per-group R % run: median, % >=25, % >=50
  mac 0.5B                  5   13    42    336    10.15   4.71   5.03      68 /  15 /  17             80    67- 84              92                          67   80   80 (n=5)
  mac 1.5B                 16   40   120    960     3.96   1.62   2.27      67 /  11 /  22             83    53- 98              94                         59   75   69 (n=16)
  mac 7B                   12   31    95    760     3.70   4.67   3.35      29 /  47 /  24             37     0- 97              64                         57   58   50 (n=12)
  nobara 0.5B              19   84   184   1472     0.59   0.19   1.18      20 /   2 /  78             62    27- 77              83                         40   58   42 (n=19)
  nobara 1.5B              18   67   162   1296    0.00*   1.90   1.50       0 /  61 /  39              0     0- 40               0                         24   44   33 (n=18)
  nobara 7B                19   69   166   1304     0.08   0.09   0.24      10 /  11 /  79             32     2- 55              59                         15   42   32 (n=19)
  nobara M26                4   16    38    208     1.19   0.12   0.45      87 /   1 /  12             96    90- 97              99                          97   75   75 (n=4)
  nobara M35                4   16    38    208     0.60   0.31   0.53      49 /  13 /  38             72    49- 97              88                          97  100   50 (n=4)
  nobara gemma3-1b          4    9    22    176    0.00*   0.43   0.86       0 /  20 /  80              0     0- 17               0                           0   25   25 (n=4)
  nobara phi3-mini          7   17    44    352    0.00*   0.16   0.22       0 /  34 /  66              0     0- 35               0                           2   29   14 (n=7)

## 3b. pooled 3-level, per machine x engine class

  stratum                 grp  rst  runs  comps     sdR%   sdB%   sdC%    R/B/C % of 1 comp   R % run mean  90% boot   R % cell mean   per-group R % run: median, % >=25, % >=50
  mac goinfer              15   34    99    792    0.00*   3.68   1.80       0 /  81 /  19              0     0- 77               0                         48   53   47 (n=15)
  mac peer                 18   50   158   1264     6.79   3.56   3.92      62 /  17 /  21             76    50- 90              90                         80   83   78 (n=18)
  nobara goinfer            8   22    48    384     0.35   0.29   1.46       5 /   4 /  91             26     0- 47              51                           0   25   12 (n=8)
  nobara peer              67  256   606   4632    0.00*   1.01   0.99       0 /  51 /  49              0     0- 72               0                         31   52   39 (n=67)

## 3c. pooled 3-level, per machine x backend

  stratum                 grp  rst  runs  comps     sdR%   sdB%   sdC%    R/B/C % of 1 comp   R % run mean  90% boot   R % cell mean   per-group R % run: median, % >=25, % >=50
  mac cpu                  11   28    88    704     8.20   6.10   5.45      50 /  28 /  22             62    13- 82              83                         50   64   55 (n=11)
  mac metal                22   56   169   1352     3.08   0.38   0.91      91 /   1 /   8             98    67- 99              99                         63   73   68 (n=22)
  nobara cpu               12   35    88    704     0.40   0.12   0.43      44 /   4 /  51             81    36- 93              93                         68   58   58 (n=12)
  nobara cuda              63  243   566   4312    0.00*   1.06   1.10       0 /  48 /  52              0     0- 66               0                         24   48   32 (n=63)

## 3d. within-server components from EVERY used cell (one restart is enough to estimate run and completion)

Pooled 2-level ANOVA inside each restart (run > completion). 'W' = within-server variance of one run mean,
s2_B + s2_C/n_c. This is the noise that more completions per restart would average down.

  machine engine   model     backend cells  runs     sdB%   sdC%   sdW%   B share of W
  mac     goinfer  0.5B      cpu         8    26     0.53   1.62   0.78            46%
  mac     goinfer  0.5B      metal       6    15     1.86   2.08   2.00            86%
  mac     goinfer  1.5B      cpu         8    26     0.34   1.54   0.64            28%
  mac     goinfer  1.5B      metal      43   128     0.40   0.75   0.48            70%
  mac     goinfer  7B        cpu         7    20     8.23   3.39   8.32            98%
  mac     goinfer  7B        metal      37   113     0.18   0.62   0.28            40%
  mac     goinfer  phi3-mini cpu         1     5     0.17   0.90   0.36            22%
  mac     goinfer  phi3-mini metal       1     5     0.81   1.58   0.98            67%
  mac     peer     0.5B      cpu         6    20     6.74   7.15   7.20            88%
  mac     peer     0.5B      metal       9    27     0.20   1.00   0.41            25%
  mac     peer     1.5B      cpu         6    20     3.98   6.08   4.53            77%
  mac     peer     1.5B      metal      33    93     0.30   1.36   0.57            28%
  mac     peer     7B        cpu         4    11     8.60   8.78   9.15            88%
  mac     peer     7B        metal      21    66     0.72   1.45   0.88            66%
  mac     peer     phi3-mini cpu         1     5     4.83   4.27   5.06            91%
  mac     peer     phi3-mini metal       1     5    0.00*   0.22   0.08             0%
  nobara  goinfer  0.5B      cpu        12    29     0.63   1.43   0.81            61%
  nobara  goinfer  0.5B      cuda       57   122     0.28   1.69   0.66            19%
  nobara  goinfer  0.5B      webgpu      1     2    0.00*   3.76   1.33             0%
  nobara  goinfer  1.5B      cpu        13    34     0.22   0.46   0.27            65%
  nobara  goinfer  1.5B      cuda       75   233    0.00*   2.21   0.79             0%
  nobara  goinfer  1.5B      webgpu      1     2     1.16   2.32   1.42            67%
  nobara  goinfer  7B        cpu        12    29     0.32   0.81   0.43            56%
  nobara  goinfer  7B        cuda       65   192     0.32   1.48   0.62            27%
  nobara  goinfer  7B        webgpu      1     2     0.65   1.06   0.75            75%
  nobara  goinfer  G20       cpu         1     2     0.42   0.56   0.47            82%
  nobara  goinfer  G20       cuda        1     5     0.02   0.20   0.07             7%
  nobara  goinfer  M26       cuda       13    30    0.00*  17.59   8.03             0%
  nobara  goinfer  M35       cuda       10    23     6.01   8.72   7.20            70%
  nobara  goinfer  gemma3-1b cpu         1     2    0.00*   0.62   0.22             0%
  nobara  goinfer  gemma3-1b cuda       10    22     0.63   1.73   0.88            51%
  nobara  goinfer  gemma3-1b webgpu      1     2     0.45   0.79   0.53            72%
  nobara  goinfer  phi3-mini cpu         1     2     0.15   0.17   0.16            86%
  nobara  goinfer  phi3-mini cuda       42   166    0.00*   1.53   0.54             0%
  nobara  goinfer  phi3-mini webgpu      1     2    0.00*   3.05   1.08             0%
  nobara  peer     0.5B      cpu         9    22     0.16   0.36   0.21            62%
  nobara  peer     0.5B      cuda       65   146     0.36   1.26   0.57            39%
  nobara  peer     1.5B      cpu        11    32     0.09   0.25   0.13            51%
  nobara  peer     1.5B      cuda       60   142     2.06   1.64   2.14            92%
  nobara  peer     7B        cpu         9    22     0.08   0.28   0.13            39%
  nobara  peer     7B        cuda       64   156     0.09   0.23   0.12            56%
  nobara  peer     G20       cpu         2     4     0.12   0.30   0.16            58%
  nobara  peer     G20       cuda        2    10     0.18   0.66   0.30            37%
  nobara  peer     M26       cuda       19    46     0.10   0.43   0.21            24%
  nobara  peer     M35       cuda       17    40     0.30   0.56   0.38            60%
  nobara  peer     gemma3-1b cpu         2     4     0.35   0.66   0.42            69%
  nobara  peer     gemma3-1b cuda       16    36     0.40   0.90   0.51            62%
  nobara  peer     phi3-mini cpu         2     4     0.12   0.28   0.16            60%
  nobara  peer     phi3-mini cuda       32    84     0.08   0.43   0.17            22%

(4-level tables: a session = this host's files chained by start time with gaps < 3 h. 'R % in-session' = s2_R / (s2_R + within-server variance of a run mean): the kill line with session drift taken out. '(S+R) %' = the same with session drift left in.)

## 4. pooled 4-level (session > restart > run > completion), per machine

  stratum                 grp  sess  rst  runs     sdS%   sdR%   sdB%   sdC%   share of one run mean: S / R / within   R % in-session  90% boot   (S+R) %  90% boot   in-session restart pairs
  mac                      33    57   84   257     6.74   2.54   3.61   3.27                69 /   10 /    22                     31     0- 66        78    41- 91                   27 df
  nobara                   75   218  278   654     0.52  0.00*   0.98   1.03                20 /    0 /    80                      0     0- 33        20     8- 71                   60 df

## 4a. pooled 4-level, per machine x model

  stratum                 grp  sess  rst  runs     sdS%   sdR%   sdB%   sdC%   share of one run mean: S / R / within   R % in-session  90% boot   (S+R) %  90% boot   in-session restart pairs
  mac 0.5B                  5     8   13    42    15.43   2.82   4.71   5.03                88 /    3 /     9                     24     0- 81        91    67- 96                    5 df
  mac 1.5B                 16    27   40   120     3.77   2.91   1.62   2.27                55 /   33 /    13                     72     1- 83        87    65- 99                   13 df
  mac 7B                   12    22   31    95     4.44   1.81   4.67   3.35                43 /    7 /    50                     12     0- 67        50     0- 98                    9 df
  nobara 0.5B              19    55   84   184     0.59   0.32   0.19   1.18                53 /   15 /    32                     32    10- 50        68    33- 82                   29 df
  nobara 1.5B              18    53   67   162     0.58  0.00*   1.90   1.50                 8 /    0 /    92                      0     0-  0         8     3- 54                   14 df
  nobara 7B                19    55   69   166     0.10  0.00*   0.09   0.24                43 /    0 /    57                      0     0- 25        43    16- 64                   14 df
  nobara M26                4    16   16    38     1.19    n/a   0.12   0.45                96 /    0 /     4                      0     0-  0        96    90- 97                    0 df
  nobara M35                4    16   16    38     0.60    n/a   0.31   0.53                72 /    0 /    28                      0     0-  0        72    49- 97                    0 df
  nobara gemma3-1b          4     9    9    22    0.00*    n/a   0.43   0.86                 0 /    0 /   100                      0     0-  0         0     0- 17                    0 df
  nobara phi3-mini          7    14   17    44    0.00*  0.00*   0.16   0.22                 0 /    0 /   100                      0     0- 67         0     0- 67                    3 df

## 4b. pooled 4-level, per machine x engine class

  stratum                 grp  sess  rst  runs     sdS%   sdR%   sdB%   sdC%   share of one run mean: S / R / within   R % in-session  90% boot   (S+R) %  90% boot   in-session restart pairs
  mac goinfer              15    15   34    99      n/a  0.00*   3.68   1.80                 0 /    0 /   100                      0     0- 77         0     0- 77                   19 df
  mac peer                 18    42   50   158     5.04   5.13   3.56   3.92                38 /   40 /    22                     64    29- 91        78    61- 92                    8 df
  nobara goinfer            8     8   22    48      n/a   0.35   0.29   1.46                 0 /   26 /    74                     26     0- 47        26     0- 47                   14 df
  nobara peer              67   210  256   606     0.56  0.00*   1.01   0.99                21 /    0 /    79                      0     0- 10        21     9- 75                   46 df

## 4c. pooled 4-level, per machine x backend

  stratum                 grp  sess  rst  runs     sdS%   sdR%   sdB%   sdC%   share of one run mean: S / R / within   R % in-session  90% boot   (S+R) %  90% boot   in-session restart pairs
  mac cpu                  11    13   28    88    20.72   2.43   6.10   5.45                90 /    1 /     9                     13     0- 45        91    13- 96                   15 df
  mac metal                22    44   56   169     3.73   0.16   0.38   0.91                98 /    0 /     2                     10     0- 53        98    73- 99                   12 df
  nobara cpu               12    29   35    88     0.43  0.00*   0.12   0.43                84 /    0 /    16                      0     0-  9        84    48- 94                    6 df
  nobara cuda              63   189  243   566     0.52  0.00*   1.06   1.10                18 /    0 /    82                      0     0- 32        18     6- 71                   54 df

## 5. cross-check: restart-to-restart semivariance by separation, net of each cell's own within-server noise

For each pair of restarts of one cell: (ybar_i - ybar_j)^2 / 2 - (w_i + w_j) / 2, where w is the within-server
variance of that restart's mean from the cell's OWN 3-level s2_B and s2_C (raw, so the average is unbiased).
Its expectation is the restart component plus any drift between the two restarts. The mean is over pairs,
and 'sd%' = 100*sqrt(mean). The median is printed too, because a few pairs carry most of the sum.

  stratum          separation        pairs  cells   mean (sd%)  median (sd%)   |restart-mean diff| median / p90 / max %
  mac              O other session      37     13         6.02          0.31                0.52 / 23.35 / 44.24
  mac              P same pass          11     11         0.36          0.06                0.09 /  0.71 /  1.41
  mac              S same session       23     10         2.83          0.49                3.45 / 11.59 / 14.51
  mac cpu          O other session       2      2        20.78         25.36               44.24 / 44.24 / 44.24
  mac cpu          S same session       21      9         2.96          0.49                4.06 / 11.59 / 14.51
  mac goinfer      P same pass           9      9         0.23          0.06                0.09 /  0.71 /  0.71
  mac goinfer      S same session       14      7        0.00*          0.42                0.77 /  4.24 /  9.39
  mac metal        O other session      35     11         3.69          0.31                0.52 /  2.67 / 24.53
  mac metal        P same pass          11     11         0.36          0.06                0.09 /  0.71 /  1.41
  mac metal        S same session        2      1         0.32          0.49                0.76 /  0.76 /  0.76
  mac peer         O other session      37     13         6.02          0.31                0.52 / 23.35 / 44.24
  mac peer         P same pass           2      2         0.69          0.97                1.41 /  1.41 /  1.41
  mac peer         S same session        9      3         4.65          4.03                8.39 / 14.51 / 14.51
  nobara           O other session     466     54        0.00*          0.04                0.15 /  0.99 /  5.03
  nobara           S same session       87     37         0.19         0.00*                0.17 /  0.84 /  1.39
  nobara cpu       O other session      40      9         0.51          0.28                0.42 /  1.54 /  2.08
  nobara cpu       S same session        6      6        0.00*         0.00*                0.08 /  0.37 /  0.37
  nobara cuda      O other session     426     45        0.00*          0.04                0.14 /  0.96 /  5.03
  nobara cuda      S same session       81     31         0.20          0.02                0.19 /  0.84 /  1.39
  nobara goinfer   S same session       23      8         0.27         0.00*                0.61 /  1.35 /  1.39
  nobara peer      O other session     466     54        0.00*          0.04                0.15 /  0.99 /  5.03
  nobara peer      S same session       64     29         0.16         0.00*                0.11 /  0.43 /  1.14

## 5b. within-server trend: last run vs first run inside one server lifetime (every used restart with >= 2 runs)

A run level that only adds noise averages away with more runs. A TREND does not: if the rate drifts over a
server's lifetime, a cell with more completions per restart measures a different mean, not a sharper one.
d = 100*(exp(mean ln rate of last run - of first run) - 1) per restart; 't' = mean(d) / (sd(d)/sqrt(n)).

  stratum                  restarts   mean d% median d%      t   % d<-2% % d>+2%
  mac cpu goinfer                24     -2.05      0.17   -1.4       17%      4%
  mac cpu peer                   17     -6.08     -4.03   -2.5       53%     12%
  mac metal goinfer              87      0.04     -0.04    0.3        2%      5%
  mac metal peer                 64      0.11     -0.01    1.0        2%      3%
  nobara cpu goinfer             40     -0.18     -0.04   -1.3        5%      0%
  nobara cpu peer                35     -0.07     -0.04   -1.6        0%      0%
  nobara cuda goinfer           273      0.16     -0.01    1.1        0%      3%
  nobara cuda peer              275     -0.14     -0.05   -1.5        1%      1%
  nobara webgpu goinfer           5      0.93      0.75    3.0        0%     20%

  per-model strata flagged (>= 25% of restarts move > 2% first run -> last run):
    mac cpu goinfer 7B               restarts   7  moved >2%:   4  median d   -3.5%  min  -26.7%  max    0.0%
    mac cpu peer 0.5B                restarts   6  moved >2%:   3  median d   -0.8%  min  -26.2%  max    1.4%
    mac cpu peer 1.5B                restarts   6  moved >2%:   4  median d    2.0%  min  -17.6%  max    9.9%
    mac cpu peer 7B                  restarts   4  moved >2%:   3  median d  -14.2%  min  -21.8%  max    0.8%
    mac cpu peer phi3-mini           restarts   1  moved >2%:   1  median d  -10.8%  min  -10.8%  max  -10.8%
    mac metal goinfer 0.5B           restarts   6  moved >2%:   2  median d   -0.4%  min   -2.9%  max    7.9%
    mac metal goinfer phi3-mini      restarts   1  moved >2%:   1  median d   -2.6%  min   -2.6%  max   -2.6%
    nobara cuda goinfer M35          restarts  10  moved >2%:   3  median d    0.6%  min   -0.4%  max   29.6%
    nobara webgpu goinfer 1.5B       restarts   1  moved >2%:   1  median d    2.0%  min    2.0%  max    2.0%

  the largest |d|, with the cell (how often a trend reproduces across restarts is the tell):
       29.6%  nobara goinfer     M35       cuda   d128   greedy           peer-matrix-2026-09-15/nobara-pc-cuda-sweep.json
      -26.7%  mac    goinfer     7B        cpu    d128   greedy           cpu-decode-peer-gap-2026-09-27/arm64-speed-served.json
      -26.2%  mac    ollama      0.5B      cpu    d128   greedy           peer-claim-2026-09-25-mac/i-cpu-decode.json
      -22.5%  mac    goinfer_old 7B        cpu    d128   greedy           cpu-decode-peer-gap-2026-09-27/arm64-speed-served-pass2.json
      -22.1%  nobara goinfer     M26       cuda   d128   greedy           peer-matrix-2026-09-15/nobara-pc-cuda-sweep.json
      -21.8%  mac    ollama      7B        cpu    d128   greedy           cpu-decode-peer-gap-2026-09-27/arm64-fix-served.json
      -17.6%  mac    ollama      1.5B      cpu    d128   greedy           cpu-decode-peer-gap-2026-09-27/arm64-speed-served.json
      -16.4%  nobara ollama      1.5B      cuda   d512   greedy           peer-matrix-2026-09-17/nobara-pc-cuda-sweep.json
      -15.7%  mac    ollama      7B        cpu    d128   greedy           cpu-decode-peer-gap-2026-09-27/arm64-speed-served-pass2.json
      -14.7%  nobara ollama      1.5B      cuda   d512   greedy           peer-matrix-2026-09-15/nobara-pc-cuda-sweep.json
      -14.2%  mac    ollama      7B        cpu    d128   greedy           cpu-decode-peer-gap-2026-09-27/arm64-speed-served.json
       11.3%  nobara goinfer     M35       cuda   d128   greedy           peer-matrix-2026-09/nobara-w1-d7-m35-m26_2026-09-04.json

## 6. what a restart costs, and what that buys

Per cell: T = secs (start, load, warm-up, all timed completions with their prefill, teardown, the 3 s settle)
and D = timed decode seconds, summed over its completions as (tokens - 1) / rate. Overhead O = T - D. How O
splits between per-start and per-completion work (each completion's prompt prefill) is not recorded, so a
restart's cost a is bracketed: a_hi = O (all overhead is per-start), a_lo = 3 s settle + one warm-up
completion's decode (D/n). The per-completion cost b runs the other way: b_lo = D/n, b_hi = D/n + (O - a_lo)/n.
The idle-gate wait is NOT included, and it is paid once per restart (section 1.2 of the task doc: 44% of
gated wall).

  machine engine   model     backend cells   T med s D med s O med s    a_lo   a_hi    b_lo  b_hi   O share of T
  mac     goinfer  0.5B      cpu         8      19.8    14.2     6.0     3.6    6.0    0.59  0.69            30%
  mac     goinfer  0.5B      metal       3      12.8     6.4     5.9     3.4    5.9    0.40  0.56            46%
  mac     goinfer  1.5B      cpu         8      37.9    29.9     8.1     4.2    8.1    1.25  1.41            21%
  mac     goinfer  1.5B      metal      21      24.7    18.6     6.5     3.8    6.5    0.77  0.89            26%
  mac     goinfer  7B        cpu         7     108.6    86.2    28.4     6.6   28.4    3.59  4.50            26%
  mac     goinfer  7B        metal      19      81.8    68.9    12.8     5.9   12.8    2.87  3.16            16%
  mac     peer     0.5B      cpu         6      20.5    11.7     8.9     3.5    8.9    0.49  0.71            43%
  mac     peer     0.5B      metal       4      19.8    10.5     9.3     3.4    9.3    0.44  0.68            47%
  mac     peer     1.5B      cpu         6      35.7    24.7    11.0     4.0   11.0    1.03  1.32            31%
  mac     peer     1.5B      metal      15      27.2    17.7     9.4     3.7    9.4    0.74  0.97            34%
  mac     peer     7B        cpu         4     112.9    89.4    23.6     6.7   23.6    3.72  4.43            21%
  mac     peer     7B        metal      11      65.1    48.0    16.3     5.0   16.3    2.00  2.47            25%
  nobara  goinfer  0.5B      cpu        12      40.1    27.2    10.0     4.7   10.0    1.70  2.03            25%
  nobara  goinfer  0.5B      cuda       18      11.0     3.0     8.0     3.2    8.0    0.19  0.49            73%
  nobara  goinfer  1.5B      cpu        13      86.8    64.5    15.7     7.0   15.7    4.03  4.57            18%
  nobara  goinfer  1.5B      cuda       19      16.4     4.4    12.2     3.3   12.2    0.28  0.83            74%
  nobara  goinfer  7B        cpu        12     330.8   223.3    45.4    17.0   45.4   13.96 15.73            14%
  nobara  goinfer  7B        cuda       18      44.8    13.8    31.4     3.9   31.4    0.86  2.58            70%
  nobara  goinfer  M26       cuda        5     130.8    42.4    82.8     5.7   82.8    2.65  7.48            63%
  nobara  goinfer  M35       cuda        4     496.5    55.6   440.9     6.5  440.9    3.47 30.63            89%
  nobara  goinfer  gemma3-1b cuda        3      12.5     2.8     9.7     3.2    9.7    0.17  0.58            78%
  nobara  goinfer  phi3-mini cuda       10      38.0    16.2    21.8     3.7   21.8    0.67  1.43            57%
  nobara  peer     0.5B      cpu         9      34.8    17.7    17.3     4.1   17.3    1.10  1.93            50%
  nobara  peer     0.5B      cuda       20      18.2     3.8    14.4     3.2   14.4    0.24  0.94            79%
  nobara  peer     1.5B      cpu        11      64.5    55.6    18.4     5.3   18.4    2.31  2.86            28%
  nobara  peer     1.5B      cuda       18      19.9     5.2    14.7     3.3   14.7    0.32  1.04            74%
  nobara  peer     7B        cpu         9     182.0   145.5    36.7    12.1   36.7    9.09 10.63            20%
  nobara  peer     7B        cuda       20      29.3    11.0    18.3     3.5   18.3    0.46  1.08            62%
  nobara  peer     G20       cpu         2     139.5    93.9    45.6     8.9   45.6    5.87  8.16            33%
  nobara  peer     G20       cuda        2     160.2    95.1    65.1     5.4   65.1    2.38  3.87            41%
  nobara  peer     M26       cuda       10      95.7    45.5    50.0     5.8   50.0    2.84  5.60            52%
  nobara  peer     M35       cuda        8      88.2    41.8    45.0     5.6   45.0    2.61  5.07            51%
  nobara  peer     gemma3-1b cpu         2      52.4    34.1    18.3     5.1   18.3    2.13  2.95            35%
  nobara  peer     gemma3-1b cuda        5      22.6     6.7    15.9     3.4   15.9    0.42  1.20            70%
  nobara  peer     phi3-mini cpu         2     114.6    93.9    20.7     8.9   20.7    5.87  6.61            18%
  nobara  peer     phi3-mini cuda       11      18.0    11.9    10.0     3.5   10.0    0.49  0.76            56%

Design comparison for one goinfer cell mean inside one session (a gate's paired reading), per machine x
model x backend. A design is R restarts x r runs x 8 completions in the same session:
  Var = s2_R/R + s2_B/(R r) + s2_C/(8 R r),   wall = R (a + 8 r b)
s2_R = the machine's pooled in-session restart component for GOINFER cells (section 4b), at its point estimate,
at the 95th percentile of its group bootstrap, and at zero (the best case for fewer restarts).
s2_B and s2_C = that stratum's within-server components
from every used cell (section 3d); W = s2_B + s2_C/8. D0 = today's two-pass gate: R=2 (pass 1 + reversed
pass 2) x r=3. BREAK-EVEN: one restart with any r can match D0 only if s2_R < W/3, i.e. only if the
restart share of a run mean is < 25%. That bar is tighter than the task doc's 50% kill line. 'best' = the
cheapest (R, r), R 1-4 and r 1-60, with Var <= Var(D0), at the dear-restart end (a_hi, b_lo) and the
cheap-restart end (a_lo, b_hi) of the cost bracket. 'trend' = section 5b flags this stratum: a quarter or
more of its restarts move > 2% between first and last run, so more runs per restart shift the mean and the
row does not apply. Session drift s2_S is not in Var: nothing inside one session reduces it.

  stratum                 sdW% trend  s2_R   sdR% R%run R=1 ok  sd(D0)% sd(1x6)%           best, dear restart          best, cheap restart
  mac 0.5B cpu            0.78     -  point  0.00     0    yes     0.32     0.32    1 x 6     34s/  40s   15%    1 x 6     37s/  40s    9%
                                      p95    1.28    73     no     0.96     1.32    3 x 1     32s/  40s   20%    3 x 1     27s/  40s   32%
                                      zero   0.00     0    yes     0.32     0.32    1 x 6     34s/  40s   15%    1 x 6     37s/  40s    9%
  mac 0.5B metal          2.00   YES  point  0.00     0    yes     0.82     0.82    1 x 6     25s/  31s   19%    1 x 6     30s/  34s   10%
                                      p95    1.28    29     no     1.22     1.52    2 x 3     31s/  31s    0%    4 x 1     32s/  34s    6%
                                      zero   0.00     0    yes     0.82     0.82    1 x 6     25s/  31s   19%    1 x 6     30s/  34s   10%
  mac 1.5B cpu            0.64     -  point  0.00     0    yes     0.26     0.26    1 x 6     68s/  76s   11%    1 x 6     72s/  76s    6%
                                      p95    1.28    80     no     0.94     1.31    3 x 1     54s/  76s   29%    3 x 1     46s/  76s   39%
                                      zero   0.00     0    yes     0.26     0.26    1 x 6     68s/  76s   11%    1 x 6     72s/  76s    6%
  mac 1.5B metal          0.48     -  point  0.00     0    yes     0.20     0.20    1 x 6     44s/  50s   13%    1 x 6     46s/  50s    8%
                                      p95    1.28    88     no     0.93     1.30    3 x 1     38s/  50s   24%    3 x 1     33s/  50s   35%
                                      zero   0.00     0    yes     0.20     0.20    1 x 6     44s/  50s   13%    1 x 6     46s/  50s    8%
  mac 7B cpu              8.32   YES  point  0.00     0    yes     3.40     3.40    1 x 6    201s/ 229s   12%    1 x 6    223s/ 229s    3%
                                      p95    1.28     2    yes     3.52     3.63    2 x 3    229s/ 229s    0%    2 x 3    229s/ 229s    0%
                                      zero   0.00     0    yes     3.40     3.40    1 x 6    201s/ 229s   12%    1 x 6    223s/ 229s    3%
  mac 7B metal            0.28     -  point  0.00     0    yes     0.11     0.11    1 x 6    151s/ 163s    8%    1 x 6    158s/ 163s    4%
                                      p95    1.28    95     no     0.91     1.29    3 x 1    107s/ 163s   34%    3 x 1     93s/ 163s   43%
                                      zero   0.00     0    yes     0.11     0.11    1 x 6    151s/ 163s    8%    1 x 6    158s/ 163s    4%
  nobara 0.5B cpu         0.81     -  point  0.35    16    yes     0.41     0.48    2 x 3    102s/ 102s    0%    2 x 3    107s/ 107s    0%
                                      p95    0.52    29     no     0.49     0.61    4 x 1     94s/ 102s    7%    4 x 1     84s/ 107s   22%
                                      zero   0.00     0    yes     0.33     0.33    1 x 6     92s/ 102s   10%    1 x 6    102s/ 107s    4%
  nobara 0.5B cuda        0.66     -  point  0.35    22    yes     0.37     0.44    2 x 3     25s/  25s    0%    2 x 3     30s/  30s    0%
                                      p95    0.52    38     no     0.45     0.58    2 x 3     25s/  25s    0%    4 x 1     28s/  30s    5%
                                      zero   0.00     0    yes     0.27     0.27    1 x 6     17s/  25s   32%    1 x 6     27s/  30s   11%
  nobara 1.5B cpu         0.27     -  point  0.35    62     no     0.27     0.37    3 x 1    144s/ 225s   36%    3 x 1    131s/ 233s   44%
                                      p95    0.52    78     no     0.38     0.53    3 x 1    144s/ 225s   36%    3 x 1    131s/ 233s   44%
                                      zero   0.00     0    yes     0.11     0.11    1 x 6    209s/ 225s    7%    1 x 6    226s/ 233s    3%
  nobara 1.5B cuda        0.78     -  point  0.35    17    yes     0.40     0.47    2 x 3     38s/  38s    0%    2 x 3     47s/  47s    0%
                                      p95    0.52    30     no     0.48     0.61    2 x 3     38s/  38s    0%    4 x 1     40s/  47s   15%
                                      zero   0.00     0    yes     0.32     0.32    1 x 6     25s/  38s   32%    1 x 6     43s/  47s    7%
  nobara 7B cpu           0.43     -  point  0.35    39     no     0.30     0.39    4 x 1    628s/ 761s   17%    4 x 1    571s/ 789s   28%
                                      p95    0.52    59     no     0.41     0.55    3 x 1    471s/ 761s   38%    3 x 1    428s/ 789s   46%
                                      zero   0.00     0    yes     0.18     0.18    1 x 6    715s/ 761s    6%    1 x 6    772s/ 789s    2%
  nobara 7B cuda          0.61     -  point  0.35    25    yes     0.35     0.43    2 x 3    104s/ 104s    0%    2 x 3    132s/ 132s    0%
                                      p95    0.52    41     no     0.44     0.57    2 x 3    104s/ 104s    0%    4 x 1     98s/ 132s   26%
                                      zero   0.00     0    yes     0.25     0.25    1 x 6     73s/ 104s   30%    1 x 6    128s/ 132s    3%
  nobara M26 cuda         6.22     -  point  0.35     0    yes     2.55     2.56    1 x 7    231s/ 293s   21%    2 x 3    370s/ 370s    0%
                                      p95    0.52     1    yes     2.57     2.59    1 x 7    231s/ 293s   21%    2 x 3    370s/ 370s    0%
                                      zero   0.00     0    yes     2.54     2.54    1 x 6    210s/ 293s   28%    1 x 6    364s/ 370s    2%
  nobara M35 cuda         6.76   YES  point  0.35     0    yes     2.77     2.78    1 x 7    635s/1049s   39%    2 x 3   1483s/1483s    0%
                                      p95    0.52     1    yes     2.78     2.81    1 x 7    635s/1049s   39%    2 x 3   1483s/1483s    0%
                                      zero   0.00     0    yes     2.76     2.76    1 x 6    608s/1049s   42%    1 x 6   1477s/1483s    0%
  nobara gemma3-1b cuda   0.88     -  point  0.35    14    yes     0.43     0.50    1 x 12    26s/  28s    5%    2 x 3     34s/  34s    0%
                                      p95    0.52    26     no     0.51     0.63    2 x 3     28s/  28s    0%    4 x 1     31s/  34s    9%
                                      zero   0.00     0    yes     0.36     0.36    1 x 6     18s/  28s   35%    1 x 6     31s/  34s    9%
  nobara phi3-mini cuda   0.54     -  point  0.35    30     no     0.33     0.41    2 x 3     76s/  76s    0%    4 x 1     60s/  76s   20%
                                      p95    0.52    48     no     0.43     0.56    2 x 3     76s/  76s    0%    4 x 1     60s/  76s   20%
                                      zero   0.00     0    yes     0.22     0.22    1 x 6     54s/  76s   29%    1 x 6     72s/  76s    5%

  Read with section 6's O share: where a restart is cheap next to its completions (a_hi small against 48 b),
  dropping one saves little wall whatever the variance says.

## 7. sensitivity: the per-machine headline with the curation and the prompt key removed

  as analysed (curated, prompt in key)                 mac     groups  33   3-level R % run mean   67   4-level: R in-session   31%  (S+R)   78%
  as analysed (curated, prompt in key)                 nobara  groups  75   3-level R % run mean    0   4-level: R in-session    0%  (S+R)   20%
  no curation (void + arm files pooled)                mac     groups  41   3-level R % run mean   90   4-level: R in-session   93%  (S+R)   93%
  no curation (void + arm files pooled)                nobara  groups 115   3-level R % run mean   90   4-level: R in-session   94%  (S+R)   94%
  no prompt key (essay-v2 pooled with older prompts)   mac     groups  29   3-level R % run mean   68   4-level: R in-session   33%  (S+R)   76%
  no prompt key (essay-v2 pooled with older prompts)   nobara  groups  65   3-level R % run mean   26   4-level: R in-session    0%  (S+R)   44%
  neither                                              mac     groups  35   3-level R % run mean   90   4-level: R in-session   93%  (S+R)   93%
  neither                                              nobara  groups  98   3-level R % run mean   90   4-level: R in-session   95%  (S+R)   95%
```

</details>
