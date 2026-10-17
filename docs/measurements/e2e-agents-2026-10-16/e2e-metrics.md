# End-to-end agent run: extractor output (2026-10-16 18:24:07 PDT)

### this run (serve 968c50d6)

`/home/francis/goinfer-bench/e2e-agents-2026-10-16/e2e-out`

| agent | passed | session start (s) | follow-up TTFT | cold restarts | compactions (agent / wire) | snapshot restores | carried-id fallbacks | fast-alias | side requests | unaligned runs |
|---|---|---|---|---|---|---|---|---|---|---|
| claude-code | 1/3 (read+,search-,edit-) | 198.2, 198.4, 197.2 | n=5 median 193.2 s, p90 208.2 s, max 208.2 s | 3 of 5 | 2 / 0 | 0 (0 rows) | none | 0 | 0 | 0 |
| codex | 2/3 (read+,search+,edit-) | 62.6, 0.3, 0.4 | n=45 median 2.2 s, p90 3.5 s, max 66.7 s | 2 of 45 | 3 / 1 | 3 (18240 rows) | history=27 | 0 | 3 | 0 |
| opencode | 3/3 (read+,search+,edit+) | 84.2, 5.6, 5.7 | n=21 median 0.5 s, p90 3.2 s, max 6.6 s | 0 of 21 | 0 / 0 | 3 (22464 rows) | none | 3 | 4 | 0 |

## Baselines (cross-session: different builds, different days; read as context, not as a paired difference)

### opencode, the day run of 2026-10-15 23:05 (serve 2a0d11bb, one rep)

`/home/francis/goinfer-bench/ids-cc-2026-10/out`

| agent | passed | session start (s) | follow-up TTFT | cold restarts | compactions (agent / wire) | snapshot restores | carried-id fallbacks | fast-alias | side requests | unaligned runs |
|---|---|---|---|---|---|---|---|---|---|---|
| opencode | 3/3 (read+,search+,edit+) | 106.9, 97.5, 121.8 | n=18 median 1.3 s, p90 3.4 s, max 95.4 s | 1 of 18 | 0 / 0 | 0 (0 rows) | none | 0 | 4 | 0 |

### I3 job B, 2026-10-13 (before carried ids; five tasks, so rows hold read,search,edit,write,fix)

`/home/francis/goinfer-bench/harness-2026-10-13/i3-out-2026-10-13-mellum-16k+mellum-32k-cc`

| agent | passed | session start (s) | follow-up TTFT | cold restarts | compactions (agent / wire) | snapshot restores | carried-id fallbacks | fast-alias | side requests | unaligned runs |
|---|---|---|---|---|---|---|---|---|---|---|
| claude-code | 1/5 (read+,search-,edit-,write-,fix-) | 458.2, 451.3, 451.7 | n=4 median 225.6 s, p90 491.5 s, max 491.5 s | 0 of 4 | 2 / 0 | 0 (0 rows) | none | 0 | 0 | 2 |
| codex | 2/5 (read-,search+,edit-,write+,fix-) | 61.3, 61.4, 61.0, 61.2, 61.3 | n=53 median 62.4 s, p90 80.7 s, max 98.0 s | 0 of 53 | 1 / 0 | 0 (0 rows) | none | 0 | 1 | 0 |
| opencode | 4/5 (read+,search+,edit+,write-,fix+) | 130.6, 128.9, 124.8, 109.0, 94.2 | n=31 median 0.9 s, p90 3.2 s, max 3.3 s | 0 of 31 | 0 / 0 | 0 (0 rows) | none | 0 | 5 | 0 |

### I3 after-run, Codex with carried ids on (2026-10-15; three reps of five tasks)

`/home/francis/goinfer-bench/ids3-2026-10/ids3-out`

| agent | passed | session start (s) | follow-up TTFT | cold restarts | compactions (agent / wire) | snapshot restores | carried-id fallbacks | fast-alias | side requests | unaligned runs |
|---|---|---|---|---|---|---|---|---|---|---|
| codex | 10/15 (read+,search+,edit+,write+,fix-,read+,search+,edit-,write-,fix-,read+,search+,edit+,write+,fix-) | 64.5, 63.2, 64.5, 64.1, 64.2, 64.2, 63.6, 64.8, 65.0, 64.0, 64.1, 63.5, 65.2, 65.1, 63.0 | n=247 median 1.1 s, p90 4.3 s, max 129.2 s | 16 of 247 | 15 / 12 | 0 (0 rows) | decode=19, history=185 | 0 | 15 | 0 |
