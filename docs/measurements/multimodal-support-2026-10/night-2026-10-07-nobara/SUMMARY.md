# Night run 2026-10-07 — nobara-pc

Runner finished. Started 2026-10-07 20:30 PDT, deadline 06:30, ended 00:15 (3 h 45 min).

9 of 11 job(s) ok.

| # | job | result | ran | est | queued at rev | log |
|---|---|---|---|---|---|---|
| 1 | gate-gpu-cuda | stopped (re-queued) | 35 min | 1 h 30 min | f95fabfd | `/home/francis/goinfer-logs/night/runs/2026-10-07/gate-gpu-cuda.log` |
| 2 | gate-gpu-cuda | FAILED rc=1 | 2 h 12 min | 1 h 30 min | f95fabfd | `/home/francis/goinfer-logs/night/runs/2026-10-07/gate-gpu-cuda.log` |
| 3 | s1c-e2b-speed | ok | 4 min | 30 min | 928f9a41 | `/home/francis/goinfer-logs/night/runs/2026-10-07/s1c-e2b-speed.log` |
| 4 | s5b-e2b-audio | ok | 4 min | 20 min | ec97c37b | `/home/francis/goinfer-logs/night/runs/2026-10-07/s5b-e2b-audio.log` |
| 5 | s4-tower-speed | ok | 15 min | 30 min | c80e9983 +dirty | `/home/francis/goinfer-logs/night/runs/2026-10-07/s4-tower-speed.log` |
| 6 | s9a-e2b-prefill-ttft | ok | 9 min | 15 min | 698e8a31 | `/home/francis/goinfer-logs/night/runs/2026-10-07/s9a-e2b-prefill-ttft.log` |
| 7 | s9b-e2b-image-ttft | ok | 10 min | 20 min | e6d18adb | `/home/francis/goinfer-logs/night/runs/2026-10-07/s9b-e2b-image-ttft.log` |
| 8 | peer-vetted-nobara | ok | 27 min | 50 min | 7e1c1275 | `/home/francis/goinfer-logs/night/runs/2026-10-07/peer-vetted-nobara.log` |
| 9 | s4sig-speed | ok | 3 min | 12 min | f37fe277 | `/home/francis/goinfer-logs/night/runs/2026-10-07/s4sig-speed.log` |
| 10 | s7-nobara | ok | 5 min | 1 h 00 min | 7a67be44 | `/home/francis/goinfer-logs/night/runs/2026-10-07/s7-nobara.log` |
| 11 | s13lite-nobara | ok | 4 min | 55 min | 7a67be44 +dirty | `/home/francis/goinfer-logs/night/runs/2026-10-07/s13lite-nobara.log` |

## Not ok — last 25 lines of each log

### gate-gpu-cuda — stopped (re-queued)

```
      | github.com/townsendmerino/goinfer/gpu.(*Context).Close(0x3cdafd84ce08)
      | github.com/townsendmerino/goinfer/gpu.TestRMSNormBatched_parity(0x3cdafd7ca488)
      | 	/home/francis/goinfer-bench/gate-gpu-night-wt/gpu/prefillrunner_batched_test.go:164 +0x18d4
      | FAIL	github.com/townsendmerino/goinfer/gpu	600.446s

[1m== 5. repo hygiene (derived from .github/workflows/ci.yml) ==[0m
  [32mPASS[0m    GITIGNORED DESTINATIONS: none. A committed document citing a path under a gitignored directory is red without an existence test — the gitignore status decides, because an uncommitted target has no history to audit.
  [33mSKIP[0m  21 darwin-only CI hygiene step(s) — wrong platform for this host
  [32mPASS[0m  9 CI hygiene check(s) reproduced locally, derived from ci.yml

[1m== verdict ==[0m
  check groups: 10 declared -> 10 reported   |   verdicts within them: 12 pass, 1 skip, 2 fail
  of which: kernel-level suite = reported   |   resident PARITY gates (forward asserted) = reported

  [33mSkipped — a skip is not a pass; this gate does NOT cover:[0m
    - heavy tier skipped 47 test(s) — see the 2c census for which
    - graphs are FORCED in 2d (GOINFER_CUDA_GRAPHS_UNSAFE): this proves the capture/replay CODE, not that graphs are admitted in production — admitGraphs still declines here (DEFAULT compute mode, no MPS).
    - SKIPPED: 21 darwin-only CI hygiene step(s) — wrong platform for this host

  [31mFAIL[0m — cuda on Linux @ 420f4404. Do not tag.
exit status 1
!! the gate printed no PASS verdict (rc=1); see /home/francis/goinfer-logs/gate-gpu-night-2026-10-07/gate-gpu.log
finished: 2026-10-07 22:41:49 PDT rc=1

=== END      2026-10-07 22:41:50 PDT  rc=1
```

### gate-gpu-cuda — FAILED rc=1

```
      | github.com/townsendmerino/goinfer/gpu.(*Context).Close(0x3cdafd84ce08)
      | github.com/townsendmerino/goinfer/gpu.TestRMSNormBatched_parity(0x3cdafd7ca488)
      | 	/home/francis/goinfer-bench/gate-gpu-night-wt/gpu/prefillrunner_batched_test.go:164 +0x18d4
      | FAIL	github.com/townsendmerino/goinfer/gpu	600.446s

[1m== 5. repo hygiene (derived from .github/workflows/ci.yml) ==[0m
  [32mPASS[0m    GITIGNORED DESTINATIONS: none. A committed document citing a path under a gitignored directory is red without an existence test — the gitignore status decides, because an uncommitted target has no history to audit.
  [33mSKIP[0m  21 darwin-only CI hygiene step(s) — wrong platform for this host
  [32mPASS[0m  9 CI hygiene check(s) reproduced locally, derived from ci.yml

[1m== verdict ==[0m
  check groups: 10 declared -> 10 reported   |   verdicts within them: 12 pass, 1 skip, 2 fail
  of which: kernel-level suite = reported   |   resident PARITY gates (forward asserted) = reported

  [33mSkipped — a skip is not a pass; this gate does NOT cover:[0m
    - heavy tier skipped 47 test(s) — see the 2c census for which
    - graphs are FORCED in 2d (GOINFER_CUDA_GRAPHS_UNSAFE): this proves the capture/replay CODE, not that graphs are admitted in production — admitGraphs still declines here (DEFAULT compute mode, no MPS).
    - SKIPPED: 21 darwin-only CI hygiene step(s) — wrong platform for this host

  [31mFAIL[0m — cuda on Linux @ 420f4404. Do not tag.
exit status 1
!! the gate printed no PASS verdict (rc=1); see /home/francis/goinfer-logs/gate-gpu-night-2026-10-07/gate-gpu.log
finished: 2026-10-07 22:41:49 PDT rc=1

=== END      2026-10-07 22:41:50 PDT  rc=1
```


## Who picks these up

- gate-gpu-cuda: nobara session, R1 re-run after f95fabfd · docs/queue-release.md
- gate-gpu-cuda: nobara session, R1 re-run after f95fabfd · docs/queue-release.md
- s1c-e2b-speed: nobara session, S1 on CUDA · docs/tasks/task-multimodal-support-2026-10.md
- s5b-e2b-audio: Claude (Mac), S5 · docs/tasks/task-multimodal-support-2026-10.md
- s4-tower-speed: nobara session, S4 · docs/tasks/task-multimodal-support-2026-10.md
- s9a-e2b-prefill-ttft: nobara session, S9 on CUDA part A · docs/tasks/task-multimodal-support-2026-10.md
- s9b-e2b-image-ttft: nobara session, S9 on CUDA part B · docs/tasks/task-multimodal-support-2026-10.md
- peer-vetted-nobara: Claude (Mac), peer-vetted · docs/measurements/peer-vetted-2026-10-07-nobara-pc.md · pre-registered at 7e1c1275; serve built at 65b2c22a; grade.jsonl in /home/francis/goinfer-bench/peer-vetted-2026-10-07/results
- s4sig-speed: nobara session, S4 addendum · docs/tasks/task-multimodal-support-2026-10.md
- s7-nobara: nobara session, S7 · docs/tasks/task-multimodal-support-2026-10.md
- s13lite-nobara: nobara session, S13-lite · docs/tasks/task-multimodal-support-2026-10.md
