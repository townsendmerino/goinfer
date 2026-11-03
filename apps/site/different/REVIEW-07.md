# Review notes: writeup 07 "It refuses rather than swaps" (drafted 2026-09-29)

Draft only (`reviewed:` empty). Body about 830 words. The site build passes with `-drafts`. Nothing was run or measured; every figure is copied from a record.

## Claims and sources

| Claim | Source |
|---|---|
| v0.16.0, 16 GB M1 Pro, 35B-A3B Q4_K_M: no warning, swap +7,819 MB in 5 s | `docs/measurements/cold-user-2026-09-06.md`, "D", numbers table |
| gpt-oss-20b plain-CPU load took swap to 22.9 GB, on 2026-09-04/05 | `task-never-swap-2026-09.md` "What this is" (22.6 to 22.9 GB); `swap-tripwire-2026-09-22.md` (22.9 GB) |
| Real refusal text (24.3 / 6.0 / 11.3 / 41.6 / 5.2 / 3.7 GB), zero swap growth | `swap-tripwire-2026-09-22.md`, "Why the fit guard had to be bypassed". Wording cross-checked against `fitCheck.arithmetic()` and `declineErr()` in `decoder/fitguard.go` |
| Refusal names -stream-weights, smaller model/quant, `GOINFER_NO_FIT_GUARD=1` | `decoder/fitguard.go` `remedy()`, `declineErr()` |
| Estimate = weights + KV + mapped source file vs 70% of currently available memory; live, not total | `decoder/fitguard.go` (`fitMemFraction`, `arithmetic`); `CHANGELOG.md` v0.17.2; `task-first-hour.md` "R13-follow-on" |
| Third live re-run: refused in under 5 seconds, zero Swapouts, 8.9 GB weights vs 5.0 GB budget (70% of ~7.1-7.2 GB) | `task-first-hour.md` "Re-verified live, third time" |
| Sidecar is the default for a .gguf on macOS and Linux; a .giw load prices KV + scratch only | `internal/prequant/prequant.go` `DefaultToSidecar`; `task-never-swap-2026-09.md` S4 item 1; `docs/env-vars.md` `GOINFER_GGUF_DIRECT` |
| Automatic retry: dense only, one retry, note on stderr, `--fit=off` keeps the refusal | `internal/modelload/modelload.go` (the retry block); `loadflags.go` `fitHelp` |
| Retry cost: streamed/resident 0.944, ~5.6%, 7B q4_k_m, window 7 of 28, 6 runs per arm, 2026-09-09 | `task-gpu-paths-2026-09.md` (entry dated 2026-09-09); `task-fit-to-hardware.md` header |
| Tripwire: swap-used, first sample is baseline, +512 MB, 2 s poll, direct .gguf load aborts naming growth and priced terms | `decoder/swapwatch.go` defaults; `internal/swapguard/swapguard.go`; `docs/env-vars.md` `GOINFER_SWAP_GUARD` |
| Serving half: 503, recovers after 30 s within threshold, running requests finish, banner line | `docs/server.md`; `internal/serveapp/swapguard.go` |
| Positive control: tripped at +0.72-0.80 GB, peak +1.69 GB, external kill, machine recovered, cause inferred not traced | `swap-tripwire-2026-09-22.md` (result, table, "Why the peak overshoots") |
| Negative controls: 100/100 each, 0 trips, 467 readings flat at 1,346.12 MB, M1 Pro 16 GB | `swap-tripwire-negative-controls-2026-09-24.md` |
| Sidecar vs direct: anonymous footprint 16-21%, swap delta 0 MB | `sidecar-default-2026-09-22.md` |
| Load-time abort covers only a direct .gguf build, not sidecar/streamed/safetensors | `internal/modelload/modelload.go` `loadGuarded`; `task-never-swap-2026-09.md` S3 "Scope, deliberately narrow" |
| Swap probe on macOS and Linux only, else unknown | `decoder/memwatch_darwin.go`, `memwatch_linux.go`, `memwatch_other.go` |
| MoE excluded from the auto retry; manual CPU MoE streaming ran 2h10 with zero completions | `decoder/fitguard.go` `FitDeclineError` comment; `task-never-swap-2026-09.md` S0 |
| Guard is an "every unknown proceeds" design; 70% came from one measured failure | `decoder/fitguard.go` header and `fitMemFraction` comment |
| `goinfer-chat fit <path>` loads the checkpoint | `internal/fitcmd/fit.go` usage; only `internal/chatapp/main.go` dispatches `fit` (serve does not) |

## Conflicts between records

- **The brief said `--fit=off` can "keep or drop" the refusal. The code and records say it only keeps it.** `--fit=off` disables the automatic retry and the fit-by-default sizing. It does not bypass the guard: `swap-tripwire-2026-09-22.md` says "`-fit=off` does NOT bypass it", and only `GOINFER_NO_FIT_GUARD=1` does. The draft says this.
- The task doc's S3 build text says the serving refusal is a **529**; the newer `docs/server.md` and the code (`halt.go`) say **503**. I used 503.
- The task doc S1 says linux defaults to direct; its own later note (2026-09-24) and `DefaultToSidecar` say linux uses the sidecar. I used the newer.
- S4 item 4 (`GOMEMLIMIT`) is marked dropped in the LATEST note, so it is not described as a feature.
- The task doc's older S4 text calls the MoE working-set predictor never validated; the LATEST note says it over-predicted by ~15-20% on M35. I left the predictor and `-accept-slow` out (see below).

## Left out as unverified or out of scope

- **No `docs/releases/v0.17.2.md` exists.** The "live-verified memory guard" evidence is in `CHANGELOG.md` v0.17.2 and `task-first-hour.md`; I cited those.
- The `--quant q4k` load path (no .giw form, always direct) and what the fit-guard retry does there: not checked. The draft says a plain .gguf "often" meets the sidecar path first.
- What today's default darwin load of gpt-oss-20b prints on a 16 GB Mac (S2 made it sidecar-transcodable on 2026-09-24): no record, so the refusal example is dated 2026-09-22 and labelled as such.
- The MoE working-set predictor, `-accept-slow`, the pool/mmap pager, S6 Metal aliasing, `GOMEMLIMIT`: real but not needed for this claim.
- The chip in the 2026-09-22 and 2026-09-09 records is not named ("MacBook", "this 16 GB Mac"); only the 09-06 and 09-24 records say M1 Pro. The draft says "the MacBook" for those rows.
- The exact date of the third live re-run: the records give only "before v0.17.2, tagged 2026-09-08".
- Windows and any non-macOS/Linux behaviour: nothing measured. The draft says so.

## Questions for the owner

1. The 2026-09-22 refusal example predates the 2026-09-24 sidecar default for every family. Do you want a fresh refusal captured on the Mac (a quick, safe run, since it refuses before allocating) so the page shows current wording?
2. `--fit=off` is worded here as "keeps the refusal, does not turn the guard off". Is that the behaviour you want people to rely on, or is `--fit=off` meant to become the off switch?
3. The 0.944 streaming cost comes from a throwaway scratchpad script (6 runs per arm, one model). Comfortable quoting it, or should it wait for a repeat?
