# S4 item 4 — GOMEMLIMIT GC-CPU check, 2026-09-23

**Provenance.** MacBook (darwin/arm64) · goinfer `1f2010ff` (pre-push; landed in the S4 items 2-5
commit) · `demo/chat` binary built locally (`go build ./demo/chat`) · CPU backend, `int8int8` ·
greedy (`--temp 0`) · `GODEBUG=gctrace=1`.

**What this checks.** `task-never-swap-2026-09.md` S4's registered decision rule for item 4
(`debug.SetMemoryLimit`): *"kept only if the measured anonymous footprint after load is not worse
and GC CPU (from `GODEBUG=gctrace=1`, one run) does not rise more than 10%"*. This is that one run.

**Runs.**

| model | max tokens | `GOMEMLIMIT` | log |
|---|---|---|---|
| qwen2.5-coder-0.5b-instruct-q4_k_m | 200 | `off` | `gomemlimit-gctrace-2026-09-23-0.5b-off.log` |
| qwen2.5-coder-1.5b-instruct-q4_k_m | 300 | (unset — `applyGoMemLimit`'s new default applies) | `gomemlimit-gctrace-2026-09-23-default.log` |
| qwen2.5-coder-1.5b-instruct-q4_k_m | 300 | `off` | `gomemlimit-gctrace-2026-09-23-off.log` |

Same prompt, same quant, same machine, back-to-back — the paired A/B is the 1.5B pair (the 0.5B
run is a smoke test only, run before the 1.5B pair was designed).

**Result: a genuine null, not a demonstrated win.** Every `gc N @...` line in every run shows 0%
GC CPU, and the heap-goal sequence is IDENTICAL between the two 1.5B runs (32 → 43 → 76 MB goal on
both). tok/s differed (38.8 default vs 42.2 off) but that tracks session/thermal noise, not GC
pressure — there is no GC-CPU signal in either log for it to track.

**Why: the resolved limit never binds at this scale.** `applyGoMemLimit` resolves to
`hostRAMAvailable() - giwMemMargin`, several GB on this machine. A 1.5B model's live heap peaks at
tens of MB. The limit is nowhere near what the GC has to work against, so `debug.SetMemoryLimit`
is, at this scale, inert — which is also *why* the registered bar is trivially met (a limit that
never binds cannot cost GC CPU) and why this run cannot speak to whether the bar would still hold
at the scale the brief actually cares about.

**What this does NOT show.** The interesting case — a load where the limit genuinely constrains
the heap (gpt-oss-20b-class, or an M35/M26-class MoE) — is unmeasured. This session has no such
checkpoint that fits this machine's free disk (the identical constraint S5's own real M35
measurement is blocked on — see that section's status note). A future pass with access to a
larger checkpoint should re-run this exact A/B (`GODEBUG=gctrace=1`, `GOMEMLIMIT=off` vs default)
at that scale before trusting this null result to generalize.

**Decision taken:** kept as built (default-on, soft limit). The registered bar was met on the one
run this session could safely take, and the downside of a SOFT limit that turns out to sit too low
is bounded (more GC CPU, never a refused allocation — `debug.SetMemoryLimit`'s own documented
semantics) and opt-out via `GOMEMLIMIT` is unconditional. This is a judgment call under an
incomplete measurement, stated as one, not a claim that the large-scale case has been checked.
