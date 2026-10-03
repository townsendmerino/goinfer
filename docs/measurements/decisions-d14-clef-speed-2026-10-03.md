# D14 — Clef-flash against JEV-9B, speed: the Mac CPU, 1 and 5 questions about one state — PRE-REGISTRATION DRAFT 2026-10-03

**Status: a draft written on nobara for the Mac's session to finish and queue. Nothing here has run on the Mac.** The Mac's night queue is its own (`night.py` is per machine), and the run must be sized from a Mac smoke first (section 5). The Mac session fills in section 5, commits this file, and only then queues the job; the harness, the run script and the projection are done.

## 1. The question

D14 of `docs/tasks/task-constrained-confidence.md`: **how much does a record of 1 and of 5 questions about one state cost on Clef-flash, against JEV-9B, on the Mac CPU?** And so: **does Clef make D8 (shared state across questions on hybrid models) unnecessary for TypeSafe-shaped requests?** D7 measured the problem on nobara: on this family each question pays a full prefill, so five JEV-style decisions cost five times one (4.62 s against 5 x 0.92 s at K=256, to 0.1%), and D8's trigger fired at 0.909. Clef reads the whole record in one backbone pass, so its cost should grow with the schema text and not with the state.

Both models are 9B `qwen3_5` on the same CPU path at the same precision, so the comparison is a comparison of how many tokens each route prefills. That is what the projection below prices, and what the measurement can contradict.

## 2. Projection, written before any Mac number exists

Per-token prefill cost is taken as equal for the two (same architecture, same quant, same backend), so time follows input tokens. Input tokens:

- **Clef**, exact, from the official `encode_record` and Clef-flash's tokenizer on D7's frozen prompts (`decisions-d7-2026-09-28/prompts.json`, sha256 `2d56d87f...`), mean over the 8 states per K: 1 question 428 / 1,187 / 4,252 tokens at K = 256 / 1,024 / 4,096; 5 questions 836 / 1,595 / 4,660. A first question costs the state plus about 172 tokens of scaffolding; each further question about 100.
- **JEV**, from D7's measured label-route requests on the same prompts (the nearest measured stand-in: JEV's bare-v1 template has no chat wrapper, so its true counts are somewhat lower and the projected ratio somewhat lower with them): 1 question 317 / 1,076 / 4,141; **5 questions = five prefills**, 1,584 / 5,378 / 20,704.

| K | JEV time / Clef time, 1 question | JEV time / Clef time, 5 questions | Clef 5q / 1q | JEV 5q / 1q |
|---|---|---|---|---|
| 256 | 0.74 | 1.89 | 1.95 | 5.0 |
| 1,024 | 0.91 | 3.37 | 1.34 | 5.0 |
| 4,096 | 0.97 | 4.44 | 1.10 | 5.0 |

A ratio above 1 means Clef is faster. **The projection says Clef is slower for ONE question (its scaffolding costs 100 to 170 extra tokens) and increasingly faster as questions are added**, by 1.9x to 4.4x at five.

What it leaves out, deliberately named: (a) **the head's own cost**, unmeasured, on the CPU in f32 with a scalar attention (six attentions over the whole sequence); it can only add time to the Clef side, so the registered bands are widened on that side; (b) **attention's quadratic share at long K**, which D7 saw as a lower prefill rate at 4,096 (265 tokens per second against about 340 below) and which applies to both routes; (c) any per-request overhead.

**Bands** (the projected ratio x 0.80 to x 1.10, the lower edge wider for the head): a cell whose 95% interval overlaps its band is "as projected"; otherwise it is OFF PROJECTION and the record names the wrong input from the measured requests (usage tokens divided by time) before the number is quoted, as D7 had to.

## 2a. Amendment (2026-10-03, after the projection above and before any Mac run): D8 changes what the JEV arm costs

`decisions-d8-shared-state-2026-10-03.md` (on `main`) makes the CPU decision path prefill what a request's questions share ONCE. The Mac's 9B does not run resident, so **the JEV arm on the Mac now runs with sharing on**, and section 2's "JEV 5q = five prefills" no longer holds there. JEV's bare-v1 template puts `[kind]` before the state, so only same-kind questions share: D7's five questions (noul, score, choice, noul, choice) form three groups, measured on the real weights as 3 prefills for 5 prompts (52.7 s against 80.0 s at K=256, exploratory). Clef's cost is unchanged.

Revised projection for the JEV arm at 5 questions (about 3K + 170 tokens; the structure is the same at every K): **JEV/Clef = 1.12 (K=256), 2.03 (K=1,024), 2.67 (K=4,096)** in place of 1.89, 3.37 and 4.44. The 1-question ratios (0.74, 0.91, 0.97) and Clef's 5q/1q growth are unchanged; **JEV's 5q/1q growth falls from 5.0 to about 2.9 at K=256, 3.0 at 1,024 and 3.0 at 4,096** (3 prefills for 1). Bands are the revised projection x 0.80 to x 1.10, as before.

What this means for the measurement: the question is no longer "Clef against an unshared JEV" but "Clef against JEV as it is served now". **The serve binary must be built from a revision that contains D8 (`main` after it lands), and its sha256 recorded; a binary from before it measures the old JEV arm and must say so.** The unshared JEV arm is what `decisions-d7-2026-09-28.md` section 5 already measured on CUDA. The registered D8 rule (Clef's own 5q/1q <= 2.0) is about the Clef route and is unaffected.

## 3. Design

- **Machine and path:** the Mac, `-backend cpu` (the 9B does not run resident on the Mac, `decisions-d7-2026-09-28.md` section 2), both models at `int8int8` (the decision-model default). The harness checks the server's `decode path:` line and voids the run on a resident path.
- **Models:** `~/models/JEV-9B` (with `head=`, Route B) and `~/models/clef-flash`, both on the internal SSD; never `/Volumes/` and never the archive. Each is about 9.5 GB at int8int8, so **one is resident at a time** on a 16 GB machine.
- **Prompts:** D7's frozen `prompts.json` and question wording (`d7_bench.Q`, `FIVE`), unchanged, so both arms read the same states and the same questions. The 1-question shape rotates noul / score / choice by state as D7 did. The score question has six levels, which Route B requires.
- **Arms:** `jev` and `clef`, each at 1 and at 5 questions, all as `POST /v1/systemone` with the same body; only the route differs.
- **Order:** two passes, the second with the block order reversed (pass 1 JEV then Clef, pass 2 Clef then JEV), because the models cannot be interleaved request by request. Inside a block, per K, per state, the 1-question request then the 5-question request. A warm-up of 2 requests is discarded after each server start.
- **Quiet box:** an idle gate before every state (on the Mac the instant gate, the share of CPUs busy over a one-second window must be at most 10%; on Linux the per-CPU load). A run with that raised is void as a result.
- **Cells:** K = 256 and 1,024; K = 4,096 only if the smoke's rate puts it inside the budget (a five-question JEV request at K = 4,096 is about 20,700 tokens).
- **States per K:** the first N of D7's eight; N is set from the smoke (section 5), default 6.

## 4. What is computed, and the rules

- **Per state, per cell:** the ratio (JEV time) / (Clef time), each time the geometric mean of that state's request over the two passes. The headline is the geometric mean over states with a t-interval on the logs. The two passes' ratios are also printed separately; a difference between them that the reversed order did not cancel is reported as drift.
- **Growth:** time(5 questions) / time(1 question) per arm and K.
- **D8 rule, registered:** D8 is **unnecessary for the five-question shape on Clef** if Clef's 5q/1q is at most 2.0 at every K measured; **ambiguous, goes to the owner,** between 2.0 and 3.0; **still wanted** above 3.0. (The projection puts it at 1.1 to 1.95, so it predicts "unnecessary".) The rule is about Clef serving this shape; it says nothing about a dense family, which reuses a prefix already, nor about JEV, which still needs D8.
- **Validity:** every answer valid, every JEV answer on route `head` and every Clef answer on route `clef`; an invalid row is recorded, never dropped.
- **Quoted:** the measured ratios with the machine, the date and the interval. The projection is never quoted where it was wrong.

## 5. To be completed on the Mac before queueing

1. **Pull `main`** (the Clef serve route is `internal/serveapp/systemone_clef.go`) and build `goinfer-serve` for darwin/arm64 from a committed revision; record its sha256 here.
2. **Get both models onto the internal SSD**: `models-pull clef-flash` (the archive has it); JEV-9B if it is not there already. Check `free` memory and that the Metal memory guard does not interfere with a CPU load.
3. **Smoke, by day, exploratory and not quotable** (`--smoke`: K = 256, one state, one pass): confirms the machinery on the Mac and gives the Mac's per-token rate for both models. From it, compute the job's estimate: tokens per pass (above) times the rate, times 2 passes; shrink N (and drop K = 4,096) until it is at most 3 hours, as the run budget in `CLAUDE.md` requires. Record the smoke's numbers, the chosen N and Ks, and the estimate here.
4. **Commit this file with those fields filled in, then queue** (`python3 scripts/night.py add d14-clef-speed --est <min> -- env SERVE=... SERVE_SHA=... D14_STATES=<N> D14_KS=<ks> bash docs/measurements/decisions-d14-clef-speed-2026-10/run-d14.sh`). Do not start the queue; the owner does.

**A machinery smoke already ran on nobara's CPU (2026-10-03, exploratory, one state, K = 256 and 1,024, one pass, int8int8, `serve` built from `main`; NOT the Mac and not quotable as a speed).** It shows the harness works end to end on both routes with the real models: 8 of 8 requests valid, every JEV answer on route `head` and every Clef answer on route `clef`, input tokens close to the projection's (JEV 291 / 1,478 / 1,038 / 5,213, Clef 404 / 839 / 1,151 / 1,586 against projected 317 / 1,584 / 1,076 / 5,378 and 428 / 836 / 1,187 / 1,595). Its four time ratios (JEV / Clef: 0.72 and 1.69 at K = 256, 0.89 and 2.85 at K = 1,024) sit inside the section 2 bands, on one state. It also hints at what to watch: Clef's time per token was about 51 to 54 ms up to 1,151 tokens and about 62 ms at 1,586, which fits the head costing more as the sequence grows; the Mac measurement will say. The smoke took 19 minutes, not the 6 to 10 I estimated: the K = 1,024 five-question JEV request alone was 282 s. **Size the Mac job from the Mac's own smoke, not from these figures.**

Prerequisite on nobara's side: none outstanding. D13's fidelity run is independent of this one (it grades the answers, not the time).

## 6. Not in scope

The resident cells: **CUDA is now possible** (D11's follow-up, `decisions-d11-resident-hidden-2026-10-03.md`; one exploratory run of three records measured 3.2 ms per token on nobara's 2070 SUPER, ungraded for fidelity) and is not part of this Mac draft; **Metal does not implement the seam yet**, so the Mac runs the Clef route's backbone on the CPU, which is what this draft measures. Clef 27B is not measured here.

## 7. Result

*Not yet run.*
