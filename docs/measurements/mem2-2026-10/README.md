# MEM2: the Metal towers free their scratch on grow, and close, 2026-10-21

Gates: `docs/tasks/task-memory-campaign-2026-10.md`, §4 MEM2, "MEM2 gates" (MEM2-G), registered before the code.
Machine: MacBook Pro, M1 Pro, 16 GB, macOS 26.6.2, Go 1.27.2, darwin/arm64; the owner's desktop open (load average 4.0–4.7
at the start of G4, not a timed run). G4 read its checkpoint from `~/models/qwen3-vl-2b-instruct`.

| gate | reading | result |
|---|---|---|
| MEM2-G1 grow | seven towers on their tiny fixtures: the ledger's buffer count grown stepwise equals grown once on every tower (52, 58, 134, 54, 58, 54, 53), every output byte-equal. Metal's own byte reading failed the registered 64 KiB bar on two towers (below); under the amended bar every tower is within 128 KiB (`g1-g3.log`) | **FAIL as registered (byte cross-check only); PASS as amended** |
| MEM2-G2 close | after `Close` every ledger reads (0, 0) and Metal's allocated size is back to +0 bytes on all seven (after the amendment's warm build); a second `Close` returns nil; a forward after it is an error | **PASS** (as amended: the gemma4-vision tower read +256 KiB before it) |
| MEM2-G3 unload | serve's admin unload closes the Gemma 4 vision tower, the Qwen3.5+ and GLM-OCR accelerators and the Gemma 4 audio accelerator once each; an entry that built none closes nothing | **PASS** |
| MEM2-G4 tier 1 | Qwen3-VL-2B tower, f32 (its projections upload as f16): weights 587.4 MiB; 1080p (8,160 patches) then 4K (32,160 patches) holds 2622.5 MiB of scratch, 4K alone 2622.5 MiB (+0.00%); after `Close` +0.3 and +0.2 MiB (`g4-branch.log`). Main, the same test, arm A only: 3288.0 MiB of scratch after the 4K forward (the 1080p scratch's 665.5 MiB stranded), and **3875.4 MiB still allocated after `Close`** (`g4-main.log`) | **PASS** |

**G1's byte cross-check, diagnosed (one pass, not re-run as registered).** The ledger counts, the gate's exact
instrument, matched on all seven. Metal's `CurrentAllocatedSize` did not, twice:

- gemma4-vision, +256 KiB stepwise and +256 KiB left after `Close`: a cost of the first `g4vAccel` build in the process.
  It is present right after the build, before any forward (built +4,456,448 bytes against +4,194,304 on a second
  build), and a second build in the same process neither has it nor leaves it.
- embeddinggemma2, +128 KiB stepwise with the same 134 buffers: reproducible in both orders, run twice each
  (7,389,184 ascending, 7,258,112 descending, every time), and back to +0 after `Close`. Same buffers, different bytes:
  Metal's suballocation of small buffers, not stranded scratch.

Amendment, made after the reading and stated as such: each tower is built and closed once before the readings, and G1's
byte bar between the two orders is 256 KiB (twice the suballocation effect); G2's 64 KiB after `Close` is unchanged.
The amended gate still sees the defects it exists for: the planted dropped release in the grid tower's `grow` turned the
three grid towers that grow red on the ledger (80 buffers against 54, 84 against 58, 79 against 53) and two of them on
bytes (+557,056 and +1,753,088). At real size the effect is not close to either bar (G4: 665.5 MiB stranded on main).

Planted defects, run once each on the committed tree and reverted: the release dropped from `gridVAccel.grow` (G1 red, above);
`gridVAccel.Close` left a no-op (G2 red on the four grid towers: 53–58 buffers and 39–40 objects left, 4.2–5.5 MB held, a
forward after `Close` succeeding); serve's Gemma 4 vision close dropped (G3 red: closed 0 times). The towers' matches-CPU
gates (`TestG4VMetal_matchesCPU`, `TestGridVisionMetal_matchesCPU`, `TestS3Towers_tiny`, `TestG4AMetal_matchesCPU`,
`TestEG2Metal_matchesCPU`) PASS, and `scripts/touched.py` is green (the whole `metal` package, 89 s).
