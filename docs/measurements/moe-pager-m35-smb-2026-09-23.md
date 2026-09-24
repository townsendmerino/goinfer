# S5 — real M35 through the CPU MoE pager, over SMB, 2026-09-23 (correctness + safety only)

**THIS IS NOT A BENCHMARK. Every timing in this file is VOID.** The checkpoint was read over a
Wi-Fi link (~10 MB/s) from an SMB mount of nobara-pc's NVMe `~/models`, which `docs/benchmarks.md`
§ "Model storage" forbids as a read path for any number. It is recorded because it answers two
different questions that do not depend on I/O speed: does the mechanism produce correct output at
M35 scale, and does the machine stay safe while doing it. No tok/s figure from it may be quoted.
The registered S5 rule (tok/s ratio pool vs mmap at an equal budget) is therefore **still owed** and
needs the checkpoint on local disk.

**Provenance.** MacBook (darwin/arm64, 16 GB) · goinfer built from the tree at `041abc97` + this
session's S4/S5 work · `cmd/serve`, `-backend cpu -stream-weights -weight-cache 1 -ctx 512` ·
checkpoint `qwen3.6-35b-a3b-int4.giw` (22.1 GB, 10240 experts, 15.6 GB of them paged) ·
mounted `//192.168.1.240/francis` → `/home/francis/models` (nobara's NVMe, NOT the `/srv/models`
SMR archive share) · greedy, `max_tokens 4`, prompt `The capital of France is`.

**Guards armed for both runs.** decoder's S3 in-process tripwire (+512 MB over baseline, on by
default in `goinfer-serve`) AND an external out-of-process watcher (`scripts/swap_killwatch.sh`,
kills at +2 GB swap-used over its own baseline or <2 GB free disk; it reads `sysctl vm.swapusage`
directly rather than trusting the process's own RSS, per R11(c)). Neither fired.

| | mmap mode | pool mode (`-moe-pager=pool`) |
|---|---|---|
| banner | `expert paging (mmap)` | `expert paging (pread)` |
| load | 28m 10s | 27m 01s |
| output (4 greedy tokens) | `" Paris.\n\n<think>"` | `" Paris.\n\n<think>"` — **identical** |
| max swap-used delta | +0 MB | +0 MB |
| max RSS (whole run, incl. load) | 5.85 GB | 6.26 GB |
| predicted decode rate (S4 item 5, a prior) | ~2.12 tok/s | ~2.12 tok/s |

Raw logs: `moe-pager-m35-smb-2026-09-23-{mmap,pool}-{load,killwatch}.log`.

## What this shows

- **Correct output at M35 scale in both modes, and the two agree.** A real 35B-A3B MoE decoded a
  correct, coherent answer through the pager at a 1 GB budget against 15.6 GB of experts, in both
  backing modes, with byte-identical greedy tokens.
- **The machine stayed safe.** Swap-used never rose above baseline in either run. The S3 tripwire
  and the external kill switch stayed quiet. This is the scenario the 2026-09-04/05 kernel panic
  came from; here it did not repeat. It does NOT show the guards would have held — nothing tripped,
  so they were not exercised.
- **The S4 item 5 gate did not refuse** (predicted 2.12 tok/s, above the 2.0 floor).

## What this does NOT show (read before citing)

- **No speed.** Void, as above. In particular it does not settle the S5 ship rule (pool ≥ 0.9× mmap
  tok/s) or the darwin-default flip.
- **Wi-Fi throttled the I/O to ~10 MB/s**, so this run could not generate the page-fault storm the
  historical incidents were about. "Swap stayed flat" is weaker evidence than it would be on a local
  SSD, where re-fault rate is ~300× higher. The safety claim is real but conditional on that.
- **Only 4 tokens, no paged-vs-resident baseline.** M35 cannot load resident on this machine, so
  "identical" means mmap ≡ pool, not paged ≡ unpaged. The unit-level paged-≡-resident gate
  (`TestExpertPaging_bitExact`) still needs `GOINFER_MOE_GIW` and was not run.
- **The 2.12 tok/s prediction is unvalidated** — one request took ~4–6 min for 4 tokens over the
  link, which says nothing about the prior either way.

## Two findings from the run

1. **The load reads the whole file.** 27–28 min at ~10 MB/s ≈ the full 22 GB, even though
   `-stream-weights` is meant to page lazily. Something in the `.giw` load path touches every byte
   up front (candidate: full-blob validation/checksum in `giw.Read`/`LoadSerializedWeights`, or
   the tokenizer/metadata split). On local NVMe this is invisible; on any slow read path it is the
   entire load time, and it defeats the point of a lazy pager. Not investigated.
2. **`-moe-pager` overrides an exported `GOINFER_MOE_PREAD_CPU`.** `applyMoEPagerEnv` sets the env var
   explicitly from the flag (default `mmap`), by design, so the first attempt here ran mmap mode
   despite `GOINFER_MOE_PREAD_CPU=1` in the environment. Caught from the banner, not assumed.
   Pass `-moe-pager=pool`, not the env var, through `goinfer-serve`.

## Also

macOS TCC blocked reading the SMB mount until the terminal app was granted access in System
Settings → Privacy & Security; `mount_smbfs` itself succeeded first, which made it look like a
share problem.
