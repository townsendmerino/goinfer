# Reasoning items on /v1/responses: live Codex round trip, 2026-10-14

**Check.** Codex should send serve's reasoning items back. The turns after the first should then reuse the previous
prompt plus the generated tokens from the KV, instead of re-prefilling.

- **Setup.** nobara-pc, RTX 2070 SUPER, driver 595.91.07. serve built from branch `responses-reasoning-items-2026-10`
  (uncommitted tree on 743ecc50; the binary and its rev are archived in the log directory). Mellum2.1-thinking int4 `.giw`
  from `~/models`, `--backend cuda --ctx 16384 --moe-cache-experts --log-requests`. Codex 0.162.1 `exec --json` with
  the I3 `read` prompt, through a capture proxy that saved every request body. Run under the timing lock (label
  `responses-reasoning-live`). One run, exploratory. Logs and bodies: `~/goinfer-bench/responses-reasoning-live-2026-10-14/`.
- **What Codex sent back.** Every reasoning item, with `content` (the `reasoning_text`), `encrypted_content`
  (`goinfer-reasoning-v1:…`) and an empty `summary`, ahead of its `function_call`, on both later requests.
- **Reuse.** serve's request log and the response's `usage.input_tokens_details.cached_tokens`:

  | request | prompt tokens | cached | completion | time to first token |
  |---|---|---|---|---|
  | 1 | 6,204 | 0 | 300 | 65.5 s |
  | 2 | 6,554 | 6,504 (= 6,204 + 300) | 319 | 0.62 s |
  | 3 | 6,949 | 6,873 (= 6,554 + 319) | 290 | 0.94 s |

  Each later turn reused the whole of the previous prompt plus every token generated. Codex answered `5` (correct) and
  exited 0, reporting `cached_input_tokens: 13377`.
- **Not compared in this run.** The before-state is the I3 night's Codex x Mellum2.1 cell, where every Codex request
  took about 66 s to its first byte.
