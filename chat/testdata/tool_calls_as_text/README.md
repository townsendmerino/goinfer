The three replies of tack's E1 pilot job 1 (2026-10-17, Qwen2.5-7B q4_k_m, opencode over /v1/chat/completions) that came back as text, verbatim from opencode's `text` parts.
`pilot_task01_raw_newlines.txt`: a `<tool_call>` block whose `prompt` string holds raw newlines. `pilot_task02_function_key.txt`: a block that names the tool under `"function"`.
`pilot_task05_garbled_markers.txt`: prose, then a bare call between two non-marker tokens where `<tool_call>` and `</tool_call>` belong; it must stay text.
