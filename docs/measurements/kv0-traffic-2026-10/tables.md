
### Configuration `qwen3-4b-20k`: qwen3-4b, `-ctx 20480`

| harness | session | task | requests | main-thread turns | threads | thread switches | first request (tokens) | last request | strictly extends its thread's previous request | other requests | overlapping another |
|---|---|---|---|---|---|---|---|---|---|---|---|
| opencode | r1 | chain | 10 | 9 | 2 | 1 | 7,216 | 11,536 | 8 of 8 | 1 | 2 |
| opencode | r2 | chain | 10 | 9 | 2 | 1 | 7,216 | 11,817 | 8 of 8 | 1 | 2 |
| opencode | r1 | fix | 24 | 22 | 3 | 3 | 7,170 | 7,996 | 20 of 21 | 2 | 2 |
| opencode | r2 | fix | 4 | 3 | 2 | 1 | 7,170 | 8,726 | 2 of 2 | 1 | 2 |
| opencode | r1 | edit | 4 | 3 | 2 | 1 | 7,167 | 8,911 | 2 of 2 | 1 | 2 |
| codex | r1 | chain | 9 | 9 | 1 | 0 | 5,859 | 8,777 | 8 of 8 | 0 | 0 |
| codex | r2 | chain | 9 | 9 | 1 | 0 | 5,859 | 6,609 | 8 of 8 | 0 | 0 |
| codex | r1 | fix | 1 | 1 | 1 | 0 | 5,814 | 5,814 | 0 of 0 | 0 | 0 |
| codex | r2 | fix | 2 | 2 | 1 | 0 | 5,814 | 5,941 | 1 of 1 | 0 | 0 |
| codex | r1 | edit | 3 | 3 | 1 | 0 | 5,810 | 5,974 | 2 of 2 | 0 | 0 |
| codex | r3 | chain | 8 | 8 | 1 | 0 | 5,859 | 6,493 | 7 of 7 | 0 | 0 |

**Requests that do not strictly extend their thread's previous request** (the divergence and what sits there):

| harness | session | request | tokens | previous in thread | common prefix | first divergence at |
|---|---|---|---|---|---|---|
| opencode | r1 fix | 24 | 7,996 | 15,280 | 7,151 | user#1 (+3 of 12 tokens) |

**New conversations: the first main-thread request of each session after the harness's first**

| harness | session | tokens | shared with an earlier session (longest common prefix) | share | the server reused (today) | gain at B=64 (tokens) | saved at nobara1.5B (s, projection) | saved at mac1.5B (s, projection) |
|---|---|---|---|---|---|---|---|---|
| opencode | r2 chain | 7,216 | 7,216 | 100.0% | 0 | 7,168 | 1.8 | 11.7 |
| opencode | r1 fix | 7,170 | 7,151 | 99.7% | 0 | 7,104 | 1.8 | 11.6 |
| opencode | r2 fix | 7,170 | 7,170 | 100.0% | 0 | 7,168 | 1.8 | 11.7 |
| opencode | r1 edit | 7,167 | 7,152 | 99.8% | 0 | 7,104 | 1.8 | 11.6 |
| codex | r2 chain | 5,859 | 5,859 | 100.0% | 5,859* | 0 | 0.0 | 0.0 |
| codex | r1 fix | 5,814 | 5,795 | 99.7% | 5,795* | 0 | 0.0 | 0.0 |
| codex | r2 fix | 5,814 | 5,814 | 100.0% | 5,814* | 0 | 0.0 | 0.0 |
| codex | r1 edit | 5,810 | 5,795 | 99.7% | 5,795* | 0 | 0.0 | 0.0 |
| codex | r3 chain | 5,859 | 5,859 | 100.0% | 5,795* | 29 | 0.0 | 0.0 |

(* the Responses route does not report reuse; estimated as the overlap with the previous request on the server)

- opencode: 4 new conversations; shared-with-earlier-session median 7,161 of 7,170 tokens; gain at B=64 median 7,136 (min 7,104, max 7,168).
- codex: 5 new conversations; shared-with-earlier-session median 5,814 of 5,814 tokens; gain at B=64 median 0 (min 0, max 29).

**Per turn: tokens a block-sharing manager would skip beyond what the single-slot server skips today**, over every request:

| harness | requests | with gain > 0 | total gain (tokens) | median / max gain of those | prompt tokens, all requests | gain as a share of all prompt tokens |
|---|---|---|---|---|---|---|
| opencode | 52 | 9 | 37,760 | 7,104 / 7,168 | 478,646 | 7.9% |
| codex | 32 | 1 | 29 | 29 / 29 | 205,752 | 0.0% |


### Configuration `qwen3-4b-28k-cc`: qwen3-4b, `-ctx 28672`

| harness | session | task | requests | main-thread turns | threads | thread switches | first request (tokens) | last request | strictly extends its thread's previous request | other requests | overlapping another |
|---|---|---|---|---|---|---|---|---|---|---|---|
| claude-code | r1 | chain | 8 | 2 | 2 | 1 | 17,755 | 20,568 | 1 of 1 | 6 | 0 |
| claude-code | r11 | read | 9 | 2 | 4 | 5 | 17,751 | 20,261 | 1 of 1 | 7 | 6 |

**Requests that do not strictly extend their thread's previous request** (the divergence and what sits there):

| harness | session | request | tokens | previous in thread | common prefix | first divergence at |
|---|---|---|---|---|---|---|
| claude-code | r1 chain | 5 | 19,360 | 18,067 | 18,044 | user#4 (+3 of 1316 tokens) |
| claude-code | r1 chain | 6 | 18,118 | 19,360 | 15,641 | user#1 (+131 of 1004 tokens) |
| claude-code | r1 chain | 7 | 19,411 | 18,118 | 17,242 | user#4 (+132 of 2298 tokens) |
| claude-code | r1 chain | 8 | 18,109 | 19,411 | 15,708 | user#1 (+198 of 1010 tokens) |
| claude-code | r11 read | 5 | 8,430 | 8,430 | 8,430 | assistant#5 (+3 of 3 tokens) |
| claude-code | r11 read | 7 | 9,723 | 8,430 | 8,407 | user#4 (+3 of 1316 tokens) |

**New conversations: the first main-thread request of each session after the harness's first**

| harness | session | tokens | shared with an earlier session (longest common prefix) | share | the server reused (today) | gain at B=64 (tokens) | saved at nobara1.5B (s, projection) | saved at mac1.5B (s, projection) |
|---|---|---|---|---|---|---|---|---|
| claude-code | r11 read | 17,751 | 22 | 0.1% | 21* | 0 | 0.0 | 0.0 |

(* the Responses route does not report reuse; estimated as the overlap with the previous request on the server)

- claude-code: 1 new conversations; shared-with-earlier-session median 22 of 17,751 tokens; gain at B=64 median 0 (min 0, max 0).

**Per turn: tokens a block-sharing manager would skip beyond what the single-slot server skips today**, over every request:

| harness | requests | with gain > 0 | total gain (tokens) | median / max gain of those | prompt tokens, all requests | gain as a share of all prompt tokens |
|---|---|---|---|---|---|---|
| claude-code | 17 | 3 | 41,270 | 15,531 / 17,376 | 264,051 | 15.6% |

