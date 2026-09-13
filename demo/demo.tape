# goinfer chat — "an entire LLM in one file" demo (1.5B tier)
#
# Render from the repo root:   vhs demo.tape    ->  docs/assets/demo.gif + .mp4
#   (needs `vhs` on PATH + `ttyd` + `ffmpeg`)
# The Hide block cd's into demo/chat/dist and warms the page cache so the
# visible boot shows the instant-load hook (a cold mmap reads slower).
# --temp 0 keeps every render byte-for-byte identical.

Output docs/assets/demo.gif
Output docs/assets/demo.mp4   # mp4 for Twitter/Bluesky (autoplays, sharper, no GIF size limit)

Set Shell "bash"
Set FontSize 22
Set Width 1100
Set Height 760
Set Padding 20
Set TypingSpeed 35ms

# --- invisible setup: warm the binary's pages so the boot is fast + clean ---
Hide
Type "cd demo/chat/dist && cat goinfer-chat-1.5b-linux-amd64 > /dev/null && clear" Enter
Sleep 2s
Show

# --- the hook: an entire 1.5B LLM boots from one static binary in ~0.4s ---
Type "./goinfer-chat-1.5b-linux-amd64 --temp 0" Sleep 600ms Enter
Sleep 2.5s        # "loaded 28-layer model … in ~0.4s" — the instant-boot moment

# --- steer it live (the system-prompt control) ---
Type "/system Reply with a single Go code block. No prose." Sleep 400ms Enter
Sleep 1.2s

# --- a real prompt: a clean generic, streamed live (the 1.5B writes idiomatic Go) ---
Type "Write a generic Dedup[T comparable](xs []T) []T that removes duplicates, preserving first-seen order." Sleep 500ms Enter
Sleep 16s         # the generic function streams out + the dim [tok/s] line lands

# --- the party trick: a model that *cannot* emit invalid JSON (constrained decoding) ---
Type "/reset" Sleep 300ms Enter
Sleep 500ms
Type "/json" Sleep 300ms Enter
Sleep 500ms
Type "Extract repo, version, language and license as JSON from: ken v0.4.0 — an MIT-licensed Go code-search tool." Sleep 500ms Enter
Sleep 7s

# --- clean exit ---
Type "/quit" Sleep 400ms Enter
Sleep 1.5s
