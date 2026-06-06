# goinfer chat — "an entire LLM in one file" demo
#
# Render from the repo root:   vhs demo.tape    ->  demo.gif
# The Hide block cd's into demo/chat/dist and warms the page cache so the
# visible boot shows the ~0.5s instant-load hook (a cold mmap reads ~2s).
# --temp 0 keeps every render byte-for-byte identical.

Output docs/assets/demo.gif
Output docs/assets/demo.mp4   # mp4 for Twitter/Bluesky (autoplays, sharper, no GIF size limit)

Set Shell "bash"
Set FontSize 22
Set Width 1100
Set Height 760
Set Padding 20
Set TypingSpeed 55ms

# --- invisible setup: warm the binary's pages so the boot is fast + clean ---
Hide
Type "cd demo/chat/dist && cat goinfer-chat-0.5b-darwin-arm64 > /dev/null && clear" Enter
Sleep 2s
Show

# --- the hook: an entire LLM boots from one static binary in ~0.5s ---
Type "./goinfer-chat-0.5b-darwin-arm64 --temp 0" Sleep 600ms Enter
Sleep 2.5s        # "mapping weights…" then "loaded … in ~0.47s" — the instant-boot moment

# --- steer it (shows the live system-prompt control) ---
Type "/system Reply with only a fenced Go code block, no prose." Sleep 400ms Enter
Sleep 1.5s

# --- a real prompt, streamed live ---
Type "write a Go function that reverses a slice of ints in place" Sleep 500ms Enter
Sleep 7s          # the function streams out + the dim [tok/s] line lands

# --- the party trick: guaranteed-valid JSON via constrained decoding ---
Type "/demo json" Sleep 500ms Enter
Sleep 7s

# --- clean exit ---
Type "/quit" Sleep 400ms Enter
Sleep 1.5s
