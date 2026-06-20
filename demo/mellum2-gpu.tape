# Mellum2 — a 12B coding MoE running GPU-resident on an 8 GB card, in pure Go.
#
# Render from the repo root:
#   PATH="$HOME/go/bin:$HOME/.local/bin:$PATH" vhs demo/mellum2-gpu.tape
#     -> docs/assets/mellum2-gpu.gif + .mp4   (needs vhs + ttyd + ffmpeg)
#
# Prereqs: a prebuilt GPU demo binary + the prequant int4 bundle:
#   go build -tags gpu -o /tmp/goinfer-chat-gpu ./demo/chat
#   go run ./cmd/prequant -quant int4 -o ~/models/mellum2.int4.giw ~/models/mellum2-unq
#
# The Hide block warms the 7 GB .giw page cache and waits out the ~13 s int4 resident
# load, so the visible recording opens on the ready chat (the load isn't the story — a
# 12B coding model decoding RESIDENT on a consumer 8 GB card, no CUDA/Python, is).
# --temp 0 keeps the render byte-for-byte identical.

Output docs/assets/mellum2-gpu.gif
Output docs/assets/mellum2-gpu.mp4

Set Shell "bash"
Set FontSize 22
Set Width 1100
Set Height 760
Set Padding 20
Set TypingSpeed 35ms

# --- invisible setup: warm the bundle's pages, boot the 12B onto the GPU, land on the prompt ---
Hide
Type "cat ~/models/mellum2.int4.giw > /dev/null" Enter
Sleep 6s
Type "clear && /tmp/goinfer-chat-gpu --model ~/models/mellum2.int4.giw --backend webgpu --temp 0" Enter
Sleep 22s
Show

# --- the screen now shows: loaded 28-layer 12B model [backend=webgpu], ready prompt ---
Sleep 2.5s

# --- a real coding task: the 12B streams idiomatic Go, GPU-resident on 8 GB ---
Type "Write a Go function MergeSorted(a, b []int) []int that merges two sorted slices into one sorted slice." Sleep 600ms Enter
Sleep 14s        # the function streams + the dim [tok, tok/s] line lands

# --- the party trick: a model that *cannot* emit invalid JSON (constrained decoding) ---
Type "/json" Sleep 400ms Enter
Sleep 600ms
Type "Extract name, language and license as JSON: Mellum2 is a 12B Go-friendly coding model under Apache-2.0." Sleep 500ms Enter
Sleep 8s

Type "/quit" Sleep 400ms Enter
Sleep 1.5s
