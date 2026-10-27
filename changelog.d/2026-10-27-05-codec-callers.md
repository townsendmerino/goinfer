- **`goinfer-serve`, `goinfer-chat` and the decide tool encode prompts through `internal/codec`** (SB8, C2). The text,
  tool, overflow and vision routes, goinfer-chat's text, batch and image paths, and the decide tool's chat encoding call
  the codec, where each composed template detection, rendering, the image splice and encoding itself. Same calls, same
  order: no prompt or token changes. The census's run through the codec is what holds that.
