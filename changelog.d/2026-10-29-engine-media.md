- **Image and audio turns run through `engine.Engine`** (SB7b). `engine.Request.Media` carries an image or audio
  input (`engine.Media`: its `MediaKind`, the placeholder spans, the lazy features and Qwen's grid arguments), and
  `engine.Local` starts it on the decoder generator its kind names. `engine.Report` gains the image-prefill fields
  `ImgPrefillResident`, `ImgPrefillDecline` and `DecodeResident`. The serving shell's vision and audio routes start
  on the engine instead of calling the model, so a fake engine can serve them. No behavior changes.
