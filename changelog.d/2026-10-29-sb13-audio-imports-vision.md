- **`internal/audio` may import `internal/vision`** (SB13, Francis 2026-10-29). `towers-standalone` in
  `internal/layout/imports.json` gains its one exception, so the audio towers can walk the tower program type. A rule's
  new `allow_goinfer` field names the exception exactly. `TestLayout_towersStandalone` holds every other goinfer
  import from either tower forbidden.
