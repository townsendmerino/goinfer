- **`engine.Engine.Start` returns one `*engine.Stream`** (SB7b), with `Tokens()`, `Report()` and `Close()`, in place
  of a channel and two functions. `Report()` is nil-safe and `Close()` runs once however often it is called.
  `NewStream` and `GenerationStream` (over a `*decoder.Generation`) build one; `ReportOf` and `Reporter` are gone.
  The package is Experimental; nothing a client sees changes.
