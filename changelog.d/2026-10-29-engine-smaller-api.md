- **The `engine` package exports less** (SB7b). `engine.Model`, `BlockDrafter` and the helpers `HandleOf`,
  `DrafterOf`, `Call`, `IsNil`, `AsDecoder`, `ReportNow` and `LogprobsOf` are gone: the serving shell's model handle is
  internal again. `Decoder.Model` is a `*decoder.Model` and `Decoder.Drafter` a `*decoder.BlockSpec`, and `engine`'s
  copies of `SampleInfo`, `TokenLogprob` and `KVScope` give way to `decoder`'s (`Report.Logprobs`, `Request.Scope`).
  The package is Experimental; nothing a client sees changes.
