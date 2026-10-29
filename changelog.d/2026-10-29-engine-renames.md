- **`engine.Decoder` is now `engine.Local`** (SB7b), with its fields `Spec` → `Speculative` and `Adapter` →
  `AdapterBound`, and `engine.Request.PlainConstrained` is now `NoSpeculation`, so each name says what it does. The
  behavior is unchanged. The package is Experimental; nothing a client sees changes.
