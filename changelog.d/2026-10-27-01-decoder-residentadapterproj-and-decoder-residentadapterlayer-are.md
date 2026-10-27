- **`decoder.ResidentAdapterProj` and `decoder.ResidentAdapterLayer` are aliases now** of the same plain-data types in
  `internal/adapter`, so a backend's kernel layer can take a LoRA delta without importing `decoder` (SB1). Every caller
  compiles unchanged; apidiff reports the two Experimental-tier names (and `ResidentAdapter.SetAdapter`, which names one)
  as a type-identity change, and `%T` and reflection print `adapter.ResidentAdapterLayer`.
