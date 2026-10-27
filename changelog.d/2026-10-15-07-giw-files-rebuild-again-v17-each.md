- **.giw files rebuild again (v17).** Each cached sidecar is transcoded once from its source on first load; a bundle named
  with `--model` or built into a binary must be rebuilt with `cmd/prequant`. From v17 every tensor carries its encoding,
  so a new encoding no longer needs a format bump (`docs/giw-bundles.md`, "From v17").
