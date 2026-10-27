- **`scripts/touched.py` runs every package's tests every time again** (`-count=1`). The test-cache classification that let
  Go replay 14 packages' results is removed: it saved about 9 s on a 133 s re-run, and each review round found a new way a
  hidden input could slip past it and replay a stale pass.
