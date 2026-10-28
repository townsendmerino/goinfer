- **CI: a PR's decoder tail no longer waits on the decision census, and a test summary no longer reports a second error.**
  MM1's two census tests run in their own -race steps in `test-decoder-a` and `test-decoder-d`, so `test-decoder-tail`
  takes about 8 minutes again instead of 19 to 24 (#325). Every `go test` summary step now runs only when its test step
  ran, so a failed earlier step reads as one error instead of two. No test was dropped and no required check renamed.
