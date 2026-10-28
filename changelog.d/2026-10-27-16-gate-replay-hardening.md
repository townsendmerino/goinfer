- **`gate quick -replay` and `gate identity -replay` refuse a record that has lost what its run had** (SB12, Gemini's
  review of #373). A quick test job with no stream and no error, an identity dump whose logits file is missing or of the
  wrong size, and a dump present at the run but gone from the record each exit 2 instead of grading. A live identity run
  now reads a corrupt logits file as no result rather than as identical. A `-full` quick run's record line is replayed.
