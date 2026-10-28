- **`TestBuildDemand_matchesDecoderDefault` holds CUDA's rounding report to 0 only where that is the true figure.** It
  compared every checkpoint in `testdata`, so a real model placed there by the night gate (llama3.2-1b, qwen2.5-0.5b and
  six more), whose buffers are far over the 2 MiB quantum, failed it although CUDA prices it correctly. It now compares the
  checkpoints whose every weight is under the quantum (the committed fixtures, as before) and logs the rest. Pricing is
  unchanged.
