- **Cost counters on Metal** (SB11, tests only; nothing new on the wire). The Metal device now counts every dispatch encoded,
  command buffer committed and host wait, and a counted token attributes each dispatch to the pipeline that ran it, with its
  allocations from the memory ledger. On llama-tiny a decode token is 46 dispatches over 11 pipelines, one command buffer and
  one wait, with no allocation, checked exactly against a committed golden; a planted extra command buffer turns it red.
  Logits are unchanged with the counters on.
