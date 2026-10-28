- **Metal cost counters count at the commit** (SB11, tests only). The device now counts a command buffer's dispatches
  when the buffer is committed, so a token that commits a buffer encoded before counting began reports those dispatches as
  unattributed instead of missing them. A buffer allocated and released within a token is still counted as one allocation.
  Decode outputs are unchanged.
