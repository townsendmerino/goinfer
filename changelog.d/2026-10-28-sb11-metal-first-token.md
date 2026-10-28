- **Metal cost counters attribute the first counted token** (SB11, tests only). Counting one token after uncounted decode
  now attributes every dispatch to its pipeline. Before, the buffer the previous token had already encoded was reported as
  unattributed. Nothing extra is recorded while counting is off, and decode outputs are unchanged.
