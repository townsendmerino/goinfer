- **`engine.Scripted`, a fake engine** (SB7b). It streams fixed token ids within each request's budget and keeps no
  report, so a serving shell can be tested with no model. `serve`'s own scripted test seam now uses it.
