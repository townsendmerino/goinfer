- **Fixed: `serve check`'s long-prompt row no longer fails on reasoning models.** The row asks for 8 tokens and counted only content, so a model
  that thinks first read "stream produced no tokens". Reasoning tokens now count, and the row says how many of them there were.
