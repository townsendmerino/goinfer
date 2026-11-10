- **Changes behavior: on CUDA, a context the fit guard chose is trimmed to fit, as an unrequested one is.** A load without `--ctx` whose
  guard-chosen context missed free VRAM by a few MB fell back to the CPU; it now runs resident at the largest context that fits, never
  below 4096, and with several KV slots gives up positions before slots. An explicit `--ctx` is unchanged: never trimmed, refused with
  the numbers when it does not fit.
