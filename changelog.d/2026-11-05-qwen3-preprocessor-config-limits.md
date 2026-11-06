- **A Qwen3-family `preprocessor_config.json` with sizes no released model has is refused when the model loads.** It used to load,
  and then each image request either panicked (a `temporal_patch_size` past what a slice can hold), returned empty pixel values with no
  error (a `patch_size` times `merge_size` that wraps to zero), or allocated until the process was killed. The limits are
  `patch_size * merge_size` at most 1024, `temporal_patch_size` at most 4, and `longest_edge` at most the 32 MiB-pixel input limit, far
  above any released config's 32, 2 and 16,777,216.
