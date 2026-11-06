- **A Qwen VL or GLM-OCR `preprocessor_config.json` with sizes no released model has is refused when the model loads.** It used to
  load, and then each image request either panicked (a `temporal_patch_size` past what a slice can hold), returned empty pixel values
  with no error (a `patch_size` times `merge_size` that wraps to zero), or allocated until the process was killed. `temporal_patch_size`
  may be at most 2 and the pixel budget (`max_pixels`, or `size.longest_edge`) at most 16,777,216, the largest any released
  Qwen2.5-VL, Qwen3-VL, Qwen3.5, Qwen3.6 or GLM-OCR config sets, and `patch_size * merge_size` at most 1024 (released configs use 28
  and 32). Qwen2.5-VL's loader, which had no limits, now checks the same ones.
