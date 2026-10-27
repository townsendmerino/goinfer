- **`fit` and `serve check` now judge resident eligibility the way the load does.** `Model.Plan` used to judge eligibility from the
  architecture alone. It now asks the load's own admission, with the model's own options. Plan's answer changes in four cases:
  - Nemotron 3 Nano is now planned resident on webgpu. Plan used to call it "not eligible", although the load builds it there.
  - A dense Nemotron-H at int8 or f32 now declines on every GPU backend unless `GOINFER_SSM_RESIDENT` is set, as the load does,
    and Metal's context fit treats it as CPU-only.
  - Per-32 activations (`ActQuantGroup` 32) now decline on metal and webgpu, as the load does.
  - Phi-3 at per-32 W8A8 on cuda is now planned resident, as the load builds it.

  A declined backend's reason quotes the admission's own sentence. How a model is placed and priced is unchanged.
