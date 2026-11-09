- **`fit` prices a Metal load through the sidecar it builds.** A `.gguf` or a safetensors directory with no fresh metal sidecar is
  loaded on Metal by building that sidecar first, and `fit`'s Metal line now prices it as the sidecar holds it: canonical int4,
  without the CPU's row4 repack, and a host copy the bundle maps. It priced the source itself before, which on an int4 model could
  double the weights and add a heap copy of them. A directory's existing sidecar is now read too. The line says when it prices a
  sidecar the load has not built yet.
