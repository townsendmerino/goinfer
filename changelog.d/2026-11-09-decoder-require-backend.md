- **`decoder.Options.RequireBackend` makes a backend's decline a load error.** When `Backend` names a backend other than
  the CPU and the load builds no resident runner on it (not linked, no device, or the model declined), `Load` returns an
  error that `errors.Is` matches against the new `decoder.ErrBackendDeclined`, with the reason, instead of running on the
  CPU: `serve -require-backend`'s resident half, for library callers. Experimental tier.
