- **Closing a model with an int8 SigLIP image tower on CUDA now releases the tower's device** (SB12). Before, closing the
  encoder freed the tower's buffers but left its compiled kernels, stream and CUDA context open, both on a normal close and
  when the tower failed to set up. A server that loads and unloads such a model no longer holds them after each unload.
