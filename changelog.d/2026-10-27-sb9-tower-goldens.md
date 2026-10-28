- **Per-tower goldens** (SB9, step S, tests only). Every image and audio tower's output on its fixed inputs is now compared
  bit for bit with a committed golden on the CPU, CUDA and WebGPU. Each golden names what its bits depend on (the CPU's
  kernel set, the GPU and its driver, the WebGPU adapter), and a device golden from another driver fails until it is
  deliberately re-recorded. A planted one-ulp change in any tower turns exactly that tower red. On nobara every GPU tower
  agrees with the CPU at cosine 1.0. No tower's output changed.
