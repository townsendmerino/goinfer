- The host fit guard prices a CUDA-resident load's KV as VRAM, not host memory, and prices the build's packed copy of the model where there is no source file. A 7B at a 32K context is no longer priced at 11.9 GiB of host against 8.5 GiB measured; the guard's total is within 2% of the measured peak on the 7B and 8% on phi3.

