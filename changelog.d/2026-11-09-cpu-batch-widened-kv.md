- **CPU batched decode reads f16 and int8 KV caches.** With `--cpu-batch`, concurrent CPU generations on an `--kv f16` or
  `--kv i8` cache now join their decode tokens into one batched forward, with replies bit-identical to decoding each alone.
  Before, those tokens decoded one sequence at a time, and `--cpu-batch on` with such a cache refused the load.
