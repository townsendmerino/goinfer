- **Fix: on a Linux host whose transparent huge pages are set to `always`, a large CPU load no longer swaps when its KV cache is
  reserved.** The CPU KV reservation is now taken on ordinary pages, while the weights keep their huge pages. A burst of huge-page
  faults for a multi-GiB reservation beside 25 GiB of resident weights woke the kernel's high-order reclaim, which swapped the
  process even with tens of GiB free: Inkling-2-Mini through `goinfer-chat` at its default context crossed 512 MB of its own swap
  in 7 of 8 runs, and in 0 of 8 with the change, at the same decode speed (paired ratio 0.994–1.003 over four rounds). Hosts in
  `madvise` or `never` mode are unchanged.
