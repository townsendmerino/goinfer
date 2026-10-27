- **Changes behavior on macOS. The fit guard's available memory counts cached files as available** (B41). It is now the free,
  file-backed and purgeable pages (speculative read-ahead counted once, as file-backed), read through `host_statistics64` without an exec, where it was free + inactive +
  speculative + purgeable pages from `vm_stat`, which counted the active file cache as used: right after a large model ran, a load
  could be refused for memory the kernel would hand back at once. Inactive app memory no longer counts as available, so with many
  apps open it can read lower than before. Linux keeps `MemAvailable`. The night runner and the benchmark harness read the same
  number through a new `cmd/hostmem`, which also reads a process's own memory (`-pid`, `task_info` `TASK_VM_INFO`).
