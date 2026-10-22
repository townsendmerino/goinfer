# MEM4(b): the towers drop their host copy after a device tower attaches (D11 (a)), 2026-10-22

Gates: `docs/tasks/task-memory-campaign-2026-10.md`, §4 MEM4, "MEM4(b) gates" (TWR-G), registered before the code.
Machine: MacBook Pro, M1 Pro, 16 GB, macOS 26.6.2, Go 1.27.2, darwin/arm64. aikit v1.72.0 (tagged 2026-10-22 on 4b3c37d8;
the API is aikit #7). The tier 1 checkpoint is `~/models/qwen3-vl-2b-instruct`, on local disk.

| gate | reading | result |
|---|---|---|
| TWR-G1 release (aikit) | on each tower's tiny fixture (aikit's Qwen3.5, GLM-OCR, Qwen2.5-VL; Gemma 4's from goinfer's testdata): the release freed 100–106% of the blocks' bytes (234,752; 1,323,008; 84,736; 83,072), the forward after it byte-equal to the one before, the export hashing the same; a `LoadQwen3VisionEncoderFrom` tower refuses and keeps working. The plant (`ensureBlocks` not reloading) red on all four. aikit `preflight` PASS 10/10 | **PASS** |
| TWR-G2 serve | the Qwen3.5+ and GLM-OCR towers release once when their device tower starts; after a device memory error, `deviceFallback`'s CPU path reloads the blocks and returns features byte-equal to a never-released encoder's; the Gemma 4 and Qwen2.5-VL startup towers release once on an attached device and not on the CPU (`twr-g2.log`). Plants, each red: the Qwen3.5 release dropped (0 releases); the Gemma 4 release dropped (0 releases) | **PASS** |
| TWR-G3 tier 1 | Qwen3-VL-2B tower on Metal, float32: the blocks are 1,153.2 MiB; the Go heap after the attach 1,553.2 MiB, after the release 400.0 MiB, **−1,153.2 MiB (100.0% of the blocks)**; bar ≥ 80% (`twr-g3.log`) | **PASS** |

The release runs once per tower, after its device tower is built and before any request runs it (`internal/serveapp`:
`releaseTowerBlocks`). SigLIP (Gemma 3) already loaded head-only; Pixtral and LFM2-VL have no device tower, so nothing changes for them.
