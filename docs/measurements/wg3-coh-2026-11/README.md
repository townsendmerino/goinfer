# WG3-coh: Command-R7B's position-13 gap (results)

Registered in [registration.md](registration.md) before the run. nobara-pc, RTX 2070 SUPER, driver 595.112.07, WebGPU over
Vulkan; real Command-R7B at int4 on both arms, the resident at the gate's context of 512, from nobara's local model store;
`TestResidentCaptureParityWebGPU` at `cfefb0b8d`, 20 positions of README.md's tokens, exclusive lock 2026-11-24
13:40:34-13:41:13 PST. Log: [capture-pos13.log.gz](capture-pos13.log.gz). An earlier attempt (12:22:49-12:23:15) failed
before measuring, the resident's KV cache not fitting at the test's default context (`residency KV alloc (layer 23): ...
Not enough memory left`); the registration's amendment records it.

**Reproduced.** Position 13 reads a logit cosine of 0.860009 (the gate read 0.860); every other position reads 0.979 or
above, argmax agreeing at all 20. The CPU against itself with ±1-ULP norm noise reads 0.948670 at the same position, its
own worst (`TestCPUQuantSensitivityREADME`, 2026-11-24 09:29).

**Outcome, by the registration: 4, no clear cause.** No (layer, column) meets outcome 2's test (the largest at position 13
is the MLP contribution of layer 23, 4.2× its baseline over positions 0-12 but 1.6× the median over layers), and 52
columns exceed outcome 3's 2× limit.

The shape, position 13 against each column's baseline (the max over positions 0-12):

| layers | ctx | attn-contrib | mlp (post-attn) |
|---|---|---|---|
| 0-11 | 0.0-1.3× | 0.0-1.3× | 0.0-1.3× |
| 12-14 | 2.1-2.7× | 2.0-2.9× | 1.3-2.2× |
| 15-26 | 2.2-3.9× | 2.4-4.0× | 2.9-4.2× |
| 27-31 | 1.7-2.3× | 1.7-2.7× | 2.7-4.1× |

The excess starts at layer 12 in attention (context and contribution together), the MLP following from layer 14, and
holds through the last layer; no layer stands out from its neighbours. **The instrument's limit:** each layer starts from
the GPU's own residual, but its CPU sublayer outputs are the CPU's own forward, so a column carries the CPU-side input
difference as well as that layer's own error; the columns are not isolated per layer. Telling a WebGPU defect that opens
at layer 12 from int4/int8 rounding amplified at a sensitive token needs a capture that runs the CPU layer on the GPU's
input. #1255 stays unarmed; this goes to Francis.
