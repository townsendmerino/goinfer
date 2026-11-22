- **Faster: gpt-oss decodes 1.64x as fast on the CPU.** Each token's chosen experts now run their gate and up projections
  as one batched W4A8 call instead of one call per expert. Outputs are bit-identical; the speed rule's ten timed pairs on a
  Ryzen were all faster (#1129).
